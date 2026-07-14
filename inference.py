import cv2
import torch
import numpy as np
import yaml
import os
import time
import argparse
from torchvision import transforms
from PIL import Image

from feature_extractor import FeatureExtractor
from model             import load_checkpoint
from rl_agent          import DQNAgent

# Audio configuration removed, using native winsound.

# ── Transform for inference ───────────────────────────────────────────────────

def _transform(size: int = 224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ── EMA smoother ──────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self.v     = 0.5

    def update(self, x: float) -> float:
        self.v = self.alpha * x + (1 - self.alpha) * self.v
        return self.v


# ── Audio ─────────────────────────────────────────────────────────────────────

class Alarm:
    def __init__(self, cooldown: float = 4.0):
        self.cooldown = cooldown  # Kept for signature compatibility
        self.playing  = False

    def trigger(self):
        if self.playing:
            return
        try:
            import winsound
            # SystemHand is the Windows Critical Error sound. 
            # SND_LOOP makes it play continuously until stopped.
            winsound.PlaySound("SystemHand", winsound.SND_ALIAS | winsound.SND_ASYNC | winsound.SND_LOOP)
            self.playing = True
        except Exception as e:
            print(f"[Audio] Error: {e}")

    def stop(self):
        if not self.playing:
            return
        try:
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
            self.playing = False
        except Exception:
            pass


# ── Display ───────────────────────────────────────────────────────────────────

def draw(frame: np.ndarray, is_drowsy: bool, p_drowsy: float,
         ear: float, mar: float, perclos: float,
         consecutive: int, fps: float) -> np.ndarray:
    """Draw status overlay on frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Border
    color = (0, 0, 220) if is_drowsy else (0, 210, 0)
    cv2.rectangle(out, (0, 0), (w-1, h-1), color, 8)

    # Top bar background
    ov = out.copy()
    cv2.rectangle(ov, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, out, 0.45, 0, out)

    # Feature values
    cv2.putText(out, f"EAR:{ear:.3f}  MAR:{mar:.3f}  "
                     f"PERCLOS:{perclos:.2f}  FPS:{fps:.0f}",
                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(out, f"P(drowsy):{p_drowsy:.3f}  "
                     f"Consecutive:{consecutive}",
                (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (180, 180, 180), 1, cv2.LINE_AA)

    # Status label
    label = "DROWSY" if is_drowsy else "ALERT"
    lc    = (0, 0, 255) if is_drowsy else (0, 230, 0)
    fs    = 2.2
    tk    = 3
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, tk)
    lx = 20
    ly = 130
    cv2.putText(out, label, (lx+2, ly+2),
                cv2.FONT_HERSHEY_DUPLEX, fs, (0,0,0), tk+2, cv2.LINE_AA)
    cv2.putText(out, label, (lx, ly),
                cv2.FONT_HERSHEY_DUPLEX, fs, lc, tk, cv2.LINE_AA)

    # P(drowsy) bar
    by = h - 18
    bx0, bx1 = 20, w - 20
    bf = int(bx0 + p_drowsy * (bx1 - bx0))
    cv2.rectangle(out, (bx0, by-6), (bx1, by+5), (50,50,50), -1)
    cv2.rectangle(out, (bx0, by-6), (bf,  by+5), lc, -1)

    return out


# ── Rule-based fallback ───────────────────────────────────────────────────────

def rule_prob(ear: float, mar: float, perclos: float) -> float:
    """
    Accurate rule-based P(drowsy) using validated EAR/MAR thresholds.
    Used when no CNN checkpoint is available.
    """
    score = 0.0
    # EAR thresholds (validated in literature)
    if ear < 0.20:  score += 0.60   # eyes clearly closed
    elif ear < 0.25: score += 0.35  # eyes partially closed
    elif ear < 0.28: score += 0.10  # eyes slightly drooping
    # MAR thresholds (Calibrated for inner-lip tracking)
    if mar > 0.50:   score += 0.15  # clear yawn
    elif mar > 0.35: score += 0.05  # slight yawn
    # PERCLOS
    if perclos > 0.5: score += 0.15
    elif perclos > 0.3: score += 0.08
    return min(1.0, score)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(config: dict, source=0, checkpoint: str = None,
        rule_only: bool = False, save_output: str = None):

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size  = config['model']['input_size']
    tf        = _transform(img_size)

    # ── Load CNN ──────────────────────────────────────────────────────────────
    model = None
    if not rule_only:
        ckpt = checkpoint or config['paths']['best_model']
        if ckpt and os.path.exists(ckpt):
            model = load_checkpoint(ckpt, device, config)
            model.eval()
            print(f"[Inference] CNN loaded from {ckpt}")
        else:
            print("[Inference] No checkpoint — using rule-based detection.")
            rule_only = True

    # ── RL Agent ──────────────────────────────────────────────────────────────
    agent  = DQNAgent(config['rl'])
    rl_path = config['paths']['global_rl']
    agent.load(rl_path)

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config['inference']['display_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['inference']['display_height'])
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Output writer ─────────────────────────────────────────────────────────
    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_output, fourcc, 20, (actual_w, actual_h))

    # ── Feature extractor ─────────────────────────────────────────────────────
    extractor = FeatureExtractor()

    # ── State ─────────────────────────────────────────────────────────────────
    ema            = EMA(alpha=config['inference']['ema_alpha'])
    alarm          = Alarm(cooldown=config['inference']['alarm_cooldown_sec'])
    threshold      = config['inference']['drowsy_threshold']
    req_consecutive = config['inference']['consecutive_frames']

    consecutive = 0       # consecutive DROWSY frames
    prev_state  = None
    prev_action = None
    frame_count = 0
    fps_time    = time.time()
    fps         = 0.0

    print("[Inference] Running. Press Q or ESC to quit. Press F to toggle fullscreen.\n")

    cv2.namedWindow("Drowsiness Detector", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Drowsiness Detector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    is_fullscreen = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            fps      = 30 / (time.time() - fps_time)
            fps_time = time.time()

        # ── Extract features ──────────────────────────────────────────────────
        feat = extractor.extract(frame)

        if not feat['face_detected']:
            cv2.putText(frame, "No face detected — look at camera",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 130, 255), 2, cv2.LINE_AA)
            cv2.imshow("Drowsiness Detector", frame)
            consecutive = 0
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('f'):
                is_fullscreen = not is_fullscreen
                cv2.setWindowProperty("Drowsiness Detector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)
            continue

        ear     = feat['EAR']
        mar     = feat['MAR']
        perclos = feat['PERCLOS']
        crop    = feat['face_crop']

        # ── CNN or rule-based P(drowsy) ───────────────────────────────────────
        if not rule_only and model is not None and crop is not None:
            try:
                img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                img_t   = tf(img_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    p_raw = torch.sigmoid(model(img_t)).item()
            except Exception:
                p_raw = rule_prob(ear, mar, perclos)
        else:
            p_raw = rule_prob(ear, mar, perclos)

        # ── Hard overrides (physics-based, very reliable) ─────────────────────
        # If EAR is definitively closed, CNN might be wrong — override it
        if ear < 0.18:
            p_raw = max(p_raw, 0.85)   # eyes definitely closed
        # Note: Yawning (MAR) hard override removed to avoid false alarms.

        # ── EMA smoothing ─────────────────────────────────────────────────────
        p_smooth = ema.update(p_raw)

        # ── Build RL state ────────────────────────────────────────────────────
        consec_norm = min(1.0, consecutive / req_consecutive)
        state = np.array([p_smooth, ear, mar, perclos, consec_norm],
                         dtype=np.float32)

        # ── RL decision ───────────────────────────────────────────────────────
        action    = agent.select_action(state, p_drowsy=p_smooth,
                                        cnn_threshold=threshold)
        is_drowsy = (action == 1)

        # ── Consecutive frames gate ───────────────────────────────────────────
        # Require N consecutive DROWSY frames before actually triggering alarm
        # This eliminates virtually all single-frame false positives
        if is_drowsy:
            consecutive += 1
        else:
            consecutive = 0

        alarm_triggered = consecutive >= req_consecutive

        # ── RL learning ───────────────────────────────────────────────────────
        reward = agent.compute_reward(action, p_drowsy=p_smooth)
        if prev_state is not None:
            agent.store(prev_state, prev_action, reward, state, False)
            agent.update()
        prev_state  = state
        prev_action = action

        # ── Audio ─────────────────────────────────────────────────────────────
        if alarm_triggered:
            alarm.trigger()
        else:
            alarm.stop()

        # ── Display ───────────────────────────────────────────────────────────
        out = draw(frame, alarm_triggered, p_smooth,
                   ear, mar, perclos, consecutive, fps)
        cv2.imshow("Drowsiness Detector", out)
        if writer:
            writer.write(out)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('f'):
            is_fullscreen = not is_fullscreen
            cv2.setWindowProperty("Drowsiness Detector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    extractor.close()

    # Save updated RL weights
    agent.save(rl_path)
    print("[Inference] Session saved.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config',      default='config.yaml')
    p.add_argument('--checkpoint',  default=None)
    p.add_argument('--source',      default=0)
    p.add_argument('--rule-only',   action='store_true')
    p.add_argument('--save-output', default=None)
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    try:
        src = int(args.source)
    except ValueError:
        src = args.source

    run(cfg, source=src, checkpoint=args.checkpoint,
        rule_only=args.rule_only, save_output=args.save_output)
