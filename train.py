"""
train.py
--------
High-accuracy training loop with:
  • EfficientNet-B2 backbone
  • Focal Loss + label smoothing
  • MixUp augmentation
  • Cosine annealing with warm restarts
  • Backbone freeze warm-up → gradual unfreeze
  • Early stopping on val loss
  • Best model auto-saved

Usage:
    python train.py
    python train.py --epochs 60 --batch-size 32
    python train.py --dry-run       (2-epoch smoke test)
    python train.py --resume outputs/checkpoints/checkpoint_epoch020.pt
"""

import os
import time
import argparse
import yaml
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from model   import (DrowsinessNet, FocalLoss, build_model, save_checkpoint,
                     mixup_batch, mixup_criterion)
from dataset import build_dataloaders


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',     default='config.yaml')
    p.add_argument('--epochs',     type=int,   default=None)
    p.add_argument('--batch-size', type=int,   default=None)
    p.add_argument('--lr',         type=float, default=None)
    p.add_argument('--dry-run',    action='store_true')
    p.add_argument('--resume',     default=None)
    return p.parse_args()


# ── One epoch ─────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scaler, device, cfg, epoch):
    """Train one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    mixup_alpha = cfg['training']['mixup_alpha']
    grad_clip   = cfg['training']['grad_clip']

    total_loss = correct = total = 0

    pbar = tqdm(loader, desc=f"Ep {epoch:03d} [TRAIN]", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        # MixUp augmentation
        imgs_m, y_a, y_b, lam = mixup_batch(imgs, labels, alpha=mixup_alpha)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits = model(imgs_m).squeeze(1)
            loss   = mixup_criterion(criterion, logits, y_a, y_b, lam)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        preds       = (torch.sigmoid(logits) > 0.5).long()
        # Accuracy against the dominant label
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    """Validate. Returns (avg_loss, accuracy, recall_drowsy, precision_drowsy)."""
    model.eval()
    total_loss = correct = total = 0
    tp = fp = fn = tn = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast('cuda'):
            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        probs  = torch.sigmoid(logits)
        preds  = (probs > 0.5).long()

        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()

    acc     = correct / total
    recall  = tp / (tp + fn + 1e-8)
    prec    = tp / (tp + fp + 1e-8)
    f1      = 2 * prec * recall / (prec + recall + 1e-8)

    return total_loss / total, acc, recall, f1


# ── Main ──────────────────────────────────────────────────────────────────────

def train(config: dict, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Train] Device: {device}")
    if torch.cuda.is_available():
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")

    # CLI overrides
    if args.epochs:     config['training']['epochs']       = args.epochs
    if args.batch_size: config['training']['batch_size']   = args.batch_size
    if args.lr:         config['training']['learning_rate'] = args.lr
    if args.dry_run:    config['training']['epochs']       = 2

    epochs       = config['training']['epochs']
    patience     = config['training']['early_stopping_patience']
    freeze_until = config['model']['unfreeze_at_epoch']
    ckpt_dir     = config['paths']['checkpoints']
    best_path    = config['paths']['best_model']
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = build_dataloaders(config)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(config, device)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt['epoch'] + 1
        print(f"[Train] Resumed from epoch {ckpt['epoch']}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = FocalLoss(
        gamma          = config['training']['focal_loss_gamma'],
        alpha          = config['training']['focal_loss_alpha'],
        label_smoothing= config['training']['label_smoothing']
    )

    # ── Optimizer: separate LR for backbone vs head ───────────────────────────
    # Backbone gets 10x lower LR (pretrained features are fragile)
    backbone_params = list(model.backbone.parameters())
    head_params     = list(model.head.parameters())
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config['training']['learning_rate'] / 10},
        {'params': head_params,     'lr': config['training']['learning_rate']},
    ], weight_decay=config['training']['weight_decay'])

    # ── Scheduler: cosine warm restarts ──────────────────────────────────────
    # T_0=10: restart every 10 epochs, T_mult=2: double period each restart
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # ── Freeze backbone for warm-up ───────────────────────────────────────────
    model.freeze_backbone()
    print(f"[Train] Backbone frozen for first {freeze_until} epochs.")
    print(f"[Train] Starting {epochs} epochs...\n")

    best_val_loss = float('inf')
    no_improve    = 0
    best_recall   = 0.0
    scaler        = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        # Unfreeze backbone after warm-up
        if epoch == freeze_until + 1:
            model.unfreeze_backbone()
            print(f"\n[Train] Backbone UNFROZEN at epoch {epoch} — full fine-tuning begins.")

        tr_loss, tr_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, config, epoch)
        vl_loss, vl_acc, vl_recall, vl_f1 = val_epoch(
            model, val_loader, criterion, device)

        scheduler.step(epoch)
        elapsed = time.time() - t0

        print(f"Ep {epoch:03d}/{epochs} | "
              f"Loss {tr_loss:.4f}/{vl_loss:.4f} | "
              f"Acc {tr_acc:.4f}/{vl_acc:.4f} | "
              f"Recall(drowsy) {vl_recall:.4f} | "
              f"F1 {vl_f1:.4f} | "
              f"{elapsed:.0f}s")

        # ── Save periodic checkpoint ──────────────────────────────────────────
        if epoch % 5 == 0:
            path = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch:03d}.pt")
            save_checkpoint(model, optimizer, scheduler,
                            epoch, vl_loss, vl_acc, path)

        # ── Save best model (by val loss AND recall) ──────────────────────────
        # Prioritize recall — missing a drowsy event is worse than false alarm
        improved = (vl_loss < best_val_loss) or \
                   (vl_recall > best_recall + 0.01)

        if improved:
            best_val_loss = vl_loss
            best_recall   = vl_recall
            no_improve    = 0
            save_checkpoint(model, optimizer, scheduler,
                            epoch, vl_loss, vl_acc, best_path)
            print(f"  ✓ Best model saved  "
                  f"(recall={vl_recall:.4f}  loss={vl_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n[Train] Early stopping at epoch {epoch}.")
                break

    print(f"\n[Train] Complete. Best recall={best_recall:.4f}")
    print(f"[Train] Model saved → {best_path}")


if __name__ == '__main__':
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train(config, args)
