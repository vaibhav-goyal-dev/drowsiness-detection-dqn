"""
model.py
--------
EfficientNet-B2 binary classifier for drowsiness detection.
Stronger than MobileNetV3 — significantly better accuracy on image datasets.

Architecture:
  EfficientNet-B2 backbone (pretrained ImageNet)
      ↓ 1408-d features
  Custom head: Linear(512) → BN → GELU → Dropout → Linear(256) → Linear(1)
      ↓
  Sigmoid → P(DROWSY)
"""

import torch
import torch.nn as nn
import timm
import os
import yaml
from typing import Optional


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary Focal Loss with label smoothing.
    Aggressively down-weights easy negatives so training focuses on hard cases.
    gamma=2.5 is stronger than the default 2.0 — better for imbalanced data.
    """

    def __init__(self, gamma: float = 2.5, alpha: float = 0.75,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.gamma           = gamma
        self.alpha           = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing to targets
        targets_smooth = targets.float() * (1 - self.label_smoothing) \
                         + 0.5 * self.label_smoothing

        bce  = nn.functional.binary_cross_entropy_with_logits(
            logits, targets_smooth, reduction='none')
        prob = torch.sigmoid(logits)
        p_t  = prob * targets + (1 - prob) * (1 - targets)
        a_t  = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = a_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


# ── Model ─────────────────────────────────────────────────────────────────────

class DrowsinessNet(nn.Module):
    """
    EfficientNet-B2 fine-tuned for binary drowsiness classification.

    Why EfficientNet-B2 over MobileNetV3:
      - Compound scaling: balances depth/width/resolution simultaneously
      - ~4% higher accuracy on ImageNet at similar speed
      - Squeeze-and-Excitation blocks improve feature recalibration
      - Better gradient flow via MBConv blocks

    Input:  [B, 3, 224, 224]
    Output: [B, 1] logit
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.4):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            'efficientnet_b2',
            pretrained   = pretrained,
            num_classes  = 0,        # remove original classifier
            global_pool  = 'avg'     # global average pooling
        )
        feat_dim = self.backbone.num_features   # 1408 for EfficientNet-B2

        # ── Classification head ───────────────────────────────────────────────
        # Deeper head captures more complex drowsiness patterns
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, 1)        # single logit
        )
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logit [B, 1]."""
        features = self.backbone(x)
        return self.head(features)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns P(DROWSY) in [0, 1]."""
        self.eval()
        return torch.sigmoid(self.forward(x))

    def freeze_backbone(self):
        """Freeze backbone weights for warm-up phase."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze for full fine-tuning — use lower LR after unfreezing."""
        for p in self.backbone.parameters():
            p.requires_grad = True


# ── MixUp augmentation ────────────────────────────────────────────────────────

def mixup_batch(images: torch.Tensor, labels: torch.Tensor,
                alpha: float = 0.2):
    """
    MixUp: blend two random training samples.
    Forces model to learn linear interpolations → better generalization.
    Returns (mixed_images, label_a, label_b, lambda).
    """
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam    = float(torch.distributions.Beta(alpha, alpha).sample())
    batch  = images.size(0)
    index  = torch.randperm(batch, device=images.device)
    mixed  = lam * images + (1 - lam) * images[index]
    return mixed, labels, labels[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixed loss for MixUp."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Factory / checkpoint helpers ──────────────────────────────────────────────

def build_model(config: dict, device: torch.device) -> DrowsinessNet:
    model = DrowsinessNet(
        pretrained = config['model']['pretrained'],
        dropout    = config['model']['dropout']
    )
    return model.to(device)


def save_checkpoint(model: DrowsinessNet, optimizer, scheduler,
                    epoch: int, val_loss: float, val_acc: float,
                    path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch':      epoch,
        'model':      model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'scheduler':  scheduler.state_dict() if scheduler else None,
        'val_loss':   val_loss,
        'val_acc':    val_acc,
    }, path)


def load_checkpoint(path: str, device: torch.device,
                    config: Optional[dict] = None) -> DrowsinessNet:
    if config is None:
        with open('config.yaml', encoding="utf-8") as f:
            config = yaml.safe_load(f)
    model = build_model(config, device)
    ckpt  = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"[Model] Loaded epoch={ckpt['epoch']}  "
          f"val_acc={ckpt['val_acc']:.4f}  val_loss={ckpt['val_loss']:.4f}")
    return model
