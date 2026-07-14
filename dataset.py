"""
dataset.py
----------
Dataset and DataLoader factory for binary drowsiness classification.
Uses albumentations for much stronger augmentation than torchvision.

Strong augmentation rationale:
  - Drivers appear in varying lighting (day/night/tunnel)
  - Camera angle varies per vehicle
  - Glasses, beard, makeup cause appearance variation
  - Motion blur from vehicle vibration
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional


# ── Albumentations transforms ─────────────────────────────────────────────────

def get_train_transform(cfg: dict) -> A.Compose:
    """
    Aggressive training augmentation.
    Each augmentation simulates a real-world variation drivers experience.
    """
    aug = cfg.get('augmentation', {})
    size = cfg['dataset']['image_size']

    return A.Compose([
        A.Resize(size + 32, size + 32),
        A.RandomCrop(size, size),
        A.HorizontalFlip(p=aug.get('horizontal_flip_p', 0.5)),

        # Lighting variations (car interior has huge lighting range)
        A.RandomBrightnessContrast(
            brightness_limit = aug.get('random_brightness_limit', 0.3),
            contrast_limit   = aug.get('random_contrast_limit', 0.3),
            p=0.7
        ),
        A.HueSaturationValue(
            hue_shift_limit   = aug.get('hue_shift_limit', 10),
            sat_shift_limit   = 15,
            val_shift_limit   = 15,
            p=0.4
        ),

        # Simulates night driving / shadows from sun
        A.RandomShadow(p=aug.get('random_shadow_p', 0.2)),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),

        # Geometric variations (camera angle, head tilt)
        A.Rotate(limit=aug.get('rotate_limit', 15), p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=0, p=0.3),

        # Blur (vibration, focus issues, eyelid blur)
        A.OneOf([
            A.MotionBlur(blur_limit=aug.get('blur_limit', 5)),
            A.GaussianBlur(blur_limit=aug.get('blur_limit', 5)),
            A.MedianBlur(blur_limit=3),
        ], p=0.3),

        # Occlusion simulation (glasses, hair, hands on face)
        A.CoarseDropout(
            max_holes=6, max_height=20, max_width=20,
            min_holes=1, fill_value=0,
            p=aug.get('coarse_dropout_p', 0.3)
        ),

        # Elastic transforms (slight face deformation robustness)
        A.ElasticTransform(alpha=1, sigma=5, p=aug.get('elastic_transform_p', 0.1)),

        # Normalization
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transform(size: int = 224) -> A.Compose:
    """Clean transform — no augmentation for validation/test."""
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_inference_transform(size: int = 224):
    """For single-frame inference on numpy arrays (OpenCV frames)."""
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class DrowsinessDataset(Dataset):
    """
    Loads images from:
        root/alert/   → label 0
        root/drowsy/  → label 1

    Accepts: .jpg .jpeg .png .bmp .webp
    """

    CLASSES  = {'alert': 0, 'drowsy': 1}
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(self, root: str, transform=None):
        self.root      = root
        self.transform = transform
        self.samples   = []
        self.labels    = []
        self._load()

    def _load(self):
        for cls, label in self.CLASSES.items():
            d = os.path.join(self.root, cls)
            if not os.path.isdir(d):
                print(f"[Dataset] WARNING: {d} not found")
                continue
            for f in sorted(os.listdir(d)):
                if os.path.splitext(f)[1].lower() in self.IMG_EXTS:
                    self.samples.append((os.path.join(d, f), label))
                    self.labels.append(label)
        n_alert  = self.labels.count(0)
        n_drowsy = self.labels.count(1)
        print(f"[Dataset] {os.path.basename(self.root):8s} → "
              f"ALERT={n_alert:5d}  DROWSY={n_drowsy:5d}  "
              f"Total={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        # Load as numpy for albumentations
        img = np.array(Image.open(path).convert('RGB'))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, label

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for WeightedRandomSampler."""
        counts = np.bincount(self.labels)
        w      = 1.0 / counts
        return torch.DoubleTensor([w[l] for l in self.labels])


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloaders(config: dict):
    """Returns (train_loader, val_loader, test_loader)."""
    root       = config['dataset']['root']
    size       = config['dataset']['image_size']
    bs         = config['training']['batch_size']
    workers    = config['training']['num_workers']

    train_ds = DrowsinessDataset(os.path.join(root, 'train'),
                                  get_train_transform(config))
    val_ds   = DrowsinessDataset(os.path.join(root, 'val'),
                                  get_val_transform(size))
    test_ds  = DrowsinessDataset(os.path.join(root, 'test'),
                                  get_val_transform(size))

    # Weighted sampler balances class frequency per batch
    sampler = WeightedRandomSampler(
        weights     = train_ds.class_weights(),
        num_samples = len(train_ds),
        replacement = True
    )

    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                              num_workers=workers, pin_memory=True,
                              persistent_workers=workers > 0)
    val_loader   = DataLoader(val_ds, batch_size=bs, shuffle=False,
                              num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=bs, shuffle=False,
                              num_workers=workers, pin_memory=True)

    return train_loader, val_loader, test_loader
