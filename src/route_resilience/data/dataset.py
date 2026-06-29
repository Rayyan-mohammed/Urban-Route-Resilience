"""RoadTileDataset — the PyTorch Dataset shared by M3/M4 training and M5 eval.

Pipeline per item:
    manifest row -> load mask GeoTIFF (binary target)
                 -> load real image (image_path) OR synthesize pseudo image
                 -> [train only] apply synthetic occlusion to the IMAGE (§3.3)
                 -> albumentations augment + ImageNet normalise -> CHW tensors

Design choices:
  - Occlusion is a TRAIN-TIME augmentation on the image; the mask target stays
    complete. That is the occlusion-recall learning signal (M2).
  - Pseudo images are seeded by row index so the base image is STABLE across
    epochs (only occlusion/augmentation vary) — important until real imagery
    (image_path) arrives, at which point this synthesis is bypassed.
  - num_workers=0 friendly: rasterio handles are opened/closed per __getitem__.
"""

from __future__ import annotations

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")  # no network version check

import albumentations as A
import numpy as np
import rasterio
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from torch.utils.data import Dataset

from .occlusion import apply_from_config
from .synth_image import synthesize_image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(train: bool) -> A.Compose:
    """Augmentation + normalisation. Geometric aug is safe for road masks."""
    augs: list = []
    if train:
        augs += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
        ]
    augs += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(augs)


class RoadTileDataset(Dataset):
    def __init__(
        self,
        manifest,
        split: str,
        cfg: DictConfig,
        *,
        occlude: bool | None = None,
    ):
        self.df = manifest[manifest["split"] == split].reset_index(drop=True)
        self.cfg = cfg
        self.split = split
        # Occlude on train by default (if enabled); never on val/test unless asked.
        if occlude is None:
            occlude = split == "train" and bool(cfg.data.occlusion.enabled)
        self.occlude = occlude
        self.tf = build_transforms(train=(split == "train"))

    def __len__(self) -> int:
        return len(self.df)

    def _load_mask(self, row) -> np.ndarray:
        with rasterio.open(row["mask_path"]) as ds:
            m = ds.read(1)
        return (m > 0).astype(np.uint8)

    def _load_image(self, row, mask: np.ndarray, idx: int) -> np.ndarray:
        img_path = row.get("image_path", "")
        if isinstance(img_path, str) and img_path:
            with rasterio.open(img_path) as ds:
                img = ds.read([1, 2, 3]).transpose(1, 2, 0)  # CHW -> HWC
            return img.astype(np.uint8)
        # No real imagery yet: stable pseudo image seeded by index.
        return synthesize_image(mask, seed=1000 + idx)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        mask = self._load_mask(row)
        img = self._load_image(row, mask, idx)
        if self.occlude:
            img = apply_from_config(img, mask, self.cfg.data.occlusion).image
        out = self.tf(image=img, mask=mask)
        image = out["image"].float()                  # (3,H,W)
        target = out["mask"].unsqueeze(0).float()     # (1,H,W)
        return image, target
