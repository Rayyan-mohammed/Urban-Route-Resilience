"""Segmentation losses.

M3 baseline: Dice + Focal — the conventional region-overlap objective. Dice
handles the heavy road/background class imbalance; Focal focuses on hard pixels.

NOTE: this is deliberately a PURELY REGION-based loss. It has no notion of
connectivity — a 1-pixel gap barely changes Dice. That blindness is exactly what
M4's clDice (topology-aware) loss will fix, and the contrast is the USP story.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class DiceFocalLoss(nn.Module):
    def __init__(self, dice_w: float = 0.5, focal_w: float = 0.5):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.focal = smp.losses.FocalLoss(mode="binary")
        self.dice_w = float(dice_w)
        self.focal_w = float(focal_w)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_w * self.dice(logits, target) + self.focal_w * self.focal(logits, target)
