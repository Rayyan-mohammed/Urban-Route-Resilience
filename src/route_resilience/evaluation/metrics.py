"""Pixel-wise segmentation metrics (M3).

A streaming accumulator over batches that computes IoU, Dice, precision, recall
from confusion counts (micro-averaged over all pixels seen). Dependency-light and
exact — no per-batch averaging artefacts.

M5 will ADD topology metrics (clDice, occlusion-recall, connectivity ratio, APLS)
in this same module; these pixel metrics remain the baseline yardstick.
"""

from __future__ import annotations

import torch


class SegMetrics:
    """Accumulate TP/FP/FN over batches; compute IoU/Dice/precision/recall."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._tp = 0.0
        self._fp = 0.0
        self._fn = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = torch.sigmoid(logits) > self.threshold
        t = target.bool()
        self._tp += (pred & t).sum().item()
        self._fp += (pred & ~t).sum().item()
        self._fn += (~pred & t).sum().item()

    def compute(self) -> dict[str, float]:
        tp, fp, fn = self._tp, self._fp, self._fn
        eps = 1e-7
        return {
            "iou": tp / (tp + fp + fn + eps),
            "dice": 2 * tp / (2 * tp + fp + fn + eps),
            "precision": tp / (tp + fp + eps),
            "recall": tp / (tp + fn + eps),
        }
