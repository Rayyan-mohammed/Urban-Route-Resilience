"""Pixel-wise segmentation metrics (M3).

A streaming accumulator over batches that computes IoU, Dice, precision, recall
from confusion counts (micro-averaged over all pixels seen). Dependency-light and
exact — no per-batch averaging artefacts.

M5 will ADD topology metrics (clDice, occlusion-recall, connectivity ratio, APLS)
in this same module; these pixel metrics remain the baseline yardstick.
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.measure import label
from skimage.morphology import skeletonize

_EPS = 1e-7


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


# --------------------- topology / connectivity metrics --------------------
# These operate on 2D numpy BOOL masks (one tile) — the M5 evaluation metrics.
def cldice_score(pred: np.ndarray, true: np.ndarray) -> float:
    """clDice metric (Shit et al.) on hard skeletons — connectivity quality.

    Harmonic mean of topology-precision (pred skeleton lies on true mask) and
    topology-sensitivity (true skeleton is covered by pred). 1.0 = perfect.
    """
    pred = pred.astype(bool)
    true = true.astype(bool)
    if not pred.any() or not true.any():
        return 0.0
    sp = skeletonize(pred)
    st = skeletonize(true)
    tprec = (sp & true).sum() / (sp.sum() + _EPS)
    tsens = (st & pred).sum() / (st.sum() + _EPS)
    if tprec + tsens == 0:
        return 0.0
    return float(2 * tprec * tsens / (tprec + tsens))


def occlusion_recall_counts(pred: np.ndarray, true: np.ndarray, occ_mask: np.ndarray):
    """(recovered, hidden) counts on OCCLUDED road pixels — the headline signal.

    hidden = true road that was occluded; recovered = hidden road the model still
    predicted. Return raw counts so callers can micro-average across tiles.
    """
    hidden = true.astype(bool) & occ_mask.astype(bool)
    recovered = int((pred.astype(bool) & hidden).sum())
    return recovered, int(hidden.sum())


def n_components(mask: np.ndarray) -> int:
    """Number of connected road components (8-connectivity)."""
    return int(label(mask.astype(bool), connectivity=2).max())


def connectivity_ratio(pred: np.ndarray, true: np.ndarray) -> float:
    """n_components(true) / n_components(pred).

    1.0 = pred is as fragmented as ground truth (ideal). <1 = pred is OVER-
    fragmented (broken roads — the failure mode clDice training prevents).
    """
    ncc_p = n_components(pred)
    if ncc_p == 0:
        return 0.0
    return float(n_components(true) / ncc_p)
