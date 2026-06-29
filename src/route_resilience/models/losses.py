"""Segmentation losses + a config-driven loss factory.

Two objectives, and the contrast between them IS the USP (§8, §10):

  DiceFocalLoss      region-overlap only. A 1-pixel gap barely moves Dice, so a
                     model trained on it produces broken, UN-routable masks.

  DiceClDiceLoss     Dice (region) + soft-clDice (TOPOLOGY). clDice compares the
                     soft-skeletons of prediction and target, so a broken road is
                     heavily penalised -> the mask stays connected and routable.
                     This is what makes our extraction "connectivity-complete".

soft-clDice (Shit et al., CVPR 2021) uses a differentiable soft-skeleton built
from iterated soft morphology (min/max pooling), so the whole thing is trainable.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- baseline loss -----------------------------
class DiceFocalLoss(nn.Module):
    """Region-overlap loss (M3 baseline). No notion of connectivity."""

    def __init__(self, dice_w: float = 0.5, focal_w: float = 0.5):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.focal = smp.losses.FocalLoss(mode="binary")
        self.dice_w = float(dice_w)
        self.focal_w = float(focal_w)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_w * self.dice(logits, target) + self.focal_w * self.focal(logits, target)


# ----------------------------- soft morphology ---------------------------
def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    # erosion = min filter; min via -maxpool(-x). Separable 3x1 then 1x3.
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeleton(img: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Differentiable soft-skeleton of a probability map in [0,1] (N,1,H,W)."""
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SoftClDiceLoss(nn.Module):
    """Topology loss: 1 - clDice between soft-skeletons of pred and target."""

    def __init__(self, iters: int = 10, smooth: float = 1.0):
        super().__init__()
        self.iters = int(iters)
        self.smooth = float(smooth)

    def forward(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        skel_pred = soft_skeleton(probs, self.iters)
        skel_true = soft_skeleton(target, self.iters)
        s = self.smooth
        # topology precision: predicted skeleton lies on true mask
        tprec = (torch.sum(skel_pred * target) + s) / (torch.sum(skel_pred) + s)
        # topology sensitivity: true skeleton is covered by prediction
        tsens = (torch.sum(skel_true * probs) + s) / (torch.sum(skel_true) + s)
        cl_dice = 2.0 * tprec * tsens / (tprec + tsens)
        return 1.0 - cl_dice


class DiceClDiceLoss(nn.Module):
    """Our loss: (1-alpha)*Dice region term + alpha*soft-clDice topology term."""

    def __init__(self, alpha: float = 0.5, iters: int = 10, smooth: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.cldice = SoftClDiceLoss(iters=iters, smooth=smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        return (1.0 - self.alpha) * self.dice(logits, target) + self.alpha * self.cldice(probs, target)


# ------------------------------- factory ---------------------------------
def build_loss(cfg) -> nn.Module:
    """Select the loss from cfg.train.loss.name: 'dice_focal' | 'dice_cldice'."""
    lc = cfg.train.loss
    name = lc.get("name", "dice_focal")
    if name == "dice_focal":
        return DiceFocalLoss(lc.get("dice_w", 0.5), lc.get("focal_w", 0.5))
    if name == "dice_cldice":
        return DiceClDiceLoss(
            alpha=lc.get("cldice_alpha", 0.5),
            iters=lc.get("cldice_iters", 10),
            smooth=lc.get("cldice_smooth", 1.0),
        )
    raise ValueError(f"unknown loss name: {name!r}")
