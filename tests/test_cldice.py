"""M4 tests — soft-skeleton, clDice, and the loss factory.

The headline test (`test_cldice_more_sensitive_to_break_than_dice`) verifies the
USP mechanism: introducing a small GAP in a road raises the clDice loss MORE than
it raises the region (Dice) loss. That sensitivity to connectivity is exactly why
we add clDice on top of Dice.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from route_resilience.config import load_config  # noqa: E402
from route_resilience.models.losses import (  # noqa: E402
    DiceClDiceLoss,
    DiceFocalLoss,
    SoftClDiceLoss,
    build_loss,
    soft_skeleton,
)

CFG = ["base.yaml", "data.yaml", "model_baseline.yaml", "train.yaml"]


def _line(h=64, w=64, width=3, gap=None):
    """A horizontal road band; optionally a vertical gap [c0,c1) zeroed out."""
    t = torch.zeros(1, 1, h, w)
    r0 = h // 2 - width // 2
    t[:, :, r0 : r0 + width, :] = 1.0
    if gap is not None:
        t[:, :, :, gap[0] : gap[1]] = 0.0
    return t


def _bridge(broken=False, h=64, w=64):
    """Two thick blocks joined by a THIN bridge. Removing the bridge changes
    area only slightly (small Dice impact) but severs connectivity (large clDice
    impact) — the exact regime where topology loss matters."""
    t = torch.zeros(1, 1, h, w)
    t[:, :, 24:40, 6:26] = 1.0    # left block
    t[:, :, 24:40, 38:58] = 1.0   # right block
    if not broken:
        t[:, :, 30:33, 26:38] = 1.0  # thin connecting bridge (3 x 12 px)
    return t


def test_soft_skeleton_thins_and_nonempty():
    band = _line(width=7)
    skel = soft_skeleton(band, iters=10)
    assert skel.sum() > 0                 # skeleton exists
    assert skel.sum() < band.sum()        # it is thinner than the band


def test_cldice_zero_when_perfect():
    t = _line()
    loss = SoftClDiceLoss(iters=10)(t.clone(), t)
    assert loss.item() < 0.05             # near-zero for a perfect match


def test_cldice_penalises_break():
    t = _bridge(broken=False)
    pred_full = t.clone()
    pred_broken = _bridge(broken=True)    # bridge severed
    cl_full = SoftClDiceLoss(iters=10)(pred_full, t).item()
    cl_broken = SoftClDiceLoss(iters=10)(pred_broken, t).item()
    assert cl_broken > cl_full + 0.05     # severing the bridge clearly hurts clDice


def test_cldice_more_sensitive_to_break_than_dice():
    """Severing a thin bridge should move clDice more than it moves region-Dice."""
    t = _bridge(broken=False)
    pred_full = t.clone()
    pred_broken = _bridge(broken=True)

    def soft_dice_loss(p, q):
        inter = (p * q).sum()
        return 1.0 - (2 * inter + 1.0) / (p.sum() + q.sum() + 1.0)

    dice_delta = soft_dice_loss(pred_broken, t).item() - soft_dice_loss(pred_full, t).item()
    cl = SoftClDiceLoss(iters=10)
    cldice_delta = cl(pred_broken, t).item() - cl(pred_full, t).item()
    # The topology loss reacts more strongly to the connectivity break.
    assert cldice_delta > dice_delta


def test_dice_cldice_differentiable():
    t = _line()
    logits = torch.randn(1, 1, 64, 64, requires_grad=True)
    loss = DiceClDiceLoss(alpha=0.5, iters=5)(logits, t)
    assert loss.item() > 0
    loss.backward()
    assert logits.grad is not None


def test_build_loss_factory():
    base = build_loss(load_config(*CFG))
    assert isinstance(base, DiceFocalLoss)
    ours = build_loss(load_config(*CFG, overrides=["train.loss.name=dice_cldice"]))
    assert isinstance(ours, DiceClDiceLoss)
    with pytest.raises(ValueError):
        build_loss(load_config(*CFG, overrides=["train.loss.name=bogus"]))
