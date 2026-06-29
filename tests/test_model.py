"""M3 tests — model factory, loss, metrics, and Dataset item contract.

Torch-dependent, so skipped if torch isn't installed. The Dataset test is skipped
if no manifest/tiles exist yet (run scripts/build_dataset.py first).
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from route_resilience.config import load_config  # noqa: E402

CFG = ["base.yaml", "data.yaml", "model_baseline.yaml", "train.yaml"]


def _cfg(overrides=None):
    return load_config(*CFG, overrides=overrides or [])


def test_model_forward_shape():
    from route_resilience.models.baseline import build_model

    model = build_model(_cfg(overrides=["model.encoder_weights=null"]))
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 64, 64)  # 1-channel logits, same HxW


def test_dice_focal_loss_positive_and_differentiable():
    from route_resilience.models.losses import DiceFocalLoss

    loss_fn = DiceFocalLoss(0.5, 0.5)
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    target = (torch.rand(2, 1, 32, 32) > 0.5).float()
    loss = loss_fn(logits, target)
    assert loss.item() > 0
    loss.backward()
    assert logits.grad is not None


def test_metrics_perfect_and_empty():
    from route_resilience.evaluation.metrics import SegMetrics

    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    perfect_logits = torch.where(target.bool(), 10.0, -10.0)
    m = SegMetrics()
    m.update(perfect_logits, target)
    r = m.compute()
    assert r["iou"] > 0.99 and r["recall"] > 0.99 and r["precision"] > 0.99

    # All-wrong prediction -> IoU ~ 0.
    m2 = SegMetrics()
    m2.update(-perfect_logits, target)
    assert m2.compute()["iou"] < 0.01


@pytest.mark.skipif(
    not Path("data/processed/manifest.csv").exists(),
    reason="dataset not built; run scripts/build_dataset.py",
)
def test_dataset_item_contract():
    from route_resilience.data.build import read_manifest
    from route_resilience.data.dataset import RoadTileDataset
    from route_resilience.paths import PROCESSED

    cfg = _cfg()
    manifest = read_manifest(PROCESSED / "manifest.csv")
    ds = RoadTileDataset(manifest, "train", cfg)
    assert len(ds) > 0
    img, target = ds[0]
    assert img.shape[0] == 3 and img.ndim == 3       # (3,H,W)
    assert target.shape[0] == 1 and target.ndim == 3  # (1,H,W)
    assert img.shape[1:] == target.shape[1:]          # aligned
    assert img.dtype == torch.float32
    assert set(torch.unique(target).tolist()) <= {0.0, 1.0}  # binary target
