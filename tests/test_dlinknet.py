"""Tests for the D-LinkNet baseline (§4.1, §10).

Torch-dependent, so skipped if torch isn't installed. `encoder_weights=null`
everywhere so no ImageNet download is needed offline / in CI.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from route_resilience.config import load_config  # noqa: E402

CFG = ["base.yaml", "data.yaml", "model_dlinknet.yaml", "train.yaml"]


def _cfg(overrides=None):
    return load_config(*CFG, overrides=(overrides or []) + ["model.encoder_weights=null"])


def test_dlinknet_selected_by_factory():
    from route_resilience.models.baseline import build_model
    from route_resilience.models.dlinknet import DLinkNet

    model = build_model(_cfg())
    assert isinstance(model, DLinkNet)


def test_dlinknet_forward_shape():
    from route_resilience.models.baseline import build_model

    model = build_model(_cfg()).eval()
    x = torch.randn(2, 3, 64, 64)  # divisible by 32 (stride-32 encoder)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 64, 64)  # 1-channel logits, same HxW


def test_dlinknet_trainable():
    from route_resilience.models.baseline import build_model
    from route_resilience.models.losses import DiceFocalLoss

    model = build_model(_cfg())
    loss_fn = DiceFocalLoss(0.5, 0.5)
    x = torch.randn(1, 3, 64, 64)
    target = (torch.rand(1, 1, 64, 64) > 0.5).float()
    loss = loss_fn(model(x), target)
    loss.backward()
    # gradients flow into the dilated centre block -> the model actually trains.
    grad = model.center.dilate1.weight.grad
    assert grad is not None and torch.isfinite(grad).all()


def test_dblock_preserves_shape():
    from route_resilience.models.dlinknet import DBlock

    block = DBlock(16)
    x = torch.randn(1, 16, 8, 8)
    assert block(x).shape == x.shape  # dilation keeps resolution
