"""Tests for multi-scale + flip test-time augmentation (§7.1, §10).

Torch-only (no dataset needed). A tiny deterministic conv stands in for the
segmentation model so assertions are exact.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from route_resilience.models.tta import tta_predict, tta_predict_mask  # noqa: E402


class _TinyNet(torch.nn.Module):
    """1x1 conv -> logits; resolution-preserving, so TTA math is checkable."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1)

    def forward(self, x):
        return self.conv(x)


def test_tta_identity_matches_plain_forward():
    # scales=(1.0,), no flips -> TTA must equal a single sigmoid(forward).
    net = _TinyNet().eval()
    x = torch.randn(3, 32, 32)
    with torch.no_grad():
        plain = torch.sigmoid(net(x.unsqueeze(0)))[0]
    tta = tta_predict(net, x, scales=(1.0,), flips=False)
    assert tta.shape == (1, 32, 32)
    assert torch.allclose(tta, plain, atol=1e-6)


def test_tta_output_is_probability_and_right_shape():
    net = _TinyNet().eval()
    x = torch.randn(3, 64, 64)
    p = tta_predict(net, x, scales=(0.75, 1.0, 1.25), flips=True)
    assert p.shape == (1, 64, 64)
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0


def test_tta_batched_input():
    net = _TinyNet().eval()
    x = torch.randn(2, 3, 32, 32)
    p = tta_predict(net, x, scales=(1.0, 1.25), flips=True)
    assert p.shape == (2, 1, 32, 32)


def test_tta_mask_thresholds():
    net = _TinyNet().eval()
    x = torch.randn(3, 32, 32)
    m = tta_predict_mask(net, x, threshold=0.5, scales=(1.0,), flips=True)
    assert m.dtype == torch.bool and m.shape == (1, 32, 32)
