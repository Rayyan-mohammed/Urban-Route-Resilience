"""M5 tests — evaluation metrics (clDice score, occlusion-recall, connectivity).

Offline; needs scikit-image (skeletonize/label).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skimage")

from route_resilience.evaluation.metrics import (  # noqa: E402
    cldice_score,
    connectivity_ratio,
    n_components,
    occlusion_recall_counts,
)


def _line(h=64, w=64, width=3, gap=None):
    m = np.zeros((h, w), dtype=np.uint8)
    r0 = h // 2 - width // 2
    m[r0 : r0 + width, :] = 1
    if gap is not None:
        m[:, gap[0] : gap[1]] = 0
    return m


def test_cldice_perfect_is_one():
    t = _line()
    assert cldice_score(t, t) > 0.99


def test_cldice_drops_on_break():
    t = _line()
    broken = _line(gap=(30, 34))
    assert cldice_score(broken, t) < cldice_score(t, t)


def test_cldice_empty_is_zero():
    t = _line()
    assert cldice_score(np.zeros_like(t), t) == 0.0


def test_occlusion_recall_counts():
    true = _line()
    occ = np.zeros_like(true)
    occ[:, 20:40] = 1                     # occlude a span
    hidden = (true.astype(bool) & occ.astype(bool)).sum()
    # Prediction recovers exactly half of the hidden span.
    pred = true.copy()
    pred[:, 30:40] = 0                    # miss half the occluded region
    rec, denom = occlusion_recall_counts(pred, true, occ)
    assert denom == hidden
    assert rec == pytest.approx(hidden / 2, rel=0.01)


def test_occlusion_recall_full_recovery():
    true = _line()
    occ = np.zeros_like(true)
    occ[:, 20:40] = 1
    rec, denom = occlusion_recall_counts(true.copy(), true, occ)  # perfect pred
    assert rec == denom and denom > 0


def test_connectivity_ratio():
    t = _line()                          # 1 component
    assert n_components(t) == 1
    assert connectivity_ratio(t, t) == pytest.approx(1.0)
    # Over-fragmented prediction (gap splits into 2) -> ratio < 1.
    broken = _line(gap=(30, 34))
    assert n_components(broken) == 2
    assert connectivity_ratio(broken, t) < 1.0
