"""M2 tests — synthetic occlusion generator (offline, no network).

Verify the core training-signal invariants: the target mask is never touched,
occlusion lands on roads, coverage is honoured, and runs are reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skimage")
pytest.importorskip("scipy")

from route_resilience.data.occlusion import apply_occlusion  # noqa: E402


def _toy():
    """A 128x128 grey image with a horizontal road band through the middle."""
    img = np.full((128, 128, 3), 100, dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[58:70, :] = 1  # road band
    return img, mask


def test_target_mask_is_never_modified():
    img, mask = _toy()
    mask_before = mask.copy()
    apply_occlusion(img, mask, apply_prob=1.0, seed=0)
    assert np.array_equal(mask, mask_before)


def test_input_image_not_mutated_in_place():
    img, mask = _toy()
    img_before = img.copy()
    res = apply_occlusion(img, mask, apply_prob=1.0, seed=1)
    assert np.array_equal(img, img_before)  # original untouched
    assert not np.array_equal(res.image, img_before)  # output changed


def test_occluded_road_is_subset_of_road_and_occlusion():
    img, mask = _toy()
    res = apply_occlusion(img, mask, apply_prob=1.0, seed=2)
    road = mask > 0
    assert res.occlusion_mask.dtype == bool
    assert np.all(res.occluded_road_mask <= road)  # subset of road
    assert np.all(res.occluded_road_mask <= res.occlusion_mask)  # subset of occlusion


def test_coverage_reaches_target_range():
    img, mask = _toy()
    res = apply_occlusion(
        img, mask, apply_prob=1.0, coverage_range=(0.4, 0.4), max_occluders=200, seed=3
    )
    assert res.coverage >= 0.35  # got close to the 0.4 target
    assert res.n_occluders > 0


def test_apply_prob_zero_is_noop():
    img, mask = _toy()
    res = apply_occlusion(img, mask, apply_prob=0.0, seed=4)
    assert res.n_occluders == 0
    assert res.coverage == 0.0
    assert np.array_equal(res.image, img)
    assert res.occlusion_mask.sum() == 0


def test_no_road_returns_unchanged():
    img = np.full((64, 64, 3), 100, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)  # no road at all
    res = apply_occlusion(img, mask, apply_prob=1.0, seed=5)
    assert res.n_occluders == 0
    assert np.array_equal(res.image, img)


def test_determinism_same_seed():
    img, mask = _toy()
    a = apply_occlusion(img, mask, apply_prob=1.0, seed=7)
    b = apply_occlusion(img, mask, apply_prob=1.0, seed=7)
    assert np.array_equal(a.image, b.image)
    assert np.array_equal(a.occlusion_mask, b.occlusion_mask)
    assert a.coverage == b.coverage


def test_different_seeds_differ():
    img, mask = _toy()
    a = apply_occlusion(img, mask, apply_prob=1.0, seed=7)
    b = apply_occlusion(img, mask, apply_prob=1.0, seed=8)
    assert not np.array_equal(a.image, b.image)
