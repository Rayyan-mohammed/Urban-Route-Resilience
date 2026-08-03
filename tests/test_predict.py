"""Tests for sliding-window inference (imagery -> geo-referenced mask).

This is the finale's critical path, so it is tested end to end: a tiny stand-in
model is run over a synthetic GeoTIFF and the resulting mask is fed to the SAME
loader the dashboard uses, proving the Phase I -> Phase II handoff works.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402

from route_resilience.inference.predict import (  # noqa: E402
    _blend_weight,
    _origins,
    predict_array,
    predict_geotiff,
    to_uint8_rgb,
)


class _ConstNet(torch.nn.Module):
    """Emits a constant logit regardless of input — makes blending checkable."""

    def __init__(self, logit: float = 2.0):
        super().__init__()
        self.logit = logit
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), self.logit)


class _LeftHalfNet(torch.nn.Module):
    """Positive logits on the left half of whatever window it sees."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        n, _, h, w = x.shape
        out = torch.full((n, 1, h, w), -5.0)
        out[..., : w // 2] = 5.0
        return out


# ------------------------------ radiometry ------------------------------
def test_uint8_passes_through_untouched():
    img = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
    assert np.array_equal(to_uint8_rgb(img), img)


def test_uint16_is_percentile_stretched_to_full_range():
    # 12-bit-style data: an ImageNet-normalised model would see near-black without stretch.
    img = np.linspace(0, 4095, 8 * 8 * 3).reshape(8, 8, 3).astype(np.uint16)
    out = to_uint8_rgb(img)
    assert out.dtype == np.uint8
    assert out.max() == 255 and out.min() == 0


def test_flat_band_does_not_divide_by_zero():
    img = np.full((4, 4, 3), 700, dtype=np.uint16)
    out = to_uint8_rgb(img)
    assert out.dtype == np.uint8 and np.isfinite(out).all()


# ------------------------------ window maths ------------------------------
def test_origins_cover_the_full_extent():
    for extent, win, stride in [(512, 512, 384), (1000, 512, 384), (1300, 512, 256)]:
        xs = _origins(extent, win, stride)
        assert xs[0] == 0
        assert xs[-1] + win == extent          # flush with the far edge
        covered = np.zeros(extent, dtype=bool)
        for x in xs:
            covered[x:x + win] = True
        assert covered.all()


def test_origins_single_window_when_image_is_small():
    assert _origins(300, 512, 384) == [0]


def test_blend_weight_is_strictly_positive():
    w = _blend_weight(16, 16)
    assert w.shape == (16, 16)
    assert w.min() > 0.0                        # never divides by zero
    assert w[8, 8] > w[0, 0]                    # tapers toward the window edges


# ------------------------------ predict_array ------------------------------
def test_constant_model_gives_seamless_constant_probability():
    """Overlapping windows must not leave visible seams — a seam is a topology break."""
    img = np.random.randint(0, 256, (900, 900, 3), dtype=np.uint8)
    probs = predict_array(_ConstNet(2.0).eval(), img, window=512, overlap=0.25)
    expected = float(torch.sigmoid(torch.tensor(2.0)))
    assert probs.shape == (900, 900)
    assert np.allclose(probs, expected, atol=1e-5)   # flat everywhere, no seams
    assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_image_smaller_than_window_is_padded_and_cropped():
    img = np.random.randint(0, 256, (100, 130, 3), dtype=np.uint8)
    probs = predict_array(_ConstNet().eval(), img, window=512)
    assert probs.shape == (100, 130)


def test_spatial_structure_is_preserved():
    img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    probs = predict_array(_LeftHalfNet().eval(), img, window=512, overlap=0.0)
    assert probs[:, :250].mean() > 0.9
    assert probs[:, 262:].mean() < 0.1


def test_rejects_non_rgb_input():
    with pytest.raises(ValueError):
        predict_array(_ConstNet().eval(), np.zeros((10, 10), dtype=np.uint8))


# ------------------------------ predict_geotiff ------------------------------
def _write_image(path, h=600, w=600, crs="EPSG:32643", res=0.5):
    arr = np.random.randint(0, 256, (3, h, w), dtype=np.uint8)
    profile = {
        "driver": "GTiff", "height": h, "width": w, "count": 3, "dtype": "uint8",
        "crs": crs, "transform": from_origin(785000.0, 1437000.0, res, res),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
    return path


def test_predicted_mask_keeps_crs_and_transform(tmp_path):
    img = _write_image(tmp_path / "tile.tif")
    out = tmp_path / "pred.tif"
    info = predict_geotiff(img, _ConstNet(5.0).eval(), out_path=out)

    assert out.exists()
    with rasterio.open(img) as src, rasterio.open(out) as dst:
        assert dst.crs == src.crs
        assert dst.transform == src.transform      # georeferencing survives
        assert dst.count == 1 and dst.dtypes[0] == "uint8"
        mask = dst.read(1)
    assert set(np.unique(mask)).issubset({0, 1})
    assert info["road_frac"] == pytest.approx(1.0)  # constant +5 logit -> all road
    assert info["pixel_size_m"] == pytest.approx(0.5)


def test_empty_prediction_is_reported_not_crashed(tmp_path):
    img = _write_image(tmp_path / "tile.tif", h=300, w=300)
    info = predict_geotiff(img, _ConstNet(-5.0).eval(), out_path=tmp_path / "pred.tif")
    assert info["road_frac"] == 0.0


def test_probability_map_is_written_when_asked(tmp_path):
    img = _write_image(tmp_path / "tile.tif", h=300, w=300)
    info = predict_geotiff(img, _ConstNet().eval(), out_path=tmp_path / "pred.tif",
                           save_probs=True)
    with rasterio.open(info["prob_path"]) as ds:
        assert ds.dtypes[0] == "float32"
        assert 0.0 <= float(ds.read(1).min()) <= 1.0


def test_predicted_mask_feeds_the_graph_pipeline(tmp_path):
    """The whole point: a predicted mask must load like any manifest mask."""
    pytest.importorskip("networkx")
    pytest.importorskip("pyproj")
    from route_resilience.dashboard.service import load_tile_graph

    img = _write_image(tmp_path / "tile.tif", h=512, w=512)
    info = predict_geotiff(img, _LeftHalfNet().eval(), out_path=tmp_path / "pred.tif")
    g, crs = load_tile_graph(info["mask_path"], resolution_m=0.5, heal=False)

    assert crs == "EPSG:32643"
    assert g.number_of_nodes() > 0
    # Geo-referenced: nodes carry real-world coords and lon/lat.
    n0 = next(iter(g.nodes(data=True)))[1]
    assert "x" in n0 and "lon" in n0
