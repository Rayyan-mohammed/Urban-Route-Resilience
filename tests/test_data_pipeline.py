"""M1 tests — exercise the data pipeline WITHOUT any network/OSM download.

- Tiling + rasterisation are tested with hand-built shapely geometry (offline).
- The split is tested with a synthetic manifest (pure pandas/numpy).

`importorskip` lets the split tests run even before the conda geo stack exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from route_resilience.config import load_config


# ----------------------------- tiling geometry -----------------------------
def test_tile_grid_count_and_georef():
    rasterio = pytest.importorskip("rasterio")  # noqa: F841
    from route_resilience.data.geo import make_tile_grid

    # 1000m x 1000m extent, 100px tiles @ 1 m/px (=100m tiles), no overlap -> 10x10.
    tiles = make_tile_grid(
        bounds=(0, 0, 1000, 1000), crs="EPSG:32643",
        tile_size=100, resolution_m=1.0, overlap=0.0, place="t",
    )
    assert len(tiles) == 100
    # Each tile is 100x100 px and 100m on a side.
    t = tiles[0]
    w, s, e, n = t.bounds
    assert (e - w) == pytest.approx(100.0)
    assert t.width == t.height == 100
    # Pixel (0,0) of a tile maps to its top-left (west, north) corner.
    x, y = t.transform * (0, 0)
    assert x == pytest.approx(w)
    assert y == pytest.approx(n)


def test_tile_grid_overlap_increases_count():
    pytest.importorskip("rasterio")
    from route_resilience.data.geo import make_tile_grid

    no_ov = make_tile_grid((0, 0, 1000, 1000), "EPSG:32643", 100, 1.0, 0.0)
    ov = make_tile_grid((0, 0, 1000, 1000), "EPSG:32643", 100, 1.0, 0.25)
    assert len(ov) > len(no_ov)  # 25% overlap -> more, smaller-step tiles


def test_rasterize_roads_connectivity():
    pytest.importorskip("rasterio")
    shapely = pytest.importorskip("shapely")
    from shapely.geometry import LineString

    from route_resilience.data.geo import make_tile_grid, rasterize_roads

    tile = make_tile_grid((0, 0, 100, 100), "EPSG:32643", 100, 1.0, 0.0)[0]
    # A horizontal road across the middle of the tile.
    road = LineString([(5, 50), (95, 50)])
    mask = rasterize_roads([road], tile, buffer_m=2.0)
    assert mask.dtype == np.uint8
    assert mask.max() == 1
    assert mask.sum() > 0
    # Buffered road must be an unbroken horizontal band (connectivity preserved).
    row_hits = mask.sum(axis=1)
    assert row_hits.max() >= 80  # spans most of the 100px width


def test_rasterize_empty_returns_zeros():
    pytest.importorskip("rasterio")
    from route_resilience.data.geo import make_tile_grid, rasterize_roads

    tile = make_tile_grid((0, 0, 100, 100), "EPSG:32643", 100, 1.0, 0.0)[0]
    mask = rasterize_roads([], tile, buffer_m=2.0)
    assert mask.shape == (100, 100)
    assert mask.sum() == 0


# ----------------------------- split logic --------------------------------
def _synthetic_manifest(n_per_terrain=20):
    rows = []
    for terrain in ["urban", "suburban", "forested", "rural"]:
        for i in range(n_per_terrain):
            rows.append({"tile_id": f"{terrain}_{i}", "terrain": terrain})
    return pd.DataFrame(rows)


def test_stratified_split_covers_every_terrain():
    from route_resilience.data.split import stratified_split

    cfg = load_config("base.yaml", "data.yaml")
    df = _synthetic_manifest(20)
    out = stratified_split(df, cfg, seed=cfg.seed)

    # Every terrain must appear in train, val AND test (the whole point).
    for terrain in df["terrain"].unique():
        sub = out[out["terrain"] == terrain]
        assert set(sub["split"].unique()) == {"train", "val", "test"}

    # Roughly the configured ratios overall.
    frac_train = (out["split"] == "train").mean()
    assert 0.6 < frac_train < 0.8  # ~0.7


def test_split_is_deterministic():
    from route_resilience.data.split import stratified_split

    cfg = load_config("base.yaml", "data.yaml")
    df = _synthetic_manifest(15)
    a = stratified_split(df, cfg, seed=7)
    b = stratified_split(df, cfg, seed=7)
    assert (a["split"].values == b["split"].values).all()
