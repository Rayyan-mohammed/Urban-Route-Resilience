"""Tests for the real (raster-grounded) hazard layer.

RasterHazard is what turns the "hazard-grounded twin" claim from synthetic into
real: a DEM or flood-depth product decides which junctions go out of service.
Needs rasterio; skips cleanly without it.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402

from route_resilience.resilience.hazard import RasterHazard  # noqa: E402
from route_resilience.resilience.simulate import resilience_report  # noqa: E402

ORIGIN_X, ORIGIN_Y, RES = 1000.0, 1000.0, 1.0


def _graph_at(coords):
    """Line graph whose nodes sit at the given world (x, y) points."""
    g = nx.Graph()
    for i, (x, y) in enumerate(coords):
        g.add_node(i, x=float(x), y=float(y))
    for i in range(len(coords) - 1):
        g.add_edge(i, i + 1, length=1.0, weight=1.0)
    return g


def _write_raster(path, values, *, crs="EPSG:32643", nodata=None):
    arr = np.asarray(values, dtype="float32")
    profile = {
        "driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1],
        "count": 1, "dtype": "float32", "crs": crs,
        "transform": from_origin(ORIGIN_X, ORIGIN_Y, RES, RES),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    return path


def _xy(col, row):
    """World coordinate at the centre of raster cell (col, row)."""
    return ORIGIN_X + (col + 0.5) * RES, ORIGIN_Y - (row + 0.5) * RES


def test_depth_mode_floods_only_deep_nodes(tmp_path):
    # Row 0 is dry (0.0 m), row 1 is under 0.8 m of water.
    ras = _write_raster(tmp_path / "depth.tif", [[0.0, 0.0], [0.8, 0.8]])
    g = _graph_at([_xy(0, 0), _xy(1, 0), _xy(0, 1), _xy(1, 1)])

    hz = RasterHazard(ras, mode="depth", threshold=0.3)
    assert hz.impacted_nodes(g) == {2, 3}


def test_depth_threshold_is_respected(tmp_path):
    ras = _write_raster(tmp_path / "depth.tif", [[0.5, 0.5], [0.5, 0.5]])
    g = _graph_at([_xy(0, 0), _xy(1, 1)])

    assert RasterHazard(ras, mode="depth", threshold=0.3).impacted_nodes(g) == {0, 1}
    assert RasterHazard(ras, mode="depth", threshold=0.9).impacted_nodes(g) == set()


def test_elevation_mode_floods_low_ground(tmp_path):
    # A DEM: low-lying row 1 sits below an 812 m water level.
    ras = _write_raster(tmp_path / "dem.tif", [[820.0, 818.0], [805.0, 800.0]])
    g = _graph_at([_xy(0, 0), _xy(1, 0), _xy(0, 1), _xy(1, 1)])

    hz = RasterHazard(ras, mode="elevation", threshold=812.0)
    assert hz.impacted_nodes(g) == {2, 3}


def test_nodes_outside_the_raster_are_unaffected(tmp_path):
    ras = _write_raster(tmp_path / "depth.tif", [[5.0, 5.0], [5.0, 5.0]])
    g = _graph_at([_xy(0, 0), (ORIGIN_X + 500.0, ORIGIN_Y - 500.0)])

    # Node 0 is inside and flooded; node 1 is off-raster -> no evidence, no hit.
    assert RasterHazard(ras, mode="depth", threshold=0.3).impacted_nodes(g) == {0}


def test_nodata_is_not_treated_as_flooded(tmp_path):
    ras = _write_raster(tmp_path / "depth.tif", [[-9999.0, 2.0]], nodata=-9999.0)
    g = _graph_at([_xy(0, 0), _xy(1, 0)])

    assert RasterHazard(ras, mode="depth", threshold=0.3).impacted_nodes(g) == {1}


def test_node_coords_are_reprojected_to_the_raster_crs(tmp_path):
    pytest.importorskip("pyproj")
    from pyproj import Transformer

    ras = _write_raster(tmp_path / "depth.tif", [[0.0, 0.0], [0.9, 0.9]], crs="EPSG:32643")
    # Same two points, expressed in WGS84 instead of UTM.
    to_wgs = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True).transform
    dry, wet = _xy(0, 0), _xy(0, 1)
    g = _graph_at([to_wgs(*dry), to_wgs(*wet)])

    hz = RasterHazard(ras, mode="depth", threshold=0.3, node_crs="EPSG:4326")
    assert hz.impacted_nodes(g) == {1}


def test_rejects_unknown_mode(tmp_path):
    ras = _write_raster(tmp_path / "d.tif", [[1.0]])
    with pytest.raises(ValueError):
        RasterHazard(ras, mode="rainfall")


def test_plugs_into_resilience_report_unchanged(tmp_path):
    """The whole point of the interface: the twin needs no change for real data."""
    ras = _write_raster(tmp_path / "depth.tif", [[0.0, 0.0, 0.9, 0.0, 0.0]])
    g = _graph_at([_xy(c, 0) for c in range(5)])

    rep = resilience_report(g, RasterHazard(ras, mode="depth", threshold=0.3))
    assert rep["n_impacted"] == 1                 # the middle junction floods
    assert 0.0 <= rep["resilience_index"] <= 1.0
    assert rep["components_after"] == 2           # cutting the middle splits the line
