"""M9 tests — dashboard service layer (load, reproject, flood, reroute, map).

Needs a built dataset (manifest + mask tiles) and folium/pyproj/rasterio.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pyproj")
pytest.importorskip("folium")
pytest.importorskip("rasterio")

from route_resilience.dashboard import service  # noqa: E402

_HAS_DATA = Path("data/processed/manifest.csv").exists()
pytestmark = pytest.mark.skipif(not _HAS_DATA, reason="dataset not built")


def _dense_tile():
    import pandas as pd
    df = pd.read_csv("data/processed/manifest.csv").sort_values("road_frac", ascending=False)
    return df.iloc[0]["mask_path"]


def test_heat_color_is_hex():
    c = service.heat_color(0.0)
    assert c.startswith("#") and len(c) == 7
    assert service.heat_color(1.0) != service.heat_color(0.0)


def test_load_tile_graph_has_lonlat_and_heals():
    g, crs = service.load_tile_graph(_dense_tile(), resolution_m=0.5)
    assert g.number_of_nodes() > 10
    assert crs.upper().startswith("EPSG")
    n = next(iter(g.nodes()))
    d = g.nodes[n]
    assert "lon" in d and "lat" in d
    assert 77.0 < d["lon"] < 78.0 and 12.0 < d["lat"] < 14.0   # Bengaluru bbox


def test_nearest_node_round_trips():
    g, crs = service.load_tile_graph(_dense_tile(), resolution_m=0.5)
    target = next(iter(g.nodes()))
    found = service.nearest_node(g, g.nodes[target]["lon"], g.nodes[target]["lat"], crs)
    assert found == target


def test_flood_reduces_and_reports():
    g, crs = service.load_tile_graph(_dense_tile(), resolution_m=0.5)
    from route_resilience.resilience.centrality import critical_nodes
    center = critical_nodes(g, top=1)[0][0]
    report, impacted, bc_after = service.flood(g, center, radius_m=40.0)
    assert 0.0 <= report["resilience_index"] <= 1.0
    assert len(impacted) >= 1
    assert isinstance(bc_after, dict)


def test_reroute_returns_paths():
    g, crs = service.load_tile_graph(_dense_tile(), resolution_m=0.5)
    nodes = list(g.nodes())
    src, dst = nodes[0], nodes[-1]
    out = service.reroute(g, src, dst)
    assert "cost" in out and "before" in out["cost"]
    # before_path is either a coordinate list or None (if disconnected)
    assert out["before_path"] is None or isinstance(out["before_path"], list)


def test_build_map_returns_folium_html():
    import folium
    g, crs = service.load_tile_graph(_dense_tile(), resolution_m=0.5)
    from route_resilience.resilience.centrality import node_betweenness
    m = service.build_map(g, node_betweenness(g))
    assert isinstance(m, folium.Map)
    assert "leaflet" in m._repr_html_().lower()
