"""M11 test — end-to-end pipeline integration (mask -> graph -> heal -> resilience).

Skipped if the dataset isn't built. This is the integration the finale relies on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rasterio")
pytest.importorskip("pyproj")

from route_resilience.config import load_config  # noqa: E402

_HAS_DATA = Path("data/processed/manifest.csv").exists()
pytestmark = pytest.mark.skipif(not _HAS_DATA, reason="dataset not built")


def _dense_tile():
    import pandas as pd
    df = pd.read_csv("data/processed/manifest.csv").sort_values("road_frac", ascending=False)
    return df.iloc[0]["mask_path"]


def test_pipeline_report_is_complete_and_serialisable():
    import json

    from route_resilience.pipeline import run_tile_pipeline

    cfg = load_config("base.yaml", "data.yaml", "graph.yaml")
    rep = run_tile_pipeline(_dense_tile(), cfg, hazard_radius_m=50.0, top_k=5)

    # structure
    for key in ("graph", "critical_nodes", "sample_hazard", "resilience", "crs"):
        assert key in rep
    assert rep["graph"]["n_nodes"] > 10
    assert 0.0 <= rep["resilience"]["resilience_index"] <= 1.0
    assert len(rep["critical_nodes"]) == 5
    # critical nodes carry geo coords in the Bengaluru bbox
    top = rep["critical_nodes"][0]
    assert 77.0 < top["lon"] < 78.0 and 12.0 < top["lat"] < 14.0
    # fully JSON-serialisable (no tuple keys / numpy types leaking)
    assert isinstance(json.dumps(rep), str)


def test_pipeline_healing_reduces_components():
    from route_resilience.pipeline import run_tile_pipeline

    cfg = load_config("base.yaml", "data.yaml", "graph.yaml")
    healed = run_tile_pipeline(_dense_tile(), cfg, heal=True)
    unhealed = run_tile_pipeline(_dense_tile(), cfg, heal=False)
    assert healed["graph"]["n_components"] <= unhealed["graph"]["n_components"]
