"""End-to-end integration pipeline (M11) — tile -> graph -> heal -> resilience.

Ties M1/M6/M7/M8 into one call so the finale only has to swap the *input mask*
(GT OSM mask now; Cartosat-3 model prediction at the finale) — the downstream
graph -> heal -> twin chain is already built and tested.

    report = run_tile_pipeline(mask_path, cfg)

Returns a JSON-serialisable dict: graph stats, healing info, top critical
junctions (with lon/lat), and a resilience report for a sample hazard.
"""

from __future__ import annotations

from .dashboard.service import load_tile_graph
from .graph.build import graph_stats
from .resilience.centrality import critical_nodes, node_betweenness
from .resilience.hazard import RadiusHazard
from .resilience.simulate import resilience_report
from .utils import get_logger

log = get_logger(__name__)


def run_tile_pipeline(
    mask_path,
    cfg,
    *,
    heal: bool = True,
    hazard_radius_m: float = 50.0,
    top_k: int = 5,
) -> dict:
    """Run the full pixels->topology->resilience chain on one tile."""
    res = float(cfg.data.resolution_m)
    max_gap = float(cfg.graph.healing.max_gap_m)
    g, crs = load_tile_graph(mask_path, res, heal=heal, max_gap_m=max_gap)

    if g.number_of_nodes() == 0:
        return {"mask_path": str(mask_path), "crs": crs, "empty": True}

    bc = node_betweenness(g)
    crit = critical_nodes(g, top=top_k)
    critical = [
        {"lon": g.nodes[n]["lon"], "lat": g.nodes[n]["lat"], "betweenness": float(v)}
        for n, v in crit
    ]

    # Sample hazard: flood the single most critical junction.
    top_node = crit[0][0]
    hz = RadiusHazard(center=(g.nodes[top_node]["x"], g.nodes[top_node]["y"]),
                      radius_m=hazard_radius_m)
    resilience = resilience_report(g, hz)

    return {
        "mask_path": str(mask_path),
        "crs": crs,
        "graph": graph_stats(g),
        "critical_nodes": critical,
        "sample_hazard": {
            "type": "RadiusHazard",
            "center_lonlat": [g.nodes[top_node]["lon"], g.nodes[top_node]["lat"]],
            "radius_m": hazard_radius_m,
        },
        "resilience": resilience,
    }
