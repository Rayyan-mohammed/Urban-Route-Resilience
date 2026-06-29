"""Resilience simulation (M8) — ablate a hazard, measure the damage, reroute.

resilience_report : efficiency before vs after the hazard ablation ->
                    Resilience Index = E_after / E_before in [0,1]
                    (1 = no impact, 0 = network destroyed). This is the headline
                    decision-support number.
reroute_cost      : shortest-path length between two points before vs after the
                    closure -> the "click two points, flood, see the detour cost"
                    interaction the dashboard exposes.
"""

from __future__ import annotations

import networkx as nx

from .centrality import global_efficiency
from .hazard import Hazard


def ablate(g: nx.Graph, nodes) -> nx.Graph:
    """Return a copy of g with `nodes` removed (the damaged network)."""
    h = g.copy()
    h.remove_nodes_from([n for n in nodes if n in h])
    return h


def _lcc_fraction(g: nx.Graph) -> float:
    if g.number_of_nodes() == 0:
        return 0.0
    return max((len(c) for c in nx.connected_components(g)), default=0) / g.number_of_nodes()


def resilience_report(
    g: nx.Graph, hazard: Hazard, *, weight: str = "length", sample_k: int | None = None
) -> dict:
    """Quantify how much mobility a hazard destroys."""
    impacted = set(hazard.impacted_nodes(g))
    e0 = global_efficiency(g, weight=weight, sample_k=sample_k)
    g2 = ablate(g, impacted)
    e1 = global_efficiency(g2, weight=weight, sample_k=sample_k)
    ri = (e1 / e0) if e0 > 0 else 0.0
    return {
        "n_nodes": g.number_of_nodes(),
        "n_impacted": len(impacted),
        "efficiency_before": e0,
        "efficiency_after": e1,
        "resilience_index": ri,
        "efficiency_drop": (1.0 - ri),
        "components_after": nx.number_connected_components(g2) if g2.number_of_nodes() else 0,
        "lcc_fraction_after": _lcc_fraction(g2),
    }


def reroute_cost(g: nx.Graph, source, target, *, ablated_nodes=(), weight: str = "length") -> dict:
    """Shortest-path length source->target before vs after ablating nodes.

    `after` is inf if the closure disconnects the pair; `ratio` is the detour
    multiplier (inf if disconnected, None if unmeasurable).
    """
    def splen(gg):
        try:
            return float(nx.shortest_path_length(gg, source, target, weight=weight))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float("inf")

    before = splen(g)
    after = splen(ablate(g, ablated_nodes))
    if before in (0.0, float("inf")):
        ratio = None
    elif after == float("inf"):
        ratio = float("inf")
    else:
        ratio = after / before
    return {"before": before, "after": after, "ratio": ratio}
