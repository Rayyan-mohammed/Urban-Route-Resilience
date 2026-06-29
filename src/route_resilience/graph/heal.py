"""Graph healing (M7) — bridge occlusion gaps to make the network routable.

Road extraction can leave a graph split into several connected components where
occlusion hid a segment. Healing reconnects them with the minimum-cost set of
bridges, so the resilience twin (M8) runs on a connected, routable network.

Algorithm (Disjoint-Set / MST baseline, roadmap §7):
  1. Candidate bridges = node pairs in DIFFERENT components within `max_gap_m`
     (KDTree), preferring degree-1 endpoints (where real roads break).
  2. Sort candidates by length; Union-Find over components; add a bridge only if
     it merges two components (Kruskal). Stop when connected / no candidate left.

Bridges are added with attribute healed=True (length = straight-line gap distance)
so they can be visualised and, if desired, weighted differently in routing.

NOT healed: gaps wider than max_gap_m (treated as genuine separations, not
artifacts). Endpoint-to-edge (T-junction) projection is a documented future
enhancement; the GNN link-predictor is the stretch goal.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree


class _DSU:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        self.parent[rx] = ry
        return True


def heal_graph(
    g: nx.Graph,
    *,
    max_gap_m: float = 40.0,
    endpoints_only: bool = True,
) -> tuple[nx.Graph, dict]:
    """Return (healed_graph, info). Adds bridge edges (healed=True) across gaps."""
    g = g.copy()
    comps_before = nx.number_connected_components(g) if g.number_of_nodes() else 0
    info = {
        "components_before": comps_before,
        "components_after": comps_before,
        "n_bridges": 0,
        "bridge_length_m": 0.0,
    }
    if g.number_of_nodes() < 2 or comps_before < 2:
        return g, info

    ids = list(g.nodes())
    xy = np.array([[g.nodes[n]["x"], g.nodes[n]["y"]] for n in ids], dtype=float)
    comp_of = {}
    for ci, cc in enumerate(nx.connected_components(g)):
        for n in cc:
            comp_of[n] = ci

    tree = cKDTree(xy)
    candidates = []
    for i, j in tree.query_pairs(r=max_gap_m):
        a, b = ids[i], ids[j]
        if comp_of[a] == comp_of[b]:
            continue
        if endpoints_only and g.degree(a) != 1 and g.degree(b) != 1:
            continue
        d = float(np.hypot(*(xy[i] - xy[j])))
        candidates.append((d, a, b))
    candidates.sort(key=lambda t: t[0])

    dsu = _DSU(set(comp_of.values()))
    bridge_len = 0.0
    n_bridges = 0
    for d, a, b in candidates:
        if dsu.union(comp_of[a], comp_of[b]):
            g.add_edge(a, b, length=d, weight=d, healed=True)
            bridge_len += d
            n_bridges += 1

    info.update(
        components_after=nx.number_connected_components(g),
        n_bridges=n_bridges,
        bridge_length_m=bridge_len,
    )
    return g, info
