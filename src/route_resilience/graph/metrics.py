"""Graph-based metric: APLS (Average Path Length Similarity) — now unlocked (M6).

APLS (SpaceNet) compares the shortest-path lengths between corresponding nodes of
a ground-truth graph and a predicted graph. A road that is present but BROKEN
(disconnected) tanks APLS even if its pixels are mostly right — which is exactly
the connectivity failure our clDice training targets. Pixel IoU can't see this;
APLS can.

Definition (symmetric):
  For each ordered graph pair (A, B): snap every A-node to its nearest B-node
  within `snap_m`. For sampled source nodes, take single-source shortest paths in
  A; for each reachable target, compare against the matched nodes' path length in
  B with score = 1 - min(1, |L_A - L_B| / L_A). Missing match or unreachable -> 0.
  APLS = mean of the (A->B) and (B->A) half-scores.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree


def _xy(g: nx.Graph):
    ids = list(g.nodes())
    if not ids:
        return ids, np.zeros((0, 2))
    xy = np.array([[g.nodes[n]["x"], g.nodes[n]["y"]] for n in ids], dtype=float)
    return ids, xy


def _half_apls(gA: nx.Graph, gB: nx.Graph, snap_m: float, max_sources: int) -> float:
    ids_a, xy_a = _xy(gA)
    ids_b, xy_b = _xy(gB)
    if len(ids_a) == 0 or len(ids_b) == 0:
        return 0.0

    tree = cKDTree(xy_b)
    dist, idx = tree.query(xy_a, k=1)
    match = {
        ids_a[i]: (ids_b[idx[i]] if dist[i] <= snap_m else None) for i in range(len(ids_a))
    }

    if len(ids_a) <= max_sources:
        sources = ids_a
    else:
        sel = np.linspace(0, len(ids_a) - 1, max_sources).astype(int)
        sources = [ids_a[k] for k in sel]

    scores: list[float] = []
    for a in sources:
        len_a = nx.single_source_dijkstra_path_length(gA, a, weight="length")
        b = match[a]
        len_b = (
            nx.single_source_dijkstra_path_length(gB, b, weight="length")
            if b is not None else {}
        )
        for t, la in len_a.items():
            if t == a or la <= 0:
                continue
            bt = match.get(t)
            if b is None or bt is None or bt not in len_b:
                scores.append(0.0)
            else:
                lb = len_b[bt]
                scores.append(1.0 - min(1.0, abs(la - lb) / la))
    return float(np.mean(scores)) if scores else 0.0


def graph_apls(
    g_true: nx.Graph,
    g_pred: nx.Graph,
    *,
    snap_m: float | None = None,
    resolution_m: float = 0.5,
    max_sources: int = 50,
) -> float:
    """Symmetric APLS in [0,1]; 1.0 = identical routing structure."""
    if snap_m is None:
        snap_m = max(4.0, 8 * resolution_m)
    if g_true.number_of_nodes() == 0 and g_pred.number_of_nodes() == 0:
        return 1.0
    return 0.5 * (
        _half_apls(g_true, g_pred, snap_m, max_sources)
        + _half_apls(g_pred, g_true, snap_m, max_sources)
    )
