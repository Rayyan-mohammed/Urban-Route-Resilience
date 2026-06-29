"""High-level graph construction from a road mask (M6).

mask_to_graph: binary mask (+ optional geo-transform) -> NetworkX graph + skeleton.
graph_stats:   summary used by the dashboard and reports.
save/load:     GraphML persistence (pixel paths are dropped — not GraphML-safe).
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
from skimage.morphology import skeletonize

from .skeletonize import skeleton_to_graph


def merge_junction_clusters(g: nx.Graph, max_len: float) -> nx.Graph:
    """Collapse tangles of mutually-adjacent junction pixels into one node.

    8-connectivity turns a single intersection into several degree>=3 pixels joined
    by ~1px edges. We contract any short edge (<= max_len) whose BOTH endpoints are
    junctions, so each real intersection becomes one node. Long edges (real roads)
    and junction->endpoint arms are never merged.
    """
    g = g.copy()
    while True:
        target = None
        for u, v, d in g.edges(data=True):
            if d.get("length", 0.0) <= max_len and g.degree(u) >= 3 and g.degree(v) >= 3:
                target = (u, v)
                break
        if target is None:
            break
        u, v = target
        ux, uy = g.nodes[u]["x"], g.nodes[u]["y"]
        vx, vy = g.nodes[v]["x"], g.nodes[v]["y"]
        g = nx.contracted_nodes(g, u, v, self_loops=False, copy=False)
        g.nodes[u]["x"], g.nodes[u]["y"] = (ux + vx) / 2, (uy + vy) / 2
        g.nodes[u].pop("contraction", None)
    g.remove_edges_from(list(nx.selfloop_edges(g)))
    return g


def dissolve_degree2_nodes(g: nx.Graph) -> nx.Graph:
    """Remove degree-2 pass-through nodes, fusing their two edges into one.

    A degree-2 node is an interior path point, not an intersection. Junction
    merging can leave a few behind; dissolving them yields a clean graph whose
    nodes are only real endpoints and junctions. Edge length is the sum.
    """
    g = g.copy()
    while True:
        target = next((n for n in g.nodes() if g.degree(n) == 2), None)
        if target is None:
            break
        a, b = list(g.neighbors(target))
        if a == b:                       # degenerate; just drop the node
            g.remove_node(target)
            continue
        new_len = g[target][a]["length"] + g[target][b]["length"]
        if g.has_edge(a, b):
            if new_len < g[a][b]["length"]:
                g[a][b]["length"] = g[a][b]["weight"] = new_len
        else:
            g.add_edge(a, b, length=new_len, weight=new_len)
        g.remove_node(target)
    return g


def mask_to_graph(
    mask: np.ndarray,
    transform=None,
    resolution_m: float = 1.0,
    *,
    simplify: bool = True,
    merge_px: float = 2.5,
):
    """Return (graph, skeleton) for a binary road mask.

    simplify (recommended) merges junction-pixel tangles AND dissolves degree-2
    pass-through nodes, so nodes are only real endpoints/junctions. merge_px is the
    junction-cluster radius in pixels (converted to metres via resolution_m).
    """
    skel = skeletonize(mask.astype(bool))
    g = skeleton_to_graph(skel, transform=transform, resolution_m=resolution_m)
    if simplify:
        g = merge_junction_clusters(g, max_len=merge_px * resolution_m)
        g = dissolve_degree2_nodes(g)
    return g, skel


def graph_stats(g: nx.Graph) -> dict:
    total_len = float(sum(d.get("length", 0.0) for _, _, d in g.edges(data=True)))
    n_iso = sum(1 for _, deg in g.degree() if deg == 0)
    return {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "n_components": nx.number_connected_components(g) if g.number_of_nodes() else 0,
        "n_isolated_nodes": n_iso,
        "total_length_m": total_len,
    }


def save_graphml(g: nx.Graph, path: str | Path) -> None:
    """Persist a graph to GraphML. Pixel-path lists are stripped (not serialisable)."""
    h = g.copy()
    for _, _, d in h.edges(data=True):
        d.pop("pixels", None)
    # node ids are (r,c) tuples -> stringify for GraphML
    h = nx.relabel_nodes(h, lambda n: f"{n[0]}_{n[1]}" if isinstance(n, tuple) else str(n))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(h, path)


def load_graphml(path: str | Path) -> nx.Graph:
    return nx.read_graphml(path)
