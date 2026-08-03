"""Network criticality + efficiency (M8).

betweenness  : which junctions/edges carry the most shortest-path traffic. Recompute
               it on the DAMAGED graph to see how criticality shifts after a closure
               (the "dynamic" in dynamic betweenness).
efficiency   : Latora-Marchiori global efficiency, length-weighted. Average of
               1/d(i,j) over all ordered node pairs; UNREACHABLE pairs stay in the
               denominator, so fragmentation lowers efficiency. This is the basis of
               the Resilience Index.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def node_betweenness(g: nx.Graph, weight: str = "length", k: int | None = None, seed: int = 42):
    if g.number_of_nodes() == 0:
        return {}
    kk = min(k, g.number_of_nodes()) if k else None
    return nx.betweenness_centrality(g, weight=weight, normalized=True, k=kk, seed=seed)


def edge_betweenness(g: nx.Graph, weight: str = "length", k: int | None = None, seed: int = 42):
    if g.number_of_edges() == 0:
        return {}
    kk = min(k, g.number_of_nodes()) if k else None
    return nx.edge_betweenness_centrality(g, weight=weight, normalized=True, k=kk, seed=seed)


def critical_nodes(g: nx.Graph, top: int = 10, **kw):
    """Top-`top` (node, betweenness) pairs, most critical first."""
    bc = node_betweenness(g, **kw)
    return sorted(bc.items(), key=lambda kv: kv[1], reverse=True)[:top]


def global_efficiency(
    g: nx.Graph,
    weight: str = "length",
    sample_k: int | None = None,
    seed: int = 42,
    n_universe: int | None = None,
) -> float:
    """Length-weighted Latora-Marchiori global efficiency.

    sample_k samples that many source nodes for large graphs (single-source
    Dijkstra each); None = exact over all sources.

    `n_universe` pins the pair-count denominator to a node set LARGER than the
    graph itself — the fixed-universe convention used for hazard ablation. When a
    flood removes junctions, those junctions must stay in the denominator as
    permanently unreachable; otherwise removing poorly-connected nodes *raises*
    measured efficiency (a smaller, tighter network scores better) and the
    Resilience Index can exceed 1, i.e. "the flood improved the city". Defaults to
    the graph's own node count, which is the standard single-network definition.
    """
    nodes = list(g.nodes())
    m = len(nodes)
    n = int(n_universe) if n_universe is not None else m
    if n < 2 or m < 1:
        return 0.0
    if sample_k and sample_k < m:
        rng = np.random.default_rng(seed)
        sources = [nodes[i] for i in rng.choice(m, sample_k, replace=False)]
    else:
        sources = nodes

    total = 0.0
    for s in sources:
        for t, d in nx.single_source_dijkstra_path_length(g, s, weight=weight).items():
            if t != s and d > 0:
                total += 1.0 / d
    # Scale the sampled source rows up to all m surviving sources, then normalise
    # over the full universe of n(n-1) ordered pairs. Unreachable and removed
    # pairs contribute 0 to the numerator but remain in the denominator.
    if not sources:
        return 0.0
    return (m / len(sources)) * total / (n * (n - 1))
