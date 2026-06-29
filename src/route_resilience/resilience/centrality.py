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
    g: nx.Graph, weight: str = "length", sample_k: int | None = None, seed: int = 42
) -> float:
    """Length-weighted Latora-Marchiori global efficiency.

    sample_k samples that many source nodes for large graphs (single-source
    Dijkstra each); None = exact over all sources.
    """
    nodes = list(g.nodes())
    n = len(nodes)
    if n < 2:
        return 0.0
    if sample_k and sample_k < n:
        rng = np.random.default_rng(seed)
        sources = [nodes[i] for i in rng.choice(n, sample_k, replace=False)]
    else:
        sources = nodes

    total = 0.0
    for s in sources:
        for t, d in nx.single_source_dijkstra_path_length(g, s, weight=weight).items():
            if t != s and d > 0:
                total += 1.0 / d
    denom = len(sources) * (n - 1)  # unreachable pairs counted here -> penalised
    return total / denom if denom else 0.0
