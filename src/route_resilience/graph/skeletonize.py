"""Skeleton -> NetworkX graph tracing (M6 core).

A binary road mask is skeletonized to 1-px centerlines, then traced into a graph:
  - NODE  = skeleton pixel whose neighbour-count != 2 (endpoint=1, junction>=3,
            isolated=0). Degree-2 pixels are interior path points, not nodes.
  - EDGE  = the chain of degree-2 pixels connecting two nodes; its weight is the
            summed Euclidean step length (diagonal = sqrt(2)) times resolution_m,
            i.e. the real-world road length -> shortest path = physical distance.

Each node stores world coordinates (x,y) derived from the tile's affine transform,
so the graph is geo-referenced (needed by the resilience twin and the map UI).
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

_NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _neighbors(r: int, c: int, H: int, W: int):
    for dr, dc in _NEIGH:
        rr, cc = r + dr, c + dc
        if 0 <= rr < H and 0 <= cc < W:
            yield rr, cc


def _step_len(a, b, res: float) -> float:
    diag = (a[0] != b[0]) and (a[1] != b[1])
    return (math.sqrt(2) if diag else 1.0) * res


def skeleton_to_graph(skel: np.ndarray, transform=None, resolution_m: float = 1.0) -> nx.Graph:
    """Trace a 1-px skeleton (bool HxW) into a geo-referenced NetworkX graph."""
    skel = skel.astype(bool)
    H, W = skel.shape
    pix = set(zip(*np.where(skel), strict=True))

    def degree(p):
        return sum(1 for n in _neighbors(p[0], p[1], H, W) if n in pix)

    def world(p):
        r, c = p
        if transform is not None:
            x, y = transform * (c + 0.5, r + 0.5)  # pixel centre -> world
            return float(x), float(y)
        return float(c), float(r)

    nodes = {p for p in pix if degree(p) != 2}
    G = nx.Graph()
    for p in nodes:
        x, y = world(p)
        G.add_node(p, pixel=p, row=p[0], col=p[1], x=x, y=y)

    visited_steps: set[tuple] = set()  # directed pixel steps already consumed

    def trace(start, first):
        prev, cur = start, first
        length = _step_len(prev, cur, resolution_m)
        visited_steps.add((prev, cur))
        visited_steps.add((cur, prev))
        path = [start, cur]
        while cur not in nodes:
            nxt = None
            for n in _neighbors(cur[0], cur[1], H, W):
                if n in pix and n != prev and (cur, n) not in visited_steps:
                    nxt = n
                    break
            if nxt is None:
                break
            length += _step_len(cur, nxt, resolution_m)
            visited_steps.add((cur, nxt))
            visited_steps.add((nxt, cur))
            prev, cur = cur, nxt
            path.append(cur)
        return cur, length, path

    # Trace every edge leaving a node.
    for nd in nodes:
        for nb in _neighbors(nd[0], nd[1], H, W):
            if nb in pix and (nd, nb) not in visited_steps:
                end, length, path = trace(nd, nb)
                if length > 0:
                    G.add_edge(nd, end, weight=length, length=length, pixels=path)

    # Pure loops (all degree-2, no node): seed an arbitrary node and trace once.
    remaining = pix - {p for step in visited_steps for p in step} - nodes
    while remaining:
        seed = remaining.pop()
        G.add_node(seed, pixel=seed, row=seed[0], col=seed[1],
                   x=world(seed)[0], y=world(seed)[1])
        nodes.add(seed)
        for nb in _neighbors(seed[0], seed[1], H, W):
            if nb in pix and (seed, nb) not in visited_steps:
                end, length, path = trace(seed, nb)
                if length > 0:
                    G.add_edge(seed, end, weight=length, length=length, pixels=path)
        remaining = pix - {p for step in visited_steps for p in step} - nodes

    return G
