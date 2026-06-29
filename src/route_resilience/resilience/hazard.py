"""Hazard layer (M8) — what gets 'flooded' to stress-test the network.

A Hazard maps a graph to the set of impacted nodes. Synthetic generators are used
now; a real DEM/flood raster implements the SAME `impacted_nodes` interface later
(returning low-lying nodes) with zero downstream change — that's the pluggability
the roadmap's hazard-grounded story needs.

Nodes carry world (x, y) from the tile transform, so spatial hazards work in metres.
"""

from __future__ import annotations

import networkx as nx


class Hazard:
    """Interface: return the set of node ids a hazard impacts."""

    def impacted_nodes(self, g: nx.Graph) -> set:
        raise NotImplementedError


class NodeHazard(Hazard):
    """Explicit node set (e.g. a junction the planner clicked)."""

    def __init__(self, nodes):
        self.nodes = set(nodes)

    def impacted_nodes(self, g: nx.Graph) -> set:
        return {n for n in self.nodes if n in g}


class RadiusHazard(Hazard):
    """All nodes within `radius_m` of a world point (a localized flood)."""

    def __init__(self, center: tuple[float, float], radius_m: float):
        self.cx, self.cy = center
        self.r2 = radius_m * radius_m

    def impacted_nodes(self, g: nx.Graph) -> set:
        out = set()
        for n, d in g.nodes(data=True):
            if (d["x"] - self.cx) ** 2 + (d["y"] - self.cy) ** 2 <= self.r2:
                out.add(n)
        return out


class BandHazard(Hazard):
    """All nodes whose `axis` ('x' or 'y') coordinate is in [lo, hi].

    Stands in for a low-lying strip / river corridor until a DEM is wired up.
    """

    def __init__(self, axis: str = "y", lo: float = 0.0, hi: float = 0.0):
        self.axis = axis
        self.lo = lo
        self.hi = hi

    def impacted_nodes(self, g: nx.Graph) -> set:
        return {n for n, d in g.nodes(data=True) if self.lo <= d[self.axis] <= self.hi}
