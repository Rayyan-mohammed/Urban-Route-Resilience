"""M7 tests — Disjoint-Set graph healing.

Offline; needs networkx + scipy. Graphs are built directly with x,y attrs so the
tests are independent of skeleton extraction.
"""

from __future__ import annotations

import networkx as nx
import pytest

pytest.importorskip("scipy")

from route_resilience.graph.heal import heal_graph  # noqa: E402


def _two_segments(gap: float):
    """Two short horizontal segments separated by a horizontal gap of `gap` m."""
    g = nx.Graph()
    # left segment: nodes A(0,0)-B(10,0)
    g.add_node("A", x=0.0, y=0.0)
    g.add_node("B", x=10.0, y=0.0)
    g.add_edge("A", "B", length=10.0, weight=10.0)
    # right segment: C(10+gap,0)-D(20+gap,0)
    g.add_node("C", x=10.0 + gap, y=0.0)
    g.add_node("D", x=20.0 + gap, y=0.0)
    g.add_edge("C", "D", length=10.0, weight=10.0)
    return g


def test_heal_bridges_small_gap():
    g = _two_segments(gap=8.0)
    assert nx.number_connected_components(g) == 2
    healed, info = heal_graph(g, max_gap_m=40.0)
    assert info["components_before"] == 2
    assert info["components_after"] == 1
    assert info["n_bridges"] == 1
    assert nx.number_connected_components(healed) == 1
    # The bridge connects the nearest endpoints B and C (gap = 8 m).
    bridges = [(u, v, d) for u, v, d in healed.edges(data=True) if d.get("healed")]
    assert len(bridges) == 1
    assert info["bridge_length_m"] == pytest.approx(8.0, abs=0.01)


def test_heal_leaves_wide_gap_alone():
    g = _two_segments(gap=100.0)               # far apart -> genuine separation
    healed, info = heal_graph(g, max_gap_m=40.0)
    assert info["n_bridges"] == 0
    assert nx.number_connected_components(healed) == 2


def test_heal_connected_graph_is_noop():
    g = _two_segments(gap=8.0)
    g.add_edge("B", "C", length=8.0, weight=8.0)  # already connected
    healed, info = heal_graph(g, max_gap_m=40.0)
    assert info["components_before"] == 1
    assert info["n_bridges"] == 0


def test_heal_marks_bridges():
    g = _two_segments(gap=8.0)
    healed, _ = heal_graph(g, max_gap_m=40.0)
    healed_edges = [d for _, _, d in healed.edges(data=True) if d.get("healed")]
    assert all(d["length"] > 0 and d["weight"] > 0 for d in healed_edges)


def test_heal_three_components_minimum_bridges():
    g = _two_segments(gap=8.0)
    g.add_node("E", x=30.0, y=0.0)
    g.add_node("F", x=40.0, y=0.0)
    g.add_edge("E", "F", length=10.0, weight=10.0)  # third component near D
    healed, info = heal_graph(g, max_gap_m=40.0)
    assert info["components_before"] == 3
    assert info["components_after"] == 1
    assert info["n_bridges"] == 2                   # n_components - 1
