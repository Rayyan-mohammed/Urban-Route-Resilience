"""M8 tests — centrality, efficiency, hazards, ablation, resilience, reroute.

Offline; needs networkx. Graphs are built directly with x,y attrs.
"""

from __future__ import annotations

import networkx as nx
import pytest

from route_resilience.resilience.centrality import (
    critical_nodes,
    global_efficiency,
    node_betweenness,
)
from route_resilience.resilience.hazard import BandHazard, NodeHazard, RadiusHazard
from route_resilience.resilience.simulate import ablate, reroute_cost, resilience_report


def _grid(n=4):
    g = nx.Graph(nx.grid_2d_graph(n, n))
    for (i, j) in list(g.nodes()):
        g.nodes[(i, j)]["x"] = float(j)
        g.nodes[(i, j)]["y"] = float(i)
    for u, v in g.edges():
        g[u][v]["length"] = g[u][v]["weight"] = 1.0
    return g


def _line(n=5):
    g = nx.Graph()
    for i in range(n):
        g.add_node(i, x=float(i), y=0.0)
    for i in range(n - 1):
        g.add_edge(i, i + 1, length=1.0, weight=1.0)
    return g


# ----- centrality / efficiency -----
def test_betweenness_central_node_highest():
    bc = node_betweenness(_line(5))
    assert max(bc, key=bc.get) == 2          # middle of a 5-node line


def test_global_efficiency_positive_and_bounded():
    e = global_efficiency(_grid(4))
    assert 0.0 < e <= 1.0


def test_efficiency_drops_after_node_removal():
    g = _line(5)
    before = global_efficiency(g)
    after = global_efficiency(ablate(g, [2]))  # cut the line in half
    assert after < before


def test_critical_nodes_sorted():
    cn = critical_nodes(_grid(4), top=3)
    vals = [v for _, v in cn]
    assert vals == sorted(vals, reverse=True) and len(cn) == 3


# ----- hazards -----
def test_node_hazard():
    g = _grid(3)
    h = NodeHazard([(1, 1), (0, 0), ("missing",)])
    assert h.impacted_nodes(g) == {(1, 1), (0, 0)}


def test_radius_hazard():
    g = _grid(3)
    impacted = RadiusHazard(center=(0.0, 0.0), radius_m=1.1).impacted_nodes(g)
    assert impacted == {(0, 0), (0, 1), (1, 0)}   # within 1.1 m of origin


def test_band_hazard():
    g = _grid(3)
    impacted = BandHazard(axis="y", lo=0.0, hi=0.0).impacted_nodes(g)
    assert impacted == {(0, 0), (0, 1), (0, 2)}   # entire bottom row


# ----- simulation -----
def test_ablate_removes_nodes():
    g = _grid(3)
    h = ablate(g, [(1, 1)])
    assert (1, 1) not in h and h.number_of_nodes() == g.number_of_nodes() - 1


def test_resilience_report_index_in_range():
    g = _grid(4)
    rep = resilience_report(g, NodeHazard([(1, 1), (1, 2), (2, 1)]))
    assert 0.0 <= rep["resilience_index"] <= 1.0
    assert rep["resilience_index"] < 1.0          # removing core nodes hurts
    assert rep["efficiency_drop"] == pytest.approx(1 - rep["resilience_index"])


def test_resilience_full_grid_vs_severe_hazard():
    g = _grid(4)
    mild = resilience_report(g, NodeHazard([(0, 0)]))            # corner
    severe = resilience_report(g, BandHazard(axis="x", lo=1.0, hi=2.0))  # 2 inner cols
    assert severe["resilience_index"] < mild["resilience_index"]


def test_reroute_cost_disconnect_is_inf():
    g = _line(5)
    r = reroute_cost(g, 0, 4, ablated_nodes=[2])  # severs the only path
    assert r["before"] == 4.0
    assert r["after"] == float("inf")
    assert r["ratio"] == float("inf")


def test_reroute_cost_detour_ratio():
    g = _grid(4)
    r = reroute_cost(g, (0, 0), (0, 3), ablated_nodes=[(0, 1)])  # force a detour
    assert r["before"] == 3.0
    assert r["after"] >= r["before"]          # detour is at least as long
    assert r["ratio"] >= 1.0
