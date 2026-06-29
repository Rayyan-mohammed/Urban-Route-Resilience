"""M6 tests — skeleton->graph tracing, geo-referencing, stats, and APLS.

Offline; needs scikit-image + networkx + scipy.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skimage")
pytest.importorskip("networkx")
pytest.importorskip("scipy")

import networkx as nx  # noqa: E402

from route_resilience.graph.build import graph_stats, mask_to_graph  # noqa: E402
from route_resilience.graph.metrics import graph_apls  # noqa: E402


def test_line_graph_two_nodes_one_edge():
    mask = np.zeros((21, 21), np.uint8)
    mask[10, 0:21] = 1                      # horizontal 1-px line
    g, _ = mask_to_graph(mask, resolution_m=1.0)
    assert g.number_of_nodes() == 2         # two endpoints
    assert g.number_of_edges() == 1
    length = next(iter(g.edges(data=True)))[2]["length"]
    assert length == pytest.approx(20.0, abs=1.0)  # 20 unit steps


def test_cross_graph_junction_and_endpoints():
    mask = np.zeros((21, 21), np.uint8)
    mask[10, 0:21] = 1
    mask[0:21, 10] = 1                       # plus shape
    g, _ = mask_to_graph(mask, resolution_m=1.0)  # simplify=True merges the tangle
    degrees = sorted(d for _, d in g.degree())
    assert g.number_of_nodes() == 5          # 1 junction + 4 endpoints (tangle merged)
    assert degrees[-1] == 4                   # central junction has degree 4
    assert degrees.count(1) == 4              # four endpoints
    assert g.number_of_edges() == 4


def test_node_georeferencing_with_transform():
    from affine import Affine
    mask = np.zeros((10, 10), np.uint8)
    mask[5, 0:10] = 1
    # 2 m/px, origin at world (1000, 2000) top-left, north-up.
    transform = Affine(2.0, 0, 1000.0, 0, -2.0, 2000.0)
    g, _ = mask_to_graph(mask, transform=transform, resolution_m=2.0)
    # endpoint at pixel (5,0): world x = 1000 + (0+0.5)*2 = 1001; y = 2000 - (5+0.5)*2 = 1989
    n = next(p for p in g.nodes() if g.nodes[p]["col"] == 0)
    assert g.nodes[n]["x"] == pytest.approx(1001.0)
    assert g.nodes[n]["y"] == pytest.approx(1989.0)


def test_graph_stats_keys():
    mask = np.zeros((21, 21), np.uint8)
    mask[10, 0:21] = 1
    g, _ = mask_to_graph(mask, resolution_m=1.0)
    st = graph_stats(g)
    assert set(st) >= {"n_nodes", "n_edges", "n_components", "total_length_m"}
    assert st["n_components"] == 1


def _chain():
    g = nx.Graph()
    for n, x in [(0, 0.0), (1, 10.0), (2, 20.0)]:
        g.add_node(n, x=x, y=0.0)
    g.add_edge(0, 1, length=10.0)
    g.add_edge(1, 2, length=10.0)
    return g


def test_apls_identical_is_one():
    assert graph_apls(_chain(), _chain(), snap_m=1.0) == pytest.approx(1.0)


def test_apls_drops_when_broken():
    broken = _chain()
    broken.remove_edge(1, 2)                 # node 2 now unreachable from 0
    score = graph_apls(_chain(), broken, snap_m=1.0)
    assert score < 1.0
