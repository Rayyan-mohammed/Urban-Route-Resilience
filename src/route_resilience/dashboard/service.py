"""Dashboard service layer (M9) — all twin logic behind a thin, testable API.

The Streamlit UI (M10) calls ONLY these functions, so every behaviour (load, heal,
criticality, flood, reroute, map render) is unit-tested without a browser.

Tiles are geo-referenced (real Bengaluru UTM coords), so we reproject node
coordinates UTM -> WGS84 (lon/lat) and overlay the graph on a real OpenStreetMap
basemap (roadmap §15). A real DEM hazard would slot into the same flood() call.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import rasterio
from pyproj import Transformer

from ..graph.build import mask_to_graph
from ..graph.heal import heal_graph
from ..resilience.centrality import node_betweenness
from ..resilience.hazard import RadiusHazard
from ..resilience.simulate import ablate, reroute_cost, resilience_report


def add_lonlat(g: nx.Graph, crs: str) -> None:
    """Attach WGS84 lon/lat to every node (in place), reprojected from `crs`."""
    tf = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    for _, d in g.nodes(data=True):
        lon, lat = tf.transform(d["x"], d["y"])
        d["lon"], d["lat"] = float(lon), float(lat)


def load_tile_graph(mask_path, resolution_m: float, *, heal: bool = True, max_gap_m: float = 40.0):
    """Read a mask GeoTIFF -> healed, geo-referenced graph (with lon/lat). Returns (g, crs)."""
    with rasterio.open(mask_path) as ds:
        mask = (ds.read(1) > 0).astype(np.uint8)
        transform = ds.transform
        crs = str(ds.crs)
    g, _ = mask_to_graph(mask, transform=transform, resolution_m=resolution_m)
    if heal:
        g, _ = heal_graph(g, max_gap_m=max_gap_m)
    if g.number_of_nodes():
        add_lonlat(g, crs)
    return g, crs


def nearest_node(g: nx.Graph, lon: float, lat: float, crs: str):
    """Node nearest to a clicked (lon, lat), matched in projected metres."""
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = tf.transform(lon, lat)
    return min(g.nodes(), key=lambda n: (g.nodes[n]["x"] - x) ** 2 + (g.nodes[n]["y"] - y) ** 2)


def flood(g: nx.Graph, center_node, radius_m: float):
    """Flood a radius around a node; return (report, impacted_set, recomputed_betweenness)."""
    cx, cy = g.nodes[center_node]["x"], g.nodes[center_node]["y"]
    hz = RadiusHazard(center=(cx, cy), radius_m=radius_m)
    report = resilience_report(g, hz)
    impacted = hz.impacted_nodes(g)
    bc_after = node_betweenness(ablate(g, impacted))
    return report, impacted, bc_after


def reroute(g: nx.Graph, src, dst, ablated_nodes=()):
    """Shortest path src->dst before vs after a closure. Returns dict with cost + coords."""
    cost = reroute_cost(g, src, dst, ablated_nodes=ablated_nodes)

    def coords(gg):
        try:
            return [(g.nodes[n]["lat"], g.nodes[n]["lon"])
                    for n in nx.shortest_path(gg, src, dst, weight="length")]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    return {
        "cost": cost,
        "before_path": coords(g),
        "after_path": coords(ablate(g, ablated_nodes)),
    }


def graph_center(g: nx.Graph) -> tuple[float, float]:
    lats = [d["lat"] for _, d in g.nodes(data=True)]
    lons = [d["lon"] for _, d in g.nodes(data=True)]
    return (float(np.mean(lats)), float(np.mean(lons)))


def heat_color(v: float) -> str:
    """Betweenness in [0,1] -> hex colour, blue (low) to red (high)."""
    r = int(60 + v * 175)
    g = int(90 - v * 50)
    b = int(210 - v * 170)
    return f"#{max(0, r):02x}{max(0, g):02x}{max(0, b):02x}"


def build_map(g: nx.Graph, bc: dict, *, impacted=None, before_path=None,
              after_path=None, hazard_center=None, hazard_radius_m=None):
    """Build a folium map of the graph: edges, betweenness-coloured nodes, hazard,
    and optional before/after reroute paths. Returns a folium.Map.
    """
    import folium

    impacted = impacted or set()
    norm = max(bc.values()) if bc else 1.0
    m = folium.Map(location=graph_center(g), zoom_start=16, tiles="OpenStreetMap",
                   control_scale=True)

    # edges
    for u, v, d in g.edges(data=True):
        pts = [(g.nodes[u]["lat"], g.nodes[u]["lon"]), (g.nodes[v]["lat"], g.nodes[v]["lon"])]
        if d.get("healed"):
            folium.PolyLine(pts, color="#e02828", weight=2, dash_array="5").add_to(m)
        else:
            faint = u in impacted or v in impacted
            folium.PolyLine(pts, color="#c8323c" if faint else "#8a8a90",
                            weight=1, opacity=0.5 if faint else 0.8).add_to(m)

    # nodes coloured by betweenness; impacted nodes marked red
    for n, d in g.nodes(data=True):
        if n in impacted:
            folium.CircleMarker((d["lat"], d["lon"]), radius=4, color="#e03232",
                                fill=True, fill_opacity=0.9).add_to(m)
            continue
        val = bc.get(n, 0.0) / norm if norm > 0 else 0.0
        folium.CircleMarker((d["lat"], d["lon"]), radius=2 + 4 * val,
                            color=heat_color(val), fill=True, fill_opacity=0.85,
                            tooltip=f"betweenness={bc.get(n, 0.0):.3f}").add_to(m)

    # hazard zone
    if hazard_center is not None and hazard_radius_m:
        folium.Circle(hazard_center, radius=hazard_radius_m, color="#f5d73c",
                      fill=False, weight=2).add_to(m)

    # reroute paths
    if before_path:
        folium.PolyLine(before_path, color="#37c871", weight=4, opacity=0.9,
                        tooltip="route before").add_to(m)
    if after_path:
        folium.PolyLine(after_path, color="#ff9b3c", weight=4, opacity=0.9,
                        tooltip="route after").add_to(m)
    return m
