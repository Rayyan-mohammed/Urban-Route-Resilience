"""OSMnx fetch layer — place/point -> projected road-edge GeoDataFrame.

This is the *only* module that talks to OSM. It hands `build.py` a GeoDataFrame
of road centerlines already projected to a metre CRS, so all downstream tiling
maths are in metres. Isolating OSM here means swapping in real satellite imagery
later only touches `build.py`, never the tiling/rasterisation core.

Why `network_type="drive"` (configurable): we want the *routable* network — the
graph a planner reroutes on in the resilience twin (§7 Phase III). Footpaths and
service alleys would inflate the graph without adding routing value.
"""

from __future__ import annotations

from ..utils import get_logger

log = get_logger(__name__)


def road_edges_for_place(
    *,
    place: str | None = None,
    point: tuple[float, float] | None = None,
    dist_m: int = 1500,
    network_type: str = "drive",
    simplify: bool = True,
):
    """Fetch road centerlines and return (edges_gdf, crs_str).

    Provide EITHER `place` (a geocodable name) OR `point` (lat, lon) + `dist_m`.
    The returned GeoDataFrame is projected to its local UTM zone (metres), with
    one row per road segment (LineString geometry).
    """
    import osmnx as ox

    if place:
        log.info("OSMnx: graph_from_place(%r, %s)", place, network_type)
        graph = ox.graph_from_place(place, network_type=network_type, simplify=simplify)
    elif point:
        log.info("OSMnx: graph_from_point(%s, dist=%dm, %s)", point, dist_m, network_type)
        graph = ox.graph_from_point(
            point, dist=dist_m, network_type=network_type, simplify=simplify
        )
    else:
        raise ValueError("Provide either `place` or `point`.")

    # Project to local UTM (metres) so tile sizing in metres is correct.
    graph = ox.project_graph(graph)
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    crs_str = str(edges.crs)
    log.info("OSMnx: %d road segments, CRS=%s", len(edges), crs_str)
    return edges, crs_str
