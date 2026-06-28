"""Phase 0 — data pipeline: tiling, OSM auto-masks, synthetic occlusion, Dataset.

Public API (M1):
    geo.make_tile_grid / rasterize_roads / save_mask_geotiff   — tiling core
    osm.road_edges_for_place                                   — OSMnx fetch
    build.build_place / write_manifest / read_manifest         — orchestration
    split.stratified_split                                     — terrain split

M2 adds: occlusion synthetic generator.

NOTE: submodules are imported lazily (not here) so that lightweight code paths
(e.g. the terrain split) don't drag in the heavy geospatial stack (rasterio/GDAL).
Import what you need explicitly:  `from route_resilience.data import geo`.
"""

__all__ = ["geo", "osm", "build", "split"]
