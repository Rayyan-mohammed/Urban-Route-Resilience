"""Geospatial core for the data pipeline — sensor-agnostic tiling + rasterisation.

This module knows nothing about OSM or any specific sensor. It answers two
questions that every data source (OSMnx now, Sentinel/LISS/Cartosat later) needs:

  1. Given a geographic extent, what is the grid of geo-referenced tiles?
     -> `make_tile_grid` returns `TileRef`s carrying an affine transform + CRS,
        so a pixel (row, col) in any tile maps back to real-world coordinates.
        That round-trip is what lets Phase II place the extracted graph on a map.

  2. Given road vectors and a tile's transform, what is the aligned binary mask?
     -> `rasterize_roads` burns buffered road centerlines onto the tile grid.

Keeping geo-referencing first-class (not throwing away coordinates after tiling)
is the design decision that makes the whole "mask -> routable graph -> map" chain
possible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from affine import Affine
from rasterio.features import rasterize
from rasterio.transform import from_origin


@dataclass(frozen=True)
class TileRef:
    """A geo-referenced tile footprint (no pixel data — just where/how big)."""

    tile_id: str
    transform: Affine  # pixel (col,row) -> world (x,y); north-up
    width: int
    height: int
    crs: str  # e.g. "EPSG:32643" (UTM 43N for Bengaluru)
    bounds: tuple[float, float, float, float]  # (west, south, east, north) in CRS units


def make_tile_grid(
    bounds: tuple[float, float, float, float],
    crs: str,
    tile_size: int,
    resolution_m: float,
    overlap: float,
    place: str = "tile",
) -> list[TileRef]:
    """Tile a projected extent into overlapping square tiles.

    Parameters
    ----------
    bounds : (west, south, east, north) in projected (metre) CRS units.
    crs : CRS string, e.g. "EPSG:32643". Must be a *projected* CRS (metres),
          because tiles are sized in metres = tile_size * resolution_m.
    tile_size : tile edge length in pixels (e.g. 512).
    resolution_m : ground sample distance per pixel (e.g. 0.5 m).
    overlap : fractional overlap between adjacent tiles in [0, 1).

    Returns
    -------
    list[TileRef], row-major (north-to-south, west-to-east).
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    west, south, east, north = bounds
    extent_m = tile_size * resolution_m
    step_m = extent_m * (1.0 - overlap)

    # West edges (cols) and north edges (rows). Last tile may overhang the extent;
    # that is fine — rasterise just leaves the overhang empty.
    x_starts = np.arange(west, east, step_m)
    y_starts = np.arange(south, north, step_m)  # south edges

    tiles: list[TileRef] = []
    n_rows = len(y_starts)
    for ri, y0 in enumerate(reversed(y_starts)):  # north-to-south for intuitive row idx
        row = n_rows - 1 - ri  # not used in id beyond ordering clarity
        for ci, x0 in enumerate(x_starts):
            tile_bounds = (x0, y0, x0 + extent_m, y0 + extent_m)
            # north-up transform: origin at top-left = (west, north_of_tile)
            transform = from_origin(x0, y0 + extent_m, resolution_m, resolution_m)
            tile_id = f"{place}_r{ri:03d}_c{ci:03d}"
            tiles.append(
                TileRef(
                    tile_id=tile_id,
                    transform=transform,
                    width=tile_size,
                    height=tile_size,
                    crs=crs,
                    bounds=tile_bounds,
                )
            )
    return tiles


def rasterize_roads(geometries, tile: TileRef, buffer_m: float) -> np.ndarray:
    """Burn road geometries onto a tile grid as a uint8 binary mask {0,1}.

    Centerlines are buffered by `buffer_m` (half road width) so thin roads survive
    rasterisation at coarse resolution. `all_touched=True` keeps connectivity —
    critical, because a 1-pixel break is a topological break (the whole §8 USP).

    `geometries` : iterable of shapely geometries already in the tile's CRS.
    """
    shapes = []
    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        g = geom.buffer(buffer_m) if buffer_m and buffer_m > 0 else geom
        if not g.is_empty:
            shapes.append((g, 1))
    if not shapes:
        return np.zeros((tile.height, tile.width), dtype=np.uint8)
    return rasterize(
        shapes,
        out_shape=(tile.height, tile.width),
        transform=tile.transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    )


def save_mask_geotiff(path, mask: np.ndarray, tile: TileRef) -> None:
    """Persist a mask as a single-band GeoTIFF, preserving geo-referencing.

    We keep the transform + CRS so Phase II can convert mask pixels back to
    world coordinates without re-deriving them.
    """
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=tile.height,
        width=tile.width,
        count=1,
        dtype="uint8",
        crs=tile.crs,
        transform=tile.transform,
        compress="deflate",
    ) as dst:
        dst.write(mask, 1)
