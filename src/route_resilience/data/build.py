"""Dataset builder — orchestrates place -> geo-referenced mask tiles + manifest.

Flow (OSMnx-only mode):
    place/point --osm.py--> projected road edges
                --geo.make_tile_grid--> grid of TileRefs
       per tile --geo.rasterize_roads--> aligned binary mask -> GeoTIFF
                --filter by road fraction (§3.3)--> keep road-dense tiles
                --> append a row to the manifest

The MANIFEST is the contract every later milestone reads: it lists each tile's
file path, geo-bounds, terrain, and road fraction. M2 occludes these, M3/M4 train
on them, M6 reloads geo-bounds to map the graph. When real imagery arrives, the
same manifest schema gains an `image_path` column and nothing downstream changes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from ..paths import PROCESSED
from ..utils import get_logger
from . import geo, osm

log = get_logger(__name__)

MANIFEST_COLUMNS = [
    "tile_id", "place", "terrain", "mask_path", "image_path", "crs",
    "west", "south", "east", "north", "width", "height", "resolution_m",
    "road_frac", "split",
]


def build_place(
    cfg: DictConfig,
    *,
    terrain: str,
    place: str | None = None,
    point: tuple[float, float] | None = None,
    dist_m: int = 1500,
    name: str | None = None,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Build mask tiles for one place; return its manifest rows as a DataFrame."""
    d = cfg.data
    out_dir = Path(out_dir) if out_dir else PROCESSED
    mask_dir = out_dir / "masks"

    edges, crs = osm.road_edges_for_place(
        place=place, point=point, dist_m=dist_m,
        network_type=d.osm.network_type, simplify=d.osm.simplify,
    )
    slug = name or (place or "place").split(",")[0].strip().replace(" ", "_").lower()
    bounds = tuple(float(v) for v in edges.total_bounds)  # (minx,miny,maxx,maxy)

    tiles = geo.make_tile_grid(
        bounds, crs, d.tile_size, d.resolution_m, d.overlap, place=slug
    )
    log.info("%s: %d candidate tiles over bounds %s", slug, len(tiles), bounds)

    rows: list[dict] = []
    for t in tiles:
        if len(rows) >= d.max_tiles_per_place:
            log.info("%s: hit max_tiles_per_place=%d", slug, d.max_tiles_per_place)
            break
        w, s, e, n = t.bounds
        # Spatial pre-filter: only edges whose bbox intersects this tile (fast).
        sub = edges.cx[w:e, s:n]
        if len(sub) == 0:
            continue
        mask = geo.rasterize_roads(sub.geometry, t, d.osm.road_buffer_m)
        road_frac = float(mask.mean())
        if road_frac < d.min_road_frac:
            continue
        mask_path = mask_dir / f"{t.tile_id}.tif"
        geo.save_mask_geotiff(mask_path, mask, t)
        rows.append({
            "tile_id": t.tile_id, "place": slug, "terrain": terrain,
            "mask_path": str(mask_path.relative_to(out_dir.parent.parent)) if out_dir.is_absolute() else str(mask_path),
            "image_path": "", "crs": t.crs,
            "west": w, "south": s, "east": e, "north": n,
            "width": t.width, "height": t.height, "resolution_m": d.resolution_m,
            "road_frac": road_frac, "split": "",
        })

    log.info("%s: kept %d road-dense tiles", slug, len(rows))
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def write_manifest(df: pd.DataFrame, path: Path) -> None:
    """Persist the manifest as CSV (git-diffable, human-readable)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("manifest: %d tiles -> %s", len(df), path)


def read_manifest(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"split": "string"}).fillna("")
