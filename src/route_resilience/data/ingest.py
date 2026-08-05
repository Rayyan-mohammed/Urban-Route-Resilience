"""Ingest real EO road datasets into the manifest contract (roadmap §3.2).

The OSMnx pipeline (build.py) gives geo-referenced *masks* but no imagery, so
training falls back to synthesised pseudo-images. That is only a plumbing
rehearsal. This module wires in the roadmap's real datasets so `image_path` is
populated and the model trains on ACTUAL satellite pixels:

    SpaceNet Roads      georeferenced GeoTIFF + road GeoJSON  -> rasterise labels
    DeepGlobe           <id>_sat.jpg + <id>_mask.png pairs
    OpenSatMap / other  generic image-dir + mask-dir (matched by filename stem)

Everything funnels through one core, `ingest_pairs`, which tiles each big
image+mask into `tile_size` sub-tiles, writes a 3-band image GeoTIFF next to a
1-band mask GeoTIFF, filters near-empty tiles, and returns manifest rows with the
SAME schema build.py uses — so the split, training, and eval code need no change.

Non-georeferenced datasets get a synthetic north-up transform at the dataset's
nominal GSD; that is enough for training/eval (the graph/resilience map is only
geographically meaningful for the OSM demo city + finale Cartosat tiles).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rasterio.transform import from_origin

from ..paths import PROCESSED
from ..utils import get_logger
from . import geo
from .build import MANIFEST_COLUMNS
from .geo import TileRef

log = get_logger(__name__)

# Nominal ground sample distance (m/pixel) per source; overridable at call time.
DEFAULT_GSD_M = {
    "deepglobe": 0.5,      # DigitalGlobe, ~0.5 m
    "spacenet": 0.3,       # WorldView-3 PS-RGB, ~0.3 m
    "massachusetts": 1.0,  # Massachusetts Roads aerial, ~1 m
    "opensatmap": 0.5,     # mixed high-res
    "folder": 0.5,
}
_PLACEHOLDER_CRS = "EPSG:32643"  # projected metres; real geo only for the demo city


def _tile_starts(size: int, tile: int, step: int) -> list[int]:
    """Start offsets covering [0, size) with `tile`-wide windows, incl. the edge."""
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, max(1, step)))
    if starts[-1] != size - tile:
        starts.append(size - tile)  # flush-right/bottom tile so edges aren't dropped
    return starts


def _tileref_real(base_id: str, r0: int, c0: int, tile: int, transform, crs: str) -> TileRef:
    """TileRef on the source's OWN transform/CRS, for genuinely geo-referenced imagery.

    Cartosat-3 (and any GeoTIFF with real geodesy) must keep its true coordinates
    so the extracted graph lands on a real basemap in the dashboard.
    """
    from rasterio.transform import array_bounds
    from rasterio.windows import Window
    from rasterio.windows import transform as window_transform

    win_tf = window_transform(Window(c0, r0, tile, tile), transform)
    w, s, e, n = array_bounds(tile, tile, win_tf)   # (west, south, east, north)
    return TileRef(
        tile_id=f"{base_id}_r{r0:04d}_c{c0:04d}",
        transform=win_tf,
        width=tile,
        height=tile,
        crs=str(crs),
        bounds=(w, s, e, n),
    )


def _tileref(base_id: str, r0: int, c0: int, tile: int, gsd: float, img_h: int) -> TileRef:
    """Synthetic north-up TileRef for a pixel window (top-left at row r0, col c0)."""
    # World origin: top-left of the whole source at (0, img_h*gsd); north-up.
    x0 = c0 * gsd
    y_top = (img_h - r0) * gsd
    transform = from_origin(x0, y_top, gsd, gsd)
    bounds = (x0, y_top - tile * gsd, x0 + tile * gsd, y_top)
    return TileRef(
        tile_id=f"{base_id}_r{r0:04d}_c{c0:04d}",
        transform=transform,
        width=tile,
        height=tile,
        crs=_PLACEHOLDER_CRS,
        bounds=bounds,
    )


def ingest_pairs(
    pairs: Iterable[tuple[np.ndarray, np.ndarray, str]],
    cfg: DictConfig,
    *,
    source: str,
    terrain: str,
    gsd_m: float | None = None,
    out_dir: Path | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Tile (image_HWC, mask_HW, base_id) triples into manifest rows with imagery.

    - image: uint8 (H,W,3); mask: any (H,W) where >0 means road.
    - Writes `data/processed/images/<tile>.tif` (3-band) + `masks/<tile>.tif`.
    - Keeps only tiles with road fraction >= cfg.data.min_road_frac.

    Adapters may yield a 4th element, `(transform, crs)`, for imagery that carries
    real geodesy (Cartosat-3, SpaceNet). Those tiles keep their true coordinates
    instead of the synthetic north-up placeholder grid, so the graph they produce
    lands on a real basemap.
    """
    d = cfg.data
    tile = int(d.tile_size)
    overlap = float(d.overlap)
    gsd = float(gsd_m if gsd_m is not None else DEFAULT_GSD_M.get(source, 0.5))
    step = max(1, int(tile * (1.0 - overlap)))
    out_dir = Path(out_dir) if out_dir else PROCESSED
    img_dir, mask_dir = out_dir / "images", out_dir / "masks"

    rows: list[dict] = []
    n_src = 0
    for pair in pairs:
        if limit is not None and n_src >= limit:
            break
        n_src += 1
        # 3-tuple = synthetic grid; 4-tuple carries the source's real (transform, crs).
        image, mask, base_id = pair[0], pair[1], pair[2]
        georef = pair[3] if len(pair) > 3 else None
        if image.ndim != 3 or image.shape[2] != 3:
            log.warning("skip %s: image not (H,W,3), got %s", base_id, image.shape)
            continue
        h, w = image.shape[:2]
        if mask.shape[:2] != (h, w):
            log.warning("skip %s: mask %s != image %s", base_id, mask.shape, (h, w))
            continue
        mbin = (mask > 0).astype(np.uint8)
        for r0 in _tile_starts(h, tile, step):
            for c0 in _tile_starts(w, tile, step):
                m = mbin[r0 : r0 + tile, c0 : c0 + tile]
                road_frac = float(m.mean())
                if road_frac < float(d.min_road_frac):
                    continue
                if georef is not None:
                    ref = _tileref_real(f"{source}_{base_id}", r0, c0, tile,
                                        georef[0], georef[1])
                else:
                    ref = _tileref(f"{source}_{base_id}", r0, c0, tile, gsd, h)
                img_t = image[r0 : r0 + tile, c0 : c0 + tile].astype(np.uint8)
                img_path = img_dir / f"{ref.tile_id}.tif"
                mask_path = mask_dir / f"{ref.tile_id}.tif"
                geo.save_image_geotiff(img_path, img_t, ref)
                geo.save_mask_geotiff(mask_path, m, ref)
                rows.append({
                    "tile_id": ref.tile_id, "place": source, "terrain": terrain,
                    "mask_path": str(mask_path), "image_path": str(img_path),
                    "crs": ref.crs, "west": ref.bounds[0], "south": ref.bounds[1],
                    "east": ref.bounds[2], "north": ref.bounds[3],
                    # Real pixel size when the source is geo-referenced; the
                    # synthetic transform carries `gsd` here, so this covers both.
                    "width": tile, "height": tile,
                    "resolution_m": abs(float(ref.transform.a)),
                    "road_frac": road_frac, "split": "",
                })
    log.info("%s: %d source images -> %d road-dense tiles", source, n_src, len(rows))
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


# --------------------------- source adapters -----------------------------
def _read_rgb(path: Path) -> np.ndarray:
    """Read an image file as uint8 (H,W,3). PIL avoids the Windows GDAL/JPEG quirks."""
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _read_mask(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def iter_deepglobe(root: Path) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
    """DeepGlobe Road Extraction: `<id>_sat.jpg` + `<id>_mask.png` in one folder."""
    root = Path(root)
    sats = sorted(root.rglob("*_sat.jpg")) + sorted(root.rglob("*_sat.png"))
    if not sats:
        log.warning("deepglobe: no *_sat.{jpg,png} under %s", root)
    for sat in sats:
        mask = sat.with_name(sat.name.replace("_sat.", "_mask."))
        if mask.suffix == ".jpg":
            mask = mask.with_suffix(".png")
        if not mask.exists():
            # test split ships images without masks — skip silently
            continue
        base = sat.name.split("_sat.")[0]
        yield _read_rgb(sat), _read_mask(mask), base


def iter_folder(images: Path, masks: Path) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
    """Generic (OpenSatMap etc.): image dir + mask dir matched by filename stem."""
    images, masks = Path(images), Path(masks)
    mask_by_stem = {p.stem: p for p in masks.rglob("*") if p.suffix.lower() in
                    (".png", ".tif", ".tiff", ".jpg", ".jpeg")}
    for img in sorted(images.rglob("*")):
        if img.suffix.lower() not in (".png", ".tif", ".tiff", ".jpg", ".jpeg"):
            continue
        m = mask_by_stem.get(img.stem)
        if m is None:
            continue
        yield _read_rgb(img), _read_mask(m), img.stem


def iter_spacenet(root: Path, *, road_buffer_m: float = 2.0) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
    """SpaceNet Roads: georeferenced RGB GeoTIFFs + road GeoJSON labels.

    Imagery under `**/PS-RGB/*.tif` (or `**/RGB-PanSharpen/*.tif`); labels under
    `**/geojson*/**/*.geojson`, matched by the shared AOI/chip id in the filename.
    Roads are reprojected to the image CRS and rasterised on the image's own
    transform, so labels align to pixels exactly.
    """
    import geopandas as gpd
    import rasterio

    root = Path(root)
    imgs = sorted(root.rglob("*PS-RGB*.tif")) + sorted(root.rglob("*RGB-PanSharpen*.tif"))
    geojsons = list(root.rglob("*.geojson"))
    by_id: dict[str, Path] = {}
    for g in geojsons:
        by_id[_chip_id(g.stem)] = g
    if not imgs:
        log.warning("spacenet: no PS-RGB/RGB-PanSharpen GeoTIFFs under %s", root)
    for img_path in imgs:
        cid = _chip_id(img_path.stem)
        gj = by_id.get(cid)
        if gj is None:
            continue
        with rasterio.open(img_path) as ds:
            bands = min(3, ds.count)
            arr = ds.read(list(range(1, bands + 1))).transpose(1, 2, 0)
            img = _to_uint8_rgb(arr)
            ref = TileRef("sn", ds.transform, ds.width, ds.height, str(ds.crs),
                          tuple(ds.bounds))
            crs = ds.crs
        roads = gpd.read_file(gj)
        if roads.empty:
            mask = np.zeros((ref.height, ref.width), np.uint8)
        else:
            roads = roads.to_crs(crs)
            mask = geo.rasterize_roads(roads.geometry, ref, road_buffer_m)
        yield img, mask, cid


def iter_massachusetts(root: Path) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
    """Massachusetts Roads: `<split>/` image dirs beside `<split>_labels/` mask dirs.

    The Kaggle release (`balraj98/massachusetts-roads-dataset`) ships
    `tiff/{train,val,test}` next to `tiff/{train,val,test}_labels`, 1500x1500 aerial
    tiles at ~1 m GSD with white-on-black road labels. Rather than hardcoding those
    three names we pair ANY `<x>/` with its `<x>_labels/` sibling, so a re-packaged
    copy still ingests.

    Note the 1 m GSD: stacking this with DeepGlobe's 0.5 m gives the
    multi-resolution pretraining §3.3 asks for, and the `terrain` column keeps the
    two domains balanced across the train/val/test split.
    """
    root = Path(root)
    pairs: list[tuple[Path, Path]] = []
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        if d.name.endswith("_labels"):
            continue
        labels = d.with_name(d.name + "_labels")
        if labels.is_dir():
            pairs.append((d, labels))

    if not pairs:
        log.warning("massachusetts: no '<x>/ + <x>_labels/' dir pairs under %s", root)
        return
    log.info("massachusetts: %d split(s): %s", len(pairs), [p[0].name for p in pairs])
    for img_dir, mask_dir in pairs:
        yield from iter_folder(img_dir, mask_dir)


def _footprint_radius_m(bounds, crs) -> float:
    """Half-diagonal of a footprint in metres, for any CRS."""
    west, south, east, north = bounds
    if getattr(crs, "is_geographic", False):
        import math

        mid_lat = math.radians((south + north) / 2.0)
        w_m = abs(east - west) * 111_320.0 * max(0.1, math.cos(mid_lat))
        h_m = abs(north - south) * 110_540.0
    else:
        w_m, h_m = abs(east - west), abs(north - south)
    return 0.5 * (w_m**2 + h_m**2) ** 0.5


def iter_geotiff_osm(
    root: Path,
    *,
    road_buffer_m: float = 2.0,
    network_type: str = "drive",
) -> Iterator[tuple[np.ndarray, np.ndarray, str, tuple]]:
    """Geo-referenced imagery + OSM road labels fetched for each image's footprint.

    This is the finale adapter. Cartosat-3 tiles arrive as GeoTIFFs with **no
    labels**, so there is nothing to fine-tune against. Here we read each image's
    real footprint, pull the OSM drivable network covering it, reproject to the
    image CRS and burn it onto the image's own pixel grid — giving aligned
    (imagery, label) pairs with zero manual annotation.

    Needs internet (OSMnx). At a venue without it, pre-cache the masks before the
    event and use `--source folder --images ... --masks ...` instead.
    """
    import rasterio
    from rasterio.warp import transform_bounds

    from .osm import road_edges_for_place

    root = Path(root)
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".tif", ".tiff"))
    if not images:
        log.warning("geotiff-osm: no GeoTIFFs under %s", root)

    for img_path in images:
        with rasterio.open(img_path) as ds:
            if ds.crs is None:
                log.warning("skip %s: no CRS — cannot locate it on OSM", img_path.name)
                continue
            bands = min(3, ds.count)
            arr = ds.read(list(range(1, bands + 1))).transpose(1, 2, 0)
            img = _to_uint8_rgb(arr)
            transform, crs, bounds = ds.transform, ds.crs, tuple(ds.bounds)
            ref = TileRef(img_path.stem, transform, ds.width, ds.height, str(crs), bounds)

        # OSMnx wants a (lat, lon) centre + a radius that covers the whole footprint.
        west, south, east, north = transform_bounds(crs, "EPSG:4326", *bounds)
        centre = ((south + north) / 2.0, (west + east) / 2.0)
        dist_m = int(_footprint_radius_m(bounds, crs) * 1.15) + 100   # margin

        try:
            edges, _ = road_edges_for_place(
                point=centre, dist_m=dist_m, network_type=network_type)
        except Exception as exc:                      # offline / no roads / geocode fail
            log.warning("skip %s: OSM fetch failed (%s)", img_path.name, exc)
            continue
        if edges is None or edges.empty:
            log.warning("skip %s: OSM returned no roads for its footprint", img_path.name)
            continue

        mask = geo.rasterize_roads(edges.to_crs(crs).geometry, ref, road_buffer_m)
        log.info("%s: %d OSM segments -> road_frac=%.4f",
                 img_path.name, len(edges), float((mask > 0).mean()))
        yield img, mask, img_path.stem, (transform, str(crs))


def _chip_id(stem: str) -> str:
    """Best-effort shared id: the last 'chipN'/'imgN' or trailing token of a name."""
    for tok in reversed(stem.replace("-", "_").split("_")):
        if tok and (tok[0].isdigit() or tok.lower().startswith(("chip", "img"))):
            return tok
    return stem


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """Per-band percentile stretch of (possibly 16-bit) imagery to uint8 RGB."""
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    arr = arr[:, :, :3].astype(np.float32)
    out = np.zeros_like(arr, dtype=np.uint8)
    for b in range(3):
        band = arr[:, :, b]
        lo, hi = np.percentile(band, 2), np.percentile(band, 98)
        if hi <= lo:
            hi = lo + 1.0
        out[:, :, b] = np.clip((band - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return out


def ingest_source(
    source: str,
    cfg: DictConfig,
    *,
    terrain: str,
    root: Path | None = None,
    images: Path | None = None,
    masks: Path | None = None,
    gsd_m: float | None = None,
    limit: int | None = None,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Dispatch to the right adapter and tile it into manifest rows."""
    if source == "deepglobe":
        pairs = iter_deepglobe(root)
    elif source == "massachusetts":
        pairs = iter_massachusetts(root)
    elif source == "spacenet":
        pairs = iter_spacenet(root, road_buffer_m=float(cfg.data.osm.road_buffer_m))
    elif source in ("opensatmap", "folder"):
        if images is None or masks is None:
            raise ValueError(f"source={source} needs --images and --masks dirs")
        pairs = iter_folder(images, masks)
    elif source in ("geotiff-osm", "cartosat"):
        pairs = iter_geotiff_osm(
            root,
            road_buffer_m=float(cfg.data.osm.road_buffer_m),
            network_type=str(cfg.data.osm.network_type),
        )
    else:
        raise ValueError(f"unknown source: {source!r}")
    return ingest_pairs(pairs, cfg, source=source, terrain=terrain,
                        gsd_m=gsd_m, out_dir=out_dir, limit=limit)
