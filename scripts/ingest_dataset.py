"""CLI: ingest a real EO road dataset into the manifest (roadmap §3.2).

Populates `image_path` with REAL imagery so training uses satellite pixels, not
the synthetic placeholder. Supports the roadmap's datasets:

    # DeepGlobe (one folder of <id>_sat.jpg + <id>_mask.png):
    python scripts/ingest_dataset.py --source deepglobe \
        --root /kaggle/input/deepglobe-road-extraction-dataset/train --terrain deepglobe

    # SpaceNet Roads (georeferenced GeoTIFF + GeoJSON labels):
    python scripts/ingest_dataset.py --source spacenet --root /path/to/spacenet --terrain spacenet

    # Massachusetts Roads (tiff/{train,val,test} + tiff/{train,val,test}_labels):
    python scripts/ingest_dataset.py --source massachusetts \
        --root /kaggle/input/massachusetts-roads-dataset/tiff --terrain massachusetts --append

    # OpenSatMap / any image-dir + mask-dir:
    python scripts/ingest_dataset.py --source opensatmap \
        --images /path/imgs --masks /path/masks --terrain opensatmap

    # FINALE: Cartosat-3 GeoTIFFs with NO labels — OSM roads are fetched for each
    # image's own footprint and burned onto its pixel grid (needs internet):
    python scripts/ingest_dataset.py --source cartosat \
        --root data/raw/cartosat --terrain cartosat --append

Add `--append` to MERGE into the existing manifest (e.g. keep the OSM tiles and
add DeepGlobe on top); omit it to start a fresh manifest from just this source.
Re-runs the terrain-stratified split every time so `split` stays valid.
"""

from __future__ import annotations

import argparse

import pandas as pd

from route_resilience.config import load_config
from route_resilience.data import ingest, split
from route_resilience.data.build import read_manifest, write_manifest
from route_resilience.paths import PROCESSED, ensure_dirs
from route_resilience.utils import get_logger

log = get_logger("ingest_dataset")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest a real road dataset into the manifest.")
    ap.add_argument("--source", required=True,
                    choices=["deepglobe", "massachusetts", "spacenet", "opensatmap",
                             "folder", "geotiff-osm", "cartosat"])
    ap.add_argument("--root", default=None,
                    help="dataset root (deepglobe/massachusetts/spacenet/geotiff-osm)")
    ap.add_argument("--images", default=None, help="image dir (opensatmap/folder)")
    ap.add_argument("--masks", default=None, help="mask dir (opensatmap/folder)")
    ap.add_argument("--terrain", default=None, help="stratum label (default: source name)")
    ap.add_argument("--gsd", type=float, default=None, help="ground sample distance m/pixel")
    ap.add_argument("--limit", type=int, default=None, help="cap source images (quick runs)")
    ap.add_argument("--append", action="store_true", help="merge into existing manifest")
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()

    ensure_dirs()
    cfg = load_config("base.yaml", "data.yaml", overrides=args.override)
    terrain = args.terrain or args.source

    df = ingest.ingest_source(
        args.source, cfg, terrain=terrain,
        root=args.root, images=args.images, masks=args.masks,
        gsd_m=args.gsd, limit=args.limit,
    )
    if df.empty:
        # Exit non-zero: a silent return here hands the caller a missing manifest
        # and the real failure only surfaces cells later, far from its cause.
        log.error("no tiles ingested from source=%s root=%s images=%s masks=%s",
                  args.source, args.root, args.images, args.masks)
        log.error("check the path exists and matches the expected layout "
                  "(see docs/DATASETS.md); nothing was written.")
        raise SystemExit(1)

    manifest_path = PROCESSED / "manifest.csv"
    if args.append and manifest_path.exists():
        existing = read_manifest(manifest_path)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset="tile_id", keep="last").reset_index(drop=True)
        log.info("appended -> %d total tiles", len(df))

    df = split.stratified_split(df, cfg, seed=cfg.seed)
    write_manifest(df, manifest_path)
    log.info("Split summary:\n%s", split.split_summary(df).to_string())


if __name__ == "__main__":
    main()
