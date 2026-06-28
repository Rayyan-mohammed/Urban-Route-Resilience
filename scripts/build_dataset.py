"""CLI: build the OSMnx-only road-mask dataset + terrain-stratified split.

Examples
--------
    # Build all demo places from configs/places.yaml, then split:
    python scripts/build_dataset.py

    # Quick single-neighbourhood test run (small, fast):
    python scripts/build_dataset.py --only indiranagar_blr

    # Override any config key on the fly:
    python scripts/build_dataset.py -o data.resolution_m=1.0 -o data.tile_size=256

Outputs
-------
    data/processed/masks/<tile_id>.tif   geo-referenced binary road masks
    data/processed/manifest.csv          the dataset contract (with split column)
"""

from __future__ import annotations

import argparse

import pandas as pd

from route_resilience.config import load_config
from route_resilience.data import build, split
from route_resilience.paths import PROCESSED, ensure_dirs
from route_resilience.utils import get_logger

log = get_logger("build_dataset")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build OSMnx road-mask dataset.")
    ap.add_argument("--places", default="places.yaml", help="places config file")
    ap.add_argument("--only", default=None, help="build only this place `name`")
    ap.add_argument("-o", "--override", action="append", default=[],
                    help="OmegaConf dotlist override, e.g. data.tile_size=256")
    args = ap.parse_args()

    ensure_dirs()
    cfg = load_config("base.yaml", "data.yaml", overrides=args.override)
    places_cfg = load_config(args.places)

    frames = []
    for entry in places_cfg.places:
        e = dict(entry)
        if args.only and e.get("name") != args.only:
            continue
        log.info("=== building %s (%s) ===", e.get("name") or e.get("place"), e["terrain"])
        df = build.build_place(
            cfg,
            terrain=e["terrain"],
            place=e.get("place"),
            point=tuple(e["point"]) if e.get("point") else None,
            dist_m=int(e.get("dist_m", 1500)),
            name=e.get("name"),
        )
        if len(df):
            frames.append(df)

    if not frames:
        log.warning("No tiles produced — check --only name or network access.")
        return

    manifest = pd.concat(frames, ignore_index=True)
    manifest = split.stratified_split(manifest, cfg, seed=cfg.seed)
    build.write_manifest(manifest, PROCESSED / "manifest.csv")

    log.info("Split summary:\n%s", split.split_summary(manifest).to_string())


if __name__ == "__main__":
    main()
