"""CLI: run the full mask -> graph -> heal -> resilience pipeline on a tile.

Examples
--------
    # Densest tile in the manifest (default):
    python scripts/run_pipeline.py

    # A specific mask, custom hazard radius, save the JSON report:
    python scripts/run_pipeline.py --mask data/processed/masks/indiranagar_blr_r013_c004.tif \
        --radius 80 --save

Output: a human summary on stdout and (with --save) a JSON report under
artifacts/reports/.
"""

from __future__ import annotations

import argparse
import json

from route_resilience.config import load_config
from route_resilience.data.build import read_manifest
from route_resilience.paths import ARTIFACTS, PROCESSED
from route_resilience.pipeline import run_tile_pipeline
from route_resilience.utils import get_logger

log = get_logger("run_pipeline")


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end tile resilience pipeline.")
    ap.add_argument("--mask", default=None, help="mask GeoTIFF path (default: densest tile)")
    ap.add_argument("--radius", type=float, default=50.0, help="sample hazard radius (m)")
    ap.add_argument("--no-heal", action="store_true", help="skip graph healing")
    ap.add_argument("--save", action="store_true", help="write JSON report to artifacts/reports/")
    args = ap.parse_args()

    cfg = load_config("base.yaml", "data.yaml", "graph.yaml")
    mask = args.mask
    if mask is None:
        man = read_manifest(PROCESSED / "manifest.csv").sort_values("road_frac", ascending=False)
        mask = man.iloc[0]["mask_path"]

    rep = run_tile_pipeline(mask, cfg, heal=not args.no_heal, hazard_radius_m=args.radius)

    if rep.get("empty"):
        log.warning("empty graph for %s", mask)
        return

    gs, rs = rep["graph"], rep["resilience"]
    log.info("tile: %s", rep["mask_path"])
    log.info("graph: %d nodes, %d edges, %d components, %.0f m total",
             gs["n_nodes"], gs["n_edges"], gs["n_components"], gs["total_length_m"])
    log.info("top critical junction: lon=%.5f lat=%.5f (betweenness=%.3f)",
             rep["critical_nodes"][0]["lon"], rep["critical_nodes"][0]["lat"],
             rep["critical_nodes"][0]["betweenness"])
    log.info("flood r=%.0fm -> Resilience Index=%.2f (efficiency -%.0f%%), %d junctions, %d comps after",
             rep["sample_hazard"]["radius_m"], rs["resilience_index"],
             100 * rs["efficiency_drop"], rs["n_impacted"], rs["components_after"])

    if args.save:
        out_dir = ARTIFACTS / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = str(rep["mask_path"]).replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]
        out = out_dir / f"{stem}.json"
        out.write_text(json.dumps(rep, indent=2))
        log.info("saved report -> %s", out)


if __name__ == "__main__":
    main()
