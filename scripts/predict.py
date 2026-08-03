"""CLI: imagery + checkpoint -> road mask GeoTIFF (-> optional resilience report).

This is the finale's first command. Cartosat-3 tiles arrive as GeoTIFFs; this
turns them into geo-referenced masks the graph/twin/dashboard already consume.

Examples
--------
    # One tile -> mask
    python scripts/predict.py --image data/raw/cartosat_01.tif \
        --checkpoint artifacts/checkpoints/segformer_cldice_best.pth

    # With TTA, then run graph -> heal -> resilience in the same call
    python scripts/predict.py --image data/raw/cartosat_01.tif \
        --checkpoint artifacts/checkpoints/segformer_cldice_best.pth --tta --pipeline

    # A whole folder of tiles
    python scripts/predict.py --image-dir data/raw/cartosat/ \
        --checkpoint artifacts/checkpoints/segformer_cldice_best.pth --tta

Masks land in artifacts/predictions/ and reports in artifacts/reports/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from route_resilience.config import load_config
from route_resilience.inference import load_model_from_checkpoint, predict_geotiff
from route_resilience.paths import PREDICTIONS, REPORTS, ensure_dirs
from route_resilience.pipeline import run_tile_pipeline
from route_resilience.resilience.hazard import RasterHazard
from route_resilience.utils import get_logger

log = get_logger("predict")

IMAGE_SUFFIXES = (".tif", ".tiff", ".jp2", ".png", ".jpg")


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict road masks from imagery.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="single image GeoTIFF")
    src.add_argument("--image-dir", help="folder of images (all are predicted)")
    ap.add_argument("--checkpoint", required=True, help="trained *.pth")
    ap.add_argument("--out-dir", default=None, help="output dir (default artifacts/predictions)")
    ap.add_argument("--threshold", type=float, default=None, help="probability cutoff")
    ap.add_argument("--window", type=int, default=None, help="sliding-window size px")
    ap.add_argument("--overlap", type=float, default=None, help="window overlap fraction")
    ap.add_argument("--tta", action="store_true", help="multi-scale + flip TTA")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--no-stretch", action="store_true",
                    help="skip the percentile stretch (imagery is already 8-bit)")
    ap.add_argument("--save-probs", action="store_true", help="also write the probability map")
    ap.add_argument("--pipeline", action="store_true",
                    help="run graph -> heal -> resilience on each predicted mask")
    ap.add_argument("--radius", type=float, default=50.0, help="sample hazard radius (m)")
    ap.add_argument("--hazard-raster", default=None,
                    help="real DEM / flood-depth raster to drive the hazard (with --pipeline)")
    ap.add_argument("--hazard-mode", default="depth", choices=("depth", "elevation"),
                    help="depth: flooded above threshold | elevation: below flood level")
    ap.add_argument("--hazard-threshold", type=float, default=0.3,
                    help="flood depth (m) or water-level elevation (m)")
    args = ap.parse_args()

    ensure_dirs()
    cfg = load_config("base.yaml", "data.yaml", "train.yaml", "graph.yaml")
    device = _resolve_device(args.device)
    threshold = args.threshold if args.threshold is not None else float(cfg.train.threshold)
    window = args.window if args.window is not None else int(cfg.data.tile_size)
    overlap = args.overlap if args.overlap is not None else float(cfg.data.overlap)
    ecfg = cfg.get("eval", {})

    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)

    if args.image:
        images = [Path(args.image)]
    else:
        root = Path(args.image_dir)
        images = sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            log.error("no images found in %s", root)
            return

    out_dir = Path(args.out_dir) if args.out_dir else PREDICTIONS
    log.info("predicting %d image(s) on %s (tta=%s, threshold=%.2f)",
             len(images), device, args.tta, threshold)

    for img in images:
        info = predict_geotiff(
            img, model,
            out_path=out_dir / f"{img.stem}_pred.tif",
            threshold=threshold, window=window, overlap=overlap, device=device,
            tta=args.tta,
            tta_scales=tuple(ecfg.get("tta_scales", [0.75, 1.0, 1.25])),
            tta_flips=bool(ecfg.get("tta_flips", True)),
            bands=tuple(cfg.data.bands),
            stretch=False if args.no_stretch else None,
            save_probs=args.save_probs,
        )
        log.info("%s -> %s (road_frac=%.4f)", img.name, info["mask_path"], info["road_frac"])

        if not args.pipeline:
            continue

        hazard = None
        if args.hazard_raster:
            hazard = RasterHazard(args.hazard_raster, mode=args.hazard_mode,
                                  threshold=args.hazard_threshold, node_crs=info["crs"])
            log.info("hazard: %s (%s > %.2f)", args.hazard_raster,
                     args.hazard_mode, args.hazard_threshold)

        rep = run_tile_pipeline(info["mask_path"], cfg,
                                hazard_radius_m=args.radius, hazard=hazard)
        if rep.get("empty"):
            log.warning("empty graph for %s — nothing to route", img.name)
            continue
        rep["prediction"] = info
        gs, rs = rep["graph"], rep["resilience"]
        log.info("graph: %d nodes, %d edges, %d components, %.0f m",
                 gs["n_nodes"], gs["n_edges"], gs["n_components"], gs["total_length_m"])
        hz = rep["sample_hazard"]
        label = f"flood r={hz['radius_m']:.0f}m" if "radius_m" in hz else hz["type"]
        log.info("%s -> Resilience Index=%.2f (efficiency -%.0f%%), %d junctions hit",
                 label, rs["resilience_index"], 100 * rs["efficiency_drop"], rs["n_impacted"])
        out = REPORTS / f"{img.stem}_pred.json"
        out.write_text(json.dumps(rep, indent=2))
        log.info("saved report -> %s", out)


if __name__ == "__main__":
    main()
