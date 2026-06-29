"""CLI: train a segmentation model (baseline U-Net by default).

Examples
--------
    # Local pipeline smoke test (CPU, few tiles, 1 epoch, no weight download):
    python scripts/train.py --dry-run

    # Full baseline training (use a GPU box / Colab):
    python scripts/train.py

    # Train the M4 SegFormer later by swapping the model config:
    python scripts/train.py --config base.yaml data.yaml model_segformer.yaml train.yaml

    # Override anything:
    python scripts/train.py -o train.epochs=60 -o train.lr=1e-4
"""

from __future__ import annotations

import argparse

from route_resilience.config import load_config
from route_resilience.training.trainer import train
from route_resilience.utils import get_logger

log = get_logger("train")

DEFAULT_CONFIG = ["base.yaml", "data.yaml", "model_baseline.yaml", "train.yaml"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a road-segmentation model.")
    ap.add_argument("--config", nargs="*", default=DEFAULT_CONFIG,
                    help="config YAMLs to merge (left->right)")
    ap.add_argument("--dry-run", action="store_true",
                    help="tiny CPU run to validate the pipeline end-to-end")
    ap.add_argument("-o", "--override", action="append", default=[],
                    help="OmegaConf dotlist override, e.g. train.lr=1e-4")
    args = ap.parse_args()

    cfg = load_config(*args.config, overrides=args.override)
    res = train(cfg, dry_run=args.dry_run)
    log.info("done. best val IoU = %.4f", res["best_iou"])


if __name__ == "__main__":
    main()
