"""CLI: evaluate a trained model, or compare two (baseline vs SegFormer+clDice).

Examples
--------
    # Pipeline smoke test (untrained model, few tiles, CPU):
    python scripts/evaluate.py --dry-run

    # Evaluate one checkpoint on the test split:
    python scripts/evaluate.py --checkpoint artifacts/checkpoints/segformer_cldice_best.pth

    # The money comparison (run after both are trained):
    python scripts/evaluate.py \
        --checkpoint artifacts/checkpoints/baseline_unet_best.pth \
        --compare    artifacts/checkpoints/segformer_cldice_best.pth
"""

from __future__ import annotations

import argparse

from route_resilience.config import load_config
from route_resilience.evaluation.evaluate import evaluate, print_comparison, print_report
from route_resilience.utils import get_logger

log = get_logger("evaluate")

DEFAULT_CONFIG = ["base.yaml", "data.yaml", "model_baseline.yaml", "train.yaml"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate / compare road-segmentation models.")
    ap.add_argument("--config", nargs="*", default=DEFAULT_CONFIG)
    ap.add_argument("--checkpoint", default=None, help="checkpoint to evaluate")
    ap.add_argument("--compare", default=None, help="second checkpoint to compare against")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--apls", action="store_true", help="also compute graph APLS (slower)")
    ap.add_argument("--tta", action="store_true", help="multi-scale + flip test-time augmentation")
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()

    use_tta = True if args.tta else None  # None -> fall back to cfg.eval.tta
    cfg = load_config(*args.config, overrides=args.override)
    rep_a = evaluate(cfg, checkpoint=args.checkpoint, split=args.split,
                     dry_run=args.dry_run, max_tiles=args.max_tiles,
                     compute_apls=args.apls, use_tta=use_tta)
    print_report(rep_a)

    if args.compare:
        rep_b = evaluate(cfg, checkpoint=args.compare, split=args.split,
                         dry_run=args.dry_run, max_tiles=args.max_tiles,
                         compute_apls=args.apls, use_tta=use_tta)
        print_report(rep_b)
        print_comparison(rep_a, rep_b)


if __name__ == "__main__":
    main()
