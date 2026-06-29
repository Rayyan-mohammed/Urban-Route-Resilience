"""Evaluation orchestrator (M5) — turns a trained model into the §11 report.

For each tile in a split we run the model TWICE:
  - clean image  -> IoU, Dice, precision, recall, clDice, connectivity ratio
  - occluded image -> occlusion-recall (recall on the hidden road pixels)

Metrics are aggregated overall AND per terrain, so cross-terrain generalisation
is measured (§2.2). Output is a JSON report + a printed table. The same function
evaluates the baseline and the SegFormer+clDice model -> the money comparison.

Pixel metrics (IoU/Dice/P/R, occlusion-recall) are micro-averaged over pixels;
clDice and connectivity are macro-averaged over tiles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
from omegaconf import DictConfig, OmegaConf

from ..data.build import read_manifest
from ..data.dataset import build_transforms
from ..data.occlusion import apply_occlusion
from ..data.synth_image import synthesize_image
from ..models.baseline import build_model
from ..paths import METRICS, PROCESSED, ensure_dirs
from ..utils import get_logger
from .metrics import cldice_score, connectivity_ratio, occlusion_recall_counts

log = get_logger(__name__)


class _Acc:
    """Per-group metric accumulator."""

    def __init__(self):
        self.tp = self.fp = self.fn = 0.0
        self.cldice: list[float] = []
        self.conn: list[float] = []
        self.occ_tp = 0
        self.occ_denom = 0
        self.n = 0

    def add_clean(self, pred: np.ndarray, true: np.ndarray) -> None:
        p, t = pred.astype(bool), true.astype(bool)
        self.tp += (p & t).sum()
        self.fp += (p & ~t).sum()
        self.fn += (~p & t).sum()
        self.cldice.append(cldice_score(p, t))
        self.conn.append(connectivity_ratio(p, t))
        self.n += 1

    def add_occ(self, pred: np.ndarray, true: np.ndarray, occ_mask: np.ndarray) -> None:
        rec, denom = occlusion_recall_counts(pred, true, occ_mask)
        self.occ_tp += rec
        self.occ_denom += denom

    def compute(self) -> dict:
        e = 1e-7
        return {
            "n_tiles": self.n,
            "iou": self.tp / (self.tp + self.fp + self.fn + e),
            "dice": 2 * self.tp / (2 * self.tp + self.fp + self.fn + e),
            "precision": self.tp / (self.tp + self.fp + e),
            "recall": self.tp / (self.tp + self.fn + e),
            "cldice": float(np.mean(self.cldice)) if self.cldice else 0.0,
            "connectivity_ratio": float(np.mean(self.conn)) if self.conn else 0.0,
            "occlusion_recall": (self.occ_tp / self.occ_denom) if self.occ_denom else None,
        }


def _resolve_device(cfg) -> str:
    d = cfg.get("device", "auto")
    return ("cuda" if torch.cuda.is_available() else "cpu") if d == "auto" else d


def _load_pair(row, idx: int):
    with rasterio.open(row["mask_path"]) as ds:
        mask = (ds.read(1) > 0).astype(np.uint8)
    ip = row.get("image_path", "")
    if isinstance(ip, str) and ip:
        with rasterio.open(ip) as ds:
            img = ds.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8)
    else:
        img = synthesize_image(mask, seed=1000 + idx)
    return img, mask


def _load_model(cfg: DictConfig, checkpoint: str | None, device: str, dry_run: bool):
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        mcfg = OmegaConf.create(ckpt["cfg"])
        mcfg = OmegaConf.merge(mcfg, OmegaConf.create({"model": {"encoder_weights": None}}))
        model = build_model(mcfg)
        model.load_state_dict(ckpt["model"])
        log.info("loaded checkpoint %s (epoch %s)", Path(checkpoint).name, ckpt.get("epoch"))
    else:
        if dry_run:
            cfg = OmegaConf.merge(cfg, OmegaConf.create({"model": {"encoder_weights": None}}))
        model = build_model(cfg)
        log.warning("no checkpoint -> evaluating an UNTRAINED model (pipeline check only)")
    return model.to(device).eval()


@torch.no_grad()
def _predict(model, img_t: torch.Tensor, threshold: float, device: str) -> np.ndarray:
    logits = model(img_t.unsqueeze(0).to(device))
    return (torch.sigmoid(logits) > threshold).cpu().numpy()[0, 0]


def evaluate(
    cfg: DictConfig,
    *,
    checkpoint: str | None = None,
    split: str = "test",
    dry_run: bool = False,
    max_tiles: int | None = None,
) -> dict:
    ensure_dirs()
    device = "cpu" if dry_run else _resolve_device(cfg)
    model = _load_model(cfg, checkpoint, device, dry_run)
    tf = build_transforms(train=False)
    threshold = float(cfg.train.threshold)
    occ = cfg.data.occlusion

    rows = read_manifest(PROCESSED / "manifest.csv")
    rows = rows[rows["split"] == split].reset_index(drop=True)
    if max_tiles is None and dry_run:
        max_tiles = 8
    if max_tiles:
        rows = rows.head(max_tiles)
    log.info("evaluating %d tiles (split=%s, device=%s)", len(rows), split, device)

    overall = _Acc()
    per: dict[str, _Acc] = {t: _Acc() for t in rows["terrain"].unique()}

    for idx, row in rows.iterrows():
        img, mask = _load_pair(row, idx)
        clean_pred = _predict(model, tf(image=img, mask=mask)["image"].float(), threshold, device)
        # Deterministic, always-on occlusion for the occlusion-recall measurement.
        res = apply_occlusion(
            img, mask, types=tuple(occ.types), weights=list(occ.weights),
            coverage_range=tuple(occ.coverage_range), apply_prob=1.0,
            max_occluders=int(occ.max_occluders), seed=2000 + int(idx),
        )
        occ_pred = _predict(model, tf(image=res.image, mask=mask)["image"].float(), threshold, device)

        for acc in (overall, per[row["terrain"]]):
            acc.add_clean(clean_pred, mask)
            acc.add_occ(occ_pred, mask, res.occluded_road_mask)

    report = {
        "split": split,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "arch": f"{cfg.model.arch}/{cfg.model.encoder}",
        "overall": overall.compute(),
        "per_terrain": {k: v.compute() for k, v in per.items()},
    }
    if not dry_run:
        stem = Path(checkpoint).stem if checkpoint else cfg.model.arch.lower()
        out = METRICS / f"eval_{stem}_{split}.json"
        import json
        out.write_text(json.dumps(report, indent=2))
        log.info("saved report -> %s", out)
    return report


# ------------------------------ pretty print ------------------------------
_COLS = ["iou", "dice", "cldice", "connectivity_ratio", "recall", "occlusion_recall"]


def _fmt(v) -> str:
    return "  n/a " if v is None else f"{v:6.3f}"


def print_report(report: dict) -> None:
    log.info("=== %s | %s | split=%s ===", report["arch"], report["checkpoint"], report["split"])
    head = "  ".join(f"{c[:9]:>9s}" for c in _COLS)
    print(f"{'group':12s}  {head}")
    o = report["overall"]
    print(f"{'OVERALL':12s}  " + "  ".join(f"{_fmt(o[c]):>9s}" for c in _COLS))
    for terr, m in report["per_terrain"].items():
        print(f"{terr:12s}  " + "  ".join(f"{_fmt(m[c]):>9s}" for c in _COLS))


def print_comparison(rep_a: dict, rep_b: dict) -> None:
    """Side-by-side OVERALL comparison (e.g. baseline vs SegFormer+clDice)."""
    a, b = rep_a["overall"], rep_b["overall"]
    print(f"\n{'metric':20s}  {rep_a['arch'][:14]:>14s}  {rep_b['arch'][:14]:>14s}  {'delta':>8s}")
    for c in _COLS:
        va, vb = a[c], b[c]
        d = "" if (va is None or vb is None) else f"{vb - va:+.3f}"
        print(f"{c:20s}  {_fmt(va):>14s}  {_fmt(vb):>14s}  {d:>8s}")
