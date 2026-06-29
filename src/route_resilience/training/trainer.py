"""Training engine (M3) — device-agnostic train/val loop with checkpointing.

Auto-resolves cuda->cpu (we have no local GPU), uses AMP only on CUDA, tracks the
best validation IoU, and saves the best checkpoint + a metrics history JSON. The
same engine trains the M4 SegFormer (only the config changes).

`dry_run=True` shrinks everything (few tiles, 1 epoch, CPU, no pretrained
download) so the whole pipeline can be validated locally before a Colab GPU run.
"""

from __future__ import annotations

import json
import time

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from ..data.build import read_manifest
from ..data.dataset import RoadTileDataset
from ..evaluation.metrics import SegMetrics
from ..models.baseline import build_model
from ..models.losses import DiceFocalLoss
from ..paths import CHECKPOINTS, METRICS, PROCESSED, ensure_dirs
from ..utils import get_logger

log = get_logger(__name__)


def resolve_device(cfg) -> str:
    d = cfg.get("device", "auto")
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return d


def _make_loaders(cfg, manifest, device: str, dry_run: bool):
    train_ds = RoadTileDataset(manifest, "train", cfg)
    val_ds = RoadTileDataset(manifest, "val", cfg, occlude=False)
    if dry_run:
        train_ds = Subset(train_ds, list(range(min(6, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(4, len(val_ds)))))
    pin = device == "cuda"
    nw = int(cfg.train.num_workers)
    tl = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                    num_workers=nw, pin_memory=pin, drop_last=False)
    vl = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                    num_workers=nw, pin_memory=pin)
    return tl, vl


def _train_epoch(model, loader, loss_fn, opt, device, scaler):
    model.train()
    total = 0.0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        opt.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = loss_fn(model(img), mask)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss = loss_fn(model(img), mask)
            loss.backward()
            opt.step()
        total += loss.item() * img.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def _validate(model, loader, loss_fn, device, threshold):
    model.eval()
    metrics = SegMetrics(threshold=threshold)
    total = 0.0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        logits = model(img)
        total += loss_fn(logits, mask).item() * img.size(0)
        metrics.update(logits, mask)
    out = metrics.compute()
    out["loss"] = total / len(loader.dataset)
    return out


def train(cfg, *, dry_run: bool = False) -> dict:
    ensure_dirs()
    if dry_run:
        cfg = OmegaConf.merge(cfg, OmegaConf.create({
            "device": "cpu",
            "model": {"encoder_weights": None},
            "train": {"epochs": 1, "batch_size": 2, "num_workers": 0},
        }))
    device = resolve_device(cfg)
    log.info("device=%s | arch=%s/%s | dry_run=%s",
             device, cfg.model.arch, cfg.model.encoder, dry_run)

    manifest = read_manifest(PROCESSED / "manifest.csv")
    tl, vl = _make_loaders(cfg, manifest, device, dry_run)
    log.info("tiles: train=%d val=%d", len(tl.dataset), len(vl.dataset))

    model = build_model(cfg).to(device)
    loss_fn = DiceFocalLoss(cfg.train.loss.dice_w, cfg.train.loss.focal_w)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if (device == "cuda" and cfg.train.amp) else None

    best_iou = -1.0
    history = []
    out_name = cfg.train.out_name
    for epoch in range(1, int(cfg.train.epochs) + 1):
        t0 = time.time()
        tr = _train_epoch(model, tl, loss_fn, opt, device, scaler)
        val = _validate(model, vl, loss_fn, device, cfg.train.threshold)
        log.info("epoch %d/%d | train_loss=%.4f val_loss=%.4f IoU=%.4f Dice=%.4f "
                 "P=%.4f R=%.4f (%.1fs)", epoch, cfg.train.epochs, tr, val["loss"],
                 val["iou"], val["dice"], val["precision"], val["recall"], time.time() - t0)
        history.append({"epoch": epoch, "train_loss": tr, **val})
        if val["iou"] > best_iou:
            best_iou = val["iou"]
            ckpt = CHECKPOINTS / f"{out_name}_best.pth"
            torch.save({
                "model": model.state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                "epoch": epoch, "metrics": val,
            }, ckpt)
            log.info("  saved best -> %s (IoU=%.4f)", ckpt.name, best_iou)

    METRICS.mkdir(parents=True, exist_ok=True)
    with open(METRICS / f"{out_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    return {"best_iou": best_iou, "history": history}
