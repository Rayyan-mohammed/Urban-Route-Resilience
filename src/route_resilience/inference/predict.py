"""Sliding-window inference — imagery + checkpoint -> geo-referenced road mask.

Everything downstream of Phase I (graph, healing, twin, dashboard) starts from a
mask GeoTIFF. Training and evaluation, however, only ever see 512 px manifest
tiles. This module closes that gap: it runs a trained model over an image of ANY
size and writes a mask that carries the source CRS + affine transform, so
`pipeline.run_tile_pipeline` and `dashboard.service.load_tile_graph` accept it
with no change.

    model, mcfg = load_model_from_checkpoint("artifacts/checkpoints/segformer_cldice_best.pth")
    out = predict_geotiff("cartosat_tile.tif", model, out_path="pred.tif", tta=True)

Design notes
------------
- **Windows overlap and are Hann-blended.** A hard tile grid leaves seams, and a
  seam is a *topology* error — it breaks the very connectivity clDice is trained
  to preserve. Overlapping windows are averaged with a raised-cosine weight so
  predictions fade into each other.
- **Window origins are clamped inside the image**, so every window is full-size
  (encoders with stride-32 need that) and no edge padding is invented. Padding is
  only used when the whole image is smaller than one window.
- **Normalisation matches training exactly** (`build_transforms(train=False)` =
  ImageNet mean/std over 0-255), applied per window so memory stays bounded.
- **Non-8-bit imagery is percentile-stretched.** Cartosat-3/LISS products are
  commonly 10-12 bit; feeding raw DN values to an ImageNet-normalised model
  silently destroys accuracy, so we stretch to 0-255 first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
from omegaconf import OmegaConf

from ..data.dataset import build_transforms
from ..models.baseline import build_model
from ..models.tta import tta_predict
from ..utils import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------
def load_model_from_checkpoint(checkpoint, device: str = "cpu"):
    """Rebuild the trained architecture from a checkpoint. Returns (model, cfg).

    The checkpoint carries the config it was trained with, so the architecture is
    reconstructed exactly. Encoder weights are forced off — `load_state_dict`
    overwrites them anyway, and downloading ImageNet weights at the finale (likely
    offline) would fail.
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    mcfg = OmegaConf.create(ckpt["cfg"])
    mcfg = OmegaConf.merge(mcfg, OmegaConf.create({"model": {"encoder_weights": None}}))
    model = build_model(mcfg)
    model.load_state_dict(ckpt["model"])
    log.info(
        "loaded %s (arch=%s/%s, epoch=%s)",
        Path(checkpoint).name, mcfg.model.arch, mcfg.model.get("encoder", "-"), ckpt.get("epoch"),
    )
    return model.to(device).eval(), mcfg


# --------------------------------------------------------------------------
# radiometry
# --------------------------------------------------------------------------
def to_uint8_rgb(arr: np.ndarray, *, stretch: bool | None = None) -> np.ndarray:
    """(H,W,3) of any dtype -> uint8, percentile-stretched when not already 8-bit.

    `stretch=None` means "decide from the dtype": uint8 passes through untouched,
    anything else gets a 2-98 percentile stretch per band (robust to the few very
    bright pixels typical of satellite scenes).
    """
    if stretch is None:
        stretch = arr.dtype != np.uint8
    if not stretch:
        return arr.astype(np.uint8, copy=False)

    out = np.empty(arr.shape, dtype=np.uint8)
    for b in range(arr.shape[2]):
        band = arr[..., b].astype(np.float32)
        finite = band[np.isfinite(band)]
        lo, hi = (0.0, 1.0) if finite.size == 0 else np.percentile(finite, (2, 98))
        if hi <= lo:                       # flat band — nothing to stretch
            hi = lo + 1.0
        out[..., b] = np.clip((band - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return out


# --------------------------------------------------------------------------
# sliding window
# --------------------------------------------------------------------------
def _origins(extent: int, win: int, stride: int) -> list[int]:
    """Window start offsets covering `extent`, always ending flush with the edge."""
    if extent <= win:
        return [0]
    xs = list(range(0, extent - win + 1, stride))
    if xs[-1] != extent - win:
        xs.append(extent - win)
    return xs


def _blend_weight(h: int, w: int) -> np.ndarray:
    """2-D raised-cosine (Hann) window; strictly positive so it never divides by 0."""
    def ramp(n: int) -> np.ndarray:
        i = np.arange(1, n + 1, dtype=np.float32)
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * i / (n + 1))
    return np.outer(ramp(h), ramp(w)).astype(np.float32)


@torch.no_grad()
def _forward(model, img_u8: np.ndarray, tf, device: str, tta: dict | None) -> np.ndarray:
    """One window (H,W,3) uint8 -> (H,W) float32 probabilities."""
    x = tf(image=img_u8)["image"].float()          # (3,H,W), ImageNet-normalised
    if tta is not None:
        probs = tta_predict(model, x, scales=tuple(tta["scales"]),
                            flips=bool(tta["flips"]), device=device)   # (1,H,W)
        return probs.cpu().numpy()[0].astype(np.float32)
    logits = model(x.unsqueeze(0).to(device))
    return torch.sigmoid(logits).cpu().numpy()[0, 0].astype(np.float32)


def predict_array(
    model,
    image: np.ndarray,
    *,
    window: int = 512,
    overlap: float = 0.25,
    device: str = "cpu",
    tta: bool = False,
    tta_scales=(0.75, 1.0, 1.25),
    tta_flips: bool = True,
    stretch: bool | None = None,
) -> np.ndarray:
    """Run the model over an (H,W,3) image -> (H,W) float32 probability map.

    Windows of `window` px overlap by `overlap` and are Hann-blended, so the
    result has no tile seams. Images smaller than one window are reflect-padded
    up and cropped back.
    """
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"expected an (H,W,>=3) image, got {image.shape}")

    img = to_uint8_rgb(image[..., :3], stretch=stretch)
    h, w = img.shape[:2]
    win = max(32, int(window) // 32 * 32)          # encoders need /32 dims
    stride = max(1, int(round(win * (1.0 - float(overlap)))))
    tf = build_transforms(train=False)
    tta_cfg = {"scales": list(tta_scales), "flips": bool(tta_flips)} if tta else None

    # Whole image smaller than a window: pad up, predict once, crop back.
    if h < win or w < win:
        ph, pw = max(0, win - h), max(0, win - w)
        padded = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        return _forward(model, padded, tf, device, tta_cfg)[:h, :w]

    prob = np.zeros((h, w), dtype=np.float32)
    wsum = np.zeros((h, w), dtype=np.float32)
    blend = _blend_weight(win, win)
    ys, xs = _origins(h, win, stride), _origins(w, win, stride)
    log.info("sliding window: %dx%d image, %d windows of %d px (stride %d)",
             w, h, len(ys) * len(xs), win, stride)

    for y0 in ys:
        for x0 in xs:
            p = _forward(model, img[y0:y0 + win, x0:x0 + win], tf, device, tta_cfg)
            prob[y0:y0 + win, x0:x0 + win] += p * blend
            wsum[y0:y0 + win, x0:x0 + win] += blend

    return prob / wsum


def predict_geotiff(
    image_path,
    model,
    *,
    out_path=None,
    threshold: float = 0.5,
    window: int = 512,
    overlap: float = 0.25,
    device: str = "cpu",
    tta: bool = False,
    tta_scales=(0.75, 1.0, 1.25),
    tta_flips: bool = True,
    bands=(1, 2, 3),
    stretch: bool | None = None,
    save_probs: bool = False,
) -> dict:
    """Predict a road mask for an image GeoTIFF, preserving CRS + transform.

    Writes a single-band uint8 mask (0/1) to `out_path` — the exact input
    `run_tile_pipeline` / `load_tile_graph` expect. Returns a summary dict with
    the output path, road fraction and raster geometry.
    """
    with rasterio.open(image_path) as ds:
        if ds.count < len(bands):
            raise ValueError(f"{image_path} has {ds.count} band(s); need {len(bands)}")
        arr = ds.read(list(bands)).transpose(1, 2, 0)     # (H,W,3)
        profile = ds.profile.copy()
        crs, transform = ds.crs, ds.transform

    probs = predict_array(
        model, arr, window=window, overlap=overlap, device=device,
        tta=tta, tta_scales=tta_scales, tta_flips=tta_flips, stretch=stretch,
    )
    mask = (probs > float(threshold)).astype(np.uint8)

    if out_path is None:
        out_path = Path(image_path).with_name(Path(image_path).stem + "_pred.tif")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile.update(count=1, dtype="uint8", compress="lzw", nodata=None)
    # Block sizes are only legal on tiled rasters; the source profile may carry
    # them from a striped image, which GDAL then rejects.
    if not profile.get("tiled", False):
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)

    prob_path = None
    if save_probs:
        prob_path = out_path.with_name(out_path.stem + "_prob.tif")
        pprof = profile.copy()
        pprof.update(dtype="float32")
        with rasterio.open(prob_path, "w", **pprof) as dst:
            dst.write(probs.astype(np.float32), 1)

    road_frac = float(mask.mean())
    log.info("wrote %s (%dx%d, road_frac=%.4f, tta=%s)",
             out_path.name, mask.shape[1], mask.shape[0], road_frac, tta)
    if road_frac == 0.0:
        log.warning("prediction is EMPTY — check the checkpoint is trained and the "
                    "imagery radiometry matches training (try --no-stretch / --threshold)")

    return {
        "image_path": str(image_path),
        "mask_path": str(out_path),
        "prob_path": str(prob_path) if prob_path else None,
        "crs": str(crs),
        "width": int(mask.shape[1]),
        "height": int(mask.shape[0]),
        "pixel_size_m": abs(float(transform.a)),
        "threshold": float(threshold),
        "tta": bool(tta),
        "road_frac": road_frac,
    }
