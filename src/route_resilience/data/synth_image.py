"""Pseudo-satellite image synthesizer — a PLACEHOLDER, not a deliverable.

OSMnx gives us road *masks* but no imagery. To (a) visually validate the M2
occlusion generator and (b) dry-run the full M3 training loop on CPU before real
DeepGlobe/Cartosat imagery arrives, we fabricate a crude RGB "satellite" image
from a mask: low-frequency earthy terrain + grey roads on the mask.

This is deliberately simple. It is NOT used for final training — when real imagery
is available it drops straight in (the occlusion + dataset code is image-agnostic).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

_VEG = np.array([60, 90, 55], dtype=np.float32)
_SOIL = np.array([120, 105, 80], dtype=np.float32)
_ROAD = np.array([125, 125, 128], dtype=np.float32)


def synthesize_image(
    mask: np.ndarray,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a uint8 HxWx3 pseudo-satellite image consistent with `mask`."""
    if rng is None:
        rng = np.random.default_rng(seed)
    H, W = mask.shape

    # Low-frequency terrain field in [0,1] -> blend vegetation <-> soil.
    field = gaussian_filter(rng.random((H, W)).astype(np.float32), sigma=max(H, W) / 16.0)
    field = (field - field.min()) / (np.ptp(field) + 1e-6)
    bg = field[..., None] * _VEG + (1.0 - field[..., None]) * _SOIL
    bg += rng.normal(0, 6, (H, W, 3)).astype(np.float32)  # fine texture
    img = np.clip(bg, 0, 255).astype(np.uint8)

    # Paint roads grey with slight per-pixel variation.
    rr, cc = np.where(mask > 0)
    if len(rr):
        noise = rng.normal(0, 6, (len(rr), 3)).astype(np.float32)
        img[rr, cc] = np.clip(_ROAD + noise, 0, 255).astype(np.uint8)
    return img
