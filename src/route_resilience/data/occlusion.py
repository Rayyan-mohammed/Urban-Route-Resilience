"""Synthetic occlusion generator (M2) — the occlusion-recall training signal.

The model learns to "see through" occlusion only if we train it on roads that are
hidden in the *image* but still present in the *target mask*. This module pastes
realistic occluders onto the image, biased to land ON road pixels (otherwise the
model learns nothing about hidden roads), and returns an `occlusion_mask` so that
evaluation (M5) can measure recall specifically on the hidden road pixels.

Occluder types (roadmap §3.3):
  - tree_canopy     irregular green blobs (cluster of disks) — the dominant case
  - building_shadow elongated dark quadrilateral cast across a road
  - cloud           soft, semi-transparent white veil (Gaussian alpha)
  - vehicle         small bright/dark rectangle sitting on the carriageway

CONTRACT: the input `mask` is NEVER modified — it is the learning target. Only a
copy of the `image` is occluded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.draw import disk, polygon

from ..utils import get_logger

log = get_logger(__name__)

# Occluder appearance constants (RGB).
_CANOPY = np.array([34, 70, 30], dtype=np.float32)
_WHITE = np.array([236, 238, 242], dtype=np.float32)
_VEHICLE_COLORS = np.array(
    [[205, 205, 210], [150, 30, 30], [25, 25, 35], [20, 60, 120]], dtype=np.float32
)

DEFAULT_TYPES: tuple[str, ...] = ("tree_canopy", "building_shadow", "cloud", "vehicle")
DEFAULT_WEIGHTS: tuple[float, ...] = (0.40, 0.30, 0.20, 0.10)


@dataclass
class OcclusionResult:
    """Output of `apply_occlusion`.

    image              : occluded RGB image (uint8, HxWx3) — a modified COPY.
    occlusion_mask     : bool HxW, True where any occluder was painted.
    occluded_road_mask : bool HxW = occlusion_mask AND road — what the model must
                         recover. M5's occlusion-recall is computed on this.
    coverage           : fraction of road pixels that ended up occluded.
    n_occluders        : how many occluders were placed.
    by_type            : count per occluder type.
    """

    image: np.ndarray
    occlusion_mask: np.ndarray
    occluded_road_mask: np.ndarray
    coverage: float
    n_occluders: int
    by_type: dict[str, int]


# --------------------------- geometry helpers ---------------------------
def _rect_footprint(shape, center, w, h, angle) -> np.ndarray:
    """Boolean footprint of a rotated rectangle (clipped to image)."""
    r0, c0 = center
    dx = np.array([-w / 2, w / 2, w / 2, -w / 2])
    dy = np.array([-h / 2, -h / 2, h / 2, h / 2])
    ca, sa = np.cos(angle), np.sin(angle)
    cc = c0 + dx * ca - dy * sa
    rr = r0 + dx * sa + dy * ca
    foot = np.zeros(shape, dtype=bool)
    prr, pcc = polygon(rr, cc, shape=shape)
    foot[prr, pcc] = True
    return foot


# --------------------------- occluder painters --------------------------
# Each painter modifies `img` and `occ` in place; returns nothing.
def _tree_canopy(img, occ, center, rng, scale):
    H, W = occ.shape
    r0, c0 = center
    foot = np.zeros((H, W), dtype=bool)
    for _ in range(int(rng.integers(3, 7))):
        ro = int(rng.integers(-scale, scale + 1))
        co = int(rng.integers(-scale, scale + 1))
        rad = int(rng.integers(max(3, scale // 2), scale + 1))
        rr, cc = disk((r0 + ro, c0 + co), rad, shape=(H, W))
        foot[rr, cc] = True
    rr, cc = np.where(foot)
    img[rr, cc] = (0.12 * img[rr, cc] + 0.88 * _CANOPY).astype(np.uint8)
    occ |= foot


def _building_shadow(img, occ, center, rng, scale):
    w = int(rng.integers(scale, scale * 3 + 1))
    h = int(rng.integers(max(2, scale // 2), scale + 1))
    angle = float(rng.uniform(0, np.pi))
    foot = _rect_footprint(occ.shape, center, w, h, angle)
    rr, cc = np.where(foot)
    img[rr, cc] = (img[rr, cc] * 0.35).astype(np.uint8)  # darken, not replace
    occ |= foot


def _cloud(img, occ, center, rng, scale):
    H, W = occ.shape
    sigma = float(rng.integers(scale, scale * 2 + 1))
    yy, xx = np.ogrid[:H, :W]
    d2 = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
    alpha = np.exp(-d2 / (2.0 * sigma**2)).astype(np.float32)
    # Break the perfect circle with smooth noise -> irregular cloud edge.
    noise = gaussian_filter(rng.random((H, W)).astype(np.float32), sigma=max(1, scale / 2))
    alpha = np.clip(alpha * (0.6 + 0.8 * noise), 0.0, 1.0)
    a = alpha[..., None]
    img[:] = ((1.0 - a) * img + a * _WHITE).astype(np.uint8)
    occ |= alpha > 0.30


def _vehicle(img, occ, center, rng, scale):
    w = int(rng.integers(4, max(6, scale // 2) + 1))
    h = int(rng.integers(3, max(5, scale // 3) + 1))
    angle = float(rng.uniform(0, np.pi))
    foot = _rect_footprint(occ.shape, center, w, h, angle)
    rr, cc = np.where(foot)
    color = _VEHICLE_COLORS[int(rng.integers(len(_VEHICLE_COLORS)))]
    img[rr, cc] = color.astype(np.uint8)
    occ |= foot


_PAINTERS = {
    "tree_canopy": _tree_canopy,
    "building_shadow": _building_shadow,
    "cloud": _cloud,
    "vehicle": _vehicle,
}


# ------------------------------ main API --------------------------------
def apply_occlusion(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    types=DEFAULT_TYPES,
    weights=None,
    coverage_range=(0.05, 0.30),
    apply_prob=0.5,
    max_occluders=50,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> OcclusionResult:
    """Occlude `image` over road pixels until ~`coverage` of road is hidden.

    `image` : uint8 HxWx3.  `mask` : HxW (>0 = road). The mask is not modified.
    Occluders are centred on randomly chosen road pixels so they actually hide
    roads. Returns an `OcclusionResult`. Deterministic given `seed`.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be HxWx3 uint8, got {image.shape}")

    H, W = mask.shape
    out = image.copy()
    occ = np.zeros((H, W), dtype=bool)
    road = mask > 0
    n_road = int(road.sum())
    by_type = {t: 0 for t in types}

    # Skip: no road to hide, disabled, or the per-tile coin-flip says no.
    if n_road == 0 or apply_prob <= 0.0 or rng.random() >= apply_prob:
        return OcclusionResult(out, occ, occ & road, 0.0, 0, by_type)

    w = np.asarray(weights if weights is not None else [1.0] * len(types), dtype=float)
    w = w / w.sum()
    target = float(rng.uniform(*coverage_range))
    road_yx = np.argwhere(road)
    base_scale = max(8, int(0.04 * max(H, W)))  # ~20 px at 512

    placed = 0
    while placed < max_occluders:
        if (occ & road).sum() / n_road >= target:
            break
        t = types[int(rng.choice(len(types), p=w))]
        ry, rx = road_yx[int(rng.integers(len(road_yx)))]  # road-biased centre
        scale = int(max(5, base_scale * rng.uniform(0.6, 1.6)))
        _PAINTERS[t](out, occ, (int(ry), int(rx)), rng, scale)
        by_type[t] += 1
        placed += 1

    occluded_road = occ & road
    coverage = float(occluded_road.sum() / n_road)
    return OcclusionResult(out, occ, occluded_road, coverage, placed, by_type)


def apply_from_config(image, mask, occ_cfg, *, seed=None, rng=None) -> OcclusionResult:
    """Convenience wrapper reading params from a cfg.data.occlusion node."""
    types = tuple(occ_cfg.get("types", DEFAULT_TYPES))
    weights = list(occ_cfg.weights) if occ_cfg.get("weights") is not None else None
    return apply_occlusion(
        image, mask,
        types=types, weights=weights,
        coverage_range=tuple(occ_cfg.get("coverage_range", (0.05, 0.30))),
        apply_prob=float(occ_cfg.get("apply_prob", 0.5)),
        max_occluders=int(occ_cfg.get("max_occluders", 50)),
        seed=seed, rng=rng,
    )
