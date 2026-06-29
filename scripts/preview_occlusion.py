"""Render a before/after occlusion preview grid -> assets/occlusion_preview.png.

Visual sanity check + demo asset. Loads real OSMnx mask tiles from the manifest,
synthesizes a pseudo-satellite image (placeholder), applies occlusion, and lays
out columns: [pseudo image | occluded image | mask with hidden road in red].

    python scripts/preview_occlusion.py --n 5

NOTE: we composite with PIL, NOT matplotlib. On this Windows/conda env, importing
matplotlib's Agg savefig in the same process as rasterio/GDAL triggers a native
DLL abort (duplicate libpng/zlib). PIL avoids that and is lighter anyway.
Occlusion is forced on (apply_prob=1.0) here so the preview always shows it.
"""

from __future__ import annotations

import argparse

import numpy as np
import rasterio
from PIL import Image, ImageDraw

from route_resilience.config import load_config
from route_resilience.data.build import read_manifest
from route_resilience.data.occlusion import apply_occlusion
from route_resilience.data.synth_image import synthesize_image
from route_resilience.paths import ASSETS, PROCESSED
from route_resilience.utils import get_logger

log = get_logger("preview_occlusion")

CELL = 256        # panel size in px
CAP = 16          # caption strip height
GAP = 6
MARGIN = 8
COLS = ("pseudo image", "occluded", "hidden road (red)")


def _panel(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr).resize((CELL, CELL), Image.NEAREST)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="number of tiles to preview")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config("base.yaml", "data.yaml")
    occ = cfg.data.occlusion
    manifest = read_manifest(PROCESSED / "manifest.csv")
    manifest = manifest.sort_values("road_frac", ascending=False).head(args.n)
    rng = np.random.default_rng(args.seed)

    n = len(manifest)
    cell_h = CAP + CELL
    canvas_w = MARGIN * 2 + 3 * CELL + 2 * GAP
    canvas_h = MARGIN * 2 + n * cell_h + (n - 1) * GAP
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    for i, (_, row) in enumerate(manifest.iterrows()):
        with rasterio.open(row["mask_path"]) as ds:
            mask = ds.read(1)
        img = synthesize_image(mask, rng=rng)
        res = apply_occlusion(
            img, mask,
            types=tuple(occ.types), weights=list(occ.weights),
            coverage_range=tuple(occ.coverage_range),
            apply_prob=1.0,  # always show occlusion in the preview
            max_occluders=int(occ.max_occluders), rng=rng,
        )

        overlay = np.stack([(mask > 0).astype(np.uint8) * 80] * 3, axis=-1)
        overlay[res.occluded_road_mask] = [220, 40, 40]

        panels = [_panel(img), _panel(res.image), _panel(overlay)]
        caps = [
            f"{COLS[0]} · {row['tile_id']}",
            f"{COLS[1]} cov={res.coverage:.0%} n={res.n_occluders}",
            COLS[2],
        ]
        y = MARGIN + i * (cell_h + GAP)
        for c, (panel, cap) in enumerate(zip(panels, caps)):
            x = MARGIN + c * (CELL + GAP)
            draw.text((x, y + 3), cap, fill=(20, 20, 20))
            canvas.paste(panel, (x, y + CAP))

    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / "occlusion_preview.png"
    canvas.save(out)
    log.info("saved preview (%d tiles) -> %s", n, out)


if __name__ == "__main__":
    main()
