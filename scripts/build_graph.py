"""Build graphs from mask tiles and render a preview -> assets/graph_preview.png.

Demonstrates the pixels->topology pivot: for each tile we skeletonize the mask,
trace the NetworkX graph, print stats, and draw edges (blue) + nodes
(junction=red, endpoint=green) over the mask. PIL only (no matplotlib/GDAL clash).

    python scripts/build_graph.py --n 4
"""

from __future__ import annotations

import argparse

import numpy as np
import rasterio
from PIL import Image, ImageDraw

from route_resilience.config import load_config
from route_resilience.data.build import read_manifest
from route_resilience.graph.build import graph_stats, mask_to_graph
from route_resilience.paths import ASSETS, PROCESSED
from route_resilience.utils import get_logger

log = get_logger("build_graph")
CELL, CAP, GAP, MARGIN = 320, 16, 6, 8


def _render(mask: np.ndarray, g) -> Image.Image:
    H, W = mask.shape
    base = np.stack([(mask > 0).astype(np.uint8) * 55] * 3, axis=-1)
    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)
    for u, v, d in g.edges(data=True):
        px = d.get("pixels")
        if px:
            draw.line([(c, r) for r, c in px], fill=(70, 130, 230), width=1)
        else:  # fused edge (no pixel path) -> straight segment between nodes
            draw.line([(g.nodes[u]["col"], g.nodes[u]["row"]),
                       (g.nodes[v]["col"], g.nodes[v]["row"])], fill=(70, 130, 230), width=1)
    for n, deg in g.degree():
        r, c = g.nodes[n]["row"], g.nodes[n]["col"]
        color = (230, 60, 60) if deg >= 3 else (60, 200, 90) if deg == 1 else (200, 200, 60)
        draw.ellipse([c - 2, r - 2, c + 2, r + 2], fill=color)
    return img.resize((CELL, CELL), Image.NEAREST)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config("base.yaml", "data.yaml")
    res = float(cfg.data.resolution_m)
    manifest = read_manifest(PROCESSED / "manifest.csv")
    manifest = manifest.sort_values("road_frac", ascending=False).head(args.n)

    n = len(manifest)
    canvas = Image.new("RGB", (MARGIN * 2 + n * CELL + (n - 1) * GAP, MARGIN * 2 + CAP + CELL),
                       (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    for i, (_, row) in enumerate(manifest.iterrows()):
        with rasterio.open(row["mask_path"]) as ds:
            mask = (ds.read(1) > 0).astype(np.uint8)
        g, _ = mask_to_graph(mask, transform=None, resolution_m=res)
        st = graph_stats(g)
        log.info("%s: nodes=%d edges=%d comps=%d len=%.0fm",
                 row["tile_id"], st["n_nodes"], st["n_edges"], st["n_components"],
                 st["total_length_m"])
        x = MARGIN + i * (CELL + GAP)
        draw.text((x, MARGIN), f"{st['n_nodes']}N {st['n_edges']}E {st['n_components']}C",
                  fill=(20, 20, 20))
        canvas.paste(_render(mask, g), (x, MARGIN + CAP))

    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / "graph_preview.png"
    canvas.save(out)
    log.info("saved -> %s", out)


if __name__ == "__main__":
    main()
