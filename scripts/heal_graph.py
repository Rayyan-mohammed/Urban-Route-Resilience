"""Demonstrate graph healing on a real tile -> assets/healing_preview.png.

A real OSM tile graph is usually connected, so to show healing we first DAMAGE it
(sever a few short cut-edges, simulating occlusion gaps that fragment the network),
then heal. Left panel = damaged (components in different colours); right panel =
healed (bridges in red). PIL only.

    python scripts/heal_graph.py
"""

from __future__ import annotations

import argparse

import networkx as nx
import numpy as np
import rasterio
from PIL import Image, ImageDraw

from route_resilience.config import load_config
from route_resilience.data.build import read_manifest
from route_resilience.graph.build import mask_to_graph
from route_resilience.graph.heal import heal_graph
from route_resilience.paths import ASSETS, PROCESSED
from route_resilience.utils import get_logger

log = get_logger("heal_graph")
CELL, CAP, GAP, MARGIN = 360, 18, 8, 8
_PALETTE = [(70, 130, 230), (60, 200, 90), (230, 180, 40), (200, 80, 200),
            (40, 200, 200), (230, 120, 60), (150, 150, 250), (120, 220, 150)]


def _damage(g: nx.Graph, max_gap_m: float, k: int, seed: int) -> nx.Graph:
    """Remove up to k short cut-edges to fragment the graph (simulated gaps)."""
    g = g.copy()
    cut = [(u, v) for u, v in nx.bridges(g) if g[u][v].get("length", 0) <= max_gap_m]
    rng = np.random.default_rng(seed)
    rng.shuffle(cut)
    for u, v in cut[:k]:
        if g.has_edge(u, v):
            g.remove_edge(u, v)
    g.remove_nodes_from([n for n, d in g.degree() if d == 0])  # drop orphans
    return g


def _render(g: nx.Graph, title: str, highlight_healed: bool) -> Image.Image:
    img = Image.new("RGB", (CELL, CELL), (15, 15, 18))
    draw = ImageDraw.Draw(img)
    comp_color = {}
    for ci, cc in enumerate(nx.connected_components(g)):
        for n in cc:
            comp_color[n] = _PALETTE[ci % len(_PALETTE)]
    for u, v, d in g.edges(data=True):
        if highlight_healed and d.get("healed"):
            continue  # draw bridges last, on top
        px = d.get("pixels")
        color = comp_color.get(u, (120, 120, 120))
        if px:
            draw.line([(c, r) for r, c in px], fill=color, width=1)
        else:
            draw.line([(g.nodes[u]["col"], g.nodes[u]["row"]),
                       (g.nodes[v]["col"], g.nodes[v]["row"])], fill=color, width=1)
    if highlight_healed:
        for u, v, d in g.edges(data=True):
            if d.get("healed"):
                draw.line([(g.nodes[u]["col"], g.nodes[u]["row"]),
                           (g.nodes[v]["col"], g.nodes[v]["row"])], fill=(255, 40, 40), width=2)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8, help="number of gaps to simulate")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = load_config("base.yaml", "data.yaml", "graph.yaml")
    res = float(cfg.data.resolution_m)
    max_gap = float(cfg.graph.healing.max_gap_m)

    manifest = read_manifest(PROCESSED / "manifest.csv").sort_values("road_frac", ascending=False)
    row = manifest.iloc[0]
    with rasterio.open(row["mask_path"]) as ds:
        mask = (ds.read(1) > 0).astype(np.uint8)
    g, _ = mask_to_graph(mask, resolution_m=res)

    damaged = _damage(g, max_gap, args.k, args.seed)
    healed, info = heal_graph(
        damaged, max_gap_m=max_gap, endpoints_only=bool(cfg.graph.healing.endpoints_only)
    )
    log.info("damaged comps=%d -> healed comps=%d | bridges=%d (%.0fm)",
             info["components_before"], info["components_after"],
             info["n_bridges"], info["bridge_length_m"])

    canvas = Image.new("RGB", (MARGIN * 2 + 2 * CELL + GAP, MARGIN * 2 + CAP + CELL),
                       (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((MARGIN, MARGIN), f"damaged: {info['components_before']} components",
              fill=(20, 20, 20))
    draw.text((MARGIN + CELL + GAP, MARGIN),
              f"healed: {info['components_after']} component(s), {info['n_bridges']} bridges (red)",
              fill=(20, 20, 20))
    canvas.paste(_render(damaged, "damaged", False), (MARGIN, MARGIN + CAP))
    canvas.paste(_render(healed, "healed", True), (MARGIN + CELL + GAP, MARGIN + CAP))

    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / "healing_preview.png"
    canvas.save(out)
    log.info("saved -> %s", out)


if __name__ == "__main__":
    main()
