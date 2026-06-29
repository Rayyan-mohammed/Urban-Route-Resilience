"""Resilience digital-twin demo -> assets/resilience_preview.png.

On a real (healed) tile graph: colour nodes by betweenness criticality, then
'flood' a radius around the MOST critical junction, ablate it, and RECOMPUTE
betweenness on the damaged graph. Title reports the Latora-Marchiori Resilience
Index (E_after / E_before). PIL only.

    python scripts/resilience_demo.py
"""

from __future__ import annotations

import argparse

import numpy as np
import rasterio
from PIL import Image, ImageDraw

from route_resilience.config import load_config
from route_resilience.data.build import read_manifest
from route_resilience.graph.build import mask_to_graph
from route_resilience.graph.heal import heal_graph
from route_resilience.paths import ASSETS, PROCESSED
from route_resilience.resilience.centrality import critical_nodes, node_betweenness
from route_resilience.resilience.hazard import RadiusHazard
from route_resilience.resilience.simulate import ablate, resilience_report
from route_resilience.utils import get_logger

log = get_logger("resilience_demo")
CELL, CAP, GAP, MARGIN = 380, 18, 8, 8


def _heat(v: float):
    """Betweenness [0,1] -> blue (low) .. red (high)."""
    return (int(60 + v * 170), int(90 - v * 50 + 0.0), int(200 - v * 160))


def _draw_graph(g, bc, norm, hazard_center=None, hazard_r=None, removed=None):
    img = Image.new("RGB", (CELL, CELL), (15, 15, 18))
    draw = ImageDraw.Draw(img)
    removed = removed or set()
    # edges (gray; faint red if incident to a removed node)
    for u, v, d in g.edges(data=True):
        faint = (u in removed) or (v in removed)
        color = (90, 30, 30) if faint else (90, 90, 96)
        px = d.get("pixels")
        if px:
            draw.line([(c, r) for r, c in px], fill=color, width=1)
        else:
            draw.line([(g.nodes[u]["col"], g.nodes[u]["row"]),
                       (g.nodes[v]["col"], g.nodes[v]["row"])], fill=color, width=1)
    # hazard zone
    if hazard_center is not None:
        cr, cc = hazard_center
        draw.ellipse([cc - hazard_r, cr - hazard_r, cc + hazard_r, cr + hazard_r],
                     outline=(245, 215, 60), width=2)
    # nodes coloured by betweenness
    for n in g.nodes():
        r, c = g.nodes[n]["row"], g.nodes[n]["col"]
        if n in removed:
            draw.line([(c - 3, r - 3), (c + 3, r + 3)], fill=(235, 50, 50), width=2)
            draw.line([(c - 3, r + 3), (c + 3, r - 3)], fill=(235, 50, 50), width=2)
            continue
        v = bc.get(n, 0.0) / norm if norm > 0 else 0.0
        rad = 2 + int(4 * v)
        draw.ellipse([c - rad, r - rad, c + rad, r + rad], fill=_heat(v))
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius-px", type=float, default=70.0)
    args = ap.parse_args()

    cfg = load_config("base.yaml", "data.yaml", "graph.yaml")
    res = float(cfg.data.resolution_m)
    manifest = read_manifest(PROCESSED / "manifest.csv").sort_values("road_frac", ascending=False)
    with rasterio.open(manifest.iloc[0]["mask_path"]) as ds:
        mask = (ds.read(1) > 0).astype(np.uint8)
    g, _ = mask_to_graph(mask, transform=None, resolution_m=res)
    g, _ = heal_graph(g, max_gap_m=float(cfg.graph.healing.max_gap_m))

    bc_before = node_betweenness(g)
    norm = max(bc_before.values()) if bc_before else 1.0
    top_node = critical_nodes(g, top=1)[0][0]
    center = (g.nodes[top_node]["row"], g.nodes[top_node]["col"])  # (y,x)=(row,col) px

    # Hazard in WORLD coords (x=col, y=row since transform=None); radius in metres.
    hz = RadiusHazard(center=(g.nodes[top_node]["x"], g.nodes[top_node]["y"]),
                      radius_m=args.radius_px * res)
    rep = resilience_report(g, hz)
    impacted = hz.impacted_nodes(g)
    g2 = ablate(g, impacted)
    bc_after = node_betweenness(g2)
    log.info("RI=%.3f drop=%.1f%% impacted=%d comps_after=%d",
             rep["resilience_index"], 100 * rep["efficiency_drop"],
             rep["n_impacted"], rep["components_after"])

    canvas = Image.new("RGB", (MARGIN * 2 + 2 * CELL + GAP, MARGIN * 2 + CAP + CELL),
                       (245, 245, 245))
    d = ImageDraw.Draw(canvas)
    d.text((MARGIN, MARGIN), "before: betweenness (blue=low, red=high) + hazard zone",
           fill=(20, 20, 20))
    d.text((MARGIN + CELL + GAP, MARGIN),
           f"after flood: Resilience Index={rep['resilience_index']:.2f}  "
           f"(efficiency -{100 * rep['efficiency_drop']:.0f}%), recomputed betweenness",
           fill=(20, 20, 20))
    canvas.paste(_draw_graph(g, bc_before, norm, center, args.radius_px),
                 (MARGIN, MARGIN + CAP))
    canvas.paste(_draw_graph(g, bc_after, norm, removed=impacted),
                 (MARGIN + CELL + GAP, MARGIN + CAP))

    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / "resilience_preview.png"
    canvas.save(out)
    log.info("saved -> %s", out)


if __name__ == "__main__":
    main()
