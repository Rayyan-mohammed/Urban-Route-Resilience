# Route Resilience

**Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis for Urban Mobility**
Bharatiya Antariksh Hackathon 2026 (ISRO × Hack2skill) · NNRMS mandate.

Standard road segmentation breaks wherever a road is hidden — tree canopy,
building shadow, flyovers, clouds. A broken mask is topologically useless: you
cannot route on it, rank intersections on it, or simulate a flood closure on it.
**Route Resilience** turns occluded satellite imagery into a *connectivity-complete,
routable graph* and a **hazard-grounded resilience digital twin** that lets a
planner click a junction, "flood" it, and instantly read the rerouting cost and
the drop in network efficiency. Thank You.

> Full specification: [`roadmap.md`](roadmap.md). Live build status:
> [`PROJECT_STATE.md`](PROJECT_STATE.md).

## The pipeline (4 phases)

```
EO tiles ──▶ I.  SegFormer-B2 + clDice + synthetic occlusion  ──▶ connectivity-complete mask
         ──▶ II. skeletonize → NetworkX graph → gap healing    ──▶ routable weighted graph
         ──▶ III.dynamic betweenness + hazard ablation + R idx  ──▶ resilience digital twin
         ──▶ IV. Streamlit + Leaflet                            ──▶ interactive decision support
```

## USP
1. **Connectivity-complete extraction** — topology-aware clDice loss + synthetic
   occlusion training, so the graph is *actually routable* under occlusion.
2. **Hazard-grounded resilience twin** — node ablation driven by a real flood/DEM
   layer with *dynamically recalculated* betweenness and a Latora–Marchiori
   efficiency-based **Resilience Index**.

## Quickstart

```bash
# 1. Create the environment (conda-forge solves GDAL/Rasterio on Windows)
mamba env create -f environment.yml      # or: conda env create -f environment.yml
conda activate route-resilience

# 2. Install the package in editable mode
pip install -e .

# 3. Sanity check
pytest -q
```

> **No local GPU?** That's expected. Phase II–IV (graph, twin, dashboard) run on
> CPU. Heavy training (Phase I) runs on Colab/Kaggle — see `docs/TRAINING.md`.

## Repository layout

```
src/route_resilience/   # installable package (data, models, graph, resilience, dashboard)
configs/                # OmegaConf YAML configs (base, data, model, train)
data/                   # raw / interim / processed  (gitignored — see data/README.md)
scripts/                # CLI entry points (run tiling, train, build graph, ...)
tests/                  # pytest suite
notebooks/              # exploration (not the source of truth)
docs/                   # training guide, architecture notes
```

## License
MIT.
