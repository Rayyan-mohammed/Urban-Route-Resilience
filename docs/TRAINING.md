# Training venue & device strategy

We have **no usable local GPU**, so the roadmap's §9.2 compute split is enforced
in code: everything *except* model training runs locally on CPU, and the two
training milestones (M3 baseline, M4 SegFormer+clDice) run on free cloud GPUs.

## What runs where

| Stage | Milestone | Venue | Device |
|-------|-----------|-------|--------|
| Data pipeline (tiling, OSM masks, occlusion) | M1, M2 | Local | CPU |
| Baseline training | M3 | **Colab / Kaggle** | GPU (T4) |
| SegFormer-B2 + clDice training | M4 | **Colab / Kaggle** | GPU (T4/A100) |
| Inference, skeletonize, graph, healing | M5–M7 | Local | CPU |
| Resilience twin + dashboard | M8–M10 | Local | CPU |
| Cartosat-3 fine-tune | finale | Finale GPU | GPU |

## Device-agnostic rule

All model code resolves the device from config (`device: auto` -> CUDA if present
else CPU). The same code therefore runs unmodified on a Colab GPU and on a local
CPU for smoke tests — never hardcode `.cuda()`.

## Cloud workflow (M3/M4)

1. Push the repo to GitHub.
2. In Colab/Kaggle: `git clone`, `pip install -e .` (the conda geo stack isn't
   needed there — training only needs torch + smp + albumentations, already
   present on those platforms).
3. Upload the `data/processed/` tiles (or regenerate masks in-notebook via OSMnx).
4. Train, then download the cached weights into local `artifacts/checkpoints/`.

> Confirm at the **21 Jul induction** whether pre-trained weights are allowed
> on-site — the entire train-before-finale plan depends on it (PROJECT_STATE risk).
