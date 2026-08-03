# Project State — Route Resilience (BAH 2026)

> Single running ledger. Updated at the end of every milestone. The roadmap
> (`roadmap.md`) is the spec; this file tracks execution against it.

**Phase:** Implementation (1 Jul → 21 Jul induction). Cartosat-3 fine-tuning is
deliberately excluded — finale only.

**Legend:** ✅ done · 🔨 in progress · ⬜ not started · ⚠️ blocker/risk

---

## Milestone tracker

| ID  | Milestone                         | Roadmap | Status | Notes |
|-----|-----------------------------------|---------|--------|-------|
| M0  | Project foundation                | §12 P1  | ✅     | Repo skeleton, env, config, paths, logging, state ledger |
| M1  | Data pipeline (tiling + OSM split)| §3, P2  | ✅     | geo/osm/build/split + CLI. All 10 tests pass. Live OSMnx run produced 200 Indiranagar mask tiles + manifest. |
| M2  | Synthetic occlusion generator     | §3.3 P2 | ✅     | occlusion.py (4 occluders, road-biased) + synth_image.py + preview. 8 tests pass; assets/occlusion_preview.png. |
| M3  | Baseline segmentation model       | §10 P3  | ✅*    | U-Net(resnet34) **+ D-LinkNet** (own impl: dilated D-block + LinkNet decoder over smp resnet encoder), Dice/Focal, shared Dataset, device-agnostic trainer, IoU/Dice/P/R. CPU dry-run trains end-to-end + checkpoints (both archs). *Full train pending Colab GPU run. |
| M4  | SegFormer-B2 + clDice + TTA        | §10 P4  | ✅*    | smp Segformer/mit_b2, soft-clDice topology loss + loss factory. clDice 5x more break-sensitive than Dice (proven). **Multi-scale+flip TTA (models/tta.py) wired into evaluate (`--tta` / cfg.eval.tta).** CPU dry-run trains + TTA eval runs. *Full train pending Colab GPU. |
| M5  | Evaluation pipeline               | §11     | ✅     | clDice, occlusion-recall, connectivity ratio, APLS + per-terrain report & baseline/ours compare. APLS now wired in (M6). Full numbers pending GPU weights. |
| M6  | Skeleton → graph                  | §7 P2   | ✅     | skeletonize → NetworkX graph w/ junction-merge + degree-2 dissolve, geo-referenced nodes, weighted edges, graph stats, APLS metric. 7 tests pass; graph_preview asset. |
| M7  | Graph healing                     | §7 P2   | ✅     | Disjoint-Set/Kruskal gap bridging (endpoint-anchored, radius-limited), healed=True tags. 5 tests pass; healing_preview (4 comps→1, 3 bridges). GNN = stretch. |
| M8  | Resilience digital twin           | §7 P3   | ✅     | betweenness + Latora-Marchiori efficiency; pluggable Hazard (Node/Radius/Band); ablation→Resilience Index; reroute_cost. 12 tests; resilience_preview (RI=0.87). |
| M9  | Dashboard backend                 | §7 P4   | ✅     | service.py: load/heal/reproject(UTM→WGS84)/flood/reroute/build_map. 6 tests pass. |
| M10 | Dashboard frontend                | §15 P4  | ✅     | Streamlit + folium/Leaflet on OSM basemap; click-to-flood, radius slider, live RI + recomputed betweenness + reroute. Boots clean (headless smoke). |
| M11 | Docs / tests / demo assets        | §12 P7-8| ✅     | end-to-end pipeline.py + CLI; ARCHITECTURE.md; README demo embed (4 assets); Colab notebook; evaluate() orchestrator tests. 68 tests pass. |
| M12 | Finale integration surface        | §7, §12 | ✅     | **Inference on real imagery** (`inference/predict.py` + `scripts/predict.py`): sliding window, Hann-blended overlaps, TTA, percentile stretch, CRS/transform preserved. **Cartosat OSM auto-labelling** (`--source cartosat`) — verified live: 230 OSM segments → 9 labelled tiles. **RasterHazard** (real DEM/flood). **Ablation tables** (`scripts/ablations.py`). 118 tests pass. |

---

## Currently building
- **M0–M12 DONE and verified.** 118 tests pass, ruff clean. The finale surface is
  now code-complete: unlabelled imagery → OSM labels → fine-tune → predict → graph
  → heal → real-DEM hazard → twin → dashboard. Runbook: `docs/FINALE_RUNBOOK.md`.

### ⚠️ The one thing that is NOT done: real trained weights
All three checkpoints in `artifacts/checkpoints/` are **1-epoch CPU dry-runs**, not
models. Their metrics prove it — precision ≈ 0.10 with recall ≈ 1.0 means "predict
road everywhere"; D-LinkNet's are literally all zeros:

| checkpoint | epoch | IoU | precision | recall |
|---|---|---|---|---|
| baseline_unet | 1 | 0.099 | 0.099 | 0.98 |
| dlinknet | 1 | 0.000 | 0.000 | 0.00 |
| segformer_cldice | 1 | 0.097 | 0.097 | 1.00 |

Local training cannot fix this: `image_path` is **0/730** in the manifest, so the
Dataset falls back to `synth_image.py` — pseudo-images derived from the mask, i.e.
the model reads the answer off its own input. **Real numbers need the Kaggle
DeepGlobe run** (`notebooks/train_kaggle.ipynb`, `USE_REAL_DATA=True`). Roadmap
§13.1 wanted weights cached by 1 Aug; that is the live schedule slip.

### Ablations — 2 of 4 already have real numbers (no GPU needed)
`python scripts/ablations.py --all` → `artifacts/reports/ablations.md`. Measured on
40 held-out tiles:
- **Healing:** components/tile 2.23 → 1.27 (−42.7%); largest-component fraction
  0.901 → 0.970 (+7.7%); 0.9 bridges/tile.
- **Dynamic vs static betweenness:** RI 0.569 → 0.450 — recalculated attack does
  **11.8 pp more damage**, worse on 33/40 tiles. Directly supports the §8 USP.
- Ablations 1 & 2 (Dice vs +clDice, ±occlusion) are blocked on the GPU run.

### Fixed: Resilience Index could exceed 1
`global_efficiency` normalised by the *surviving* node count, so ablating a sparse
suburb shrank the denominator faster than the numerator — a measured case scored
**RI = 1.56**, i.e. "the flood improved the city". Both sides are now normalised
over the pre-hazard node set (`n_universe`); that case now scores 0.99. Regression
test in `tests/test_resilience.py`.
- **2026-07-15 — closed the two roadmap §12 P3/P4 code gaps:**
  D-LinkNet baseline (`models/dlinknet.py`, `configs/model_dlinknet.yaml` +
  `train_dlinknet.yaml`, `arch: DLinkNet` dispatched by the factory) and
  multi-scale+flip TTA (`models/tta.py`, wired into `evaluate()` via `--tta` /
  `cfg.eval.tta`). +8 tests (test_dlinknet, test_tta). Train D-LinkNet:
  `python scripts/train.py --config base.yaml data.yaml model_dlinknet.yaml train.yaml train_dlinknet.yaml`.
  Evaluate with TTA: add `--tta` to `scripts/evaluate.py`.
  Still pending (needs GPU, not code): the actual Colab training runs → real weights + metric tables.
- **2026-08-03 — closed the whole finale integration surface (M12).** All three
  steps ARCHITECTURE.md listed as finale work are now code, tested end to end:
  imagery→mask inference, Cartosat OSM auto-labelling, real DEM/flood hazard.
  Remaining finale work is running commands, not writing them — see
  `docs/FINALE_RUNBOOK.md`.
- Hazard layer: synthetic generators (Node/Radius/Band) **plus `RasterHazard`** for
  a real DEM or flood-depth product (`--hazard-raster`, depth or elevation mode).
- Predict from imagery: `python scripts/predict.py --image-dir <dir> --checkpoint <pth> --tta --pipeline`
- Ablation tables: `python scripts/ablations.py --all` → `artifacts/reports/ablations.md`
- Run pipeline: `python scripts/run_pipeline.py --save`
- Run dashboard: `streamlit run src/route_resilience/dashboard/app.py`
- Train (Colab GPU): `notebooks/train_colab.ipynb`
- Pending GPU (batch later on Colab): full baseline + SegFormer+clDice runs, then
  `python scripts/evaluate.py --checkpoint <baseline> --compare <segformer> --apls` for the money table.
  - baseline:  `python scripts/train.py`
  - segformer: `python scripts/train.py --config base.yaml data.yaml model_segformer.yaml train.yaml train_segformer.yaml`

## Environment (installed 2026-06-29)
- Miniforge at `C:\Users\HP\miniforge3`; env `route-resilience` (Python 3.11).
- Run tools via that env's python; GDAL_DATA/PROJ_LIB are set on `conda activate`.

## Open blockers / risks (live — 3 days to the finale)
- 🔴 **No real weights yet. This is the critical path.** Run
  `notebooks/train_kaggle.ipynb` with `USE_REAL_DATA=True` (FAST=True first to
  de-risk, then the full run) and download the three `*.pth` into
  `artifacts/checkpoints/`. Everything else is ready and waiting on this.
- ⚠️ **Finale rules still unconfirmed** (roadmap §9.5): are pre-trained weights
  allowed on-site? The 21 Jul induction has passed — if this was answered, record
  it here. If weights are *not* allowed, the train-before-finale strategy changes.
- ⚠️ **Venue may be offline.** `--source cartosat` needs internet for OSMnx.
  Pre-cache OSM masks for the demo city before travelling (runbook item 2).
- ⚠️ **No local GPU**: training must run on Colab/Kaggle. All other code is
  CPU-runnable so local dev never blocks.
- ⬜ **No fallback demo video** (roadmap Day 8 / 5 Aug). §9.5 rates "demo breaks
  live" Med/High; the video is the mitigation.
- ⬜ **Hazard raster source**: `RasterHazard` is built and tested, but no actual
  DEM/flood product for an Indian demo city is attached yet. Bhuvan/Bhukosh or
  SRTM/Copernicus DEM would all work.

## Decisions log
- **2026-06-28** Env: Conda/Mamba (conda-forge geospatial). Delivery: files written
  directly. Local: CPU-only → train on cloud. Python 3.11. Config: OmegaConf.
  Package layout: `src/` installable.

## Datasets (roadmap §3.2)
Ingest adapters are BUILT (`data/ingest.py` + `scripts/ingest_dataset.py`); they tile
real image+mask pairs into the manifest with a populated `image_path`, so training
uses real pixels instead of `synth_image.py`. See `configs/datasets.yaml` + docs/DATASETS.md.
- [x] **OSM via OSMnx** — 730 mask tiles, 4 Bengaluru terrains (masks only, no imagery)
- [x] **DeepGlobe** — adapter + wired live in `notebooks/train_kaggle.ipynb` (Kaggle dataset)
- [x] **SpaceNet Roads** — adapter built (GeoTIFF + GeoJSON → rasterised labels); needs the
      AWS (requester-pays) download attached, then `ingest_dataset.py --source spacenet --append`
- [x] **OpenSatMap** — covered by the generic image-dir/mask-dir adapter; attach + `--append`
- ⚠️ Only DeepGlobe is actually *runnable* out of the box (it's a one-click Kaggle dataset).
  SpaceNet/OpenSatMap adapters are untested against real data — no download available locally.
