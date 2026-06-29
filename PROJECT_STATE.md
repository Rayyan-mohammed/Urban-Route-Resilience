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
| M3  | Baseline segmentation model       | §10 P3  | ✅*    | U-Net(resnet34)+Dice/Focal, shared Dataset, device-agnostic trainer, IoU/Dice/P/R. 22 tests pass; CPU dry-run trains end-to-end + checkpoints. *Full train pending Colab GPU run. |
| M4  | SegFormer-B2 + clDice             | §10 P4  | ✅*    | smp Segformer/mit_b2, soft-clDice topology loss + loss factory. clDice 5x more break-sensitive than Dice (proven). CPU dry-run trains. *Full train pending Colab GPU. |
| M5  | Evaluation pipeline               | §11     | ✅     | clDice, occlusion-recall, connectivity ratio, APLS + per-terrain report & baseline/ours compare. APLS now wired in (M6). Full numbers pending GPU weights. |
| M6  | Skeleton → graph                  | §7 P2   | ✅     | skeletonize → NetworkX graph w/ junction-merge + degree-2 dissolve, geo-referenced nodes, weighted edges, graph stats, APLS metric. 7 tests pass; graph_preview asset. |
| M7  | Graph healing                     | §7 P2   | ✅     | Disjoint-Set/Kruskal gap bridging (endpoint-anchored, radius-limited), healed=True tags. 5 tests pass; healing_preview (4 comps→1, 3 bridges). GNN = stretch. |
| M8  | Resilience digital twin           | §7 P3   | ✅     | betweenness + Latora-Marchiori efficiency; pluggable Hazard (Node/Radius/Band); ablation→Resilience Index; reroute_cost. 12 tests; resilience_preview (RI=0.87). |
| M9  | Dashboard backend                 | §7 P4   | ⬜     | graph/twin service layer, GeoJSON export |
| M10 | Dashboard frontend                | §15 P4  | ⬜     | Streamlit + Leaflet, click-to-flood, live reroute |
| M11 | Docs / tests / demo assets        | §12 P7-8| ⬜     | README, pytest suite, fallback demo video assets |

---

## Currently building
- **M0–M8 done and verified.** 57 tests pass, ruff clean. Resilience twin runs on
  real healed graphs: betweenness criticality, hazard ablation, recomputed
  betweenness, Resilience Index (E_after/E_before). **USP fully built end-to-end.**
  Next up: **M9/M10 — Dashboard** (Streamlit + Leaflet, click-to-flood, live reroute).
- Hazard layer is synthetic + pluggable (Node/Radius/Band); real DEM/flood raster
  drops into the same `Hazard.impacted_nodes` interface later.
- Pending GPU (batch later on Colab): full baseline + SegFormer+clDice runs, then
  `python scripts/evaluate.py --checkpoint <baseline> --compare <segformer> --apls` for the money table.
  - baseline:  `python scripts/train.py`
  - segformer: `python scripts/train.py --config base.yaml data.yaml model_segformer.yaml train.yaml train_segformer.yaml`

## Environment (installed 2026-06-29)
- Miniforge at `C:\Users\HP\miniforge3`; env `route-resilience` (Python 3.11).
- Run tools via that env's python; GDAL_DATA/PROJ_LIB are set on `conda activate`.

## Open blockers / risks (live)
- ⚠️ **Finale rules unconfirmed** (roadmap §9.5): are pre-trained weights allowed
  on-site? Confirm at 21 Jul induction. Entire "train-before-finale" strategy
  depends on it.
- ⚠️ **No local GPU**: M3/M4 training must run on Colab/Kaggle. Keep all data,
  graph, twin, dashboard code CPU-runnable so local dev never blocks.
- ⬜ **Hazard layer source** (M8): need a concrete flood/DEM source for an Indian
  demo city. To be resolved before M8.

## Decisions log
- **2026-06-28** Env: Conda/Mamba (conda-forge geospatial). Delivery: files written
  directly. Local: CPU-only → train on cloud. Python 3.11. Config: OmegaConf.
  Package layout: `src/` installable.

## Datasets to acquire (before M1/M3)
- [ ] SpaceNet Roads (APLS GT) · [ ] DeepGlobe Road Extraction · [ ] OpenSatMap
- [ ] A demo city OSM extract via OSMnx (e.g. Bengaluru — named in the brief)
