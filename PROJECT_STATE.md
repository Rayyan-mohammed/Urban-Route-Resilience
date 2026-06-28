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
| M1  | Data pipeline (tiling + OSM split)| §3, P2  | ✅*    | geo/osm/build/split + CLI. Offline logic unit-tested (6 pass). *Live OSMnx run + 4 geo tests pending conda env install. |
| M2  | Synthetic occlusion generator     | §3.3 P2 | ⬜     | tree/shadow/cloud/vehicle paste — occlusion-recall signal |
| M3  | Baseline segmentation model       | §10 P3  | ⬜     | U-Net/D-LinkNet, Dice+Focal, IoU — the demo "contrast" |
| M4  | SegFormer-B2 + clDice             | §10 P4  | ⬜     | Topology loss + occlusion training (train on Colab/Kaggle) |
| M5  | Evaluation pipeline               | §11     | ⬜     | clDice, occlusion-recall, connectivity ratio, APLS |
| M6  | Skeleton → graph                  | §7 P2   | ⬜     | scikit-image skeletonize → NetworkX weighted graph |
| M7  | Graph healing                     | §7 P2   | ⬜     | MST/Disjoint-Set bridging (GNN link-pred = stretch) |
| M8  | Resilience digital twin           | §7 P3   | ⬜     | dynamic betweenness, hazard ablation, Resilience Index |
| M9  | Dashboard backend                 | §7 P4   | ⬜     | graph/twin service layer, GeoJSON export |
| M10 | Dashboard frontend                | §15 P4  | ⬜     | Streamlit + Leaflet, click-to-flood, live reroute |
| M11 | Docs / tests / demo assets        | §12 P7-8| ⬜     | README, pytest suite, fallback demo video assets |

---

## Currently building
- **M0 + M1 code complete & unit-tested (offline).** Next up: **M2 — Synthetic occlusion**.
- ⏳ User installing Miniforge → then run `pytest` (4 geo tests) + live
  `python scripts/build_dataset.py --only indiranagar_blr` to produce real tiles.

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
