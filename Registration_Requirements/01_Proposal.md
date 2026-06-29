# Route Resilience — Concept Proposal

**Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis for Urban Mobility**

| | |
|---|---|
| **Hackathon** | Bharatiya Antariksh Hackathon 2026 (BAH 2026) — ISRO × Hack2skill |
| **Mandate** | National Natural Resources Management System (NNRMS) |
| **Problem Statement** | Occlusion-robust road extraction → routable graph → criticality & failure simulation |
| **Team** | `<Team Name>` — `<Member 1>`, `<Member 2>`, `<Member 3>`, `<Member 4>` |
| **Submission stage** | Registration & Idea/Concept Submission (due 1 July 2026) |
| **Prepared** | June 2026 |

> **One-line pitch.** Route Resilience turns occluded satellite imagery into a
> *connectivity-complete, routable road graph* and a **hazard-grounded resilience
> digital twin** — so a planner can click a junction, "flood" it, and instantly
> read the rerouting cost and the drop in network efficiency.

---

## Table of Contents

1. [Problem Understanding](#1-problem-understanding)
2. [Proposed Solution](#2-proposed-solution)
3. [Methodology](#3-methodology)
4. [System Architecture](#4-system-architecture)
5. [Innovation / USP](#5-innovation--usp)
6. [Expected Impact](#6-expected-impact)
7. [Feasibility](#7-feasibility)
8. [Technology Stack](#8-technology-stack)
9. [Team Roles & Capabilities](#9-team-roles--capabilities)
10. [High-Level Implementation Plan](#10-high-level-implementation-plan)
11. [Timeline](#11-timeline)
12. [Risk Analysis](#12-risk-analysis)
13. [References](#13-references)

---

## 1. Problem Understanding

### 1.1 The core problem

Satellite road extraction in dense Indian metros **fails wherever the road is
hidden** — under tree canopy, building shadow, flyovers, parked vehicles, and
cloud cover. Standard pixel-wise segmentation produces **broken masks**, and a
broken mask is *topologically useless*:

- You **cannot route** on a disconnected network.
- You **cannot rank** intersections by importance (betweenness) on a wrong graph.
- You **cannot simulate** a flood closure if the graph was never connected.

The problem statement therefore asks for two things, in order:

1. **See through occlusion** — recover a *topologically connected* road network,
   not just a high pixel-accuracy mask.
2. **Reason on the result** — convert it into a *routable weighted graph*,
   identify critical "Gatekeeper Nodes," and run failure simulations.

### 1.2 What the data looks like

| Input | Resolution | Role |
|---|---|---|
| Sentinel-2 | 10 m | Free, global pretraining + generalisation breadth |
| Resourcesat LISS-IV | 5.8 m | Indian-context pretraining; bridges to Cartosat |
| Cartosat-3 | sub-metre | Provided at the finale — final fine-tune + inference |

Ground truth is **auto-generated from OpenStreetMap** (via OSMnx), so labels are
*noisy* and slightly *misaligned* with imagery — a challenge we design around
rather than ignore.

### 1.3 The hidden challenges (and why most teams miss them)

| Hidden challenge | Consequence if ignored |
|---|---|
| OSM ↔ imagery misalignment (labels offset several metres) | Vanilla IoU unfairly punishes correct predictions |
| Resolution gap (pretrain at 5–10 m, infer at sub-metre) | Model trained on Sentinel collapses on Cartosat |
| Flyovers / underpasses create false intersections | The graph gets phantom junctions → wrong routing |
| "Occlusion-recall" is invisible to pixel IoU | Teams *claim* robustness but cannot *prove* it |
| The resilience claim is empty unless the graph is connected first | A digital twin on a broken graph is *visibly wrong* |

### 1.4 What the judges actually score

- **Technical depth** — topology metrics (clDice, APLS, connectivity ratio,
  occlusion-recall), not just IoU.
- **Generalisation** — demonstrated across ≥3 terrain types (dense urban,
  forested suburban, rural).
- **Operational usability** — can a *non-technical planner* make a decision with
  the tool? ISRO / MeitY / Consumer-Affairs judges reward **decision support**.
- **Novelty & defensibility** — something competitors cannot reproduce in 30 hours.
- **Storytelling** — *"from space insights to stronger cities"* (the brief's own framing).

---

## 2. Proposed Solution

**Route Resilience** is a four-phase pipeline that takes raw Earth-observation
tiles and produces an interactive resilience decision-support tool.

```
EO tiles ─▶ I.  Occlusion-robust extraction  ─▶ connectivity-complete mask
         ─▶ II. Skeletonise → graph → healing ─▶ routable weighted graph
         ─▶ III.Resilience digital twin        ─▶ criticality + failure simulation
         ─▶ IV. Interactive dashboard          ─▶ decision support for a planner
```

- **Phase I — Occlusion-robust extraction.** A SegFormer (MiT-B2) segmentation
  model trained with a **topology-aware clDice loss** and **synthetic-occlusion
  augmentation**, so it learns to *in-paint connectivity* under canopy and shadow.
- **Phase II — Skeletonise + learned healing.** Convert the mask to a graph
  (scikit-image skeleton → NetworkX), then **heal residual gaps** using a
  distance + angular-continuity scorer (MST / Disjoint-Set baseline, optional GNN
  link-prediction upgrade).
- **Phase III — Resilience digital twin.** Rank intersections by **betweenness
  centrality** (Gatekeeper Nodes), then run **hazard-grounded node ablation**:
  remove nodes in order of flood/DEM exposure, **recalculate** betweenness after
  each step, and report a **Resilience Index** `R = E(perturbed) / E(baseline)`.
- **Phase IV — Interactive dashboard.** A Streamlit + Leaflet app: criticality
  heatmap, *click-to-flood* a junction, **live reroute** with Δtravel-time, and a
  Resilience Index gauge.

The headline result is a **decision**, not a prettier mask: *"a planner just
stress-tested the city in two clicks."*

---

## 3. Methodology

### 3.1 Data pipeline (zero manual labelling)

```
Sentinel-2 / LISS-IV / Cartosat
  → Rasterio tiling (512×512, 25% overlap)
  → per-band normalisation + CLAHE contrast
  → OSMnx vector pull → rasterise to aligned road mask (ground truth)
  → Albumentations: flips, rotations, brightness/shadow + SYNTHETIC OCCLUSION
  → train / val / test split STRATIFIED BY TERRAIN (urban / suburban / forested)
```

**Synthetic occlusion** is the key training signal: we programmatically paste
tree / shadow / cloud / vehicle patches over *known* roads, forcing the model to
recover connectivity it cannot directly see. This *is* the occlusion-recall
training signal that vanilla pipelines lack.

### 3.2 Model & training

- **Architecture:** SegFormer `MiT-B2` encoder (ImageNet-pretrained) via
  `segmentation-models-pytorch`. Long-range attention is exactly what occlusion
  reasoning needs; it also fine-tunes fast.
- **Loss:** `0.4·Dice + 0.3·Focal + 0.3·clDice` — Dice/Focal handle the heavy
  class imbalance (roads ≈ 3–5 % of pixels); **clDice preserves topology**.
- **Optimisation:** AdamW (lr 1e-4, wd 1e-5), cosine LR schedule, mixed-precision
  (`torch.cuda.amp`).
- **Early stopping on validation clDice** — *not* IoU. Connectivity is the point.
- **Inference:** multi-scale test-time augmentation (TTA) → threshold → skeleton
  → heal → graph.

### 3.3 Graph construction & healing

1. `skimage.morphology.skeletonize` → extract nodes (junctions, endpoints) and
   edges (segments).
2. Build a weighted **NetworkX** graph (edge weight = segment length / travel time).
3. **Gap healing:** score candidate links by *(Euclidean distance, angular
   continuity)*; bridge with **MST + Disjoint-Set** (guaranteed baseline) and,
   if time permits, a **PyTorch-Geometric GNN link-prediction** upgrade.

### 3.4 Resilience digital twin

- **Criticality:** betweenness centrality → Gatekeeper Nodes → criticality heatmap.
- **Hazard-grounded ablation:** rank nodes by **DEM/flood exposure**, remove in
  sequence, and **recalculate** betweenness at each step (dynamic, not one-shot).
- **Resilience Index:** `R = E(perturbed) / E(baseline)` using the
  **Latora–Marchiori global efficiency** measure; also track the giant-component
  size as nodes fail.

### 3.5 Evaluation strategy

| Metric | What it proves |
|---|---|
| IoU / Dice | Baseline pixel accuracy (table-stakes) |
| **clDice** | Topology / connectivity preservation (headline) |
| **Occlusion-Recall** | Road recovery *inside* occluded regions |
| **Connectivity Ratio** | Growth of the largest connected component after healing |
| **APLS / TOPO** | Graph-level path similarity vs OSM ground truth |
| Relaxed/Buffered IoU (3–5 px) | Fair scoring under OSM misalignment |
| **Resilience Index R** | The decision-support output |

**Ablation studies** (high judge value): Dice vs Dice+clDice · with vs without
synthetic occlusion · before vs after healing (connectivity-ratio jump) · static
vs dynamic betweenness. Validation is **terrain-stratified** so generalisation is
*measured*, not asserted.

---

## 4. System Architecture

The full architecture diagram is in
[02_Diagrams/System_Architecture.md](02_Diagrams/System_Architecture.md). Summary:

```
┌── PHASE I: OCCLUSION-ROBUST EXTRACTION ───────────────────────────────┐
│ SegFormer (MiT-B2) encoder–decoder                                    │
│ loss = Dice + Focal + clDice(topology)                                │
│ trained with SYNTHETIC OCCLUSION + multi-scale TTA                    │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼ connectivity-complete probability mask
┌── PHASE II: SKELETONISE + LEARNED HEALING ────────────────────────────┐
│ scikit-image skeletonize → nodes + edges                              │
│ gap healing: (distance, angular continuity) → MST/Disjoint-Set (+GNN) │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼ routable weighted vector graph (NetworkX)
┌── PHASE III: RESILIENCE DIGITAL TWIN ─────────────────────────────────┐
│ betweenness → Gatekeeper Nodes (criticality heatmap)                  │
│ hazard-grounded ablation → RECALCULATE betweenness each step          │
│ Resilience Index  R = E(perturbed) / E(baseline)                      │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌── PHASE IV: INTERACTIVE DASHBOARD ────────────────────────────────────┐
│ Streamlit + Leaflet: heatmap · click-to-flood · live reroute · R gauge│
└───────────────────────────────────────────────────────────────────────┘
```

**Design principle:** Phases II–IV run on **CPU**; only Phase I training needs a
GPU, and that is front-loaded into the pre-finale window. This keeps the entire
system runnable on a laptop during integration and demo.

---

## 5. Innovation / USP

> **"From broken pixels to a decision-grade resilience digital twin."**

Most teams (≈70 %) will stop at *IoU on a clean mask*. We win on two things they
cannot fake in 30 hours:

### USP 1 — Connectivity-complete extraction (the technical moat)
A **topology-aware clDice loss** + **synthetic-occlusion training** produces a
mask that is *actually routable* under occlusion. We can **prove** it with an
occlusion-recall metric and a connectivity-ratio ablation — competitors who used
plain Dice/BCE cannot.

### USP 2 — Hazard-grounded resilience digital twin (the emotional hook)
Node ablation driven by a **real flood/DEM layer**, with **dynamically
recalculated** betweenness after each closure, and a quantitative
**Resilience Index** (Latora–Marchiori efficiency) plus **live rerouting** — not
a static heatmap with a one-shot node delete.

### Why this combination wins

| | Typical team | Route Resilience |
|---|---|---|
| Mask under occlusion | Breaks → disconnected graph | Connectivity-complete |
| Occlusion proof | None | Occlusion-recall + ablation |
| Resilience | One static heatmap | Dynamic, hazard-grounded twin |
| Node failure | Delete one node once | Sequenced, recalculated, efficiency-scored |
| Output | "Here's a road mask" | "Here's where your city breaks, and the cost" |

Crucially, the two USPs **compound**: a resilience twin is only meaningful on a
connected graph, so competitors cannot retro-fit dynamic ablation late without
*also* having solved connectivity. The gap is our scoring margin.

A learned **GNN gap-healing** stage is kept as a *stretch upgrade*; a full
graph-native transformer (Sat2Graph / RNGDet++) is **deliberately rejected** as
un-trainable in the 30-hour window.

---

## 6. Expected Impact

This sits inside ISRO's **NNRMS mandate** to maximise downstream value from
indigenous EO assets (Cartosat-3, Resourcesat LISS-IV). The same
broken-connectivity failure blocks three ministries named in the brief:

| Stakeholder | Decision Route Resilience enables |
|---|---|
| **Disaster management (NDMA-style)** | Which junctions, if flooded, sever the city — and the best reroute |
| **MeitY / e-governance GIS** | Up-to-date routable road graphs from satellite, auto-generated |
| **Consumer Affairs / PDS** | Verify facility access routes and their resilience to disruption |

**Concrete impact statement.** A planner opens a map of a city, sees its
structurally weakest intersections coloured by criticality, clicks a node to
"flood" it, and instantly reads the rerouting cost and the drop in network
efficiency. That is a **usable resilience tool**, demonstrable live — aligned
with **National Space Day**'s framing of national-scale, operational solutions.

---

## 7. Feasibility

### 7.1 Technical feasibility — **High**
Every component has a mature, off-the-shelf library: `segmentation-models-pytorch`,
the clDice reference implementation, scikit-image, OSMnx, NetworkX, PyTorch
Geometric, Streamlit. **No unsolved research is required** — the novelty is in the
*combination*, not in inventing new mathematics.

### 7.2 Compute feasibility

| Stage | Where | Hardware |
|---|---|---|
| Pretraining (open data) | Pre-finale window (21 Jul – 5 Aug) | 1× T4/A100 (Colab Pro / Kaggle) |
| Cartosat fine-tune | Finale (6–7 Aug) | Finale GPU / CUDA laptop — *hours, not days* |
| Healing + graph + twin + UI | Finale | **CPU** |

The plan is feasible **only because** heavy training is front-loaded *outside*
the 30-hour finale. Training from scratch inside the finale would fail; the entire
strategy is built around avoiding that.

### 7.3 Team feasibility
Four members cover segmentation, geospatial data, graph theory, and full-stack
(see §9). Phase I and Phases II–IV run **in parallel** against a mock OSM graph,
so the dashboard is finished *before* the real masks arrive.

### 7.4 Current progress (de-risking already underway)
The data pipeline (Phase II foundations) is **already built and tested**:
Rasterio tiling, OSMnx auto-masks, and a terrain-stratified split, with a passing
test suite and 200 generated mask tiles for a Bengaluru neighbourhood. This proves
the riskiest plumbing before the competition even begins.

---

## 8. Technology Stack

| Layer | Tools |
|---|---|
| **Language / DL** | Python 3.11 · PyTorch |
| **Segmentation** | `segmentation-models-pytorch` (SegFormer / MiT-B2) · clDice loss |
| **Augmentation** | Albumentations + custom synthetic-occlusion |
| **Geospatial I/O** | Rasterio / GDAL · OSMnx · GeoPandas |
| **Graph / skeleton** | scikit-image · NetworkX · PyTorch Geometric (GNN healing) |
| **Resilience** | NetworkX (betweenness, global efficiency) · DEM/flood raster |
| **Dashboard** | Streamlit + Leaflet.js · GeoJSON export |
| **Config / infra** | OmegaConf · conda-forge env · pytest |
| **Compute** | Colab Pro / Kaggle (training) · CPU laptop (graph/twin/UI) |

---

## 9. Team Roles & Capabilities

| Role | Owns | Key responsibilities |
|---|---|---|
| **ML Lead** | Phase I | SegFormer + clDice training, occlusion augmentation, TTA, metrics, Cartosat fine-tune |
| **Data Engineer** | Phase II (data) | Rasterio/GDAL tiling, OSMnx auto-mask pipeline, terrain split, alignment handling |
| **Research / Graph Lead** | Phase III | Skeletonisation, MST/Disjoint-Set + GNN healing, NetworkX dynamic-betweenness twin, Resilience Index, literature defence |
| **Full-Stack / Deployment Lead** | Phase IV | Streamlit + Leaflet dashboard, GeoJSON export, demo flow, fallback video |

**Parallelism (as the brief encourages):** ML Lead + Data Engineer build
extraction while Research Lead + Full-Stack build the healing/twin/UI against a
*mock OSM graph* — so finale time is pure integration, not first-time assembly.

---

## 10. High-Level Implementation Plan

| Phase | Tasks | Deliverable |
|---|---|---|
| **1 — Research** | Lock architecture, study clDice / CoANet / Sat2Graph, write this proposal | Concept proposal (1 Jul) |
| **2 — Data prep** | Rasterio tiling, OSMnx auto-masks, terrain split, synthetic-occlusion aug | Reproducible dataset pipeline |
| **3 — Baseline** | U-Net + D-LinkNet, Dice/Focal, log IoU | Baseline metrics + failure cases (the demo "contrast") |
| **4 — Advanced model** | SegFormer-B2 + clDice, occlusion training, multi-scale TTA | Occlusion-robust weights |
| **5 — USP engine** | Skeleton → MST/Disjoint-Set healing; NetworkX twin (dynamic betweenness, hazard ablation, Resilience Index) | Routable graph + resilience engine |
| **6 — Dashboard** | Streamlit + Leaflet: heatmap, click-to-disable, live reroute, R gauge | Interactive demo app |
| **7 — Testing** | Ablations, per-terrain eval, demo dry-runs, fallback video | Results tables + safe demo |
| **8 — Submission** | Slides, repo, README, narrative | Final package |

> Phases 1–3 and the *scaffolding* of 5–6 happen **before** the finale. The
> 30-hour finale executes Phase 4 (Cartosat fine-tune) + Phase 5/6 integration +
> Phase 7/8.

---

## 11. Timeline

| Window | Dates | Goal / milestone |
|---|---|---|
| **Proposal sprint** | 24–30 Jun | Write & polish concept proposal; lock USP; mock dashboard → **Submit before 1 Jul** |
| Submission + wait | 1–19 Jul | De-risk quietly: stand up data pipeline + baseline U-Net |
| **Shortlist gate** | 20 Jul | If selected → go |
| Induction | 21 Jul | Confirm finale GPU / Cartosat format / whether pre-trained weights are allowed |
| Prep — data | 21–25 Jul | OSMnx masks, tiling, synthetic occlusion, terrain split → dataset frozen |
| Prep — model | 26 Jul – 1 Aug | Train SegFormer + clDice; hit target clDice/APLS → weights cached |
| Prep — engine | 1–4 Aug | Healing + resilience twin + dashboard on mock OSM graph → end-to-end on demo city |
| Dry run + buffer | 5 Aug | Full rehearsal; record fallback video |
| **Finale H0–H10** | 6 Aug | Fine-tune on Cartosat-3; run extraction |
| **Finale H10–H22** | 6–7 Aug | Heal → graph → criticality → ablation → R |
| **Finale H22–H30** | 7 Aug | Dashboard polish, ablation tables, pitch → **Final submission** |

See [02_Diagrams/Workflow_Diagram.md](02_Diagrams/Workflow_Diagram.md) for the
visual timeline.

---

## 12. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Cartosat fine-tune underperforms (domain gap) | Med | High | Multi-resolution pretrain (S2 → LISS → Cartosat); keep a Sentinel/LISS demo as fallback |
| No GPU at finale venue | Med | High | Confirm at induction; bring a CUDA laptop + cached weights |
| clDice unstable / slow | Low | Med | Warm up with Dice+Focal, add clDice after a few epochs; cap soft-skeleton iterations |
| OSM misalignment hurts metrics | High | Med | Buffered/relaxed IoU; report clDice & APLS, which tolerate offset |
| Scope creep (chasing graph-native USP) | Med | High | Freeze scope; GNN/graph-native only if everything else is done and demoed |
| Demo breaks live | Med | High | Pre-rendered fallback video + cached city graph |

---

## 13. References

Full reference list (datasets, segmentation/topology, graph-native, resilience,
event) is in [03_References.md](03_References.md). Headline anchors:

- **clDice** — Shit et al., CVPR 2021 (our core topology loss).
- **CoANet** — Mei et al., IEEE TIP 2021 (connectivity attention).
- **D-LinkNet** — Zhou et al., 2018 (the baseline competitors will use).
- **Latora & Marchiori** — global efficiency (our Resilience Index denominator).
- **Dynamic centrality** — Scientific Reports / Applied Network Science 2025
  (recalculated ablation outperforms static removal).

---

*This proposal answers the **full** problem statement — extraction → graph →
criticality → simulation — with a literature-backed, feasibility-checked plan,
while most submissions will quietly be "U-Net + heatmap." Every architectural
choice trades maximum novelty against what four people can ship in a 30-hour
finale.*
