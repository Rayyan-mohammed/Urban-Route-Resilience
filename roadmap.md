# Route Resilience — BAH 2026 Feasibility Study & Winning Execution Roadmap

**Hackathon:** Bharatiya Antariksh Hackathon 2026 (BAH 2026), ISRO × Hack2skill
**Problem Statement:** Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis for Urban Mobility (NNRMS mandate)
**Programme:** UG/PG/PhD · 3–4 member team · free · 30-hour live grand finale
**Stack (proposed):** Python · PyTorch · `segmentation-models-pytorch` · SegFormer/MiT · Albumentations · Rasterio/GDAL · scikit-image · OSMnx · NetworkX · PyTorch Geometric · Streamlit + Leaflet.js
**Prepared:** 24 June 2026 (idea/proposal phase) — plan runs to the 6–7 Aug 2026 finale

> This document follows the ISRO roadmap brief. The Kaggle brain-tumor plan was used **only** for structure and presentation style. The technical content is specific to this problem statement and is backed by recent literature (2018–2026) with sources in §17.

---

## Key Dates That Shape Everything

| Date | Event | What it means for us |
|---|---|---|
| 10 Jun – **1 Jul 2026** | Registration + **idea/concept submission** (no prototype required) | Our immediate deliverable is a *proposal*, not code. Win the idea round first. |
| 15–16 Jun 2026 | Problem-statement explainer sessions | Mine the recording for hidden judging signals. |
| **20 Jul 2026** | Shortlist announced | Gate. Only shortlisted teams build. |
| 21 Jul 2026 | Induction session | Confirm finale rules: what compute/data is on-site, whether pre-trained weights are allowed. |
| 21 Jul – 5 Aug | **Pre-finale prep window** | Where we do all heavy GPU training on open data. The finale is too short to train from scratch. |
| **6–7 Aug 2026** | 30-hour grand finale (ISRO venue) | Cartosat-3 data provided here. Finale = fine-tune + integrate + heal + simulate + demo. |
| 23 Aug 2026 | National Space Day | Judges' framing: solutions that look national-scale and operational win attention. |

**Strategic consequence:** there are two distinct games. **Game 1 (now → 1 Jul):** write a proposal that survives screening. **Game 2 (21 Jul → 7 Aug):** arrive at the finale with a *pre-trained, pipeline-complete* system so the 30 hours are spent on Cartosat-3 adaptation, healing, the resilience twin, and demo polish — not on training a U-Net from zero.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Understanding](#2-problem-understanding)
3. [Dataset Analysis](#3-dataset-analysis)
4. [Literature Review](#4-literature-review)
5. [Existing Solutions Analysis](#5-existing-solutions-analysis)
6. [Competitor Thinking](#6-competitor-thinking)
7. [Proposed Winning Solution](#7-proposed-winning-solution)
8. [USP — The Hackathon Differentiator](#8-usp--the-hackathon-differentiator)
9. [Feasibility Study](#9-feasibility-study)
10. [Model Selection Analysis](#10-model-selection-analysis)
11. [Evaluation Strategy](#11-evaluation-strategy)
12. [Implementation Roadmap (Phased)](#12-implementation-roadmap-phased)
13. [Timeline Aligned with the Hackathon](#13-timeline-aligned-with-the-hackathon)
14. [Team Structure](#14-team-structure)
15. [Demo Strategy](#15-demo-strategy)
16. [Final Winning Strategy](#16-final-winning-strategy)
17. [References](#17-references)

---

## 1. Executive Summary

**Problem overview.** Satellite road extraction in dense Indian metros fails wherever the road is hidden — under tree canopy, building shadow, flyovers, parked vehicles, and cloud. Standard pixel-wise segmentation produces *broken* masks. A broken mask is topologically useless: you cannot route on it, cannot compute betweenness on it, cannot simulate a flood closure on it. The problem statement asks us to (a) *see through* occlusion and (b) turn the result into a *routable weighted graph* on which we identify critical "Gatekeeper Nodes" and run failure simulations.

**Why it matters.** This sits inside ISRO's NNRMS mandate to maximise downstream value from indigenous EO assets (Cartosat-3, Resourcesat LISS-IV). The same broken-connectivity failure blocks disaster response routing, e-governance GIS, and PDS facility/route verification — three ministries named in the brief. The deliverable is **decision support**, not a prettier mask.

**Expected impact.** A planner can open a map of a city, see its structurally weakest intersections coloured by criticality, click a node to "flood" it, and instantly read the rerouting cost and the drop in network efficiency. That is a usable resilience tool, demonstrable live.

**Key innovation (our USP, see §8).** Most teams will stop at *IoU on a clean mask*. We win on the two things they cannot fake in 30 hours:
1. **Connectivity-complete extraction** — a topology-aware loss (clDice) + synthetic-occlusion training + a learned GNN gap-healing step, so the graph is *actually routable* under occlusion.
2. **A hazard-grounded resilience digital twin** — node ablation driven by a real flood/elevation layer with *dynamically recalculated* betweenness and a Latora–Marchiori efficiency-based **Resilience Index**, not random or one-shot node deletion.

---

## 2. Problem Understanding

### 2.1 Problem-statement breakdown

| Dimension | Specifics |
|---|---|
| **Core challenge** | Recover *topologically connected* road networks under heavy occlusion, then quantify and simulate systemic vulnerability. |
| **Input data** | Multi-resolution optical EO: Sentinel-2 (10 m), Resourcesat LISS-IV (5.8 m), Cartosat-3 (sub-metre, given at finale). |
| **Output** | (1) Occlusion-robust road mask → (2) routable weighted vector graph → (3) criticality heatmap → (4) interactive failure-simulation dashboard + Resilience Index. |
| **Constraints** | 30-hour build; GPU-bound training; auto-generated (OSM) ground truth, so labels are *noisy*; must generalise across urban/suburban/forested terrain. |
| **Hidden challenges** | (a) OSM ↔ imagery misalignment (labels are offset by several metres); (b) resolution gap — pretrain at 5–10 m, infer on sub-metre Cartosat-3; (c) flyovers/underpasses create false intersections; (d) "occlusion-recall" is not measured by vanilla IoU; (e) the resilience claim is empty unless the graph is connected first. |

### 2.2 Success criteria — what judges actually score

- **Technical depth:** IoU/Dice *and* topology metrics — clDice, APLS, TOPO, connectivity ratio, occlusion-recall, relaxed/buffered IoU (3–5 px tolerance).
- **Generalisation:** demonstrated across ≥3 terrain types (dense urban, forested suburban, rural).
- **Operational usability:** does the dashboard let a *non-technical planner* make a decision? ISRO/MeitY/Consumer-Affairs judges reward decision support.
- **Novelty + defensibility:** something competitors cannot reproduce in the room.
- **Storytelling:** "from space insights to stronger cities" — the brief's own framing. Mirror it.

---

## 3. Dataset Analysis

### 3.1 Imagery (input)

| Source | Resolution | Role | Quality concerns |
|---|---|---|---|
| Sentinel-2 | 10 m | Pretrain + generalisation breadth (free, global, multi-season) | Roads <10 m wide are sub-pixel; residential streets vanish. |
| Resourcesat LISS-IV | 5.8 m | Indian-context pretrain; bridges the resolution gap to Cartosat | Limited open tiles; radiometric differences vs Sentinel. |
| Cartosat-3 | sub-metre | **Finale fine-tune + final inference** | Provided only at finale → must arrive with a model that fine-tunes in hours, not days. |

### 3.2 Ground truth & open datasets

| Dataset | Use | Note |
|---|---|---|
| SpaceNet Roads (APLS benchmark) | Pretrain + topology evaluation | Has the official APLS metric and graph GT. |
| DeepGlobe Road Extraction | Pretrain baseline | D-LinkNet's home benchmark; easy to reproduce, so *everyone uses it* — use it for the baseline, not the headline. |
| OpenSatMap | Pretrain / generalisation | Newer large-scale road set. |
| OpenStreetMap (via OSMnx) | **Auto-generated masks + graph GT** | The "zero-manual-effort" pipeline. Pull vectors, rasterise to tiles. |

### 3.3 Data challenges → mitigations

| Challenge | Mitigation |
|---|---|
| **OSM misalignment** (labels offset from imagery) | Train with **relaxed/buffered IoU + clDice** (tolerant to ±3–5 px); optional small alignment shift search per tile. |
| **Class imbalance** (roads ≈ 3–5% of pixels) | Combined Dice + Focal/Tversky loss; oversample road-dense tiles. |
| **Occlusion under-representation** | **Synthetic occlusion augmentation**: programmatically paste tree/shadow/cloud/vehicle patches over known roads so the model learns to in-paint connectivity (this *is* the occlusion-recall training signal). |
| **Resolution domain gap** | Multi-resolution pretraining (10 m → 5.8 m) + finale fine-tune on Cartosat tiles; test-time multi-scale fusion. |
| **Flyover false-junctions** | Optional overpass/underpass handling at graph stage (degree + angular continuity heuristic; cf. SAM-Road++ overpass head). |

### 3.4 Recommended preprocessing pipeline

```
Sentinel-2 / LISS-IV / Cartosat  →  Rasterio tiling (512×512, 25% overlap)
        →  per-band normalisation + CLAHE contrast
        →  OSMnx vector pull  →  rasterise to aligned road mask (GT)
        →  Albumentations: flips, rotations, brightness/shadow, +SYNTHETIC OCCLUSION
        →  train / val / test split STRATIFIED BY TERRAIN (urban / suburban / forested)
```

---

## 4. Literature Review

The field splits into two lineages: **segmentation-then-graph** (pixel masks, then post-process to a graph) and **graph-native** (predict the vector graph directly). Connectivity under occlusion is the open problem in both.

### 4.1 Segmentation lineage

| Work | Year | Method | Strength | Weakness | Relevance |
|---|---|---|---|---|---|
| U-Net (Ronneberger) | 2015 | Encoder–decoder | Simple, strong baseline | No topology awareness; breaks under occlusion | Our baseline only |
| D-LinkNet (Zhou) | 2018 | LinkNet + dilated centre, pretrained encoder | DeepGlobe winner; large receptive field | Pixel loss → fragmented roads | What competitors will use → we must beat it |
| Mosinska et al. | 2018 | Topology-aware loss + iterative refinement | First to penalise broken topology | Heavy iterative inference | Motivates our clDice choice |
| CoANet (Mei et al.) | 2021 | Connectivity Attention Network | Explicitly models connectivity to bridge occlusions | CNN context limited | Direct ancestor of our occlusion approach |
| clDice (Shit et al.) | 2021 | Centerline-Dice topology-preserving loss | Differentiable, guarantees connectivity, plug-in | Needs soft-skeletonisation | **Our core loss** |
| Seg-Road (Tao et al.) | 2023 | Transformer+CNN with connectivity structures | Long-range context + connectivity | More compute | Backbone inspiration |
| RADANet (Dai et al.) | 2023 | Road-augmented deformable attention | Handles complex road geometry | Custom ops | Attention design reference |
| ISPRS survey (Liu et al.) | 2025 | Survey of DL road extraction | Maps the whole field; confirms topology is the gap | — | Frames our positioning |

### 4.2 Graph-native lineage

| Work | Year | Method | Strength | Weakness | Relevance |
|---|---|---|---|---|---|
| RoadTracer (Bastani et al.) | 2018 | Iterative CNN-guided tracing | First graph-native | Sequential, error-accumulating, slow | Context |
| Sat2Graph (He et al.) | 2020 | One-shot graph-tensor encoding | Global coordination, fast | Directional-encoding mis-connections | Strong baseline to cite |
| RNGDet / RNGDet++ (Xu et al.) | 2022/23 | DETR-style transformer + imitation learning | Accurate topology | Slow stepwise inference | SOTA reference |
| SAM-Road (Hetang et al.) | 2024 | Segment-Anything adapted + lightweight GNN edges | Foundation-model generalisation, fast | Large model | **USP option / stretch goal** |
| SAM-Road++ / global dataset (Yin et al.) | 2025 (CVPR) | Node-guided resampling + overpass head | SOTA F1, overpass handling | Heavy | Overpass idea worth borrowing |
| GLD-Road (Zhang et al.) | 2025 | Global detect + local heal two-stage | 40–92% faster retrieval, +APLS | New | Validates our "heal broken roads" stage |
| LineGraph2Road | 2026 | Structural reasoning on line graphs | New SOTA on City-scale TOPO | Complex | Cutting-edge cite for proposal credibility |

### 4.3 Resilience / criticality lineage

| Work | Finding | Relevance |
|---|---|---|
| Latora & Marchiori | **Global efficiency** as a network-performance measure | Defines our Resilience Index denominator |
| Furno et al. | Correlate **betweenness centrality** with global efficiency to rank vulnerable links | Justifies betweenness for Gatekeeper Nodes |
| Ahmadzai et al.; multiple 2024–25 studies | Betweenness is the **most effective** single measure of road-segment vulnerability | Core metric choice |
| Sci. Reports / App. Net. Sci. (2025) | **Recalculated (dynamic)** centrality disrupts networks far more than static, one-shot removal | **USP detail:** recompute betweenness after each ablation |
| Flood-percolation studies (Houston, Messina) | Rank nodes by hazard exposure, remove in sequence, track giant-component + efficiency | **USP detail:** ground ablation in a real flood/DEM layer, not random |

**Take-away that shapes the build:** the technical frontier is *connectivity/topology*, and the resilience frontier is *dynamic, hazard-grounded* ablation. Both are buildable in a hackathon with off-the-shelf libraries (clDice, NetworkX, PyG). Most teams will use neither.

---

## 5. Existing Solutions Analysis

| Solution type | Typical architecture | Pros | Cons / why it fails in deployment |
|---|---|---|---|
| DeepGlobe/Kaggle notebooks | U-Net / D-LinkNet → threshold → skeletonise | Fast to stand up, well-documented | Masks fragment under occlusion; "graph" is a disconnected pixel skeleton → no routing |
| OSMnx-only GIS pipelines | Pull OSM graph, run NetworkX centrality | Instant graph + centrality | Uses *existing* OSM, not the satellite — defeats the EO purpose; no occlusion modelling |
| Sat2Graph / RNGDet++ repos | Graph-native transformers | True topology output, SOTA | Training is multi-GPU/day-scale; *not finishable from scratch in 30 h*; brittle to fine-tune on new sensor |
| SAM-Road / SAM-Road++ | SAM backbone + GNN edges | Excellent generalisation | Large weights, integration overhead; risky as the *whole* pipeline in a sprint |

**Design conclusion:** use a **segmentation backbone we can actually fine-tune in hours** (SegFormer/D-LinkNet with clDice), then *borrow the graph-native insight* (learned link prediction for healing) as a lightweight GNN bolt-on rather than betting the whole system on a heavy graph-native transformer. This is the feasibility-vs-novelty sweet spot.

---

## 6. Competitor Thinking

**Typical team approach (≈70% of teams).**

```
DeepGlobe → U-Net/D-LinkNet → IoU number → cv2 skeletonize
   → NetworkX graph → betweenness heatmap → static Streamlit map
```

**Expected architecture:** single CNN, BCE/Dice loss, pixel-IoU as the headline metric, a screenshot of a betweenness heatmap, maybe a "remove a node" button that deletes one node once.

**Limitations:**
- Masks break under canopy/shadow → graph is disconnected → centrality is computed on a *wrong* graph.
- No occlusion-specific training or metric → they can't *prove* occlusion robustness.
- Resilience is cosmetic: one static heatmap, random/one-shot node deletion, no efficiency metric, no hazard grounding, no live rerouting.

**Why it will not win:** it answers "can I segment a road?" Judges are asking "can I *route, rank, and simulate failure* on an Indian city under occlusion?" The gap between those two questions is exactly our scoring margin.

---

## 7. Proposed Winning Solution

### 7.1 System architecture (end-to-end)

```
            ┌────────────────────── PHASE I: OCCLUSION-ROBUST EXTRACTION ──────────────────────┐
  EO tiles  │  SegFormer (MiT-B2) / D-LinkNet encoder–decoder                                   │
 (S2/LISS/  │     loss = Dice + Focal + clDice(topology) + (optional) connectivity penalty      │
  Cartosat) │     trained with SYNTHETIC OCCLUSION augmentation + multi-scale TTA               │
            └──────────────────────────────────┬───────────────────────────────────────────────┘
                                                ▼  connectivity-complete probability mask
            ┌────────────────────── PHASE II: SKELETONISE + LEARNED HEALING ───────────────────┐
            │  scikit-image skeletonize → nodes (junctions/endpoints) + edges (segments)        │
            │  GAP HEALING: candidate links scored by (Euclidean dist, angular continuity)      │
            │     baseline = MST + Disjoint-Set bridging; UPGRADE = PyG GNN link-prediction     │
            └──────────────────────────────────┬───────────────────────────────────────────────┘
                                                ▼  routable weighted vector graph (NetworkX)
            ┌────────────────────── PHASE III: RESILIENCE DIGITAL TWIN ────────────────────────┐
            │  Betweenness centrality → Gatekeeper Nodes (criticality heatmap)                  │
            │  HAZARD-GROUNDED ablation: rank nodes by DEM/flood exposure, remove in sequence   │
            │  RECALCULATE betweenness each step; track global efficiency + giant component     │
            │  Resilience Index  R = E(perturbed) / E(baseline)   [Latora–Marchiori efficiency] │
            └──────────────────────────────────┬───────────────────────────────────────────────┘
                                                ▼
            ┌────────────────────── PHASE IV: INTERACTIVE DASHBOARD ───────────────────────────┐
            │  Streamlit + Leaflet.js: criticality heatmap overlay · click-to-disable node ·    │
            │  live reroute + Δtravel-time · Resilience Index gauge · terrain selector          │
            └───────────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Component detail

- **Data pipeline.** Rasterio tiling → OSMnx auto-masks → terrain-stratified split → Albumentations + synthetic occlusion. Fully scriptable, zero manual labelling.
- **Model pipeline.** `segmentation-models-pytorch` SegFormer/`mit_b2` encoder (ImageNet pretrained). Single PyTorch process — simpler than the brain-tumor plan's hybrid TF+PyTorch setup; no GPU-VRAM-coexistence headache.
- **Training pipeline.** Loss = `0.4·Dice + 0.3·Focal + 0.3·clDice`; AdamW (lr 1e-4, wd 1e-5); cosine LR; `torch.cuda.amp` mixed precision; early stop on **val clDice** (not val IoU — that's the whole point). *Done in the pre-finale window.*
- **Inference pipeline.** Multi-scale TTA → threshold → skeletonise → heal → graph. CPU-feasible after extraction.
- **Deployment pipeline.** Streamlit app loads the fine-tuned model once (`@st.cache_resource`); Leaflet front-end for map interaction; export GeoJSON graph + Resilience report.

---

## 8. USP — The Hackathon Differentiator

### 8.1 Five candidates (scored)

| # | USP | Novelty /10 | Complexity | Judge impact | Feasibility (30 h) | Risk |
|---|---|---|---|---|---|---|
| 1 | clDice + synthetic-occlusion training (provable occlusion-recall) | 7 | Medium | High | High (plug-in loss) | Low |
| 2 | Learned **GNN gap-healing** (PyG link prediction) vs plain MST | 8 | Medium-High | High | Medium | Medium |
| 3 | **Hazard-grounded dynamic-betweenness resilience twin** + Resilience Index + live reroute | 9 | Medium | **Very High** | High (CPU, NetworkX) | Low |
| 4 | Full graph-native (SAM-Road / RNGDet++) pipeline | 9 | Very High | High | **Low** (can't train in time) | High |
| 5 | Multi-resolution generalisation (S2→LISS→Cartosat) shown live across 3 terrains | 6 | Medium | Medium-High | High | Low |

### 8.2 Final USP (selected): **#3 + #1, framed as one product**

> **"From broken pixels to a decision-grade resilience digital twin."**
> A connectivity-complete extractor (clDice + synthetic occlusion — candidate #1, the *technical moat*) feeds a **hazard-grounded resilience digital twin** that recalculates betweenness after each simulated flood closure and reports a quantitative Resilience Index with live rerouting (candidate #3, the *emotional hook*).

**Why this is the best choice:**
- **Adds real value:** answers the ministries' actual question (disaster routing, infrastructure resilience) — not "can I segment a road."
- **Feasible in time:** clDice is a drop-in loss; the twin runs on CPU with NetworkX. No risky from-scratch transformer training.
- **Hard to replicate quickly:** competitors who built a static heatmap cannot retro-fit dynamic recalculation + hazard grounding + a connectivity-complete graph in the remaining hours — and without the connectivity-complete graph their twin is computing on a broken network, which is *visibly wrong* in a side-by-side.
- **Maximises score:** it is demonstrable *live*, it hits both the technical and the usability rubrics, and it tells the brief's own "stronger cities" story.

**Candidate #2 (GNN healing)** is kept as the *stretch upgrade* if Phase I/II finish early; **#4** is explicitly rejected as too risky for the 30-hour window.

---

## 9. Feasibility Study

### 9.1 Technical feasibility — **High.** Every component has a mature library (smp, clDice reference impl, scikit-image, OSMnx, NetworkX, PyG, Streamlit). No unsolved research is required; the novelty is in the *combination*, not in inventing new math.

### 9.2 Resource & compute feasibility

| Stage | Where | Hardware | Notes |
|---|---|---|---|
| Pretraining (open data) | **Pre-finale window (21 Jul–5 Aug)** | 1× T4/A100 (Colab Pro / Kaggle / local) | The heavy lift, *outside* the 30 h. |
| Cartosat fine-tune | Finale (6–7 Aug) | finale GPU / local workstation | Hours, not days — confirm on-site GPU at induction. |
| Healing + graph + twin + UI | Finale | CPU | Lightweight per the brief. |

### 9.3 Team-skill feasibility — 4 members cover segmentation, geospatial data, graph theory, full-stack (see §14). The split lets Phase I and Phase II/III run in parallel using mock/OSM graphs — exactly the parallelism the brief praises.

### 9.4 Time feasibility — feasible **only because** training is front-loaded into the prep window. If we tried to train in the 30 h, it would fail. The plan is built around this.

### 9.5 Risk matrix

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Cartosat fine-tune underperforms / domain gap | Med | High | Multi-resolution pretrain; keep a Sentinel/LISS demo as fallback. |
| No GPU at finale venue | Med | High | Confirm at induction; bring a CUDA laptop + cached weights. |
| clDice unstable / slow | Low | Med | Warm up with Dice+Focal, add clDice after a few epochs; cap soft-skeleton iterations. |
| OSM misalignment hurts metrics | High | Med | Buffered/relaxed IoU; report clDice & APLS which tolerate it. |
| Scope creep (trying USP #4) | Med | High | Freeze scope; #4 only if everything else is done and demoed. |
| Demo breaks live | Med | High | Pre-render a fallback video + cached city graph. |

---

## 10. Model Selection Analysis

| Class | Candidate | Pros | Cons | Verdict |
|---|---|---|---|---|
| Baseline | U-Net (smp, ResNet34) | Trivial, fast | Breaks under occlusion | Build it — as the *contrast* in the demo |
| Strong CNN | D-LinkNet | DeepGlobe-proven, large RF | Pixel loss only | Fine, but everyone has it |
| **Transformer (chosen)** | **SegFormer MiT-B2 + clDice** | Long-range context for occlusion; efficient; fine-tunes fast | Slightly more setup | **Final segmentation model** |
| Connectivity-specialised | CoANet-style head | Explicit connectivity | More custom code | Borrow ideas, not full impl |
| Graph-native | Sat2Graph / RNGDet++ / SAM-Road | SOTA topology | Can't train in time | Cite, don't build |
| Healing | MST+Disjoint-Set → **PyG GNN link-pred** | Routable graph; GNN = stretch novelty | GNN needs care | MST baseline guaranteed; GNN if time |

**Recommended final architecture:** SegFormer-B2 (smp) with `Dice + Focal + clDice`, synthetic-occlusion training, multi-scale TTA → scikit-image skeleton → MST/Disjoint-Set healing (GNN upgrade optional) → NetworkX resilience twin → Streamlit/Leaflet dashboard.

---

## 11. Evaluation Strategy

**Metrics.**

| Metric | What it proves |
|---|---|
| IoU / Dice | Baseline pixel accuracy (table-stakes) |
| **clDice** | Topology/connectivity preservation (our headline) |
| **Occlusion-Recall** | Road recovery specifically inside synthetically/naturally occluded regions |
| **Connectivity Ratio** | % growth of the largest connected component after healing |
| **APLS / TOPO** | Graph-level path similarity vs OSM ground truth |
| Relaxed/Buffered IoU (3–5 px) | Fair scoring under OSM misalignment |
| **Resilience Index R** | E(perturbed)/E(baseline) — the decision-support output |

**Validation strategy.** Terrain-stratified hold-out (urban/suburban/forested) so generalisation is *measured*, not claimed. **Error analysis:** confusion on occluded vs clean tiles; per-terrain breakdown. **Ablation studies (high judge value):** (1) Dice vs Dice+clDice; (2) with vs without synthetic occlusion; (3) before vs after healing (connectivity ratio jump); (4) static vs dynamic betweenness ablation. **Benchmarking:** quote our numbers next to D-LinkNet/Sat2Graph reported APLS to position credibly.

---

## 12. Implementation Roadmap (Phased)

| Phase | Tasks | Deliverables | Est. hours |
|---|---|---|---|
| **1 — Research** | Lock architecture, read clDice/CoANet/Sat2Graph, write proposal | Concept proposal (for 1 Jul submission) | 15 |
| **2 — Data prep** | Rasterio tiling, OSMnx auto-masks, terrain split, synthetic-occlusion aug | Reproducible dataset pipeline | 20 |
| **3 — Baseline** | U-Net + D-LinkNet, Dice/Focal, log IoU | Baseline metrics + failure cases | 15 |
| **4 — Advanced model** | SegFormer-B2 + clDice, occlusion training, multi-scale TTA | Trained occlusion-robust weights | 25 |
| **5 — USP** | Skeleton→MST/Disjoint-Set healing; NetworkX twin (dynamic betweenness, hazard ablation, Resilience Index); GNN healing if time | Routable graph + resilience engine | 25 |
| **6 — Dashboard** | Streamlit + Leaflet: heatmap, click-to-disable, live reroute, R gauge | Interactive demo app | 18 |
| **7 — Testing** | Ablations, per-terrain eval, demo dry-runs, fallback video | Results tables + safe demo | 12 |
| **8 — Submission** | Slides, repo, README, narrative | Final package | 8 |

> Phases 1–3 and the *pipeline scaffolding* of 5–6 happen **before** the finale. The finale (6–7 Aug) executes Phase 4 fine-tune on Cartosat + Phase 5/6 integration + Phase 7/8.

---

## 13. Timeline Aligned with the Hackathon

### 13.1 Gantt-style plan

| Window | Dates | Goal / milestone | Checkpoint |
|---|---|---|---|
| Proposal sprint | **24 Jun – 30 Jun** | Write & polish concept proposal; lock USP; mock dashboard screenshot | ✅ Submit before **1 Jul** |
| Submission + wait | 1 Jul – 19 Jul | (Buffer) De-risk: stand up data pipeline + baseline U-Net quietly | Baseline IoU logged |
| Shortlist gate | **20 Jul** | If selected → go. Induction **21 Jul**: confirm finale GPU/data/rules | Rules confirmed |
| Prep — data | 21 Jul – 25 Jul | OSMnx masks, tiling, synthetic occlusion, terrain split | Dataset frozen |
| Prep — model | 26 Jul – 1 Aug | Train SegFormer+clDice; hit target clDice/APLS | Weights cached |
| Prep — engine | 1 Aug – 4 Aug | Healing + resilience twin + dashboard on *mock/OSM* graph | App works end-to-end on demo city |
| Dry run + buffer | 5 Aug | Full rehearsal; record fallback video | Demo-ready |
| **Finale H0–H10** | 6 Aug | Fine-tune on Cartosat-3; run extraction | Cartosat masks produced |
| **Finale H10–H22** | 6–7 Aug | Heal → graph → criticality → ablation → R | Twin runs on Cartosat graph |
| **Finale H22–H30** | 7 Aug | Dashboard polish, ablation tables, pitch, submit | ✅ Final submission |

### 13.2 Daily goals during prep (illustrative)

- **Days 1–2 (data):** every tile has an aligned OSM mask; occlusion augmentation visibly works.
- **Days 3–5 (model):** clDice beats Dice-only on connectivity ratio — the ablation that wins.
- **Days 6–7 (engine):** click a node → reroute renders → R updates. The money shot.
- **Day 8 (buffer):** break it, fix it, film it.

---

## 14. Team Structure (4 members)

| Role | Owner of | Key responsibilities |
|---|---|---|
| **ML Lead** | Phase 4 | SegFormer+clDice training, occlusion aug, TTA, metrics, fine-tune on Cartosat |
| **Data Engineer** | Phase 2 | Rasterio/GDAL tiling, OSMnx auto-mask pipeline, terrain split, alignment handling |
| **Research / Graph Lead** | Phase 5 | Skeletonisation, MST/Disjoint-Set + GNN healing, NetworkX dynamic-betweenness twin, Resilience Index, literature defence |
| **Full-Stack / Deployment Lead** | Phase 6 | Streamlit + Leaflet dashboard, GeoJSON export, demo flow, fallback video |

**Parallelism (as the brief encourages):** ML Lead + Data Engineer build extraction; Research Lead + Full-Stack build healing/twin/UI against a *mock OSM graph* — so the dashboard is finished before the real masks even arrive, and finale time is pure integration.

---

## 15. Demo Strategy

**Narrative arc (90 seconds to hook judges):**
1. **The pain:** show a raw Cartosat tile with tree-canopy/shadow over a road. Run a vanilla U-Net → *broken* mask. "This graph cannot be routed."
2. **The fix:** same tile, our clDice + occlusion model → *connectivity-complete* mask. Toggle the ablation table: connectivity ratio jumps. "Now it's routable."
3. **The product:** the dashboard. Criticality heatmap glows — Gatekeeper Nodes in red. **Click one** ("flood this junction"). Routes reroute live; Δtravel-time and the Resilience Index gauge update. "A planner just stress-tested the city in two clicks."
4. **The scale:** terrain selector — urban → forested → rural, model holds. "Sentinel-to-Cartosat, any Indian city."
5. **The close:** the brief's line — *"from space insights to stronger cities."*

**Screens to prepare:** before/after mask comparison · ablation bar chart (Dice vs +clDice) · live criticality map · click-to-disable reroute · Resilience Index gauge · per-terrain generalisation strip.

**Live-prediction examples:** a Bengaluru tile (named in the brief) plus one forested and one rural tile.

**Storytelling rule:** lead with the *decision*, not the architecture. Judges remember "I clicked and the city rerouted," not "we used a MiT-B2 encoder."

---

## 16. Final Winning Strategy

**Why this reaches the finals.** The proposal answers the *full* problem statement (extraction → graph → criticality → simulation) with a credible, literature-backed, feasible plan — while most proposals will quietly be "U-Net + heatmap."

**Why judges will care.** It maps directly onto the named stakeholders' needs: disaster routing (NDMA-style), MeitY GIS e-governance, Consumer-Affairs PDS route verification. It is decision support, demonstrable live.

**What makes it unique.** Connectivity-complete extraction (clDice + synthetic occlusion) *plus* a hazard-grounded, dynamically-recalculated resilience twin with a quantitative Resilience Index — a combination competitors cannot assemble late.

**Biggest risks.** Cartosat domain gap; no finale GPU; demo failure. All mitigated (§9.5) with multi-resolution pretraining, cached weights, and a fallback video.

**How to maximise score.** Show ablations (they prove, not assert). Measure generalisation across 3 terrains. Make the demo a *decision*, not a dashboard tour. Mirror the brief's language.

### Final action checklist

- [ ] **Submit concept proposal before 1 Jul** (Game 1) — USP, architecture diagram, metrics, mock dashboard.
- [ ] Watch the 15–16 Jun explainer recording; extract hidden judging signals.
- [ ] Quietly build data pipeline + baseline during the wait (de-risk early).
- [ ] At **21 Jul induction**, confirm finale GPU + Cartosat format + whether pre-trained weights are allowed.
- [ ] Pretrain SegFormer+clDice in the prep window; hit target clDice/APLS; cache weights.
- [ ] Finish healing + resilience twin + dashboard on a mock OSM graph **before** the finale.
- [ ] Record a fallback demo video on Day 8.
- [ ] Finale: fine-tune on Cartosat → heal → twin → polish → pitch the *decision*.
- [ ] Prepare the ablation tables that justify clDice, occlusion training, healing, and dynamic betweenness.

---

## 17. References

*Datasets:* SpaceNet Roads (APLS benchmark); DeepGlobe Road Extraction Challenge; OpenSatMap; OpenStreetMap (via OSMnx); ISRO Sentinel-2 / Resourcesat LISS-IV / Cartosat-3.

*Segmentation & topology:* Ronneberger et al., U-Net (2015); Zhou et al., D-LinkNet (2018); Mosinska et al., topology-aware delineation loss (CVPR 2018); Mei et al., CoANet — connectivity attention (IEEE TIP 2021); Shit et al., clDice topology-preserving loss (CVPR 2021); Tao et al., Seg-Road transformer+CNN (Remote Sensing 2023); Dai et al., RADANet (IEEE TGRS 2023); Liu et al., "Deep learning-based road extraction: progress, problems, perspectives" survey (ISPRS J. P&RS, 2025).

*Graph-native extraction:* Bastani et al., RoadTracer (CVPR 2018); He et al., Sat2Graph (ECCV 2020); Xu et al., RNGDet (2022) / RNGDet++ (IEEE RA-L 2023); Hetang et al., SAM-Road (2024); Yin et al., SAM-Road++ & global-scale dataset (CVPR 2025); Zhang et al., GLD-Road (2025); LineGraph2Road (2026); DOGE Bezier graph optimisation (2025).

*Resilience & criticality:* Latora & Marchiori, global efficiency; Furno et al., betweenness–efficiency framework; Ahmadzai et al. (2019) and 2024–25 road-vulnerability studies confirming betweenness as the strongest segment-vulnerability measure; Scientific Reports / Applied Network Science (2025) on recalculated (dynamic) centrality outperforming static removal; flood-percolation road studies (Houston, Messina) for hazard-grounded ablation.

*Event:* Bharatiya Antariksh Hackathon 2026 official listing (hack2skill.com/event/bah2026); timeline (reg. closes 1 Jul, shortlist 20 Jul, induction 21 Jul, 30-hour finale 6–7 Aug 2026).

---

*This roadmap is a complete, self-consistent execution plan for BAH 2026. It is engineered around two facts: the proposal is due before any code (win Game 1), and all heavy training must precede the 30-hour finale (win Game 2). Every architectural choice trades maximum novelty against what four people can actually ship.*
