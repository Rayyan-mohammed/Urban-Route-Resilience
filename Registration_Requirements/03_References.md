# References & Supporting Material

Sources backing the Route Resilience proposal. The field spans
**segmentation-then-graph**, **graph-native** extraction, and
**network-resilience** literature (2015–2026).

## Datasets

- **SpaceNet Roads** — APLS benchmark with graph ground truth (pretrain + topology eval).
- **DeepGlobe Road Extraction Challenge** — D-LinkNet's home benchmark (baseline).
- **OpenSatMap** — newer large-scale road dataset (pretrain / generalisation).
- **OpenStreetMap** (via OSMnx) — auto-generated masks + graph ground truth.
- **ISRO EO** — Sentinel-2 (10 m), Resourcesat LISS-IV (5.8 m), Cartosat-3 (sub-metre, finale).

## Segmentation & topology

- Ronneberger et al. — **U-Net** (MICCAI 2015). Baseline encoder–decoder.
- Zhou et al. — **D-LinkNet** (CVPRW 2018). DeepGlobe winner; large receptive field.
- Mosinska et al. — Topology-aware delineation loss (CVPR 2018).
- Mei et al. — **CoANet**, Connectivity Attention Network (IEEE TIP 2021).
- Shit et al. — **clDice**, centerline-Dice topology-preserving loss (CVPR 2021). **Our core loss.**
- Tao et al. — **Seg-Road**, transformer + CNN with connectivity (Remote Sensing 2023).
- Dai et al. — **RADANet**, road-augmented deformable attention (IEEE TGRS 2023).
- Liu et al. — "Deep learning-based road extraction: progress, problems, perspectives" survey (ISPRS J. P&RS 2025).

## Graph-native extraction

- Bastani et al. — **RoadTracer** (CVPR 2018).
- He et al. — **Sat2Graph** (ECCV 2020).
- Xu et al. — **RNGDet** (2022) / **RNGDet++** (IEEE RA-L 2023).
- Hetang et al. — **SAM-Road** (2024).
- Yin et al. — **SAM-Road++** & global-scale dataset (CVPR 2025); overpass head.
- Zhang et al. — **GLD-Road**, global-detect + local-heal two-stage (2025).
- **LineGraph2Road** (2026) — structural reasoning on line graphs.
- **DOGE** — Bezier graph optimisation (2025).

## Resilience & criticality

- **Latora & Marchiori** — global efficiency as a network-performance measure.
  *Defines our Resilience Index denominator.*
- **Furno et al.** — betweenness–efficiency framework for ranking vulnerable links.
- **Ahmadzai et al.** (2019) and 2024–25 road-vulnerability studies — betweenness
  is the strongest single segment-vulnerability measure.
- **Scientific Reports / Applied Network Science** (2025) — recalculated (dynamic)
  centrality disrupts networks more than static, one-shot removal. *Justifies our
  recompute-after-each-ablation design.*
- **Flood-percolation road studies** (Houston, Messina) — rank nodes by hazard
  exposure, remove in sequence, track giant component + efficiency. *Justifies our
  hazard-grounded ablation.*

## Event

- **Bharatiya Antariksh Hackathon 2026** — official listing:
  https://hack2skill.com/event/bah2026
- Timeline: registration + idea submission close **1 Jul 2026**; shortlist
  **20 Jul**; induction **21 Jul**; 30-hour grand finale **6–7 Aug 2026**;
  National Space Day **23 Aug 2026**.

---

*Full feasibility study with per-work strengths/weaknesses tables:
[`../roadmap.md`](../roadmap.md) §4 (Literature Review) and §17 (References).*
