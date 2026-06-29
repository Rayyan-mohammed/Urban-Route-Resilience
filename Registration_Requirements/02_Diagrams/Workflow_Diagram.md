# Workflow Diagram — Team & Timeline

Two views: (1) how the four roles work in **parallel**, and (2) the project
**timeline** from proposal to finale.

## 1. Parallel team workflow

```mermaid
flowchart TB
    subgraph TrackA["Track A — Extraction (ML Lead + Data Engineer)"]
        A1["Rasterio tiling + OSMnx masks"] --> A2["Synthetic occlusion aug"]
        A2 --> A3["Train SegFormer + clDice"] --> A4["Cartosat fine-tune (finale)"]
    end
    subgraph TrackB["Track B — Engine + UI (Graph Lead + Full-Stack)"]
        B1["Mock OSM graph"] --> B2["Healing + NetworkX twin"]
        B2 --> B3["Dashboard (Streamlit + Leaflet)"] --> B4["Demo + fallback video"]
    end
    A4 --> INT["FINALE INTEGRATION:<br/>real Cartosat masks into finished engine"]
    B4 --> INT
    INT --> PITCH["Pitch the decision"]
```

The two tracks run independently because Track B builds against a **mock OSM
graph** — so the dashboard is finished before the real masks arrive, and finale
time is pure integration.

## 2. Project timeline (Gantt)

```mermaid
gantt
    dateFormat  YYYY-MM-DD
    title Route Resilience — BAH 2026
    section Game 1 (Proposal)
    Proposal sprint            :done,    p1, 2026-06-24, 2026-06-30
    Submit before 1 Jul        :milestone, m1, 2026-07-01, 0d
    De-risk (pipeline+baseline) :active,  p2, 2026-07-01, 2026-07-19
    section Gate
    Shortlist announced        :milestone, m2, 2026-07-20, 0d
    Induction (confirm rules)  :milestone, m3, 2026-07-21, 0d
    section Game 2 (Prep)
    Prep - data                :         p3, 2026-07-21, 2026-07-25
    Prep - model (train+cache) :         p4, 2026-07-26, 2026-08-01
    Prep - engine + dashboard  :         p5, 2026-08-01, 2026-08-04
    Dry run + fallback video   :         p6, 2026-08-05, 1d
    section Finale (30h)
    Cartosat fine-tune + extract :       f1, 2026-08-06, 1d
    Heal -> twin -> ablation     :       f2, 2026-08-06, 1d
    Polish + pitch + submit      :       f3, 2026-08-07, 1d
```

## ASCII fallback (timeline)

```
24-30 Jun  Proposal sprint ............ SUBMIT before 1 Jul  *
01-19 Jul  De-risk: data pipeline + baseline U-Net
20 Jul     Shortlist gate  *
21 Jul     Induction (confirm GPU / Cartosat / pre-trained weights)  *
21-25 Jul  Prep: data (masks, tiling, occlusion, terrain split)
26 Jul-1 Aug  Prep: train SegFormer+clDice -> cache weights
01-04 Aug  Prep: healing + twin + dashboard on mock OSM graph
05 Aug     Dry run + record fallback video
06 Aug     Finale H0-H10:  Cartosat fine-tune + extraction
06-07 Aug  Finale H10-H22: heal -> graph -> ablation -> R
07 Aug     Finale H22-H30: polish, ablation tables, pitch, SUBMIT  *
```
