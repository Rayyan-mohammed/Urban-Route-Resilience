# System Architecture Diagram

End-to-end architecture of **Route Resilience**. Mermaid renders on GitHub, in
VS Code (Mermaid extension), and most Markdown→PDF exporters. An ASCII fallback
follows.

## Mermaid

```mermaid
flowchart TD
    subgraph IN["INPUT — Earth Observation"]
        S2["Sentinel-2 (10 m)"]
        LISS["Resourcesat LISS-IV (5.8 m)"]
        CART["Cartosat-3 (sub-metre, finale)"]
    end

    subgraph P1["PHASE I — Occlusion-Robust Extraction (GPU, pre-finale)"]
        TILE["Rasterio tiling 512x512 + CLAHE"]
        OSM["OSMnx vectors -> rasterised mask (GT)"]
        AUG["Albumentations + SYNTHETIC OCCLUSION"]
        SEG["SegFormer MiT-B2<br/>loss = Dice + Focal + clDice"]
        TTA["Multi-scale TTA + threshold"]
    end

    subgraph P2["PHASE II — Skeletonise + Healing (CPU)"]
        SK["scikit-image skeletonize"]
        GR["NetworkX nodes + edges (weighted)"]
        HEAL["Gap healing: dist + angular continuity<br/>MST / Disjoint-Set (+ optional GNN)"]
    end

    subgraph P3["PHASE III — Resilience Digital Twin (CPU)"]
        BTW["Betweenness -> Gatekeeper Nodes"]
        ABL["Hazard-grounded ablation (DEM/flood)<br/>RECALCULATE betweenness each step"]
        RIDX["Resilience Index R = E(pert)/E(base)"]
    end

    subgraph P4["PHASE IV — Interactive Dashboard"]
        DASH["Streamlit + Leaflet:<br/>heatmap | click-to-flood | live reroute | R gauge"]
    end

    S2 --> TILE
    LISS --> TILE
    CART --> TILE
    TILE --> OSM --> AUG --> SEG --> TTA
    TTA -->|connectivity-complete mask| SK --> GR --> HEAL
    HEAL -->|routable weighted graph| BTW --> ABL --> RIDX
    RIDX --> DASH
    HEAL --> DASH
```

## ASCII fallback

```
INPUT (EO)            Sentinel-2 10m | LISS-IV 5.8m | Cartosat-3 sub-m
                                       |
                                       v
PHASE I  (GPU)   Rasterio tiling + CLAHE -> OSMnx mask (GT)
                 -> Albumentations + SYNTHETIC OCCLUSION
                 -> SegFormer MiT-B2 [Dice + Focal + clDice]
                 -> multi-scale TTA + threshold
                                       |  connectivity-complete mask
                                       v
PHASE II (CPU)   skeletonize -> NetworkX graph (nodes+edges)
                 -> gap healing (dist + angle) MST/Disjoint-Set (+GNN)
                                       |  routable weighted graph
                                       v
PHASE III(CPU)   betweenness -> Gatekeeper Nodes
                 -> hazard-grounded ablation (DEM/flood), recalc each step
                 -> Resilience Index R = E(perturbed)/E(baseline)
                                       |
                                       v
PHASE IV         Streamlit + Leaflet dashboard
                 heatmap | click-to-flood | live reroute | R gauge
```

**Key property:** only Phase I needs a GPU and it runs *before* the finale.
Phases II–IV are CPU-only, so the system runs end-to-end on a laptop during
integration and the live demo.
