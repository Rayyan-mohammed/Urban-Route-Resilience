# End-to-End Pipeline Flowchart

The data → decision pipeline, showing what each stage *consumes* and *produces*.

## Mermaid

```mermaid
flowchart LR
    A["Raw EO tile<br/>(occluded road)"] --> B["Tiling + normalisation"]
    B --> C["Segmentation<br/>SegFormer + clDice"]
    C --> D{"Connectivity<br/>complete?"}
    D -->|mask| E["Skeletonise"]
    E --> F["Build NetworkX graph"]
    F --> G["Gap healing<br/>MST / Disjoint-Set"]
    G --> H["Routable weighted graph"]
    H --> I["Betweenness<br/>criticality heatmap"]
    H --> J["Hazard ablation<br/>(flood/DEM)"]
    J --> K["Recalculate betweenness"]
    K --> L["Resilience Index R"]
    I --> M["Dashboard"]
    L --> M
    M --> N["Planner decision:<br/>click-to-flood + reroute"]
```

## ASCII fallback

```
Raw EO tile (occluded)
   -> Tiling + normalisation
   -> Segmentation (SegFormer + clDice)
   -> [connectivity-complete mask]
   -> Skeletonise
   -> Build NetworkX graph
   -> Gap healing (MST / Disjoint-Set, +GNN)
   -> [routable weighted graph]
        |-> Betweenness -> criticality heatmap ----+
        |-> Hazard ablation (flood/DEM)            |
              -> recalc betweenness                |
              -> Resilience Index R ---------------+
                                                   v
                                              Dashboard
                                   (click-to-flood + live reroute)
                                                   v
                                          Planner decision
```

## Stage I/O summary

| Stage | Input | Output |
|---|---|---|
| Tiling | Raw EO scene | 512×512 normalised tiles |
| Segmentation | Tile | Connectivity-complete probability mask |
| Skeletonise | Mask | Pixel skeleton |
| Graph build | Skeleton | NetworkX nodes + weighted edges |
| Healing | Fragmented graph | **Routable** weighted graph |
| Criticality | Graph | Betweenness heatmap (Gatekeeper Nodes) |
| Ablation | Graph + flood/DEM | Sequenced node removal + recalculated betweenness |
| Resilience Index | Efficiency curve | `R = E(perturbed)/E(baseline)` |
| Dashboard | Graph + heatmap + R | Interactive decision support |
