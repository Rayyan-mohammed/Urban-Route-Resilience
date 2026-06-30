# Architecture

Route Resilience is a four-phase pipeline that turns occluded satellite imagery
into an interactive resilience digital twin. Each phase is an installable Python
subpackage with a clean, tested interface, so phases can be developed and verified
independently.

```
            ┌──────────────────────────────────────────────────────────────┐
            │                     route_resilience/                          │
            └──────────────────────────────────────────────────────────────┘

 Phase 0  DATA            data/        tiling · OSM auto-masks · occlusion · Dataset
 Phase I  EXTRACTION      models/      U-Net (baseline) · SegFormer-B2 + clDice
                          training/    device-agnostic train loop
                          evaluation/  IoU · clDice · occlusion-recall · APLS
 Phase II GRAPH           graph/       skeletonize → NetworkX graph → healing
 Phase III TWIN           resilience/  betweenness · hazard · ablation · Resilience Index
 Phase IV UI              dashboard/   Streamlit + Leaflet decision support
```

## Data flow

```
 EO tile (GeoTIFF)
   │  data.geo.make_tile_grid + data.osm.road_edges_for_place
   ▼
 (image tile, road mask)  ──[train only]── data.occlusion.apply_occlusion
   │  models.* + training.trainer            (synthetic tree/shadow/cloud/vehicle)
   ▼
 connectivity-complete MASK   ── evaluation.metrics (clDice, occlusion-recall, APLS)
   │  graph.build.mask_to_graph   (skeletonize → trace → simplify; geo-referenced)
   ▼
 routable GRAPH (NetworkX, UTM coords)
   │  graph.heal.heal_graph        (Disjoint-Set gap bridging)
   ▼
 healed GRAPH  ── resilience.centrality (betweenness, global efficiency)
   │  resilience.hazard + resilience.simulate
   ▼
 RESILIENCE REPORT  (Resilience Index, reroute cost, critical junctions)
   │  dashboard.service (reproject UTM→WGS84) + dashboard.app
   ▼
 interactive map (Leaflet/OSM): click-to-flood, live RI, reroute
```

`pipeline.run_tile_pipeline` runs mask → graph → heal → resilience in one call —
the integration the finale reuses unchanged.

## Key design decisions

- **Geo-referencing is first-class.** Tiles keep their affine transform + CRS, so
  graph nodes carry real-world coordinates and the twin lands on a real basemap.
- **Topology is the objective, not a by-product.** clDice (Phase I) trains for
  connectivity; clDice/connectivity-ratio/APLS (evaluation) measure it; healing
  (Phase II) guarantees it. A region-only model (the baseline) is kept as the
  control to demonstrate the contrast.
- **The hazard layer is an interface.** `resilience.hazard.Hazard.impacted_nodes`
  is satisfied today by synthetic generators and at the finale by a real DEM/flood
  raster — no downstream change.
- **Config-driven and device-agnostic.** OmegaConf YAML + CLI overrides; CUDA→CPU
  auto-resolve so the same code trains on Colab and runs locally on CPU.

## Where Cartosat-3 slots in (finale only)

Everything above is built and tested on OSM-derived masks + a baseline/SegFormer
trained on available datasets. At the finale, the **only** changes are:

1. **Fine-tune** the SegFormer on Cartosat-3 imagery (`configs/model_segformer.yaml`,
   `scripts/train.py` — already device-agnostic, GPU-ready).
2. **Feed the predicted mask** into `pipeline.run_tile_pipeline` instead of the OSM
   mask. The graph → heal → twin → dashboard chain is untouched.
3. **Wire a real hazard** (DEM/flood) by implementing one `Hazard` subclass.

That is the entire finale integration surface — by design.

## Testing

Every phase has offline unit tests (`tests/`) plus cross-phase integration checks.
Heavy/geo tests skip gracefully when the dataset or geo stack is absent, so CI on a
bare machine still validates the pure-Python logic.
