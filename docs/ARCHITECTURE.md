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
                          inference/   sliding-window predict on real imagery
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
   ▲  inference.predict_geotiff  (real imagery -> mask, sliding window + TTA)
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
trained on available datasets. The finale integration surface is **code-complete** —
each step below is a command, not a build task. See
[`FINALE_RUNBOOK.md`](FINALE_RUNBOOK.md) for the hour-by-hour sequence.

1. **Label the imagery.** Cartosat-3 tiles arrive with no ground truth, so
   `data.ingest.iter_geotiff_osm` (`--source cartosat`) pulls the OSM drivable
   network for each image's own footprint and burns it onto the image's pixel
   grid. Geo-referenced sources keep their true CRS/transform through tiling.
2. **Fine-tune** the SegFormer on those pairs (`scripts/train.py` — device-agnostic,
   GPU-ready). Low LR, few epochs: adaptation, not training.
3. **Predict.** `inference.predict_geotiff` (`scripts/predict.py`) runs the model
   over imagery of any size with overlapping, Hann-blended windows and optional
   multi-scale TTA, writing a mask GeoTIFF that carries the source CRS/transform.
   Non-8-bit imagery is percentile-stretched to match training radiometry.
4. **Feed it in.** That mask is the input `pipeline.run_tile_pipeline` and
   `dashboard.service.load_tile_graph` already take — graph → heal → twin →
   dashboard is untouched. `scripts/predict.py --pipeline` does 3 and 4 in one call.
5. **Wire a real hazard.** `resilience.hazard.RasterHazard` reads a DEM or
   flood-depth raster (`--hazard-raster`), reprojecting node coordinates as needed.
   No downstream change — it is just another `Hazard`.

### Why inference is not "just run the model"

Training and evaluation only ever see 512 px manifest tiles, but a delivered scene
is arbitrarily large. Cutting it into a hard tile grid leaves **seams**, and a seam
is a topology error — precisely the connectivity clDice is trained to preserve.
Windows therefore overlap and are blended, and window origins are clamped inside
the image so every window is full-size for stride-32 encoders.

## Testing

Every phase has offline unit tests (`tests/`) plus cross-phase integration checks.
Heavy/geo tests skip gracefully when the dataset or geo stack is absent, so CI on a
bare machine still validates the pure-Python logic.
