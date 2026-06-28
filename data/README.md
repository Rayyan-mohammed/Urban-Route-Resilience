# Data layout

Heavy geospatial data is **gitignored**. This file documents the contract; the
actual files are produced by the pipeline (M1+) or downloaded locally.

```
data/
├── raw/         # immutable sources — NEVER edited by code
│   ├── spacenet/      SpaceNet Roads (APLS GT)
│   ├── deepglobe/     DeepGlobe Road Extraction
│   ├── opensatmap/    OpenSatMap
│   └── sensors/       Sentinel-2 / LISS-IV tiles (Cartosat-3 arrives at finale)
├── interim/     # tiles + raw OSM pulls (regenerable)
└── processed/   # model-ready: aligned mask tiles + train/val/test split manifests
```

**Rule:** code reads from `raw/`, writes to `interim/` and `processed/`. Never
write back into `raw/`. Reproduce everything from `raw/` + configs.
