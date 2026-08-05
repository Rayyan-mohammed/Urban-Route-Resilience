# Datasets

Maps the roadmap's §3.2 datasets onto what the code actually does.

## The problem this solves

The OSMnx pipeline (`scripts/build_dataset.py`) produces geo-referenced road
**masks** but no imagery, so `RoadTileDataset` falls back to `synth_image.py` —
a pseudo-image generated *from the mask*. That exercises the plumbing, but any
metric from it is meaningless: the model would be reading the answer off its own
input. **Real numbers need real imagery.**

`scripts/ingest_dataset.py` fixes that: it tiles real image+mask pairs into the
manifest and sets `image_path`, so training reads actual satellite pixels. Nothing
downstream (split, training, TTA, eval, graph, twin, dashboard) changes.

## Sources

## The active training set

Training runs on **DeepGlobe + Massachusetts Roads** — two real-imagery datasets,
both one-click on Kaggle, no external downloads. Everything else below is built
but not part of the current run.

| Source | Role (§3.2) | GSD | Layout | Status |
|---|---|---|---|---|
| **DeepGlobe** | pretrain baseline (D-LinkNet's benchmark) | 0.5 m | `<root>/*_sat.jpg` + `*_mask.png` | ✅ **training** — Kaggle `balraj98/deepglobe-road-extraction-dataset` |
| **Massachusetts Roads** | second real domain (different sensor + region) | 1.0 m | `<split>/*.tiff` + `<split>_labels/*.tif` | ✅ **training** — Kaggle `balraj98/massachusetts-roads-dataset` |
| **OSM** (OSMnx) | demo city, geo-referenced graph GT | 0.5 m | `configs/places.yaml` | ✅ built — 730 tiles, 4 Bengaluru terrains, **masks only** (no imagery) |
| **SpaceNet Roads** | pretrain + APLS topology GT | 0.3 m | `**/PS-RGB/*.tif` + `**/*.geojson` | ⚠️ adapter built, **untested on real data** (AWS requester-pays) |
| **OpenSatMap** | pretrain / generalisation breadth | 0.5 m | image dir + mask dir (matched by stem) | ⚠️ generic adapter, **untested on real data** |
| Any geo-referenced GeoTIFF | auto-label from OSM | source | `--source geotiff-osm` | ✅ built + verified live |

### Why these two

- **Both are real pixels.** OSM is masks-only, so it falls back to `synth_image.py`
  — pseudo-imagery derived from the mask, which means the model reads the answer
  off its own input. Any metric from that is meaningless.
- **Different GSD (0.5 m vs 1.0 m)** gives the multi-resolution pretraining §3.3
  asks for, instead of overfitting one sensor's scale.
- **Different regions and sensors** (DigitalGlobe satellite over varied terrain vs
  Massachusetts aerial survey) make the per-domain generalisation report real
  rather than a formality.
- **Neither needs a download.** Both mount read-only under `/kaggle/input/`.

The `terrain` column carries the domain (`deepglobe` / `massachusetts`), so the
stratified split puts each dataset in train **and** val **and** test — no source
silently owns the test set.

Honest caveat on the rest: the SpaceNet and OpenSatMap adapters are written
against their documented layouts but have never been run against the real
archives — expect to adjust the file-matching when you first attach them.

## Usage

```bash
# 1. DeepGlobe — starts a fresh manifest
python scripts/ingest_dataset.py --source deepglobe \
    --root /kaggle/input/deepglobe-road-extraction-dataset/train \
    --terrain deepglobe --limit 200

# 2. Massachusetts Roads — appended into the same manifest
python scripts/ingest_dataset.py --source massachusetts \
    --root /kaggle/input/massachusetts-roads-dataset/tiff \
    --terrain massachusetts --limit 150 --append

# --- not part of the current run ---

# SpaceNet — georeferenced imagery + GeoJSON labels rasterised onto the image grid
python scripts/ingest_dataset.py --source spacenet --root /path/to/spacenet \
    --terrain spacenet --append

# OpenSatMap / anything with an image dir + mask dir
python scripts/ingest_dataset.py --source opensatmap \
    --images /path/imgs --masks /path/masks --terrain opensatmap --append

# Any geo-referenced imagery with NO labels — OSM roads are fetched per footprint
python scripts/ingest_dataset.py --source geotiff-osm --root /path/tiles \
    --terrain mycity --append
```

- `--append` merges into the existing manifest (so you can stack all sources) and
  re-runs the terrain-stratified split. Without it, the manifest is rebuilt from
  just that source.
- `--limit N` caps source images — DeepGlobe's train split is 6,226 images of
  1024×1024, which tiles to ~56k tiles. Use a limit on the free tier.
- `--gsd` overrides the nominal ground sample distance (defaults per source in
  `data/ingest.py: DEFAULT_GSD_M`).

## How ingest works

```
image+mask pair  ->  tile to cfg.data.tile_size (with cfg.data.overlap)
                 ->  drop tiles below cfg.data.min_road_frac (class imbalance, §3.3)
                 ->  write images/<tile>.tif (3-band) + masks/<tile>.tif (1-band)
                 ->  manifest row with image_path set  ->  stratified split
```

Non-georeferenced sources (DeepGlobe/OpenSatMap) get a **synthetic north-up
transform** at the dataset's nominal GSD. That is fine for training and eval; real
geography only matters for the OSM demo city and the finale Cartosat tiles, which
carry true CRS/transforms. SpaceNet keeps its real transform and CRS, and its road
GeoJSON is reprojected and rasterised onto the image's own grid so labels align to
pixels exactly.

## The `terrain` column

For OSM it means what it says (urban/suburban/forested/rural) and drives the
per-terrain generalisation report (§11). For external datasets it acts as a
**domain** label (`--terrain deepglobe`), which keeps the stratified split honest
when several sources are stacked — each source appears in train *and* val *and*
test rather than one source silently owning the test set.
