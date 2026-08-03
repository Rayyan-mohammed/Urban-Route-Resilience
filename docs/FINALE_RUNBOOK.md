# Finale runbook — 30 hours, 6–7 Aug 2026

Every command here has been run end-to-end on this repo. The finale is
integration and fine-tuning, not construction: the extraction → graph → heal →
twin → dashboard chain is already built and tested.

**Golden rule:** get a *complete* demo working on the very first Cartosat tile
within the first six hours, then improve it. A finished pipeline on one tile
beats a half-finished pipeline on fifty.

---

## Before you leave for the venue

| # | Item | Why it matters |
|---|---|---|
| 1 | Trained weights in `artifacts/checkpoints/` | The venue may have no GPU and no internet. Without these there is no demo. |
| 2 | **Pre-cache OSM** for the demo city (`python scripts/build_dataset.py`) | `--source cartosat` fetches OSM labels over the internet. If the venue is offline, cached masks are the fallback. |
| 3 | `pip download` / conda pack of the environment | No wheel downloads on site. |
| 4 | Fallback demo video + a pre-built graph JSON | §9.5 rates "demo breaks live" as high impact. |
| 5 | `pytest -q` green on the laptop you are bringing | Prove the machine itself is fine before the clock starts. |
| 6 | A DEM or flood raster for the demo city | Drives `RasterHazard`; without it the hazard is synthetic. |

---

## H0–H2 · Land the data

Cartosat-3 tiles arrive as GeoTIFFs **with no labels**. Generate labels from OSM
over each tile's own footprint:

```bash
python scripts/ingest_dataset.py --source cartosat \
    --root data/raw/cartosat --terrain cartosat --append
```

This reprojects the OSM drivable network onto each image's exact pixel grid,
tiles it, drops near-empty tiles, keeps the true CRS/transform, and re-runs the
terrain-stratified split. Check the summary it prints: `terrain=cartosat` must
show a non-trivial tile count in train **and** val **and** test.

> **Offline venue?** Skip OSMnx and use pre-cached masks:
> `python scripts/ingest_dataset.py --source folder --images <imgs> --masks <masks> --terrain cartosat --append`

**Sanity gate before moving on** — confirm the labels actually align with the
imagery. Open one ingested pair; if the roads are offset, adjust
`data.osm.road_buffer_m` or fall back to cached masks. Misaligned labels will
quietly destroy the fine-tune.

## H2–H8 · Fine-tune

Start from the pre-trained SegFormer+clDice rather than from scratch:

```bash
python scripts/train.py --config base.yaml data.yaml model_segformer.yaml \
    train.yaml train_segformer.yaml \
    -o train.epochs=12 -o train.lr=5e-5 -o train.out_name=segformer_cartosat
```

Low LR, few epochs — this is domain adaptation, not training. Watch val IoU in
`artifacts/metrics/segformer_cartosat_history.json`. **If it is not clearly
beating the pre-finale checkpoint by H8, stop and demo with the existing
weights.** A working demo on pre-trained weights outranks a better model nobody sees.

## H8–H12 · Extract → graph → twin

One command takes imagery all the way to a resilience report:

```bash
python scripts/predict.py --image-dir data/raw/cartosat \
    --checkpoint artifacts/checkpoints/segformer_cartosat_best.pth \
    --tta --pipeline
```

Masks land in `artifacts/predictions/`, reports in `artifacts/reports/`. The
sliding window handles tiles of any size and blends overlaps so there are no
seams; `--tta` adds multi-scale + flip averaging.

Add the real hazard once you have a DEM or flood layer:

```bash
python scripts/predict.py --image data/raw/cartosat/tile_01.tif \
    --checkpoint artifacts/checkpoints/segformer_cartosat_best.pth \
    --tta --pipeline --hazard-raster data/raw/dem.tif \
    --hazard-mode elevation --hazard-threshold 812
```

Use `--hazard-mode depth --hazard-threshold 0.3` for a flood-depth product
instead of a bare DEM.

**Watch for:** `road_frac` near 0 or near 1 in the predict log. Either means the
model is not discriminating — check radiometry (`--no-stretch`) and `--threshold`
before blaming the weights.

## H12–H18 · Numbers for the judges

```bash
# Model comparison — the headline table
python scripts/evaluate.py --checkpoint artifacts/checkpoints/baseline_unet_best.pth \
    --compare artifacts/checkpoints/segformer_cartosat_best.pth --apls --tta

# All four ablations
python scripts/ablations.py --all --limit 40 --split test \
    --reports artifacts/metrics/eval_baseline_unet_test.json \
              artifacts/metrics/eval_segformer_cartosat_best_test.json
```

`artifacts/reports/ablations.md` is paste-ready for slides. Ablations 3 and 4
(healing, dynamic betweenness) need no weights — run them early so you are never
without a results table.

## H18–H26 · Dashboard

```bash
streamlit run src/route_resilience/dashboard/app.py
```

Point it at a predicted mask from `artifacts/predictions/`. Rehearse the click
path until it is muscle memory: **pick a red junction → flood it → read the
Resilience Index drop and the reroute line.** That single interaction is the pitch.

## H26–H30 · Pitch and submit

Follow the §15 narrative: broken baseline mask → connectivity-complete mask →
click-to-flood → terrain generalisation → close on *"from space insights to
stronger cities."* Lead with the decision, not the encoder.

Final checks: `pytest -q` green · README demo assets render · fallback video
reachable offline · `artifacts/reports/ablations.md` committed and pushed.

---

## If something breaks

| Symptom | Cause | Do this |
|---|---|---|
| Predicted mask is all road or all background | Radiometry mismatch or untrained weights | `--no-stretch`, then tune `--threshold`; verify the checkpoint's epoch/metrics |
| `OSM fetch failed` during ingest | No internet at the venue | Switch to `--source folder` with pre-cached masks |
| Graph has hundreds of components | Mask is fragmented | Raise `graph.healing.max_gap_m`; confirm healing is on (not `--no-heal`) |
| Resilience Index looks implausible | Graph is a blob, not a road network | Check the mask first — the twin is only as good as the topology |
| Fine-tune diverges | LR too high for adaptation | Drop to 2e-5, or fall back to the pre-finale checkpoint |
| CUDA out of memory | Tile/batch too large | `-o train.batch_size=4`, or `--window 256` at inference |

## What is synthetic vs real

Be straight about this if a judge asks — it reads as rigour, not weakness.

- **Real:** the graph extraction, healing, betweenness, efficiency and Resilience
  Index maths; the ablation numbers; geo-referencing throughout.
- **Depends on what you attach:** the hazard layer is real if you pass
  `--hazard-raster`, synthetic (radius/band) otherwise.
- **Labels are OSM-derived**, so they carry OSM's own misalignment. That is why
  clDice and APLS are reported alongside IoU — they tolerate it (§9.5).
