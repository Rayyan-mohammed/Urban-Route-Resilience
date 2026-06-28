"""Canonical project paths — the single source of truth for *where things live*.

Every milestone imports from here instead of hardcoding strings, so if the data
layout ever moves, it moves in exactly one place.

Layout
------
    data/raw/        immutable source imagery + downloaded datasets (never edited)
    data/interim/    intermediate artifacts (tiles, raw OSM pulls)
    data/processed/  model-ready tensors / aligned mask tiles / splits
    artifacts/       trained weights, metrics, exported graphs (gitignored)
    assets/          small committed demo assets (sample tiles, screenshots)
"""

from __future__ import annotations

from pathlib import Path

# Repo root = two levels up from this file: src/route_resilience/paths.py -> repo
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

ARTIFACTS = ROOT / "artifacts"
CHECKPOINTS = ARTIFACTS / "checkpoints"
METRICS = ARTIFACTS / "metrics"
GRAPHS = ARTIFACTS / "graphs"

ASSETS = ROOT / "assets"
CONFIGS = ROOT / "configs"


def ensure_dirs() -> None:
    """Create all standard directories if missing (safe to call repeatedly)."""
    for p in (RAW, INTERIM, PROCESSED, CHECKPOINTS, METRICS, GRAPHS, ASSETS):
        p.mkdir(parents=True, exist_ok=True)
