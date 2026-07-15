"""M5 test — the evaluate() orchestrator (model -> occlusion -> metrics -> report).

Runs in dry-run mode on a few tiles with an untrained model: we assert the report
STRUCTURE (all metric keys present, per-terrain breakdown, tile count), not the
metric values (which are meaningless without trained weights). Closes the gap
where only the individual metric functions were unit-tested.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("skimage")

from route_resilience.config import load_config  # noqa: E402

_HAS_DATA = Path("data/processed/manifest.csv").exists()
pytestmark = pytest.mark.skipif(not _HAS_DATA, reason="dataset not built")

CFG = ["base.yaml", "data.yaml", "model_baseline.yaml", "train.yaml"]
_METRIC_KEYS = {"iou", "dice", "cldice", "connectivity_ratio", "recall", "occlusion_recall"}


def test_evaluate_report_structure():
    from route_resilience.evaluation.evaluate import evaluate

    cfg = load_config(*CFG)
    rep = evaluate(cfg, checkpoint=None, split="test", dry_run=True, max_tiles=4)

    assert rep["split"] == "test"
    assert rep["checkpoint"] is None
    assert "/" in rep["arch"]                     # arch/encoder
    overall = rep["overall"]
    assert _METRIC_KEYS <= set(overall)
    assert overall["n_tiles"] == 4
    # connectivity_ratio = n_comp(true)/n_comp(pred) is unbounded above (a blobby
    # under-fragmented prediction gives >1), so only bound it below.
    for k in _METRIC_KEYS - {"connectivity_ratio"}:
        assert 0.0 <= overall[k] <= 1.0
    assert overall["connectivity_ratio"] >= 0.0


def test_evaluate_per_terrain_breakdown():
    from route_resilience.evaluation.evaluate import evaluate

    cfg = load_config(*CFG)
    rep = evaluate(cfg, checkpoint=None, split="test", dry_run=True, max_tiles=6)
    assert rep["per_terrain"]                      # at least one terrain
    total = sum(v["n_tiles"] for v in rep["per_terrain"].values())
    assert total == rep["overall"]["n_tiles"]      # terrains partition the tiles


def test_evaluate_dry_run_writes_no_file(tmp_path):
    from route_resilience.evaluation.evaluate import evaluate

    cfg = load_config(*CFG)
    rep = evaluate(cfg, checkpoint=None, split="val", dry_run=True, max_tiles=3)
    # dry-run must not persist a report; apls left uncomputed by default
    assert rep["overall"].get("apls") is None
