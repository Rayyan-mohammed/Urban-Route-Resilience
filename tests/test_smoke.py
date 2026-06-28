"""M0 smoke tests — prove the foundation is wired up correctly.

These run with zero heavy deps (no torch/gdal needed) so CI and a fresh laptop
can both verify the package imports, paths resolve, and configs load.
"""

from __future__ import annotations

from pathlib import Path


def test_package_imports():
    import route_resilience

    assert route_resilience.__version__ == "0.1.0"


def test_paths_resolve():
    from route_resilience import paths

    # ROOT should contain the roadmap spec — proves we found the repo root.
    assert (paths.ROOT / "roadmap.md").exists()
    assert paths.CONFIGS.name == "configs"


def test_config_loads_and_merges():
    from route_resilience.config import load_config

    cfg = load_config("base.yaml", "data.yaml", overrides=["data.tile_size=256"])
    assert cfg.seed == 42
    assert cfg.data.tile_size == 256  # override applied
    assert cfg.data.overlap == 0.25  # base value preserved
    assert "urban" in cfg.terrains


def test_logger():
    from route_resilience.utils import get_logger

    log = get_logger("test")
    log.info("smoke test logger works")
