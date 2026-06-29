"""Config system (OmegaConf): YAML files + CLI dotlist overrides.

WHY OmegaConf instead of argparse-everywhere:
  - hierarchical YAML keeps data/model/train config readable and versioned;
  - `merge` lets one experiment override only the keys it cares about;
  - CLI dotlist overrides (e.g. `train.lr=2e-4`) make sweeps trivial — no code
    change to retune. This is the scalable, reproducible pattern the roadmap's
    "production-quality, not a PoC" mandate calls for.

Usage
-----
    from route_resilience.config import load_config
    cfg = load_config("configs/base.yaml", "configs/data.yaml",
                      overrides=["train.lr=2e-4"])
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .paths import CONFIGS


def load_config(
    *yaml_files: str | Path,
    overrides: Sequence[str] | None = None,
) -> DictConfig:
    """Load and merge one or more YAML configs, then apply CLI dotlist overrides.

    Files are merged left-to-right (later files win). Relative paths resolve
    against the repo's ``configs/`` directory.
    """
    cfgs = []
    for f in yaml_files:
        path = Path(f)
        if not path.is_absolute() and not path.exists():
            path = CONFIGS / path
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        cfgs.append(OmegaConf.load(path))

    merged = OmegaConf.merge(*cfgs) if cfgs else OmegaConf.create()
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides)))
    return merged  # type: ignore[return-value]


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Persist a resolved config next to an experiment's artifacts (provenance)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)
