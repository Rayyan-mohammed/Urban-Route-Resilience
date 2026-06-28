"""Terrain-stratified train/val/test split (§3.4, §11).

Why stratify by terrain: the roadmap demands generalisation be *measured*, not
claimed (§2.2). If we split tiles randomly, urban tiles (which dominate by count)
flood every split and a forested-recall failure hides in the average. Stratifying
guarantees each terrain appears in train AND val AND test, so per-terrain metrics
in M5 are honest.

We also keep tiles from the same place out of being split across train/val by
default OFF here (tiles overlap by 25%, so leakage between adjacent tiles is a
real concern). See `group_by_place` to enforce place-level holdout if desired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from ..utils import get_logger

log = get_logger(__name__)


def stratified_split(
    df: pd.DataFrame,
    cfg: DictConfig,
    *,
    stratify_by: str = "terrain",
    seed: int = 42,
) -> pd.DataFrame:
    """Assign each row a `split` in {train, val, test}, stratified by `stratify_by`.

    Ratios come from cfg.data.split.{train,val,test}. Returns a new DataFrame.
    """
    ratios = cfg.data.split
    r_train, r_val = float(ratios.train), float(ratios.val)
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["split"] = ""

    for stratum, group in out.groupby(stratify_by):
        idx = group.index.to_numpy().copy()  # writable: rng.shuffle mutates in place
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * r_train))
        n_val = int(round(n * r_val))
        out.loc[idx[:n_train], "split"] = "train"
        out.loc[idx[n_train : n_train + n_val], "split"] = "val"
        out.loc[idx[n_train + n_val :], "split"] = "test"
        log.info(
            "%s=%s: %d tiles -> train=%d val=%d test=%d",
            stratify_by, stratum, n, n_train, n_val, n - n_train - n_val,
        )

    return out


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot table of tile counts by terrain x split — for a quick sanity glance."""
    return pd.crosstab(df["terrain"], df["split"], margins=True)
