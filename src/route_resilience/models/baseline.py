"""Segmentation model factory (M3 baseline, reused by M4).

Thin wrapper over segmentation-models-pytorch so the architecture/encoder are
config-driven. The baseline is `Unet` + `resnet34`; M4 swaps to a SegFormer
encoder (e.g. arch=Unet encoder=mit_b2) with NO code change here.

`arch: DLinkNet` is special-cased to our in-repo implementation (smp has no
D-LinkNet); everything else is delegated to `smp.create_model`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import segmentation_models_pytorch as smp
from omegaconf import DictConfig

from .dlinknet import build_dlinknet

if TYPE_CHECKING:
    import torch.nn as nn


def build_model(cfg: DictConfig) -> nn.Module:
    m = cfg.model
    arch = m.get("arch", "Unet")
    # D-LinkNet is not in smp — dispatch to our implementation (§4.1, §10 baseline).
    if str(arch).lower() in ("dlinknet", "d-linknet", "dlinknet34"):
        return build_dlinknet(cfg)
    weights = m.get("encoder_weights", "imagenet")
    # OmegaConf may give the string "null"/"None"; normalise to actual None.
    if weights in (None, "null", "none", "None"):
        weights = None
    return smp.create_model(
        arch=arch,
        encoder_name=m.get("encoder", "resnet34"),
        encoder_weights=weights,
        in_channels=int(m.get("in_channels", 3)),
        classes=int(m.get("classes", 1)),
    )
