"""D-LinkNet (Zhou et al., DeepGlobe 2018) — the strong-CNN baseline (§4.1, §10).

smp ships U-Net/LinkNet/SegFormer but *not* D-LinkNet, and the roadmap names it as
"what competitors will use -> we must beat it". So we build it here, config-driven
like the smp factory: `arch: DLinkNet` selects this module (see `build_model`).

D-LinkNet = a LinkNet backbone with a **dilated centre block** ("D-block") inserted
between encoder and decoder. The cascade of dilated 3x3 convs (rates 1,2,4,8) blows
up the receptive field without losing resolution — exactly what long, thin, partly
occluded roads need. It is still a *pixel-loss* CNN, so unlike our SegFormer+clDice
model it has no topology objective — that contrast is the point of keeping it.

The ResNet encoder is loaded through `smp.encoders.get_encoder`, so ImageNet
pretraining and the `encoder_weights=None` (offline/dry-run) path behave exactly
like every other model in the repo.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder


class DBlock(nn.Module):
    """Dilated centre block: cascade of 3x3 convs at dilation 1,2,4,8, summed.

    Each stage feeds the next (cascade), and all stage outputs plus the input are
    added — so a pixel aggregates context from receptive fields of increasing size
    while keeping the feature-map resolution unchanged.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.dilate1 = nn.Conv2d(channels, channels, 3, padding=1, dilation=1)
        self.dilate2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.dilate3 = nn.Conv2d(channels, channels, 3, padding=4, dilation=4)
        self.dilate4 = nn.Conv2d(channels, channels, 3, padding=8, dilation=8)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.relu(self.dilate1(x))
        d2 = self.relu(self.dilate2(d1))
        d3 = self.relu(self.dilate3(d2))
        d4 = self.relu(self.dilate4(d3))
        return x + d1 + d2 + d3 + d4


class DecoderBlock(nn.Module):
    """LinkNet decoder block: 1x1 reduce -> 3x3 transpose-conv upsample x2 -> 1x1 expand."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid = in_ch // 4
        self.conv1 = nn.Conv2d(in_ch, mid, 1)
        self.norm1 = nn.BatchNorm2d(mid)
        self.deconv = nn.ConvTranspose2d(mid, mid, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_ch, 1)
        self.norm3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.deconv(x)))
        return self.relu(self.norm3(self.conv3(x)))


class DLinkNet(nn.Module):
    """D-LinkNet with a config-selectable ResNet encoder (default resnet34).

    Uses the deepest four encoder stages (strides 4/8/16/32). The decoder mirrors
    them with additive skip connections, then two transpose-conv steps take the map
    back to full input resolution. Output: 1-channel logits (same H,W as input).
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
    ):
        super().__init__()
        # depth=5 -> 6 feature maps at strides [1,2,4,8,16,32]; we use the last 4.
        self.encoder = get_encoder(
            encoder_name, in_channels=in_channels, depth=5, weights=encoder_weights
        )
        c1, c2, c3, c4 = self.encoder.out_channels[2:6]  # strides 4,8,16,32

        self.center = DBlock(c4)
        self.decoder4 = DecoderBlock(c4, c3)  # /32 -> /16
        self.decoder3 = DecoderBlock(c3, c2)  # /16 -> /8
        self.decoder2 = DecoderBlock(c2, c1)  # /8  -> /4
        self.decoder1 = DecoderBlock(c1, c1)  # /4  -> /2

        self.final = nn.Sequential(
            nn.ConvTranspose2d(c1, 32, 3, stride=2, padding=1, output_padding=1),  # /2 -> /1
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # features = [stem, s2, s4, s8, s16, s32]; e1..e4 are the last four.
        _, _, e1, e2, e3, e4 = self.encoder(x)
        c = self.center(e4)
        d4 = self.decoder4(c) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        return self.final(d1)


def build_dlinknet(cfg) -> nn.Module:
    """Construct a DLinkNet from a model config block (mirrors baseline.build_model)."""
    m = cfg.model
    weights = m.get("encoder_weights", "imagenet")
    if weights in (None, "null", "none", "None"):
        weights = None
    return DLinkNet(
        encoder_name=m.get("encoder", "resnet34"),
        encoder_weights=weights,
        in_channels=int(m.get("in_channels", 3)),
        classes=int(m.get("classes", 1)),
    )
