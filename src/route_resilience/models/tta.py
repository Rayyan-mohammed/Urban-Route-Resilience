"""Multi-scale + flip test-time augmentation (TTA) — §7.1, §10 inference.

The roadmap's inference recipe is "multi-scale TTA -> threshold -> skeletonise".
TTA runs the model on several transformed copies of a tile (horizontal/vertical
flips + a few input scales), un-transforms each prediction, and averages the
probabilities. For thin, partly occluded roads this recovers pixels a single
forward pass misses at one scale/orientation, lifting connectivity at ~no cost.

Pure inference — no training, CPU-friendly. `tta_predict` returns a probability
map in [0,1]; callers threshold it exactly like a plain forward pass.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


@torch.no_grad()
def tta_predict(
    model: torch.nn.Module,
    img: torch.Tensor,
    *,
    scales: Sequence[float] = (0.75, 1.0, 1.25),
    flips: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    """Average sigmoid probabilities over flip + multi-scale augmentations.

    Parameters
    ----------
    img : a single image tensor (C,H,W) or a batch (N,C,H,W).
    scales : input rescale factors; each is run then resized back to (H,W).
    flips : also average the horizontal and vertical flips.

    Returns a probability map matching the input's spatial size: (1,H,W) for a
    single image, or (N,1,H,W) for a batch.
    """
    model.eval()
    batched = img.dim() == 4
    x = img if batched else img.unsqueeze(0)
    x = x.float().to(device)
    h, w = x.shape[-2:]

    # (transform-to-model, inverse-on-prediction) pairs. Flips are their own inverse.
    def _ident(t):
        return t

    def _hflip(t):
        return torch.flip(t, dims=[-1])

    def _vflip(t):
        return torch.flip(t, dims=[-2])

    views = [(_ident, _ident)]
    if flips:
        views += [(_hflip, _hflip), (_vflip, _vflip)]

    probs = torch.zeros((x.shape[0], 1, h, w), dtype=torch.float32, device=device)
    n = 0
    for s in scales:
        # Keep scaled dims divisible by 32 so encoders with stride-32 don't error.
        sh = max(32, int(round(h * s / 32)) * 32)
        sw = max(32, int(round(w * s / 32)) * 32)
        xs = x if (sh == h and sw == w) else _resize(x, (sh, sw))
        for fwd, inv in views:
            logits = model(fwd(xs))
            p = torch.sigmoid(logits)
            p = inv(p)
            if p.shape[-2:] != (h, w):
                p = _resize(p, (h, w))
            probs += p
            n += 1

    probs /= n
    return probs if batched else probs[0]


@torch.no_grad()
def tta_predict_mask(
    model: torch.nn.Module,
    img: torch.Tensor,
    *,
    threshold: float = 0.5,
    scales: Sequence[float] = (0.75, 1.0, 1.25),
    flips: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    """TTA probabilities thresholded to a boolean mask (same shape as `tta_predict`)."""
    probs = tta_predict(model, img, scales=scales, flips=flips, device=device)
    return probs > threshold
