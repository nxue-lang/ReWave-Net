from __future__ import annotations

import torch
from torch.nn import functional as F


def match_spatial_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Center-crop or center-pad source tensor so its spatial size matches target."""
    source_h, source_w = source.shape[-2:]
    target_h, target_w = target.shape[-2:]

    if source_h == target_h and source_w == target_w:
        return source

    if source_h > target_h:
        start_h = (source_h - target_h) // 2
        source = source[..., start_h : start_h + target_h, :]
    elif source_h < target_h:
        pad_top = (target_h - source_h) // 2
        pad_bottom = target_h - source_h - pad_top
        source = F.pad(source, (0, 0, pad_top, pad_bottom))

    source_w = source.shape[-1]
    if source_w > target_w:
        start_w = (source_w - target_w) // 2
        source = source[..., start_w : start_w + target_w]
    elif source_w < target_w:
        pad_left = (target_w - source_w) // 2
        pad_right = target_w - source_w - pad_left
        source = F.pad(source, (pad_left, pad_right, 0, 0))

    return source
