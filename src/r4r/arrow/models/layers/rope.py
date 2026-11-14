"""2D rotary position embedding on (row, column) patch positions.

The first half of each head's features rotates with the row, the second half with the column.
Frequencies are `base^(−2k/d)` with base 100 and `d` = half the head width, as in DA3.

Adapted from https://github.com/facebookresearch/vggt (Apache-2.0), via DA3 and ARROW.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from einops import repeat

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar
    from torch import Tensor

ROPE_BASE = 100.0


def rope_tables[N: IntVar, HD: IntVar](
    positions: Tensor[[N, 2]], head_dim: Int[HD], angle_dtype: torch.dtype
) -> Tensor[[2, 2, N, HD // 2]]:
    """`[cos|sin, row|column, N, head_dim // 2]` in fp32 for integer positions:
    angle `= p · base^(−2k/(HD/2))`, k < HD/4, each frequency used twice (radians).

    Angles are rounded to `angle_dtype` first: the reference forms them with an einsum that runs
    in bf16 under CUDA autocast, and the pretrained weights expect exactly those angles.
    """

    half = head_dim // 2
    exponents = torch.arange(0, half, 2, device=positions.device).float() / half
    inverse_frequency = (1.0 / ROPE_BASE**exponents).to(angle_dtype)
    angles = positions.to(angle_dtype)[..., None] * inverse_frequency  # (N, 2, half/2)
    angles = repeat(angles.float(), "n a f -> a n (two f)", two=2)
    # 2 · (HD // 4) = HD // 2: head widths are multiples of 4
    return cast("Tensor[[2, 2, N, HD // 2]]", torch.stack((angles.cos(), angles.sin())))


def apply_rope[B: IntVar, NHead: IntVar, N: IntVar, HD: IntVar](
    x: Tensor[[B, NHead, N, HD]], tables: Tensor[[2, 2, N, HD // 2]]
) -> Tensor[[B, NHead, N, HD]]:
    """Rotate the row half and the column half of every head by their own angles:
    `x' = x cos φ + rot(x) sin φ`, `rot(a, b) = (−b, a)` on the two halves of each part.
    """

    cos, sin = tables
    rows, columns = x.chunk(2, dim=-1)
    return torch.cat((_rotate(rows, cos[0], sin[0]), _rotate(columns, cos[1], sin[1])), dim=-1)


def _rotate[B: IntVar, NHead: IntVar, N: IntVar, K: IntVar](
    x: Tensor[[B, NHead, N, K]], cos: Tensor[[N, K]], sin: Tensor[[N, K]]
) -> Tensor[[B, NHead, N, K]]:
    first, second = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-second, first), dim=-1) * sin
