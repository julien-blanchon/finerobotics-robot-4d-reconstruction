"""SwiGLU feed-forward network of the DA3/DINOv2 encoder.

Adapted from https://github.com/ByteDance-Seed/Depth-Anything-3 (Apache-2.0), via ARROW.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from shape_extensions import IntTuple
    from torch import Tensor


def swiglu_hidden(dim: int, mlp_ratio: int) -> int:
    """DINOv2's "fused" width: 2/3 of the MLP width, rounded up to a multiple of 8."""

    return (int(dim * mlp_ratio * 2 / 3) + 7) // 8 * 8


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden)
        self.w3 = nn.Linear(hidden, dim)

    def forward[S: IntTuple](self, x: Tensor[[*S, int]]) -> Tensor[[*S, int]]:
        gate, value = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(gate) * value)
