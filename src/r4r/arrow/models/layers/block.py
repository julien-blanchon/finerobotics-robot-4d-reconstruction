"""Pre-norm transformer blocks: encoder self-attention block and decoder cross-attention block.

Adapted from https://github.com/facebookresearch/vggt (Apache-2.0), via DA3 and ARROW.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from .attention import CrossAttention, SelfAttention
from .swiglu_ffn import SwiGLUFFN, swiglu_hidden

if TYPE_CHECKING:
    from shape_extensions import IntTuple, IntVar
    from torch import Tensor

ENCODER_NORM_EPS = 1e-6
MLP_RATIO = 4


class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init))

    def forward[S: IntTuple](self, x: Tensor[[*S, int]]) -> Tensor[[*S, int]]:
        return x * self.gamma


class Mlp(nn.Module):
    """GELU MLP (exact erf GELU, as `timm.layers.Mlp`)."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward[S: IntTuple](self, x: Tensor[[*S, int]]) -> Tensor[[*S, int]]:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """DA3 encoder block: SwiGLU MLP, LayerScale 1.0, QK-norm + RoPE in the later blocks."""

    def __init__(self, dim: int, num_heads: int, *, qk_norm: bool) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=ENCODER_NORM_EPS)
        self.attn = SelfAttention(dim, num_heads, qk_norm=qk_norm)
        self.ls1 = LayerScale(dim, 1.0)
        self.norm2 = nn.LayerNorm(dim, eps=ENCODER_NORM_EPS)
        self.mlp = SwiGLUFFN(dim, swiglu_hidden(dim, MLP_RATIO))
        self.ls2 = LayerScale(dim, 1.0)

    def forward[B: IntVar, N: IntVar, D: IntVar](
        self, x: Tensor[[B, N, D]], rope: Tensor[[2, 2, N, int]] | None
    ) -> Tensor[[B, N, D]]:
        x = x + self.ls1(self.attn(self.norm1(x), rope))
        return x + self.ls2(self.mlp(self.norm2(x)))


class CrossBlock(nn.Module):
    """ARROW decoder block. `norm1` is shared by queries and context (pretrained that way)."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, num_heads)
        self.ls1 = LayerScale(dim, 0.01)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, MLP_RATIO * dim)
        self.ls2 = LayerScale(dim, 0.01)

    def keys_values[B: IntVar, N: IntVar, D: IntVar](
        self, context: Tensor[[B, N, D]]
    ) -> tuple[Tensor[[B, int, N, int]], Tensor[[B, int, N, int]]]:
        return self.attn.keys_values(self.norm1(context))

    def forward[B: IntVar, Q: IntVar, N: IntVar, D: IntVar](
        self,
        query: Tensor[[B, Q, D]],
        keys: Tensor[[B, int, N, int]],
        values: Tensor[[B, int, N, int]],
    ) -> Tensor[[B, Q, D]]:
        query = query + self.ls1(self.attn(self.norm1(query), keys, values))
        return query + self.ls2(self.mlp(self.norm2(query)))
