"""ViT layers shared by the DINOv3 descriptor and the RoMa v2 multi-view matcher.

Pre-norm blocks with a GELU MLP, optional LayerScale, and DINOv3's axial RoPE: every head's
first half rotates with (row angles ‖ column angles), its second half with the same angles.

Adapted from https://github.com/facebookresearch/dinov3 (DINOv3 License), via
https://github.com/Parskatt/RoMaV2 (MIT).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

if TYPE_CHECKING:
    from shape_extensions import IntTuple, IntVar
    from torch import Tensor

MLP_RATIO = 4


class RopeEmbedding(nn.Module):
    """sin/cos tables for an `h × w` patch grid, coordinates normalised to [−1, 1] per axis.

    The tables are computed in `dtype` (bf16 in the matcher, fp32 in DINOv3) from the stored
    bf16 periods: the pretrained weights expect exactly those rounded angles.
    """

    def __init__(self, head_dim: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype
        self.register_buffer("periods", torch.empty(head_dim // 4, dtype=torch.bfloat16))

    def forward(self, h: int, w: int) -> tuple[Tensor[[int, int]], Tensor[[int, int]]]:
        periods: Tensor[[int]] = self.periods
        options = {"device": periods.device, "dtype": self.dtype}
        rows = torch.arange(0.5, h, **options) / h
        columns = torch.arange(0.5, w, **options) / w
        coords = rearrange(
            torch.stack(torch.meshgrid(rows, columns, indexing="ij"), dim=-1), "h w a -> (h w) a"
        )
        coords = 2.0 * coords - 1.0
        angles = 2 * math.pi * coords[:, :, None] / periods[None, None, :]
        angles = repeat(angles, "n a f -> n (two a f)", two=2)
        return angles.sin(), angles.cos()


def apply_rope[B: IntVar, NHead: IntVar, N: IntVar, HD: IntVar](
    x: Tensor[[B, NHead, N, HD]], sin: Tensor[[int, HD]], cos: Tensor[[int, HD]]
) -> Tensor[[B, NHead, N, HD]]:
    """Rotate the patch tokens (the last `len(sin)`), in the tables' dtype; prefix tokens pass."""

    dtype = x.dtype
    x = x.to(sin.dtype)
    prefix = x.shape[-2] - sin.shape[-2]
    patches = x[..., prefix:, :]
    first, second = patches.chunk(2, dim=-1)
    rotated = patches * cos + torch.cat((-second, first), dim=-1) * sin
    return torch.cat((x[..., :prefix, :], rotated), dim=-2).to(dtype)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward[B: IntVar, N: IntVar, D: IntVar](
        self, x: Tensor[[B, N, D]], rope: tuple[Tensor[[int, int]], Tensor[[int, int]]] | None
    ) -> Tensor[[B, N, D]]:
        q, k, v = rearrange(
            self.qkv(x), "b n (three h d) -> three b h n d", three=3, h=self.num_heads
        )
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(rearrange(out, "b h n d -> b n (h d)"))


class LayerScale(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward[S: IntTuple](self, x: Tensor[[*S, int]]) -> Tensor[[*S, int]]:
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward[S: IntTuple](self, x: Tensor[[*S, int]]) -> Tensor[[*S, int]]:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm block; LayerScale in DINOv3 only (the matcher was trained without).

    Compiled: fusing norms, RoPE and residuals halves the time. Fused ops keep fp32
    intermediates where eager rounds to bf16, so outputs differ from eager at bf16 noise level.
    """

    def __init__(self, dim: int, num_heads: int, *, eps: float, layer_scale: bool) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim) if layer_scale else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = Mlp(dim, MLP_RATIO * dim)
        self.ls2 = LayerScale(dim) if layer_scale else nn.Identity()

    @torch.compile(fullgraph=True, dynamic=True)  # type: ignore[pyrefly:missing-attribute]  # not in the shape stubs
    def forward[B: IntVar, N: IntVar, D: IntVar](
        self, x: Tensor[[B, N, D]], rope: tuple[Tensor[[int, int]], Tensor[[int, int]]] | None
    ) -> Tensor[[B, N, D]]:
        x = x + self.ls1(self.attn(self.norm1(x), rope))
        return x + self.ls2(self.mlp(self.norm2(x)))
