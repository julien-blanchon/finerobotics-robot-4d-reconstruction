"""Multi-head self-attention (encoder) and cross-attention (decoder).

Cross-attention is split in two: `keys_values` depends only on the context and is computed once
per encoding; `forward` projects the queries and attends. That split is the main inference
saving over the reference, which recomputed context keys/values for every query chunk.

Adapted from https://github.com/facebookresearch/vggt (Apache-2.0), via DA3 and ARROW.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .rope import apply_rope

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar
    from torch import Tensor


class SelfAttention[D: IntVar, NHead: IntVar](nn.Module):
    """Fused-QKV attention with optional per-head QK-LayerNorm and 2D RoPE."""

    def __init__(self, dim: Int[D], num_heads: Int[NHead], *, qk_norm: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = nn.LayerNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward[B: IntVar, N: IntVar](
        self, x: Tensor[[B, N, D]], rope: Tensor[[2, 2, N, int]] | None
    ) -> Tensor[[B, N, D]]:
        q, k, v = rearrange(
            self.qkv(x), "b n (three h d) -> three b h n d", three=3, h=self.num_heads
        )
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            # Table width is D // NHead // 2; left `int`: pyrefly normalises it to
            # D // (2 * NHead) and cannot match it with apply_rope's HD // 2.
            # Tables are fp32; normed queries are fp32 under autocast too (LayerNorm runs in fp32)
            q, k = apply_rope(q, rope.to(q.dtype)), apply_rope(k, rope.to(k.dtype))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(rearrange(out, "b h n d -> b n (h d)"))


class CrossAttention[D: IntVar, NHead: IntVar](nn.Module):
    """Queries attend to a fixed context; per-head QK-LayerNorm, no positional encoding."""

    def __init__(self, dim: Int[D], num_heads: Int[NHead]) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.q_layer = nn.Linear(dim, dim)
        self.kv_layer = nn.Linear(dim, 2 * dim)
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def keys_values[B: IntVar, N: IntVar](
        self, context: Tensor[[B, N, D]]
    ) -> tuple[Tensor[[B, NHead, N, D // NHead]], Tensor[[B, NHead, N, D // NHead]]]:
        """Normed keys and values of the (already normed) context: `[B, heads, N, head_dim]`."""

        k, v = rearrange(
            self.kv_layer(context), "b n (two h d) -> two b h n d", two=2, h=self.num_heads
        )
        return self.k_norm(k), v

    def forward[B: IntVar, Q: IntVar, N: IntVar](
        self,
        query: Tensor[[B, Q, D]],
        keys: Tensor[[B, NHead, N, D // NHead]],
        values: Tensor[[B, NHead, N, D // NHead]],
    ) -> Tensor[[B, Q, D]]:
        q = self.q_norm(rearrange(self.q_layer(query), "b q (h d) -> b h q d", h=self.num_heads))
        out = F.scaled_dot_product_attention(q, keys, values)
        # Heads merge back to NHead * (D // NHead) = D, which pyrefly cannot simplify
        merged = cast("Tensor[[B, Q, D]]", rearrange(out, "b h q d -> b q (h d)"))
        return self.proj(merged)
