"""Sparse query decoder: cross-attention from independent queries to the encoded context.

Queries never attend to each other, so they can be chunked freely (up to bf16 kernel noise),
and everything that depends only on the context (per-block normed keys and values) is
computed once.

Ref: ARROW arXiv:2610.01314, §3.1 "Sparse decoder"; D4RT.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from .layers.block import CrossBlock

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar
    from torch import Tensor


class Decoder[D: IntVar, Depth: IntVar, NHead: IntVar](nn.Module):
    def __init__(self, width: Int[D], depth: Int[Depth], num_heads: Int[NHead]) -> None:
        super().__init__()
        self.width = width
        self.blocks = nn.ModuleList(CrossBlock(width, num_heads) for _ in range(depth))

    def keys_values[B: IntVar, N: IntVar](
        self, context: Tensor[[B, N, D]], dtype: torch.dtype
    ) -> tuple[Tensor[[Depth, B, NHead, N, D // NHead]], Tensor[[Depth, B, NHead, N, D // NHead]]]:
        """Per-block keys and values in `dtype`: `[blocks, B, heads, N, head_dim]`.

        Written block by block into one preallocated buffer: for long contexts this cache is
        the largest tensor of the whole pipeline (8 blocks × 2 × N × width).
        """

        keys = values = None
        for i, block in enumerate(self.blocks):
            k, v = block.keys_values(context)
            if keys is None or values is None:
                keys = torch.empty(len(self.blocks), *k.shape, dtype=dtype, device=k.device)
                values = torch.empty(len(self.blocks), *v.shape, dtype=dtype, device=v.device)
            keys[i], values[i] = k, v
        assert keys is not None and values is not None, "decoder has no blocks"
        return keys, values

    def forward[B: IntVar, Q: IntVar, N: IntVar](
        self,
        query: Tensor[[B, Q, D]],
        keys: Tensor[[Depth, B, NHead, N, D // NHead]],
        values: Tensor[[Depth, B, NHead, N, D // NHead]],
    ) -> Tensor[[B, Q, D]]:
        for block, k, v in zip(self.blocks, keys, values, strict=True):
            query = block(query, k, v)
        return query
