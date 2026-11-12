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
    from shape_extensions import IntVar
    from torch import Tensor


class Decoder(nn.Module):
    def __init__(self, width: int, depth: int, num_heads: int) -> None:
        super().__init__()
        self.width = width
        self.blocks = nn.ModuleList(CrossBlock(width, num_heads) for _ in range(depth))

    def keys_values[B: IntVar, N: IntVar](
        self, context: Tensor[[B, N, int]]
    ) -> tuple[Tensor[[int, B, int, N, int]], Tensor[[int, B, int, N, int]]]:
        """Per-block keys and values, stacked: `[blocks, B, heads, N, head_dim]`."""

        pairs = [block.keys_values(context) for block in self.blocks]
        return torch.stack([k for k, _ in pairs]), torch.stack([v for _, v in pairs])

    def forward[B: IntVar, Q: IntVar, N: IntVar](
        self,
        query: Tensor[[B, Q, int]],
        keys: Tensor[[int, B, int, N, int]],
        values: Tensor[[int, B, int, N, int]],
    ) -> Tensor[[B, Q, int]]:
        for block, k, v in zip(self.blocks, keys, values, strict=True):
            query = block(query, k, v)
        return query
