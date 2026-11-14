"""Coarse matcher: multi-view transformer, soft nearest-neighbour matching, DPT warp head.

Both directions run in one batch: a ViT-B jointly refines the two frames' DINOv3 features
(even blocks attend over both frames, odd blocks within each frame with RoPE), every patch of
the source frame softly attends to the target frame's patches, and the attention-weighted
Fourier embedding of the target positions (the "match embedding") tells the DPT head where to
point. The joint transformer is symmetric in the two frames (no view embedding), so one pass
serves both directions.

Ref: Edstedt et al., RoMa v2, arXiv:2511.15706, §3.2–3.3.
Adapted from https://github.com/Parskatt/RoMaV2 (MIT).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .dinov3 import WIDTH as DESCRIPTOR_WIDTH
from .dpt import DPTHead
from .layers import Block, RopeEmbedding

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar
    from torch import Tensor

    from r4r.types import UV

WIDTH = 768
DEPTH = 12
NUM_HEADS = 12
NORM_EPS = 1e-6
EMBED_WIDTH = 1024  # matcher output and Fourier position embedding width


def normalized_grid[H: IntVar, W: IntVar](
    h: Int[H], w: Int[W], device: torch.device
) -> Tensor[[H, W, *UV]]:
    """Pixel centres in grid_sample's `align_corners=False` coordinates, `(x, y)` last:
    `x = 2 (j + ½) / w − 1` for column `j` (and `y` for row `i`), in (−1, 1)."""

    rows = (torch.arange(h, device=device) + 0.5) * (2 / h) - 1
    columns = (torch.arange(w, device=device) + 0.5) * (2 / w) - 1
    y, x = torch.meshgrid(rows, columns, indexing="ij")
    return torch.stack((x, y), dim=-1)


class MultiViewTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projector = nn.Linear(2 * DESCRIPTOR_WIDTH, WIDTH)
        self.blocks = nn.ModuleList(
            Block(WIDTH, NUM_HEADS, eps=NORM_EPS, layer_scale=False) for _ in range(DEPTH)
        )
        self.norm = nn.LayerNorm(WIDTH, eps=NORM_EPS)
        self.output_projector = nn.Linear(WIDTH, EMBED_WIDTH)
        self.rope_embed = RopeEmbedding(WIDTH // NUM_HEADS, torch.bfloat16)

    def forward[B: IntVar, H: IntVar, W: IntVar](
        self, a: Tensor[[B, H, W, 2048]], b: Tensor[[B, H, W, 2048]]
    ) -> Tensor[[B, 2, H, W, 1024]]:
        _, h, w, _ = a.shape
        rope = self.rope_embed(h, w)
        x = self.projector(rearrange(torch.stack((a, b), dim=1), "b v h w d -> b (v h w) d"))
        for index, block in enumerate(self.blocks):
            if index % 2:
                frames = block(rearrange(x, "b (v n) d -> (b v) n d", v=2), rope)
                x = rearrange(frames, "(b v) n d -> b (v n) d", v=2)
            else:
                x = block(x, None)
        out = self.output_projector(self.norm(x))
        return rearrange(out, "b (v h w) d -> b v h w d", v=2, h=h)


class Matcher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("omega", torch.zeros(EMBED_WIDTH // 2, 2))
        self.register_buffer("scale", torch.ones((), dtype=torch.int64))
        self.register_buffer("temp", torch.ones(()))
        self.mv_vit = MultiViewTransformer()
        self.head = DPTHead(EMBED_WIDTH)

    def forward[B: IntVar, H: IntVar, W: IntVar](
        self, a: Tensor[[B, H, W, 2048]], b: Tensor[[B, H, W, 2048]]
    ) -> Tensor[[2 * B, H * 4, W * 4, 3]]:
        """DINOv3 features of both frames → `[warp | overlap logit]`, a→b then b→a on the batch.

        Soft nearest neighbour: `m(x) = Σ_y softmax_y(⟨f̂_a(x), f̂_b(y)⟩ / τ) · φ(y)`, `f̂` the
        unit-normalised joint features, `φ(y) = (sin Ωy, cos Ωy)` the Fourier embedding of the
        target grid; the DPT head reads the warp off `m`.
        """

        _, h, w, _ = a.shape
        with torch.autocast(a.device.type, dtype=torch.bfloat16):
            joint = self.mv_vit(a, b).float()
        source = torch.cat((a, b))
        source_joint = rearrange(joint, "b v h w d -> (v b) h w d")
        target_joint = rearrange(joint.flip(1), "b v h w d -> (v b) h w d")

        # Fourier embedding of the target grid
        positions = F.linear(normalized_grid(h, w, a.device), self.scale * self.omega)
        embedding = torch.cat((positions.sin(), positions.cos()), dim=-1)

        # Soft nearest neighbour by cosine similarity at temperature `temp`
        query = F.normalize(source_joint, dim=-1) / self.temp
        query = rearrange(query, "s h w d -> s 1 (h w) d")
        key = rearrange(F.normalize(target_joint, dim=-1), "s h w d -> s 1 (h w) d")
        value = rearrange(embedding, "h w d -> 1 1 (h w) d").expand(len(query), -1, -1, -1)
        matched = F.scaled_dot_product_attention(query, key, value, scale=1.0)
        matched = rearrange(matched, "s 1 (h w) d -> s h w d", h=h)

        shallow, deep = source[..., :DESCRIPTOR_WIDTH], source[..., DESCRIPTOR_WIDTH:]
        return self.head(shallow, deep + source_joint + matched)
