"""DINOv3 ViT-L/16 descriptor: normed patch tokens of blocks 11 and 17.

The weights ship inside the RoMa v2 checkpoint in bf16, and run under bf16 autocast as in the
reference. Blocks 18–23 never reach the output, so they are not built (a quarter of the cost).

Ref: Siméoni et al., DINOv3, arXiv:2508.10104.
Adapted from https://github.com/facebookresearch/dinov3 (DINOv3 License), via
https://github.com/Parskatt/RoMaV2 (MIT).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from einops import rearrange
from torch import nn

from .layers import Block, RopeEmbedding

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

PATCH_SIZE = 16
WIDTH = 1024
NUM_HEADS = 16
STORAGE_TOKENS = 4
NORM_EPS = 1e-5
OUTPUT_BLOCKS = (11, 17)


class PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, WIDTH, PATCH_SIZE, PATCH_SIZE)

    def forward[S: IntVar, H: IntVar, W: IntVar](
        self, images: Tensor[[S, 3, H, W]]
    ) -> Tensor[[S, int, 1024]]:
        return rearrange(self.proj(images), "s c h w -> s (h w) c")


class DINOv3(nn.Module):
    """ImageNet-normalised images `[S, 3, H, W]` (H, W multiples of 16) → `[S, h, w, 2048]`.

    The two output blocks are concatenated along features, block 11 first.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, WIDTH))
        self.storage_tokens = nn.Parameter(torch.zeros(1, STORAGE_TOKENS, WIDTH))
        self.patch_embed = PatchEmbed()
        self.rope_embed = RopeEmbedding(WIDTH // NUM_HEADS, torch.float32)
        self.blocks = nn.ModuleList(
            Block(WIDTH, NUM_HEADS, eps=NORM_EPS, layer_scale=True)
            for _ in range(OUTPUT_BLOCKS[-1] + 1)
        )
        self.norm = nn.LayerNorm(WIDTH, eps=NORM_EPS)

    def forward[S: IntVar, H: IntVar, W: IntVar](
        self, images: Tensor[[S, 3, H, W]]
    ) -> Tensor[[S, H // 16, W // 16, 2048]]:
        s, _, h, w = images.shape
        patches = self.patch_embed(images)
        x = torch.cat(
            (self.cls_token.expand(s, -1, -1), self.storage_tokens.expand(s, -1, -1), patches),
            dim=1,
        )
        rope = self.rope_embed(h // PATCH_SIZE, w // PATCH_SIZE)
        outputs = []
        for index, block in enumerate(self.blocks):
            x = block(x, rope)
            if index in OUTPUT_BLOCKS:
                outputs.append(self.norm(x)[:, 1 + STORAGE_TOKENS :])
        return rearrange(
            torch.cat(outputs, dim=-1), "s (h w) d -> s h w d", h=h // PATCH_SIZE, w=w // PATCH_SIZE
        )
