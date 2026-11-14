"""DA3-Giant multi-view encoder: patch tokens of every frame plus one ID token per frame.

A DINOv2 ViT-g/14 (40 blocks, width 1536). Blocks 0–12 attend within each frame. From block 13 on,
odd blocks attend globally over all frames and even blocks within each frame, both with
QK-norm and 2D RoPE. Before block 12 the frames are reordered so that a "reference" frame comes
first; before block 13 its special token becomes camera token 0 and every other frame's becomes
camera token 1. The output concatenates the last frame-wise and last global states (3072 wide),
normalising only the global half. The ID token is the special-token slot, taken before that norm.

All of this is pinned by the pretrained weights: change nothing without retraining.

Ref: Lin et al., Depth Anything 3, arXiv:2511.10647; ARROW arXiv:2610.01314, §3.1.
Adapted from https://github.com/ByteDance-Seed/Depth-Anything-3 (Apache-2.0), via ARROW.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .layers.block import Block
from .layers.rope import rope_tables

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

PATCH_SIZE = 14
WIDTH = 1536
NUM_HEADS = 24
DEPTH = 40
FIRST_GLOBAL_BLOCK = 13
PRETRAIN_GRID = 37  # 518 px / 14
POS_EMBED_OFFSET = 0.1  # DINOv2's offset that keeps the interpolated grid from rounding down
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, WIDTH, PATCH_SIZE, PATCH_SIZE)

    def forward[N: IntVar, H: IntVar, W: IntVar](
        self, images: Tensor[[N, 3, H, W]]
    ) -> Tensor[[N, int, 1536]]:
        return rearrange(self.proj(images), "n c h w -> n (h w) c")


# Rewrites the block-13 special tokens: (pretrained tokens [B, S, 1536] in input frame order,
# reference frame [B] or None when S < 3) → tokens [B, S, 1536] in input frame order. This is the
# slot where DA3 writes its camera-encoder tokens (pose conditioning); ARROW leaves it unused.
type SlotHook = Callable[[Tensor, Tensor | None], Tensor]


class Encoder(nn.Module):
    """Encode RGB `[B, S, 3, H, W]` in [0, 1] (H, W multiples of 14).

    Returns context tokens `[B, S, P, 3072]` and ID tokens `[B, S, 3072]`, in input frame order.
    `slot` optionally rewrites the block-13 special tokens (see `SlotHook`); None keeps ARROW.
    """

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, WIDTH))
        self.camera_token = nn.Parameter(torch.zeros(1, 2, WIDTH))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + PRETRAIN_GRID**2, WIDTH))
        self.blocks = nn.ModuleList(
            Block(WIDTH, NUM_HEADS, qk_norm=i >= FIRST_GLOBAL_BLOCK) for i in range(DEPTH)
        )
        self.norm = nn.LayerNorm(WIDTH)
        self.register_buffer("image_mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1))

    def forward[B: IntVar, S: IntVar, H: IntVar, W: IntVar](
        self, images: Tensor[[B, S, 3, H, W]], slot: SlotHook | None = None
    ) -> tuple[Tensor[[B, S, int, 3072]], Tensor[[B, S, 3072]]]:
        b, f, _, h, w = images.shape
        x = self.tokens((images - self.image_mean) / self.image_std)
        local_rope, global_rope = self.rope(h // PATCH_SIZE, w // PATCH_SIZE, f, images.device)

        local_x, reference = x, None
        for i, block in enumerate(self.blocks):
            # Put DA3's reference frame first, then give it camera token 0 (others token 1)
            if i == FIRST_GLOBAL_BLOCK - 1 and f >= 3:
                reference = select_reference_frame(x[:, :, 0])
                x, local_x = (
                    to_reference_first(x, reference),
                    to_reference_first(local_x, reference),
                )
            if i == FIRST_GLOBAL_BLOCK:
                special = self.camera_tokens(b, f)
                if slot is not None:
                    special = self.rewrite_slot(special, reference, slot)
                x = torch.cat((special.to(x.dtype), x[:, :, 1:]), dim=2)

            if i >= FIRST_GLOBAL_BLOCK and i % 2 == 1:
                tokens = block(rearrange(x, "b f n c -> b (f n) c"), global_rope)
                x = rearrange(tokens, "b (f n) c -> b f n c", f=f)
            else:
                rope = local_rope if i >= FIRST_GLOBAL_BLOCK else None
                tokens = block(rearrange(x, "b f n c -> (b f) n c"), rope)
                x = local_x = rearrange(tokens, "(b f) n c -> b f n c", b=b)

        x = torch.cat((local_x, x), dim=-1)
        if reference is not None:
            x = from_reference_first(x, reference)
        id_tokens = x[:, :, 0]
        context = torch.cat((x[..., 1:, :WIDTH], self.norm(x[..., 1:, WIDTH:])), dim=-1)
        return context, id_tokens

    def tokens[B: IntVar, S: IntVar, H: IntVar, W: IntVar](
        self, images: Tensor[[B, S, 3, H, W]]
    ) -> Tensor[[B, S, int, 1536]]:
        """Special token + patch tokens, with the learned position embedding."""

        b, f, _, h, w = images.shape
        patches = self.patch_embed(rearrange(images, "b f c h w -> (b f) c h w"))
        x = torch.cat((self.cls_token.expand(b * f, -1, -1), patches), dim=1)
        x = x + self.position_embedding(h // PATCH_SIZE, w // PATCH_SIZE).to(x.dtype)
        return rearrange(x, "(b f) n c -> b f n c", b=b)

    def position_embedding(self, rows: int, columns: int) -> Tensor[[1, int, 1536]]:
        """Bicubic resize of the 37×37 pretraining grid (scale factors, not sizes, as DINOv2)."""

        if rows == columns == PRETRAIN_GRID:
            return self.pos_embed
        embed = self.pos_embed.float()
        grid = rearrange(embed[:, 1:], "1 (h w) c -> 1 c h w", h=PRETRAIN_GRID)
        scale = (
            (rows + POS_EMBED_OFFSET) / PRETRAIN_GRID,
            (columns + POS_EMBED_OFFSET) / PRETRAIN_GRID,
        )
        grid = F.interpolate(grid, scale_factor=scale, mode="bicubic", antialias=False)
        assert grid.shape[-2:] == (rows, columns), "position grid interpolation rounding"
        return torch.cat((embed[:, :1], rearrange(grid, "1 c h w -> 1 (h w) c")), dim=1)

    def camera_tokens(self, b: int, f: int) -> Tensor[[int, int, 1, 1536]]:
        reference, others = self.camera_token[:, :1], self.camera_token[:, 1:]
        tokens = torch.cat((reference, others.expand(1, f - 1, -1)), dim=1)
        return rearrange(tokens.expand(b, -1, -1), "b f c -> b f 1 c")

    @staticmethod
    def rewrite_slot(
        special: Tensor[[int, int, 1, 1536]], reference: Tensor | None, slot: SlotHook
    ) -> Tensor[[int, int, 1, 1536]]:
        """Apply `slot` in input frame order (`special` is reference-first when S ≥ 3)."""

        tokens = special[:, :, 0]
        if reference is not None:
            tokens = from_reference_first(tokens[:, :, None], reference)[:, :, 0]
        tokens = slot(tokens, reference)
        if reference is not None:
            tokens = to_reference_first(tokens[:, :, None], reference)[:, :, 0]
        return tokens[:, :, None]

    @staticmethod
    def rope(
        rows: int, columns: int, frames: int, device: torch.device
    ) -> tuple[Tensor[[2, 2, int, int]], Tensor[[2, 2, int, int]]]:
        """RoPE tables for frame-wise blocks (patch (r, c) at (r+1, c+1)) and global blocks.

        Globally every patch sits at (1, 1) and the special token at (0, 0): RoPE then only tells
        special from patch tokens, never where a patch is.
        """

        r, c = torch.meshgrid(
            torch.arange(rows, device=device), torch.arange(columns, device=device), indexing="ij"
        )
        special = torch.zeros(1, 2, dtype=torch.long, device=device)
        spatial = torch.cat((special, torch.stack((r, c), -1).flatten(0, 1) + 1))
        flat = torch.cat((special, torch.ones(rows * columns, 2, dtype=torch.long, device=device)))
        head_dim = WIDTH // NUM_HEADS
        dtype = (
            torch.get_autocast_dtype(device.type)
            if torch.is_autocast_enabled(device.type)
            else torch.float32
        )
        return (
            rope_tables(spatial, head_dim, dtype),
            rope_tables(flat.repeat(frames, 1), head_dim, dtype),
        )


def select_reference_frame[B: IntVar, S: IntVar](cls: Tensor[[B, S, 1536]]) -> Tensor[[B]]:
    """DA3's "most typical frame": similarity, norm and variance all closest to their mid-range."""

    unit = F.normalize(cls, dim=-1, eps=0.0)
    frames = cls.shape[1]
    similarity = (unit @ unit.mT - torch.eye(frames, device=cls.device)).sum(-1) / (frames - 1)

    def centred(metric: Tensor[[B, S]]) -> Tensor[[B, S]]:
        low, high = metric.amin(1, keepdim=True), metric.amax(1, keepdim=True)
        return ((metric - low) / (high - low + 1e-8) - 0.5).abs()

    balance = centred(similarity) + centred(cls.norm(dim=-1)) + centred(unit.var(dim=-1))
    return balance.argmin(dim=1)


def to_reference_first[B: IntVar, S: IntVar, N: IntVar, C: IntVar](
    x: Tensor[[B, S, N, C]], reference: Tensor[[B]]
) -> Tensor[[B, S, N, C]]:
    """Frame order [ref, 0, 1, …, ref−1, ref+1, …] (others keep their relative order)."""

    position = torch.arange(x.shape[1], device=x.device)[None]
    source = torch.where((position > 0) & (position <= reference[:, None]), position - 1, position)
    source[:, 0] = reference
    return x[torch.arange(x.shape[0], device=x.device)[:, None], source]


def from_reference_first[B: IntVar, S: IntVar, N: IntVar, C: IntVar](
    x: Tensor[[B, S, N, C]], reference: Tensor[[B]]
) -> Tensor[[B, S, N, C]]:
    """Inverse of `to_reference_first`."""

    position = torch.arange(x.shape[1], device=x.device)[None]
    source = torch.where(position < reference[:, None], position + 1, position)
    source = torch.where(position == reference[:, None], 0, source)
    return x[torch.arange(x.shape[0], device=x.device)[:, None], source]
