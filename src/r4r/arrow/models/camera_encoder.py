"""DA3's camera encoder, feeding a focal-length draft into ARROW's block-13 token slot.

DA3 turns per-frame (translation, quaternion, field of view) into one token per frame and writes
it where ARROW keeps its learned camera tokens. ARROW dropped this encoder; the gen2 adapter
(`experiments/arrow_pose`) trains it back, with LoRA on the encoder, so that a draft is corrected
rather than copied. Only focal drafts are used here (identity poses), as in gen2's training.

Adapted from https://github.com/ByteDance-Seed/Depth-Anything-3 (Apache-2.0), `model/cam_enc.py`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

from .layers.attention import SelfAttention
from .layers.block import LayerScale, Mlp

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    from .encoder import SlotHook

WIDTH = 1536
HEADS = 16
DEPTH = 4


class CameraBlock(nn.Module):
    """Pre-norm ViT block: plain attention, GELU MLP, LayerScale."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, qk_norm=False)
        self.ls1 = LayerScale(dim, 1.0)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, 4 * dim)
        self.ls2 = LayerScale(dim, 1.0)

    def forward[B: IntVar, S: IntVar](self, x: Tensor[[B, S, int]]) -> Tensor[[B, S, int]]:
        x = x + self.ls1(self.attn(self.norm1(x), None))
        return x + self.ls2(self.mlp(self.norm2(x)))


class PoseBranch(nn.Module):
    """GELU MLP from the 9-number camera encoding to the token width."""

    def __init__(self, dim: int, hidden: int, out: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out)

    def forward[B: IntVar, S: IntVar](self, x: Tensor[[B, S, int]]) -> Tensor[[B, S, int]]:
        return self.fc2(self.act(self.fc1(x)))


class CameraEncoder(nn.Module):
    """`[t (3), quaternion xyzw (4), fov_h, fov_w]` per frame → one token per frame (fp32)."""

    def __init__(self) -> None:
        super().__init__()
        self.pose_branch = PoseBranch(9, WIDTH // 2, WIDTH)
        self.token_norm = nn.LayerNorm(WIDTH)
        self.trunk = nn.Sequential(*(CameraBlock(WIDTH, HEADS) for _ in range(DEPTH)))
        self.trunk_norm = nn.LayerNorm(WIDTH)

    def forward[B: IntVar, S: IntVar](self, encoding: Tensor[[B, S, 9]]) -> Tensor[[B, S, 1536]]:
        return self.trunk_norm(self.trunk(self.token_norm(self.pose_branch(encoding))))

    def focal_slot(self, focal: float, size_wh: tuple[int, int]) -> SlotHook:
        """A slot hook giving every frame the same focal draft (native pixels) and no pose."""

        width, height = size_wh
        fov_h, fov_w = 2 * math.atan(height / 2 / focal), 2 * math.atan(width / 2 / focal)

        def hook(tokens: Tensor, _reference: Tensor | None) -> Tensor:
            batch, frames, _ = tokens.shape
            draft = torch.tensor([0, 0, 0, 0, 0, 0, 1, fov_h, fov_w], device=tokens.device)
            with torch.autocast(tokens.device.type, enabled=False):
                camera = self(draft.expand(batch, frames, 9))
            return camera.to(tokens.dtype)

        return hook
