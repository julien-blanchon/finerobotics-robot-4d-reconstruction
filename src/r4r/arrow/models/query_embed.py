"""Query encoder: appearance at the source pixel, its position, and which frames are involved.

`z = φ_rgb(9×9 crop of the source frame) + φ_uv(Fourier(u, v)) + φ_rel([h_src; h_tgt; h_ref])`
where the `h` are the encoder's per-frame ID tokens (one shared LayerNorm), so frames are named
by their content, not by an index: the decoder stays order-invariant.

Ref: ARROW arXiv:2610.01314, Eq. 3 and App. B.1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar
    from torch import Tensor

    from r4r.types import RGB, UV

    from ..queries import Queries

ID_WIDTH: Final = 3072
FOURIER_BASE = 100.0


def mlp[In: IntVar, D: IntVar](
    width_in: Int[In], width: Int[D]
) -> nn.Sequential[nn.Linear[In, D], nn.GELU, nn.Linear[D, D]]:
    return nn.Sequential(nn.Linear(width_in, width), nn.GELU(), nn.Linear(width, width))


class QueryEmbedding[D: IntVar, Crop: IntVar](nn.Module):
    def __init__(self, width: Int[D], crop: Int[Crop]) -> None:
        super().__init__()
        self.width = width
        self.crop = crop
        self.rgb_patch_embed = mlp(3 * crop * crop, width)
        self.uv_fourier_proj = mlp(width, width)
        self.camera_token_norm = nn.LayerNorm(ID_WIDTH)
        self.camera_relation_embed = mlp(3 * ID_WIDTH, width)

    def forward[B: IntVar, S: IntVar, H: IntVar, W: IntVar, Q: IntVar](
        self,
        images: Tensor[[B, S, *RGB, H, W]],
        id_tokens: Tensor[[B, S, 3072]],
        queries: Queries[B, Q],
    ) -> Tensor[[B, Q, D]]:
        _, frames, _, height, width = images.shape
        source, target, reference = (
            q.clamp(0, frames - 1) for q in (queries.source, queries.target, queries.reference)
        )

        # Start from fp32 zeros so the residual stream stays fp32 under autocast (as the reference)
        tokens = id_tokens.new_zeros(*queries.xy.shape[:2], self.width)
        tokens = tokens + self.rgb_patch_embed(self.crops(images, source, queries.xy))

        size = queries.xy.new_tensor([width, height])
        uv = (queries.xy.float() + 0.5) / size * 2 - 1
        tokens = tokens + self.uv_fourier_proj(self.fourier(uv))

        identities = [
            self.camera_token_norm(
                id_tokens[torch.arange(id_tokens.shape[0], device=frame.device)[:, None], frame]
            )
            for frame in (source, target, reference)
        ]
        return tokens + self.camera_relation_embed(torch.cat(identities, -1))

    def fourier[B: IntVar, Q: IntVar](self, uv: Tensor[[B, Q, *UV]]) -> Tensor[[B, Q, int]]:
        """`[sin uω, cos uω, sin vω, cos vω]`, `ω_k = 100^(−k / (D/4))`, k < D/4 (uv in [−1, 1])."""

        # The width is 4 · (D // 4), which pyrefly does not reduce to D (D is a multiple of 4)
        per_axis = self.width // 4
        omega = 1.0 / FOURIER_BASE ** (
            torch.arange(per_axis, dtype=torch.float32, device=uv.device) / per_axis
        )
        u, v = uv[..., :1] * omega, uv[..., 1:] * omega
        return torch.cat((u.sin(), u.cos(), v.sin(), v.cos()), -1)

    def crops[B: IntVar, S: IntVar, H: IntVar, W: IntVar, Q: IntVar](
        self, images: Tensor[[B, S, *RGB, H, W]], frame: Tensor[[B, Q]], xy: Tensor[[B, Q, *UV]]
    ) -> Tensor[[B, Q, 3 * Crop * Crop]]:
        """Raw-RGB `crop × crop` patch at the nearest pixel, replicate-padded, channel-major."""

        b, _, _, height, width = images.shape
        pad = self.crop // 2
        padded = F.pad(rearrange(images, "b f c h w -> (b f) c h w"), (pad,) * 4, mode="replicate")
        padded = rearrange(padded, "(b f) c h w -> b f h w c", b=b)

        x = xy[..., 0].round().long().clamp(0, width - 1)
        y = xy[..., 1].round().long().clamp(0, height - 1)
        offsets = torch.arange(self.crop, device=xy.device)
        rows = y[..., None, None] + offsets[:, None]  # (B, Q, crop, 1); +pad −pad cancel
        columns = x[..., None, None] + offsets[None, :]  # (B, Q, 1, crop)
        batch = torch.arange(b, device=xy.device)[:, None, None, None]
        patch = padded[batch, frame[..., None, None], rows, columns]  # (B, Q, crop, crop, 3)
        return rearrange(patch, "b q i j c -> b q (c i j)")
