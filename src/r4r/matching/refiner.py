"""Convolutional warp refiner and its local correlation.

Each refiner sees the source features, the target features sampled at the current warp, a
`(2r + 1)²` correlation window around it, and an embedding of the current displacement; a
depthwise-separable conv trunk (bf16, channels-last) predicts a warp update, an overlap-logit
update and a Cholesky-parametrised 2×2 precision update (pixels⁻², added to the previous one).

The refiner is compiled as one region: the correlation then fuses into a single kernel that
never materialises the `K × C` window features per pixel.

Ref: Edstedt et al., RoMa v2, arXiv:2511.15706, §3.4–3.5.
Adapted from https://github.com/Parskatt/RoMaV2 (MIT).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import nn

from .matcher import normalized_grid

if TYPE_CHECKING:
    from shape_extensions import Int, IntTuple, IntVar
    from torch import Tensor

    from r4r.types import UV

KERNEL_SIZE = 5
HIDDEN_BLOCKS = 8
WARP_STEP = 4.0  # displacement is predicted in units of a quarter pixel of the refiner grid
CHOLESKY_EPS = 1e-6


def window[S: IntVar, H: IntVar, W: IntVar, C: IntVar](
    target: Tensor[[S, H, W, C]], warp: Tensor[[S, H, W, *UV]], radius: int
) -> tuple[Tensor[[S, H, W, int, int, C]], Tensor[[S, H, W, 1, 1]], Tensor[[S, H, W, 1, 1]]]:
    """Target pixels on the `(2r + 2)²` integer grid around each warp position, and its weights.

    Pixels outside the image are zero (grid_sample's zero padding). Bilinear sampling at the warp
    shifted by whole pixels is the same blend (`blend`) of this grid for every shift.
    """

    s, h, w, _ = target.shape
    x = ((warp[..., 0] + 1) * w - 1) / 2
    y = ((warp[..., 1] + 1) * h - 1) / 2
    left, top = x.floor(), y.floor()
    steps = torch.arange(-radius, radius + 2, device=warp.device)
    columns = left.long()[..., None] + steps
    rows = top.long()[..., None] + steps
    inside = ((rows >= 0) & (rows < h))[..., :, None] & ((columns >= 0) & (columns < w))[
        ..., None, :
    ]
    index = rows.clamp(0, h - 1)[..., :, None] * w + columns.clamp(0, w - 1)[..., None, :]
    frames = torch.arange(s, device=warp.device)[:, None, None, None, None]
    pixels = rearrange(target, "s h w c -> s (h w) c")[frames, index] * inside[..., None]
    return pixels, (x - left)[..., None, None], (y - top)[..., None, None]


def blend[S: IntTuple, F: IntTuple](
    grid: Tensor[[*S, int, int]], fx: Tensor[[*F, 1, 1]], fy: Tensor[[*F, 1, 1]]
) -> Tensor[[*S, int, int]]:
    """Bilinear blend of each 2×2 cell of an integer-pixel grid at `(fx, fy)` (broadcast)."""

    blended = (1 - fy) * ((1 - fx) * grid[..., :-1, :-1] + fx * grid[..., :-1, 1:]) + fy * (
        (1 - fx) * grid[..., 1:, :-1] + fx * grid[..., 1:, 1:]
    )
    # Broadcasting against weights of another (compatible) shape loses the leading dims
    return cast("Tensor[[*S, int, int]]", blended)


def bilinear_sample[S: IntVar, H: IntVar, W: IntVar, C: IntVar](
    target: Tensor[[S, H, W, C]], warp: Tensor[[S, H, W, *UV]]
) -> Tensor[[S, H, W, C]]:
    """`grid_sample(target, warp)`, bilinear with zero padding (`align_corners=False`)."""

    pixels, fx, fy = window(target, warp, 0)
    corners = rearrange(pixels, "s h w i j c -> s h w c i j")
    return blend(corners, fx[..., None, :, :], fy[..., None, :, :])[..., 0, 0]


def local_correlation[S: IntVar, H: IntVar, W: IntVar, C: IntVar](
    source: Tensor[[S, H, W, C]],
    target: Tensor[[S, H, W, C]],
    warp: Tensor[[S, H, W, *UV]],
    radius: int,
) -> Tensor[[S, H, W, int]]:
    """Scaled dot products of each source pixel with the target around its warp, `[S, H, W, K]`:
    `c(x, δ) = ⟨f_a(x), f_b(w(x) + δ)⟩ / √C` for the `K = (2r + 1)²` integer offsets `δ`.

    The `(2r + 1)²` window (row-major over dy, dx) steps one target pixel. Compiled, the gather,
    dot products and blend fuse into one kernel: the `K × C` window features never exist.
    """

    pixels, fx, fy = window(target, warp, radius)
    # Product + reduce, not einsum: a bmm would materialise the window features
    products = pixels * source[..., None, None, :]
    dots = reduce(products, "s h w i j c -> s h w i j", "sum") / source.shape[-1] ** 0.5
    return rearrange(blend(dots, fx, fy), "s h w i j -> s h w (i j)")


def precision_update[S: IntVar, H: IntVar, W: IntVar](
    raw: Tensor[[S, H, W, 3]],
) -> Tensor[[S, H, W, 3]]:
    """Cholesky factor `L = [[l00, 0], [l10, l11]]` → upper triangle of `L Lᵀ` (p00, p10, p11).

    `l00 = softplus(r0) + ε`, `l10 = r1`, `l11 = softplus(r2) + ε`: `L Lᵀ` is positive definite.
    """

    l00 = F.softplus(raw[..., 0]) + CHOLESKY_EPS
    l10 = raw[..., 1]
    l11 = F.softplus(raw[..., 2]) + CHOLESKY_EPS
    return torch.stack((l00 * l00, l00 * l10, l10 * l10 + l11 * l11), dim=-1)


class Block[D: IntVar](nn.Module):
    def __init__(self, dim: Int[D]) -> None:
        super().__init__()
        self.conv_depthwise = nn.Conv2d(dim, dim, KERNEL_SIZE, padding=KERNEL_SIZE // 2, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.conv_pointwise = nn.Conv2d(dim, dim, 1)

    def forward[S: IntVar, H: IntVar, W: IntVar](
        self, x: Tensor[[S, D, H, W]]
    ) -> Tensor[[S, D, H, W]]:
        return self.conv_pointwise(F.relu(self.norm(self.conv_depthwise(x))))


class ConvRefiner[C: IntVar](nn.Module):
    def __init__(
        self, features: Int[C], projected: int, displacement: int, radius: int | None
    ) -> None:
        super().__init__()
        self.radius = radius
        window = 0 if radius is None else (2 * radius + 1) ** 2
        hidden = 2 * projected + displacement + window
        self.proj = nn.Linear(features, projected)
        self.disp_emb = nn.Conv2d(2, displacement, 1)
        self.block1 = Block(hidden)
        self.hidden_blocks = nn.Sequential(*(Block(hidden) for _ in range(HIDDEN_BLOCKS)))
        self.warp_head = nn.Conv2d(hidden, 2, 1)
        self.confidence_head = nn.Conv2d(hidden, 4, 1)

    @torch.compile(fullgraph=True, dynamic=True)  # type: ignore[pyrefly:missing-attribute]  # stubs
    def forward[B: IntVar, H: IntVar, W: IntVar](
        self,
        features: Tensor[[2 * B, H, W, C]],
        warp: Tensor[[2 * B, H, W, *UV]],
        confidence: Tensor[[2 * B, H, W, 4]],
        scale: float,
    ) -> tuple[Tensor[[2 * B, H, W, *UV]], Tensor[[2 * B, H, W, 4]]]:
        """Refine a→b (first half of the batch) and b→a; `scale` is image size / 512.

        `features` stacks frames a then b. Concatenated by the caller: with two separately
        shaped inputs, inductor saw a sum of batch symbols and materialised the correlation
        window (tens of GB) instead of fusing it.
        """

        s, h, w, _ = features.shape
        projected = self.proj(features.float())
        source, target = projected, torch.roll(projected, s // 2, dims=0)

        sampled = bilinear_sample(target, warp)
        displacement = rearrange(warp - normalized_grid(h, w, warp.device), "s h w c -> s c h w")
        inputs = [
            rearrange(source, "s h w c -> s c h w"),
            rearrange(sampled, "s h w c -> s c h w"),
            self.disp_emb(scale * displacement),
        ]
        if self.radius is not None:
            correlation = local_correlation(source, target, warp, self.radius)
            inputs.append(rearrange(correlation, "s h w k -> s k h w"))

        with torch.autocast(warp.device.type, dtype=torch.bfloat16):
            x = torch.cat(inputs, dim=1)
            x = x.contiguous(memory_format=torch.channels_last)  # type: ignore[pyrefly:unexpected-keyword]  # not in the stubs
            z = self.hidden_blocks(self.block1(x))
        z = z.float()
        step = rearrange(self.warp_head(z), "s c h w -> s h w c")
        update = rearrange(self.confidence_head(z), "s c h w -> s h w c")

        step = torch.stack((step[..., 0] / (WARP_STEP * w), step[..., 1] / (WARP_STEP * h)), dim=-1)
        warp = warp + step
        update = torch.cat((update[..., :1], precision_update(update[..., 1:])), dim=-1)
        return warp, confidence + update
