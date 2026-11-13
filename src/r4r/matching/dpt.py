"""DPT head of the matcher: two token maps → coarse warp (2) and overlap logit (1) at stride 4.

The fusion trunk runs under bf16 autocast, the last two convolutions in fp32. Both in-place
ReLUs of the reference residual unit also rewrite its skip input: the skip adds `relu(x)`,
which is what the weights were trained with.

Ref: Ranftl et al., Vision Transformers for Dense Prediction, arXiv:2103.13413.
Adapted from https://github.com/Parskatt/RoMaV2 (MIT), itself after
https://github.com/DepthAnything/Depth-Anything-V2 (Apache-2.0).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

PATCH_SIZE = 16
OUTPUT_STRIDE = 4
FEATURES = 256
PROJECTED: tuple[int, ...] = (256, 512, 1024, 1024)
HIDDEN = 32
OUTPUTS = 3  # warp x, warp y, overlap logit


def resize[S: IntVar, C: IntVar](
    x: Tensor[[S, C, int, int]], size: tuple[int, int]
) -> Tensor[[S, C, int, int]]:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=True)


class ResidualConvUnit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(FEATURES, FEATURES, 3, padding=1)
        self.conv2 = nn.Conv2d(FEATURES, FEATURES, 3, padding=1)

    def forward[S: IntVar, H: IntVar, W: IntVar](
        self, x: Tensor[[S, 256, H, W]]
    ) -> Tensor[[S, 256, H, W]]:
        x = F.relu(x)
        return self.conv2(F.relu(self.conv1(x))) + x


class FusionBlock(nn.Module):
    """Add the (refined) skip, refine, upsample to `size`, project."""

    def __init__(self, *, has_skip: bool) -> None:
        super().__init__()
        self.out_conv = nn.Conv2d(FEATURES, FEATURES, 1)
        self.resConfUnit1 = ResidualConvUnit() if has_skip else None
        self.resConfUnit2 = ResidualConvUnit()

    def forward[S: IntVar](
        self,
        x: Tensor[[S, 256, int, int]],
        skip: Tensor[[S, 256, int, int]] | None,
        size: tuple[int, int],
    ) -> Tensor[[S, 256, int, int]]:
        if self.resConfUnit1 is not None and skip is not None:
            x = x + self.resConfUnit1(skip)
        return self.out_conv(resize(self.resConfUnit2(x), size))


class Scratch(nn.Module):
    """Container that keeps the reference parameter names (`scratch.layer1_rn`, ...)."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1_rn, self.layer2_rn, self.layer3_rn, self.layer4_rn = (
            nn.Conv2d(channels, FEATURES, 3, padding=1, bias=False) for channels in PROJECTED
        )
        self.refinenet1 = FusionBlock(has_skip=True)
        self.refinenet2 = FusionBlock(has_skip=True)
        self.refinenet3 = FusionBlock(has_skip=True)
        self.refinenet4 = FusionBlock(has_skip=False)
        self.output_conv1 = nn.Conv2d(FEATURES, FEATURES // 2, 3, padding=1)
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(FEATURES // 2, HIDDEN, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(HIDDEN, OUTPUTS, 1),
        )


class DPTHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Heterogeneous layers: typed as plain modules, applied with `level`
        self.projects: nn.ModuleList = nn.ModuleList(
            nn.Conv2d(dim, channels, 1) for channels in PROJECTED
        )
        self.resize_layers: nn.ModuleList = nn.ModuleList(
            (
                nn.ConvTranspose2d(PROJECTED[0], PROJECTED[0], 4, stride=4),
                nn.ConvTranspose2d(PROJECTED[1], PROJECTED[1], 2, stride=2),
                nn.Identity(),
                nn.Conv2d(PROJECTED[3], PROJECTED[3], 3, stride=2, padding=1),
            )
        )
        self.scratch = Scratch()

    def forward[S: IntVar, H: IntVar, W: IntVar](
        self, shallow: Tensor[[S, H, W, int]], deep: Tensor[[S, H, W, int]]
    ) -> Tensor[[S, H * 4, W * 4, 3]]:
        """Token maps (DINOv3 block 11; block 17 + matching) → `[S, 4h, 4w, 3]`."""

        _, h, w, _ = shallow.shape
        scratch = self.scratch
        with torch.autocast(shallow.device.type, dtype=torch.bfloat16):
            # Each token map feeds two of the four DPT levels
            levels: list[Tensor[[S, int, int, int]]] = []
            for index, tokens in enumerate((shallow, shallow, deep, deep)):
                x = rearrange(self.norm(tokens), "s h w d -> s d h w")
                levels.append(self.resize_layers[index](self.projects[index](x)))
            layer1, layer2, layer3, layer4 = (
                rn(level)
                for rn, level in zip(
                    (scratch.layer1_rn, scratch.layer2_rn, scratch.layer3_rn, scratch.layer4_rn),
                    levels,
                    strict=True,
                )
            )
            out = scratch.refinenet4(layer4, None, layer3.shape[2:])
            out = scratch.refinenet3(out, layer3, layer2.shape[2:])
            out = scratch.refinenet2(out, layer2, layer1.shape[2:])
            out = scratch.refinenet1(out, layer1, (2 * layer1.shape[2], 2 * layer1.shape[3]))
            out = scratch.output_conv1(out)
            size = (h * PATCH_SIZE // OUTPUT_STRIDE, w * PATCH_SIZE // OUTPUT_STRIDE)
            out = resize(out, size)
        out = scratch.output_conv2(out.float())
        return rearrange(out, "s c h w -> s h w c")
