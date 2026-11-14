"""Fine features for the refiners: VGG19-BN conv stages at strides 1, 2 and 4.

Each stage's output is taken just before its max-pool (64, 128 and 256 channels). The weights
are RoMa v2's fine-tuned ones (not ImageNet's); they run under bf16 autocast as in the reference,
channels-last and compiled (cuDNN NHWC convolutions, BatchNorm + ReLU fused).

Ref: Edstedt et al., RoMa v2, arXiv:2511.15706, §3.4.
Adapted from https://github.com/Parskatt/RoMaV2 (MIT).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from einops import rearrange
from torch import nn

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.types import RGB

# (channels, convolutions) per stride 1, 2, 4
STAGES: tuple[tuple[int, int], ...] = ((64, 2), (128, 2), (256, 4))


class FineFeatures(nn.Module):
    """`layers` keeps torchvision's `vgg19_bn().features` indices, without the last max-pool."""

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels: int = 3
        for width, convolutions in STAGES:
            for _ in range(convolutions):
                layers += (nn.Conv2d(channels, width, 3, padding=1), nn.BatchNorm2d(width))
                layers.append(nn.ReLU(inplace=True))
                channels = width
            layers.append(nn.MaxPool2d(2, 2))
        self.layers = nn.Sequential(*layers[:-1])

    @torch.compile(fullgraph=True, dynamic=True)  # type: ignore[pyrefly:missing-attribute]  # not in the shape stubs
    def forward[S: IntVar, H: IntVar, W: IntVar](
        self, images: Tensor[[S, *RGB, H, W]]
    ) -> tuple[
        Tensor[[S, H, W, 64]], Tensor[[S, H // 2, W // 2, 128]], Tensor[[S, H // 4, W // 4, 256]]
    ]:
        """ImageNet-normalised images → channels-last features at strides 1, 2, 4."""

        outputs = []
        x = images.contiguous(memory_format=torch.channels_last)  # type: ignore[pyrefly:unexpected-keyword]  # not in the stubs
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                outputs.append(rearrange(x, "s c h w -> s h w c"))
            x = layer(x)
        stride1, stride2 = outputs
        return stride1, stride2, rearrange(x, "s c h w -> s h w c")
