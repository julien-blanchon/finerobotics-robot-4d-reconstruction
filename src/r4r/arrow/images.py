"""Model-input images and the map between native and model pixels.

ARROW sees every frame resized so the long side is 504 px and both sides are multiples of 14
(the reference's preparation). All frames of an episode share one native size, so no padding
is needed. Pixel centres map as `x' = (x + ½) · s − ½`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch.nn.functional as F

from .models.encoder import PATCH_SIZE

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

LONG_SIDE = 504


@dataclass(frozen=True, slots=True)
class Resize:
    native: tuple[int, int]  # (width, height)
    model: tuple[int, int]

    @property
    def scale(self) -> tuple[float, float]:
        return self.model[0] / self.native[0], self.model[1] / self.native[1]

    @classmethod
    def for_size(cls, width: int, height: int) -> Resize:
        factor = LONG_SIDE / max(width, height)

        def side(n: int) -> int:
            return max(PATCH_SIZE, (round(n * factor) + PATCH_SIZE // 2) // PATCH_SIZE * PATCH_SIZE)

        return cls((width, height), (side(width), side(height)))


def model_images[N: IntVar, H: IntVar, W: IntVar](
    frames: Tensor[[N, 3, H, W]], resize: Resize
) -> Tensor[[N, 3, int, int]]:
    """uint8 frames → float RGB in [0, 1] at model size (antialiased bilinear)."""

    width, height = resize.model
    images = frames.float() / 255
    return F.interpolate(
        images, (height, width), mode="bilinear", antialias=True, align_corners=False
    )
