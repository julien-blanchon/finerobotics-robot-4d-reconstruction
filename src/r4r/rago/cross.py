"""Correspondences between cameras: RoMa v2 on chosen frame pairs, kept where it is certain.

A pair is (camera a, timestep a, camera b, timestep b). Pairs are matched in batches, each
batch encoding the frames it uses (once each, bounded memory). Only matches RoMa is certain
of (its overlap probability above `certainty`) are kept: pairs that do not see the same scene
yield none.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from r4r.matching import RoMaV2, sample

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.types import RGB, UV

CERTAINTY = 0.8
PER_PAIR = 2000
BATCH = 4


@dataclass(frozen=True, slots=True)
class CrossMatches[K: IntVar]:
    """`K` correspondences: (camera, timestep, pixel) at each end."""

    camera_a: Tensor[[K]]
    time_a: Tensor[[K]]
    pixel_a: Tensor[[K, *UV]]
    camera_b: Tensor[[K]]
    time_b: Tensor[[K]]
    pixel_b: Tensor[[K, *UV]]


def cross_matches(
    model: RoMaV2,
    frames: list[Tensor[[int, *RGB, int, int]]],
    pairs: list[tuple[int, int, int, int]],
    certainty: float = CERTAINTY,
    per_pair: int = PER_PAIR,
) -> CrossMatches[int]:
    """Certain RoMa correspondences for `pairs` of (camera, time) frames (uint8 videos):
    those with `P(overlap) > certainty`."""

    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(0)
    parts: list[CrossMatches[int]] = []
    for start in range(0, len(pairs), BATCH):
        chunk = pairs[start : start + BATCH]
        keys = sorted({(c, t) for a, ta, b, tb in chunk for c, t in ((a, ta), (b, tb))})
        position = {key: i for i, key in enumerate(keys)}
        # Cameras of different sizes are matched at a common size (the largest), pixels mapped
        # back to each camera's own
        common = (
            max(frames[c].shape[-2] for c, _ in keys),
            max(frames[c].shape[-1] for c, _ in keys),
        )
        images = torch.stack(
            [_resized(frames[c][t].to(device).float() / 255, common) for c, t in keys]
        )
        encoding = model.encode(images)
        a = torch.tensor([position[(c, t)] for c, t, _, _ in chunk], device=device)
        b = torch.tensor([position[(c, t)] for _, _, c, t in chunk], device=device)
        found = sample(model.match(encoding[a], encoding[b]), per_pair, generator)
        for i, (ca, ta, cb, tb) in enumerate(chunk):
            sure = found.certainty[i] > certainty
            n = int(sure.sum())
            parts.append(
                CrossMatches(
                    torch.full((n,), ca, device=device),
                    torch.full((n,), ta, device=device),
                    _rescaled(found.pixels_a[i][sure], common, frames[ca].shape[-2:]),
                    torch.full((n,), cb, device=device),
                    torch.full((n,), tb, device=device),
                    _rescaled(found.pixels_b[i][sure], common, frames[cb].shape[-2:]),
                )
            )
    return CrossMatches(
        torch.cat([m.camera_a for m in parts]),
        torch.cat([m.time_a for m in parts]),
        torch.cat([m.pixel_a for m in parts]),
        torch.cat([m.camera_b for m in parts]),
        torch.cat([m.time_b for m in parts]),
        torch.cat([m.pixel_b for m in parts]),
    )


def _resized[H: IntVar, W: IntVar](
    image: Tensor[[*RGB, H, W]], size: tuple[int, int]
) -> Tensor[[*RGB, int, int]]:
    if tuple(image.shape[-2:]) == size:
        return image
    return torch.nn.functional.interpolate(
        image[None], size=size, mode="bilinear", align_corners=False, antialias=True
    )[0]


def _rescaled[N: IntVar](
    pixels: Tensor[[N, *UV]], common: tuple[int, int], own: tuple[int, int]
) -> Tensor[[N, *UV]]:
    """Pixel centres (x, y) at the `common` size → the camera's `own` (height, width):
    `x' = (x + ½) · s − ½`, `s = own / common` per axis."""

    if tuple(own) == common:
        return pixels
    scale = pixels.new_tensor((own[1] / common[1], own[0] / common[0]))
    return (pixels + 0.5) * scale - 0.5
