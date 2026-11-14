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
    from torch import Tensor

CERTAINTY = 0.8
PER_PAIR = 2000
BATCH = 4


@dataclass(frozen=True, slots=True)
class CrossMatches:
    """`K` correspondences: (camera, timestep, pixel) at each end."""

    camera_a: Tensor[[int]]
    time_a: Tensor[[int]]
    pixel_a: Tensor[[int, 2]]
    camera_b: Tensor[[int]]
    time_b: Tensor[[int]]
    pixel_b: Tensor[[int, 2]]


def cross_matches(
    model: RoMaV2,
    frames: list[Tensor[[int, 3, int, int]]],
    pairs: list[tuple[int, int, int, int]],
    certainty: float = CERTAINTY,
    per_pair: int = PER_PAIR,
) -> CrossMatches:
    """Certain RoMa correspondences for `pairs` of (camera, time) frames (uint8 videos)."""

    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(0)
    parts: list[list[Tensor]] = [[] for _ in range(6)]
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
            for slot, value in enumerate(
                (
                    torch.full((n,), ca, device=device),
                    torch.full((n,), ta, device=device),
                    _rescaled(found.pixels_a[i][sure], common, frames[ca].shape[-2:]),
                    torch.full((n,), cb, device=device),
                    torch.full((n,), tb, device=device),
                    _rescaled(found.pixels_b[i][sure], common, frames[cb].shape[-2:]),
                )
            ):
                parts[slot].append(value)
    return CrossMatches(*(torch.cat(p) for p in parts))


def _resized(image: Tensor[[3, int, int]], size: tuple[int, int]) -> Tensor[[3, int, int]]:
    if tuple(image.shape[-2:]) == size:
        return image
    return torch.nn.functional.interpolate(
        image[None], size=size, mode="bilinear", align_corners=False, antialias=True
    )[0]


def _rescaled(
    pixels: Tensor[[int, 2]], common: tuple[int, int], own: tuple[int, int]
) -> Tensor[[int, 2]]:
    """Pixel centres (x, y) at the `common` size → the camera's `own` (height, width)."""

    if tuple(own) == common:
        return pixels
    scale = pixels.new_tensor((own[1] / common[1], own[0] / common[0]))
    return (pixels + 0.5) * scale - 0.5
