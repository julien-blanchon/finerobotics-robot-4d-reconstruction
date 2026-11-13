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
        images = torch.stack([frames[c][t] for c, t in keys]).to(device).float() / 255
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
                    found.pixels_a[i][sure],
                    torch.full((n,), cb, device=device),
                    torch.full((n,), tb, device=device),
                    found.pixels_b[i][sure],
                )
            ):
                parts[slot].append(value)
    return CrossMatches(*(torch.cat(p) for p in parts))
