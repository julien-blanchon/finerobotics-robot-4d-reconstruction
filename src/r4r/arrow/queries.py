"""Point queries and the standard query patterns.

A query asks: the pixel `xy` of frame `source` — where is that physical point at the time of
frame `target`, expressed in the camera of frame `reference`? Pixel centres are integers
(`x = u · W − ½`), in model-image coordinates. Every task is a pattern of such queries.

Ref: ARROW arXiv:2610.01314, Eq. 1; D4RT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from einops import repeat

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor


@dataclass(frozen=True, slots=True)
class Queries[B: IntVar, Q: IntVar]:
    xy: Tensor[[B, Q, 2]]
    source: Tensor[[B, Q]]
    target: Tensor[[B, Q]]
    reference: Tensor[[B, Q]]

    @property
    def count(self) -> int:
        return self.xy.shape[1]

    def slice(self, start: int, stop: int) -> Queries[B, int]:
        return Queries(
            self.xy[:, start:stop],
            self.source[:, start:stop],
            self.target[:, start:stop],
            self.reference[:, start:stop],
        )

    @staticmethod
    def concat(parts: list[Queries]) -> Queries:
        return Queries(
            torch.cat([p.xy for p in parts], 1),
            torch.cat([p.source for p in parts], 1),
            torch.cat([p.target for p in parts], 1),
            torch.cat([p.reference for p in parts], 1),
        )


def pixel_grid(height: int, width: int, stride: int, device: torch.device) -> Tensor[[int, 2]]:
    """Integer pixel centres `(x, y)`, every `stride` pixels, row-major."""

    ys = torch.arange(0, height, stride, device=device, dtype=torch.float32)
    xs = torch.arange(0, width, stride, device=device, dtype=torch.float32)
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((x, y), -1).flatten(0, 1)


def per_frame[N: IntVar](xy: Tensor[[N, 2]], frames: int, reference: int | None) -> Queries:
    """Every frame asked about the same pixels, at its own time; `reference=None` → own camera.

    Order: frame-major, then pixel.
    """

    frame = repeat(torch.arange(frames, device=xy.device), "f -> 1 (f n)", n=xy.shape[0])
    ref = frame if reference is None else torch.full_like(frame, reference)
    return Queries(repeat(xy, "n c -> 1 (f n) c", f=frames), frame, frame, ref)


def tracks[N: IntVar](
    xy: Tensor[[N, 2]], source: Tensor[[N]], frames: int, reference: int | None
) -> Queries:
    """Follow each seed `(xy, source)` through every frame; `reference=None` → target camera.

    Order: target-major, then seed.
    """

    target = repeat(torch.arange(frames, device=xy.device), "t -> 1 (t n)", n=xy.shape[0])
    ref = target if reference is None else torch.full_like(target, reference)
    return Queries(
        repeat(xy, "n c -> 1 (t n) c", t=frames),
        repeat(source, "n -> 1 (t n)", t=frames),
        target,
        ref,
    )
