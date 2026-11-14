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
    from shape_extensions import Int, IntVar
    from torch import Tensor

    from r4r.types import UV


@dataclass(frozen=True, slots=True)
class Queries[B: IntVar, Q: IntVar]:
    xy: Tensor[[B, Q, *UV]]  # model-image pixels
    source: Tensor[[B, Q]]  # frame indices
    target: Tensor[[B, Q]]
    reference: Tensor[[B, Q]]

    @property
    def count(self) -> Int[Q]:
        return self.xy.shape[1]

    def slice(self, start: int, stop: int) -> Queries[B, int]:
        """Queries `start:stop` (clipped to the count, so the length is left `int`)."""

        return Queries(
            self.xy[:, start:stop],
            self.source[:, start:stop],
            self.target[:, start:stop],
            self.reference[:, start:stop],
        )


def pixel_grid(height: int, width: int, stride: int, device: torch.device) -> Tensor[[int, *UV]]:
    """Integer pixel centres `(x, y)`, every `stride` pixels, row-major."""

    ys = torch.arange(0, height, stride, device=device, dtype=torch.float32)
    xs = torch.arange(0, width, stride, device=device, dtype=torch.float32)
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((x, y), -1).flatten(0, 1)


def per_frame[N: IntVar, F: IntVar](
    xy: Tensor[[N, *UV]], frames: Int[F], reference: int | None
) -> Queries[1, F * N]:
    """Every frame asked about the same pixels, at its own time; `reference=None` → own camera.

    Order: frame-major, then pixel.
    """

    frame = repeat(torch.arange(frames, device=xy.device), "f -> 1 (f n)", n=xy.shape[0])
    ref = frame if reference is None else torch.full_like(frame, reference)
    return Queries(repeat(xy, "n c -> 1 (f n) c", f=frames), frame, frame, ref)
