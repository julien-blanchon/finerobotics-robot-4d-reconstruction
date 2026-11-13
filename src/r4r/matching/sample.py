"""Draw sparse pixel correspondences from dense matches: certainty-weighted, then balanced.

As the reference sampler: pool both directions, draw `4 × count` matches in proportion to
certainty (inside the image), then keep `count` of them with probability ∝ 1 / (1 + density),
so correspondences spread over the overlap instead of piling up on textured regions.

Ref: Edstedt et al., RoMa v2, arXiv:2511.15706; RoMa, arXiv:2305.15404, §3.6.
Adapted from https://github.com/Parskatt/RoMaV2 (MIT).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from einops import rearrange

from .matcher import normalized_grid

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar
    from torch import Tensor

    from .romav2 import Matches

OVERSAMPLING = 4
KDE_STD = 0.1  # in normalised 4D match coordinates
MIN_DENSITY = 10.0  # matches with fewer (kernel-weighted) neighbours are outliers
OUTLIER_WEIGHT = 1e-7
KDE_CHUNK = 1 << 26  # pairwise distances per chunk


@dataclass(frozen=True, slots=True)
class Correspondences[B: IntVar, K: IntVar]:
    """Pixel coordinates (OpenCV: pixel centres at integers) in each image, and certainty."""

    pixels_a: Tensor[[B, K, 2]]
    pixels_b: Tensor[[B, K, 2]]
    certainty: Tensor[[B, K]]


def to_pixels[B: IntVar, K: IntVar](
    normalised: Tensor[[B, K, 2]], size: tuple[int, int]
) -> Tensor[[B, K, 2]]:
    h, w = size
    return (normalised + 1) / 2 * normalised.new_tensor((w, h)) - 0.5


def density[B: IntVar, K: IntVar](points: Tensor[[B, K, 4]]) -> Tensor[[B, K]]:
    """Gaussian kernel density (unnormalised) at each point, chunked over queries."""

    b, k, _ = points.shape
    rows = max(1, KDE_CHUNK // (b * k))
    chunks = [
        (-torch.cdist(points[:, start : start + rows], points).square() / (2 * KDE_STD**2))
        .exp()
        .sum(-1)
        for start in range(0, k, rows)
    ]
    return torch.cat(chunks, dim=1)


@torch.inference_mode()
def sample[B: IntVar, K: IntVar](
    matches: Matches[B], count: Int[K], generator: torch.Generator | None = None
) -> Correspondences[B, K]:
    """`count` correspondences per pair (sampling without replacement)."""

    b, h, w, _ = matches.warp_ab.shape
    grid = normalized_grid(h, w, matches.warp_ab.device).expand(b, -1, -1, -1)
    # Every match as (x_a, y_a, x_b, y_b), a→b then b→a
    candidates = torch.cat(
        (
            rearrange(torch.cat((grid, matches.warp_ab), dim=-1), "b h w c -> b (h w) c"),
            rearrange(torch.cat((matches.warp_ba, grid), dim=-1), "b h w c -> b (h w) c"),
        ),
        dim=1,
    )
    certainty = torch.cat(
        (
            rearrange(matches.certainty_ab, "b h w -> b (h w)"),
            rearrange(matches.certainty_ba, "b h w -> b (h w)"),
        ),
        dim=1,
    )
    inside = candidates.abs().amax(dim=-1) <= 1 - 1 / h
    certainty = certainty * inside

    drawn = torch.multinomial(certainty, OVERSAMPLING * count, generator=generator)
    candidates = candidates.gather(1, drawn[..., None].expand(-1, -1, 4))
    certainty = certainty.gather(1, drawn)

    kernel_density = density(candidates)
    weight = torch.where(kernel_density < MIN_DENSITY, OUTLIER_WEIGHT, 1 / (kernel_density + 1))
    kept = torch.multinomial(weight, count, generator=generator)
    kept_matches = candidates.gather(1, kept[..., None].expand(-1, -1, 4))
    return Correspondences(
        to_pixels(kept_matches[..., :2], matches.size_a),
        to_pixels(kept_matches[..., 2:], matches.size_b),
        certainty.gather(1, kept),
    )
