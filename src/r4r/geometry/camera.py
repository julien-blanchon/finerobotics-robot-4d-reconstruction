"""Pinhole cameras: projection, back-projection, intrinsics fit, resizing.

OpenCV convention (+x right, +y down, +z forward) with pixel centres at integer coordinates:
the pixel `(x, y)` covers `[x − ½, x + ½]`. Intrinsics `K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from shape_extensions import IntTuple, IntVar
    from torch import Tensor

MIN_DEPTH = 1e-6


def project[Bs: IntTuple, N: IntVar](
    intrinsics: Tensor[[*Bs, 3, 3]], points: Tensor[[*Bs, N, 3]]
) -> Tensor[[*Bs, N, 2]]:
    """Pixel coordinates of camera-frame points (depth clamped away from zero)."""

    z = points[..., 2:].clamp_min(MIN_DEPTH)
    xy = points[..., :2] / z
    focal, centre = intrinsics[..., None, [0, 1], [0, 1]], intrinsics[..., None, :2, 2]
    return xy * focal + centre


def backproject[Bs: IntTuple, N: IntVar](
    intrinsics: Tensor[[*Bs, 3, 3]], pixels: Tensor[[*Bs, N, 2]], depth: Tensor[[*Bs, N]]
) -> Tensor[[*Bs, N, 3]]:
    """Camera-frame points at the given z-depth along each pixel's ray."""

    focal, centre = intrinsics[..., None, [0, 1], [0, 1]], intrinsics[..., None, :2, 2]
    rays = torch.cat(((pixels - centre) / focal, torch.ones_like(pixels[..., :1])), -1)
    return rays * depth[..., None]


def fit_intrinsics[N: IntVar](
    points: Tensor[[N, 3]], pixels: Tensor[[N, 2]], weight: Tensor[[N]]
) -> Tensor[[3, 3]]:
    """Weighted least squares of `x = fx · X/Z + cx` (and y) from camera-frame points.

    Ref: ARROW arXiv:2610.01314, App. C.3 (per-axis pinhole fit), here weighted.
    """

    ray = points[:, :2] / points[:, 2:].clamp_min(MIN_DEPTH)
    root_weight = weight.clamp_min(0).sqrt()[:, None]
    intrinsics = torch.eye(3, dtype=points.dtype, device=points.device)
    for axis in range(2):
        design = torch.stack((ray[:, axis], torch.ones_like(ray[:, axis])), -1) * root_weight
        target = pixels[:, axis : axis + 1] * root_weight
        focal, centre = torch.linalg.lstsq(design.double(), target.double()).solution[:, 0]
        intrinsics[axis, axis], intrinsics[axis, 2] = focal, centre
    return intrinsics


def resize_intrinsics[Bs: IntTuple](
    intrinsics: Tensor[[*Bs, 3, 3]], scale_x: float, scale_y: float
) -> Tensor[[*Bs, 3, 3]]:
    """Intrinsics after resizing the image by (scale_x, scale_y), pixel-centre convention.

    A pixel centre moves as `x' = (x + ½) · s − ½`, so `f' = f · s` and `c' = (c + ½) · s − ½`.
    """

    scale = intrinsics.new_tensor([scale_x, scale_y])
    resized = intrinsics.clone()
    resized[..., [0, 1], [0, 1]] = intrinsics[..., [0, 1], [0, 1]] * scale
    resized[..., :2, 2] = (intrinsics[..., :2, 2] + 0.5) * scale - 0.5
    return resized


def resize_pixels[Bs: IntTuple](
    pixels: Tensor[[*Bs, 2]], scale_x: float, scale_y: float
) -> Tensor[[*Bs, 2]]:
    """Pixel coordinates after resizing the image (pixel-centre convention)."""

    return (pixels + 0.5) * pixels.new_tensor([scale_x, scale_y]) - 0.5
