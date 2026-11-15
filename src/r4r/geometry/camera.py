"""Cameras: projection, back-projection, lens distortion, intrinsics fit, resizing.

OpenCV convention (+x right, +y down, +z forward) with pixel centres at integer coordinates:
the pixel `(x, y)` covers `[x − ½, x + ½]`. Intrinsics `K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`.

Lenses are radially symmetric: a ray at distance `s` from the optical axis lands at the
normalised radius `ρ = d(s) = s · (1 + k1 s² + k2 s⁴ + k3 s⁶ + k4 s⁸)`, then `pixel = f ρ + c`.
- pinhole: `s = r / z` (radius on the z = 1 plane): Brown–Conrady radial distortion, OpenCV's
  `k1, k2, k3` (k4 = 0; tangential and rational terms are not modelled);
- fisheye: `s = θ`, the angle from the optical axis: Kannala–Brandt, OpenCV's `cv2.fisheye`
  `k1..k4`. All k = 0 is the equidistant fisheye (`ρ = θ`), valid past 90°.
Both share `d`, so one Newton inverse serves both. Coefficients are per normalised radius:
resizing the image changes `K`, not them.

Ref: Kannala & Brandt, TPAMI 2006 (generic fisheye model); Brown, Photogramm. Eng. 1966.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from shape_extensions import IntTuple, IntVar
    from torch import Tensor

    from r4r.types import UV, XYZ, Mat3x3

MIN_DEPTH = 1e-6
# Distortion coefficients per camera (k1..k4)
COEFFICIENTS = 4
# Squared radius floor: keeps the radius differentiable at the optical axis
MIN_RADIUS_SQUARED = 1e-24
# Newton steps inverting d(s) (quadratic convergence from s = ρ: 6 is plenty in float64)
NEWTON_STEPS = 6
# A fisheye angle never exceeds this (rad), a pinhole's radius on the z = 1 plane this (89.4°
# off-axis); d'(s) is kept above MIN_SLOPE where the polynomial folds back (outside the lens's
# valid field). A pixel past the fold (a barrel lens, k1 < 0, or a short focal) has no ray:
# Newton then runs outwards, and unbounded it overflows into NaN rays and gradients
MAX_ANGLE = math.pi
MAX_RADIUS = 100.0
MIN_SLOPE = 1e-6


def project[Bs: IntTuple, N: IntVar](
    intrinsics: Tensor[[*Bs, 3, 3]],
    points: Tensor[[*Bs, N, 3]],
    distortion: Tensor[[*Bs, 4]] | None = None,
    fisheye: Tensor[[*Bs]] | None = None,
) -> Tensor[[*Bs, N, 2]]:
    """Pixel coordinates of camera-frame points (depth clamped away from zero), through the
    lens `distortion` (`fisheye`: Kannala–Brandt, else radial; None: ideal pinhole).

    `p = f · d(s) · (x, y)/r + c`, `r = ‖(x, y)‖`, `s = r/z` (radial) or `atan2(r, z)` (fisheye);
    the ideal pinhole is `p = f · (x, y)/z + c`. Points in metres, pixels in pixels.
    """

    focal, centre = intrinsics[..., None, [0, 1], [0, 1]], intrinsics[..., None, :2, 2]
    if distortion is None or fisheye is None:
        z = points[..., 2:].clamp_min(MIN_DEPTH)
        return points[..., :2] / z * focal + centre
    xy, z = points[..., :2], points[..., 2]
    radius = (xy**2).sum(-1).clamp_min(MIN_RADIUS_SQUARED).sqrt()
    # Distance from the axis: on the z = 1 plane (pinhole) or as an angle (fisheye)
    s = torch.where(fisheye[..., None], torch.atan2(radius, z), radius / z.clamp_min(MIN_DEPTH))
    rho = lens_radius(distortion, s)
    return xy * (rho / radius)[..., None] * focal + centre


def backproject[Bs: IntTuple, N: IntVar](
    intrinsics: Tensor[[*Bs, 3, 3]], pixels: Tensor[[*Bs, N, 2]], depth: Tensor[[*Bs, N]]
) -> Tensor[[*Bs, N, 3]]:
    """Camera-frame points at the given z-depth along each pixel's ray (ideal pinhole):
    `X = z · ((u − cx)/fx, (v − cy)/fy, 1)`."""

    focal, centre = intrinsics[..., None, [0, 1], [0, 1]], intrinsics[..., None, :2, 2]
    rays = torch.cat(((pixels - centre) / focal, torch.ones_like(pixels[..., :1])), -1)
    return rays * depth[..., None]


def rays[Bs: IntTuple, N: IntVar](
    intrinsics: Tensor[[*Bs, 3, 3]],
    pixels: Tensor[[*Bs, N, 2]],
    distortion: Tensor[[*Bs, 4]] | None = None,
    fisheye: Tensor[[*Bs]] | None = None,
) -> Tensor[[*Bs, N, 3]]:
    """Unit camera-frame direction of each pixel's ray (inverse of `project`).

    The lens radius is inverted by Newton steps on detached values, then one step with
    gradients: at convergence its derivatives are those of the exact inverse (implicit
    function theorem), at the cost of one evaluation.
    """

    if distortion is None or fisheye is None:
        return F.normalize(backproject(intrinsics, pixels, torch.ones_like(pixels[..., 0])), dim=-1)
    focal, centre = intrinsics[..., None, [0, 1], [0, 1]], intrinsics[..., None, :2, 2]
    xy = (pixels - centre) / focal
    rho = (xy**2).sum(-1).clamp_min(MIN_RADIUS_SQUARED).sqrt()
    limit = torch.full_like(rho[..., 0], MAX_RADIUS).masked_fill(fisheye, MAX_ANGLE)
    s = rho.detach()
    for _ in range(NEWTON_STEPS):
        s = _newton_step(distortion.detach(), s, rho.detach(), limit)
    s = _newton_step(distortion, s, rho, limit)
    axis = xy / rho[..., None]
    # The undistorted point on the z = 1 plane (pinhole), or the ray at angle s (fisheye)
    plane = torch.cat((axis * s[..., None], torch.ones_like(s[..., None])), -1)
    angle = torch.cat((axis * s.sin()[..., None], s.cos()[..., None]), -1)
    return torch.where(fisheye[..., None, None], angle, F.normalize(plane, dim=-1))


def lens_radius[Bs: IntTuple, N: IntVar](
    distortion: Tensor[[*Bs, 4]], s: Tensor[[*Bs, N]]
) -> Tensor[[*Bs, N]]:
    """`d(s) = s · (1 + k1 s² + k2 s⁴ + k3 s⁶ + k4 s⁸)` (Horner), one lens per row of `s`."""

    s2 = s * s
    k1, k2, k3, k4 = distortion[..., None, :].unbind(-1)
    return s * (1 + s2 * (k1 + s2 * (k2 + s2 * (k3 + s2 * k4))))


def _newton_step[Bs: IntTuple, N: IntVar](
    distortion: Tensor[[*Bs, 4]], s: Tensor[[*Bs, N]], rho: Tensor[[*Bs, N]], limit: Tensor[[*Bs]]
) -> Tensor[[*Bs, N]]:
    """One Newton step on `d(s) = ρ`, kept in `[0, limit]`."""

    s2 = s * s
    k1, k2, k3, k4 = distortion[..., None, :].unbind(-1)
    slope = 1 + s2 * (3 * k1 + s2 * (5 * k2 + s2 * (7 * k3 + s2 * 9 * k4)))
    step = (lens_radius(distortion, s) - rho) / slope.clamp_min(MIN_SLOPE)
    return torch.minimum((s - step).clamp_min(0), limit[..., None])


def fit_intrinsics[N: IntVar](
    points: Tensor[[N, *XYZ]], pixels: Tensor[[N, *UV]], weight: Tensor[[N]]
) -> Tensor[[*Mat3x3]]:
    """Weighted least squares of `x = fx · X/Z + cx` (and y) from camera-frame points:
    `min Σ w_i (fx X_i/Z_i + cx − x_i)²` per axis, in fp64. Points in any unit (only X/Z enters).

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
