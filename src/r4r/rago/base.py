"""Does the robot's base move? Evidence from one calibrated on-robot camera.

With the base fixed, forward kinematics and the camera's mount predict the camera's motion
between two frames, and every static scene point must then stay on its epipolar line. When the
base drives or turns, the true motion differs and static tracks leave their lines: the median
distance (pixels) over a frame pair is the evidence. It needs no odometry and no essential
matrix, only KLT tracks. A base translating exactly along the camera's own predicted direction
of travel goes unseen (the epipolar geometry does not change); turning and any other
translation do not.

Moving objects (the arm in view, a gripper on a wrist camera) are a minority of tracks in
useful views; the median ignores them.

Per episode this tells whether the fixed-base pipeline applies (the base never moves), on
which spans it applies (the base parks, then moves), or whether a per-frame base pose is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from r4r.geometry import se3

if TYPE_CHECKING:
    from torch import Tensor

    from .klt import VideoTracks

GAP = 3  # frames between the two views of a pair
MIN_POINTS = 20
# Pixels: above it, the pair's motion is not explained by the fixed base. Tracking noise and
# calibration error set the floor (0.5 px on clean synthetic tracks; a base turning 1.5° and
# driving 2 cm between the two views gives 2.5-3.5 px there): a per-dataset setting, read off
# an episode known to be parked.
THRESHOLD = 1.5
TRANSLATION_EPS = 1e-4  # metres: below, the predicted motion is a pure rotation
MIN_SPAN = 10  # frames: shortest still span reported


@dataclass(frozen=True, slots=True)
class BaseMotion:
    # Per frame pair (start, start + gap): median epipolar distance under the fixed-base
    # prediction (pixels; NaN where too few tracks)
    start: Tensor[[int]]
    residual: Tensor[[int]]
    # Per frame: covered by a pair the fixed base does not explain
    moving: Tensor[[int]]
    # Spans (first, last + 1) of at least MIN_SPAN frames where the base is still
    still: tuple[tuple[int, int], ...]

    @property
    def moving_share(self) -> float:
        return float(self.moving.double().mean())


def base_motion(
    video: VideoTracks,
    world_from_camera: Tensor[[int, 4, 4]],
    intrinsics: Tensor[[3, 3]],
    gap: int = GAP,
    threshold: float = THRESHOLD,
) -> BaseMotion:
    """Fixed-base evidence of one camera; `world_from_camera` from FK × mount (base fixed)."""

    frames = len(world_from_camera)
    dtype, device = world_from_camera.dtype, world_from_camera.device
    _, track = torch.unique(video.track, return_inverse=True)
    table = torch.full((int(track.max()) + 1, frames, 2), math.nan, dtype=dtype, device=device)
    table[track, video.frame] = video.pixel.to(table)

    start = torch.arange(frames - gap, device=device)
    end = start + gap
    k = intrinsics.to(table)
    focal = float(k[0, 0])
    rays_a = _rays(table[:, start], k)  # [N, P, 3]
    rays_b = _rays(table[:, end], k)
    b_from_a = se3.inverse(world_from_camera[end]) @ world_from_camera[start]  # [P, 4, 4]
    rotated = (b_from_a[:, :3, :3] @ rays_a[..., None])[..., 0]  # [N, P, 3]
    t = b_from_a[:, :3, 3]  # [P, 3]

    # Translation predicted: distance to the epipolar line t × R·ray_a in image b
    line = torch.linalg.cross(t.expand_as(rotated), rotated)
    epipolar = (rays_b * line).sum(-1).abs() / line[..., :2].norm(dim=-1) * focal
    # Pure rotation predicted: the point must land where the rotation takes it
    landed = rotated[..., :2] / rotated[..., 2:]
    rotation = (landed - rays_b[..., :2] / rays_b[..., 2:]).norm(dim=-1) * focal
    still_camera = t.norm(dim=-1) < TRANSLATION_EPS
    distance = torch.where(still_camera, rotation, epipolar)

    seen = ~distance.isnan()
    residual = distance.nanmedian(dim=0).values
    residual[seen.sum(0) < MIN_POINTS] = math.nan

    moving = torch.zeros(frames, dtype=torch.bool, device=device)
    for a in (residual > threshold).nonzero()[:, 0].tolist():
        moving[a : a + gap + 1] = True
    return BaseMotion(start, residual, moving, _spans(~moving))


def _rays(pixels: Tensor[[int, int, 2]], k: Tensor[[3, 3]]) -> Tensor[[int, int, 3]]:
    homogeneous = torch.cat((pixels, torch.ones_like(pixels[..., :1])), -1)
    return homogeneous @ torch.linalg.inv(k).T


def _spans(still: Tensor[[int]]) -> tuple[tuple[int, int], ...]:
    spans, first = [], None
    for f, value in enumerate([*still.tolist(), False]):
        if value and first is None:
            first = f
        elif not value and first is not None:
            if f - first >= MIN_SPAN:
                spans.append((first, f))
            first = None
    return tuple(spans)
