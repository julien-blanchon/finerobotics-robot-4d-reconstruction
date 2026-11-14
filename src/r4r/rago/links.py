"""Evidence about which body a camera rides on, from its KLT tracks and the kinematics.

A tool for whoever writes a dataset's priors (a person or an agent), not a decision:
- a static camera sees mostly still points (the background) between frames;
- a camera rigid on body L turns like L: with A the motion of L and B the camera's between
  two frames, A = X B X⁻¹ for the unknown mount X, so log R_A = R_X log R_B. Per body, the
  best rotation R_X (Park & Martin) leaves a residual; small means consistent.

Bodies moving together (hand, flange, prismatic fingers) tie and are reported as one group.
On real video the margin between groups can be thin: use it to shortlist bodies for `fit`.

Ref: Park & Martin, *Robot sensor calibration: solving AX = XB on the Euclidean group*, 1994.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import einops
import torch

from r4r.geometry import alignment, se3
from r4r.geometry.epipolar import relative_rotations

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.types import UV, XYZ, Mat3x3, Mat4x4

    from .klt import VideoTracks

GAPS = (4, 8, 16)  # frame gaps of the pairs
PAIR_STEP = 3  # first frames of pairs, every this many frames
MIN_POINTS = 30
MAX_POINTS = 512
STILL_PIXELS = 1.0  # a point "stays put" between two frames
MOVED_PIXELS = 2.0  # below: rigid with the camera (its own gripper), not used for its motion
MIN_ANGLE = math.radians(2)  # pairs where the camera barely turns carry no information
FOCAL_FACTORS = torch.exp(torch.linspace(math.log(0.5), math.log(1.5), 9))
TIE = 1e-4  # residuals closer than this (rad) belong to bodies moving together


@dataclass(frozen=True, slots=True)
class BodyEvidence:
    # Share of tracked points still between frames (≳ 0.7 static, ≲ 0.35 moving)
    still_share: float
    # Body groups by increasing residual (deg): each group moves as one body
    groups: tuple[tuple[int, ...], ...]
    residual: tuple[float, ...]


def body_evidence[T: IntVar, L: IntVar](
    video: VideoTracks[int],
    world_from_body: Tensor[[T, L, *Mat4x4]],
    intrinsics: Tensor[[*Mat3x3]],
) -> BodyEvidence:
    """Still share and body ranking of one camera (focal searched around `intrinsics`)."""

    frames = len(world_from_body)
    _, track = torch.unique(video.track, return_inverse=True)
    table = video.pixel.new_full((int(track.max()) + 1, frames, 2), math.nan).double()
    table[track, video.frame] = video.pixel.double()

    still, pairs, times = [], [], []
    for gap in GAPS:
        for a in range(0, frames - gap, PAIR_STEP):
            b = a + gap
            seen = ~table[:, a, 0].isnan() & ~table[:, b, 0].isnan()
            if int(seen.sum()) < MIN_POINTS:
                continue
            shift = (table[seen, a] - table[seen, b]).norm(dim=-1)
            still.append(float((shift < STILL_PIXELS).double().mean()))
            moved = seen.clone()
            moved[seen] = shift > MOVED_PIXELS
            if int(moved.sum()) >= MIN_POINTS:
                pairs.append((table[moved, a], table[moved, b]))
                times.append((a, b))
    still_share = float(torch.tensor(still).median()) if still else math.nan
    if not pairs:
        return BodyEvidence(still_share, (), ())

    pixels_a, pixels_b, mask = _pad(pairs, intrinsics.device)
    start, end = torch.tensor(times).T
    # Body motion b_from_a in its own frame, like the camera rotations below
    motion = se3.inverse(se3.inverse(world_from_body[start]) @ world_from_body[end])
    body_log = se3.so3_log(motion[..., :3, :3].double().cpu())  # [M, L, 3]
    best = None
    for factor in FOCAL_FACTORS.tolist():
        k = intrinsics.double().clone()
        k[:2, :2] *= factor
        rotation = relative_rotations(pixels_a, pixels_b, mask, k)
        camera_log = se3.so3_log(rotation.cpu())
        residual = _residuals(camera_log, body_log)
        if best is None or float(residual.min()) < float(best.min()):
            best = residual
    assert best is not None
    order = best.argsort().tolist()
    groups, values = [], []
    for body in order:
        value = float(best[body])
        if values and abs(value - values[-1]) < TIE:
            groups[-1].append(body)
        else:
            groups.append([body])
            values.append(value)
    return BodyEvidence(
        still_share, tuple(tuple(g) for g in groups), tuple(math.degrees(v) for v in values)
    )


def _residuals[M: IntVar, L: IntVar](
    camera_log: Tensor[[M, *XYZ]], body_log: Tensor[[M, L, *XYZ]]
) -> Tensor[[L]]:
    """Per body: `median_m ‖R_X a_m − b_m‖` (rad) over frame pairs m, with `a = log R_camera`,
    `b = log R_body` and the best `R_X = argmax_R Σ_m b_mᵀ R a_m` (closest rotation)."""

    keep = camera_log.norm(dim=-1) > MIN_ANGLE
    camera_log, body_log = camera_log[keep], body_log[keep]
    covariance = einops.einsum(camera_log, body_log, "m i, m l j -> l i j")
    rotation = alignment.closest_rotation(covariance)
    predicted = einops.einsum(rotation, camera_log, "l i j, m j -> m l i")
    return (predicted - body_log).norm(dim=-1).median(0).values


def _pad(
    pairs: list[tuple[Tensor[[int, *UV]], Tensor[[int, *UV]]]], device: torch.device
) -> tuple[Tensor[[int, int, *UV]], Tensor[[int, int, *UV]], Tensor[[int, int]]]:
    """Pairs of matched pixels `[M, N]`, subsampled to N ≤ `MAX_POINTS` and padded (mask)."""

    generator = torch.Generator().manual_seed(0)
    n = min(MAX_POINTS, max(len(a) for a, _ in pairs))
    pixels_a = torch.zeros(len(pairs), n, 2, dtype=torch.float64)
    pixels_b = torch.zeros_like(pixels_a)
    mask = torch.zeros(len(pairs), n, dtype=torch.bool)
    for i, (a, b) in enumerate(pairs):
        if len(a) > n:
            keep = torch.randperm(len(a), generator=generator)[:n]
            a, b = a[keep], b[keep]
        pixels_a[i, : len(a)], pixels_b[i, : len(b)], mask[i, : len(a)] = a, b, True
    return pixels_a.to(device), pixels_b.to(device), mask.to(device)
