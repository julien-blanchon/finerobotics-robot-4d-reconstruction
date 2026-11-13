"""Kinematic bundle adjustment: camera poses from forward kinematics, points from triangulation.

Every view's pose is `world_from_body(q_t) · mount` (hard constraint, metric by construction).
Each static track's 3D point is triangulated in closed form from its observations' rays under
the current poses (weighted least squares on the rays: no point parameters, differentiable),
then reprojected into every view that saw it. We minimise a robust reprojection error over
the free mounts and the focal lengths. ARROW contributes only where points appear in the
images; scale comes from the robot's metric motion, the field of view from the known motion.

Ref: Triggs et al., Bundle Adjustment — A Modern Synthesis, 1999 (variable projection);
     Calib3R arXiv:2509.08813 (joint hand-eye + reconstruction).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from r4r.geometry import camera, se3

from .losses import cauchy
from .rig import Rig

if TYPE_CHECKING:
    from torch import Tensor

    from .robot import Robot
    from .tracks import Tracks

# Observations closer than this (m) to their camera's plane are not reprojected
MIN_DEPTH = 0.05
# Ridge on the triangulation normal equations (m⁻²·weight), against parallel rays
RIDGE = 1e-6


@dataclass(frozen=True, slots=True, kw_only=True)
class BundleConfig:
    iterations: int = 800
    learning_rate: float = 5e-3
    # Robust scale of the reprojection error, in pixels / focal length (0.01 ≈ 2–3 px)
    cauchy: float = 0.01
    # Focal-length prior (log space): ~0.3 from ARROW, ~0.01 when the dataset knows it
    focal_std: float = 0.3


@dataclass(frozen=True, slots=True)
class BundleResult:
    body_from_camera: Tensor[[int, 4, 4]]
    intrinsics: Tensor[[int, 3, 3]]
    world_from_view: Tensor[[int, 4, 4]]
    # Per iteration: total, reprojection, prior
    history: Tensor[[int, 3]]
    # Median reprojection error at the end (pixels), and the fraction of inlier observations
    median_error: float
    inlier_fraction: float


def bundle_adjust(
    robot: Robot,
    qpos: Tensor[[int, int]],
    tracks: Tracks,
    names: tuple[str, ...],
    initial_mounts: Tensor[[int, 4, 4]],
    config: BundleConfig,
) -> BundleResult:
    """Refine free mounts and focal lengths by robust reprojection of triangulated tracks."""

    t = tracks
    _, track = torch.unique(t.track, return_inverse=True)
    count = int(track.max()) + 1
    scales = qpos.new_ones(1)  # unused: the bundle never touches ARROW's depth
    rig = Rig(
        robot, qpos, names, initial_mounts.to(qpos), scales, t.intrinsics.to(qpos), config.focal_std
    )
    pixel, weight = t.pixel.to(qpos), t.weight.to(qpos)

    optimizer = torch.optim.Adam(rig.parameters(), lr=config.learning_rate)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 0.5 * (1 + math.cos(math.pi * step / config.iterations))
    )
    history = []
    for _ in range(config.iterations):
        error, valid = reprojection(rig, t, track, count, pixel, weight)
        data = (cauchy(error, config.cauchy) * weight * valid).sum() / (weight * valid).sum()
        prior = rig.prior() / len(t.view_time)
        loss = data + prior
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        history.append(torch.stack((loss, data, prior)).detach())

    with torch.no_grad():
        error, valid = reprojection(rig, t, track, count, pixel, weight)
        focal = rig.intrinsics()[t.view_camera[t.view], 0, 0]
        pixels = (error * focal)[valid]
        return BundleResult(
            body_from_camera=rig.mounts(),
            intrinsics=rig.intrinsics(),
            world_from_view=rig.camera_poses(t.view_camera, t.view_time),
            history=torch.stack(history),
            median_error=float(pixels.median()),
            inlier_fraction=float((error[valid] < 3 * config.cauchy).float().mean()),
        )


def reprojection(
    rig: Rig,
    tracks: Tracks,
    track: Tensor[[int]],
    count: int,
    pixel: Tensor[[int, 2]],
    weight: Tensor[[int]],
) -> tuple[Tensor[[int]], Tensor[[int]]]:
    """Per observation: reprojection error (pixels / focal) and whether it is usable."""

    t = tracks
    world_from_view = rig.camera_poses(t.view_camera, t.view_time)
    intrinsics = rig.intrinsics()

    # Gather per-observation pose and intrinsics through embedding (sorted backward)
    table = torch.cat((world_from_view.flatten(1), intrinsics[t.view_camera].flatten(1)), -1)
    rows = F.embedding(t.view, table)
    pose, k = rows[:, :16].unflatten(-1, (4, 4)), rows[:, 16:].unflatten(-1, (3, 3))

    # Rays: centre c and unit direction d in the world
    local = camera.backproject(k, pixel[:, None], torch.ones_like(pixel[:, :1]))[:, 0]
    direction = F.normalize((pose[:, :3, :3] @ local[:, :, None])[..., 0], dim=-1)
    centre = pose[:, :3, 3]

    # Weighted least-squares triangulation per track: Σ w (I − d dᵀ) X = Σ w (I − d dᵀ) c
    eye = torch.eye(3, dtype=pixel.dtype, device=pixel.device)
    normal = weight[:, None, None] * (eye - direction[:, :, None] * direction[:, None, :])
    lhs = torch.zeros(count, 3, 3, dtype=pixel.dtype, device=pixel.device).index_add(
        0, track, normal
    )
    rhs = torch.zeros(count, 3, dtype=pixel.dtype, device=pixel.device).index_add(
        0, track, (normal @ centre[:, :, None])[..., 0]
    )
    point = torch.linalg.solve(lhs + RIDGE * eye, rhs)

    # Reproject each track's point into each view that saw it
    in_view = se3.transform_points(se3.inverse(pose), point[track][:, None])[:, 0]
    projected = camera.project(k, in_view[:, None])[:, 0]
    error = (projected - pixel).norm(dim=-1) / k[:, 0, 0]
    valid = in_view[:, 2] > MIN_DEPTH
    return torch.where(valid, error, 0.0), valid
