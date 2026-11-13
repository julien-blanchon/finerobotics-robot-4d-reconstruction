"""Calibrate one moving camera: its body among candidates, its mount and its focal length.

KLT tracks through the camera's video (sub-pixel on wrist and head videos, where ARROW's
tracks are not), a global search over candidate bodies × the 24 axis-aligned orientations ×
focal factors, then a kinematic bundle against the static world from the best start.
No silhouettes, no robot geometry: the static scene is the target, the kinematics the motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch

from .bundle import BundleConfig, bundle_adjust
from .klt import KLTConfig, klt_tracks
from .search import SearchConfig, search_mount

if TYPE_CHECKING:
    from torch import Tensor

    from .robot import Robot


@dataclass(frozen=True, slots=True, kw_only=True)
class FitConfig:
    klt: KLTConfig = field(default_factory=KLTConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    bundle: BundleConfig = field(default_factory=lambda: BundleConfig(attach_links=False))


@dataclass(frozen=True, slots=True)
class MountFit:
    body: int
    body_from_camera: Tensor[[4, 4]]
    intrinsics: Tensor[[3, 3]]
    # Robust score (dropped observations count as outliers), median reprojection (pixels)
    score: float
    median_error: float
    inlier_fraction: float
    # Best search score per candidate body (same order as `bodies`)
    body_scores: Tensor[[int]]


def fit_moving_camera(
    robot: Robot,
    qpos: Tensor[[int, int]],
    frames: Tensor[[int, 3, int, int]],
    camera: str,
    bodies: tuple[int, ...],
    intrinsics: Tensor[[3, 3]],
    config: FitConfig,
) -> MountFit:
    """Mount and focal of `camera` (a free camera of `robot`) riding on one of `bodies`.

    `intrinsics` is the initial guess (the search tries focal factors around it).
    """

    steps = len(qpos)
    device = qpos.device
    tracks = klt_tracks(
        [frames],
        torch.zeros(steps, dtype=torch.long, device=device),
        torch.arange(steps, device=device),
        intrinsics.to(qpos)[None],
        config.klt,
    )
    names = (camera,)
    found = search_mount(robot, qpos, tracks, names, camera, config.search, bodies)
    index = robot.camera(camera)
    # The camera on the body found, its prior centred on the search's answer
    mounts = robot.body_from_camera.to(qpos).clone()
    mounts[index] = found.body_from_camera[0]
    on_body = replace(
        robot,
        camera_body=tuple(found.body if i == index else b for i, b in enumerate(robot.camera_body)),
        body_from_camera=mounts,
    )
    result = bundle_adjust(
        on_body,
        qpos,
        replace(tracks, intrinsics=found.intrinsics),
        names,
        found.body_from_camera,
        config.bundle,
    )
    per_body = len(found.scores) // len(bodies)
    return MountFit(
        body=found.body,
        body_from_camera=result.body_from_camera[0],
        intrinsics=result.intrinsics[0],
        score=result.score,
        median_error=result.median_error,
        inlier_fraction=result.inlier_fraction,
        body_scores=found.scores.unflatten(0, (len(bodies), per_body)).min(1).values,
    )
