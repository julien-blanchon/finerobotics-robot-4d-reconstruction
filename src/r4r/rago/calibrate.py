"""Calibrate one moving camera: its body among candidates, its mount and its focal length.

One mount and one focal length explain all the episodes given (a setup): pooling episodes
adds motion, which pins down what a single short episode leaves loose (the mount's
translation, mostly). Each episode keeps its own tracks (its own scene).

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
    # Video/joint-state offsets tried (frames; state of frame t taken from t + shift)
    shifts: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3)
    shift_iterations: int = 100


@dataclass(frozen=True, slots=True)
class MountFit:
    body: int
    body_from_camera: Tensor[[4, 4]]
    intrinsics: Tensor[[3, 3]]
    # Robust score (dropped tracks at their capped static-world fit), median reprojection (px)
    score: float
    median_error: float
    inlier_fraction: float
    # Best search score per candidate body (same order as `bodies`)
    body_scores: Tensor[[int]]
    # Video/joint-state offset found (frames) and the score of each tried
    shift: int
    shift_scores: Tensor[[int]]


def fit_moving_camera(
    robot: Robot,
    episodes: list[tuple[Tensor[[int, int]], Tensor[[int, 3, int, int]]]],
    camera: str,
    bodies: tuple[int, ...],
    intrinsics: Tensor[[3, 3]],
    config: FitConfig,
) -> MountFit:
    """Mount and focal of `camera` (a free camera of `robot`) riding on one of `bodies`.

    `episodes` are (joint positions, video frames) pairs of the same setup. `intrinsics` is
    the initial guess (the search tries focal factors around it).
    """

    qpos = torch.cat([q for q, _ in episodes])
    device = qpos.device
    tracks, offset, first = [], 0, 0
    for q, frames in episodes:
        steps = len(q)
        found = klt_tracks(
            [frames],
            torch.zeros(steps, dtype=torch.long, device=device),
            torch.arange(steps, device=device),
            intrinsics.to(qpos)[None],
            config.klt,
        )
        tracks.append(replace(found, track=found.track + offset, view=found.view + first))
        offset += int(found.track.max()) + 1
        first += steps
    tracks = replace(
        tracks[0],
        view_camera=torch.zeros(len(qpos), dtype=torch.long, device=device),
        view_time=torch.arange(len(qpos), device=device),
        track=torch.cat([t.track for t in tracks]),
        view=torch.cat([t.view for t in tracks]),
        pixel=torch.cat([t.pixel for t in tracks]),
        weight=torch.cat([t.weight for t in tracks]),
        moving=torch.cat([t.moving for t in tracks]),
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
    unit = intrinsics[0, 0].to(qpos)[None]  # the search's unit: scores compare with it
    found_tracks = replace(tracks, intrinsics=found.intrinsics)
    result = bundle_adjust(
        on_body, qpos, found_tracks, names, found.body_from_camera, config.bundle, unit
    )

    # Time offset: short bundles from the solution with shifted joint states, then refine
    short = replace(config.bundle, iterations=config.shift_iterations)
    refined = replace(tracks, intrinsics=result.intrinsics)
    shift_scores = torch.tensor(
        [
            bundle_adjust(
                on_body, shifted(episodes, s), refined, names, result.body_from_camera, short, unit
            ).score
            for s in config.shifts
        ]
    )
    shift = config.shifts[int(shift_scores.argmin())]
    if shift != 0:
        result = bundle_adjust(
            on_body,
            shifted(episodes, shift),
            refined,
            names,
            result.body_from_camera,
            config.bundle,
            unit,
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
        shift=shift,
        shift_scores=shift_scores,
    )


def shifted(
    episodes: list[tuple[Tensor[[int, int]], Tensor[[int, 3, int, int]]]], shift: int
) -> Tensor[[int, int]]:
    """Concatenated joint positions, each episode's frame t taken from t + shift (clamped)."""

    parts = []
    for q, _ in episodes:
        index = (torch.arange(len(q), device=q.device) + shift).clamp(0, len(q) - 1)
        parts.append(q[index])
    return torch.cat(parts)
