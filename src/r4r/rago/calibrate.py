"""Calibrate one moving camera: its body among candidates, its mount and its focal length.

KLT tracks through the camera's video (sub-pixel on wrist and head videos, where ARROW's
tracks are not), a global search over candidate bodies × the 24 axis-aligned orientations ×
focal factors, then a kinematic bundle against the static world from the best start.
No silhouettes, no robot geometry: the static scene is the target, the kinematics the motion.

A camera's focal length is a dataset prior (fixed lens): given, the search only tries
factors within its range and the bundle keeps it there. Its distortion (`Tracks.lens`) is
refined by the bundle within its prior. A setup (episodes sharing one mount)
is calibrated episode by episode first: the mount most episodes agree with is the
consensus, disagreeing episodes are set aside, and one bundle over the agreeing episodes
(more motion: it pins down what one short episode leaves loose) gives the answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch

from r4r.geometry import se3

from .bundle import BundleConfig, bundle_adjust
from .kinematics import forward_kinematics
from .klt import KLTConfig, klt_tracks
from .search import SearchConfig, search_mount

if TYPE_CHECKING:
    from torch import Tensor

    from .lens import Lens
    from .robot import Robot
    from .tracks import Tracks

# A prior focal range this tight (relative) is a known focal: no focal factors in the search
KNOWN_FOCAL = 0.02


@dataclass(frozen=True, slots=True, kw_only=True)
class FitConfig:
    klt: KLTConfig = field(default_factory=KLTConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    bundle: BundleConfig = field(default_factory=lambda: BundleConfig(attach_links=False))
    # Video/joint-state offsets tried (frames; state of frame t taken from t + shift): a
    # diagnostic for writing a dataset's priors (its `state_shift`), not run by default
    shifts: tuple[int, ...] = (0,)
    shift_iterations: int = 100
    # Episodes agree on a mount within these (m, rad)
    agreement: tuple[float, float] = (0.02, math.radians(2.0))


@dataclass(frozen=True, slots=True)
class MountFit:
    body: int
    body_from_camera: Tensor[[4, 4]]
    intrinsics: Tensor[[3, 3]]
    # Its lens with the fitted distortion (None: an ideal pinhole)
    lens: Lens | None
    # Robust score (dropped tracks at their capped static-world fit), median reprojection (px)
    score: float
    median_error: float
    inlier_fraction: float
    # Best search score per candidate body (same order as `bodies`; empty without a search)
    body_scores: Tensor[[int]]
    # Video/joint-state offset found (frames) and the score of each tried
    shift: int
    shift_scores: Tensor[[int]]


@dataclass(frozen=True, slots=True)
class SetupFit:
    # Each episode on its own, which agree with the consensus, and the pooled fit over those
    episodes: tuple[MountFit, ...]
    agree: Tensor[[int]]
    fit: MountFit


def video_tracks(
    frames: Tensor[[int, 3, int, int]],
    intrinsics: Tensor[[3, 3]],
    config: KLTConfig,
    lens: Lens | None = None,
) -> Tracks:
    """KLT tracks of one camera's video (one view per frame); `lens` of that one camera."""

    steps = len(frames)
    device = intrinsics.device
    return klt_tracks(
        [frames],
        torch.zeros(steps, dtype=torch.long, device=device),
        torch.arange(steps, device=device),
        intrinsics[None],
        config,
        lens,
    )


def fit_moving_camera(
    robot: Robot,
    episodes: list[tuple[Tensor[[int, int]], Tracks]],
    camera: str,
    bodies: tuple[int, ...],
    intrinsics: Tensor[[3, 3]],
    config: FitConfig,
    focal_range: float | None = None,
) -> MountFit:
    """Mount and focal of `camera` (a free camera of `robot`) riding on one of `bodies`.

    `episodes` are (joint positions, `video_tracks`) of the same setup. `intrinsics` is the
    prior focal when `focal_range` (relative) is given, else a guess searched around.
    """

    qpos, tracks, _ = pool(episodes)
    search = config.search
    if focal_range is not None:
        factors = (1.0,) if focal_range <= KNOWN_FOCAL else (1 - focal_range, 1.0, 1 + focal_range)
        search = replace(
            search,
            focal_factors=factors,
            bundle=replace(search.bundle, focal_std=max(focal_range, 1e-3)),
        )
    tracks = replace(tracks, intrinsics=intrinsics.to(qpos)[None])
    found = search_mount(robot, qpos, tracks, (camera,), camera, search, bodies)
    per_body = len(found.scores) // len(bodies)
    fit = refine(
        robot,
        episodes,
        camera,
        found.body,
        found.body_from_camera[0],
        found.intrinsics[0],
        config,
        intrinsics[0, 0],
        focal_range,
    )
    return replace(fit, body_scores=found.scores.unflatten(0, (len(bodies), per_body)).amin(1))


def refine(
    robot: Robot,
    episodes: list[tuple[Tensor[[int, int]], Tracks]],
    camera: str,
    body: int,
    body_from_camera: Tensor[[4, 4]],
    intrinsics: Tensor[[3, 3]],
    config: FitConfig,
    unit: Tensor[[]],
    focal_range: float | None = None,
) -> MountFit:
    """Bundle from a known start (and, when asked, the video/joint-state offset scores).

    `unit` is the pixel unit of scores (the focal guess): fits to compare must share it.
    """

    qpos, tracks, episode = pool(episodes)
    index = robot.camera(camera)
    # The camera on `body`, its prior centred on the start
    mounts = robot.body_from_camera.to(qpos).clone()
    mounts[index] = body_from_camera
    on_body = replace(
        robot,
        camera_body=tuple(body if i == index else b for i, b in enumerate(robot.camera_body)),
        body_from_camera=mounts,
    )
    bundle = config.bundle
    if focal_range is not None:
        bundle = replace(bundle, focal_std=max(focal_range, 1e-3))
    names = (camera,)
    pixel_unit = unit.to(qpos)[None]
    tracks = replace(tracks, intrinsics=intrinsics.to(qpos)[None])
    start = body_from_camera[None]
    result = bundle_adjust(on_body, qpos, tracks, names, start, bundle, pixel_unit, episode)

    short = replace(bundle, iterations=config.shift_iterations)
    refined = replace(tracks, intrinsics=result.intrinsics, lens=result.lens)
    starts = result.body_from_camera
    shift_scores = torch.tensor(
        [
            bundle_adjust(
                on_body, shifted(episodes, s), refined, names, starts, short, pixel_unit
            ).score
            for s in config.shifts
        ]
    )
    shift = config.shifts[int(shift_scores.argmin())]
    if shift != 0:
        result = bundle_adjust(
            on_body, shifted(episodes, shift), refined, names, starts, bundle, pixel_unit, episode
        )
    return MountFit(
        body=body,
        body_from_camera=result.body_from_camera[0],
        intrinsics=result.intrinsics[0],
        lens=result.lens,
        score=result.score,
        median_error=result.median_error,
        inlier_fraction=result.inlier_fraction,
        body_scores=qpos.new_zeros(0),
        shift=shift,
        shift_scores=shift_scores,
    )


def fit_setup(
    robot: Robot,
    episodes: list[tuple[Tensor[[int, int]], Tracks]],
    camera: str,
    bodies: tuple[int, ...],
    intrinsics: Tensor[[3, 3]],
    config: FitConfig,
    focal_range: float | None = None,
) -> SetupFit:
    """One mount for a setup: per-episode fits, their consensus, a pooled fit over agreeing ones."""

    single = [
        fit_moving_camera(robot, [episode], camera, bodies, intrinsics, config, focal_range)
        for episode in episodes
    ]
    # Mounts compared on the first fit's body (rigidly joined bodies give the same camera pose)
    reference = single[0].body
    common = []
    for fit, (qpos, _) in zip(single, episodes, strict=True):
        poses = forward_kinematics(robot, qpos[:1])[0]
        common.append(se3.inverse(poses[reference]) @ poses[fit.body] @ fit.body_from_camera)
    mounts = torch.stack(common)
    delta = se3.inverse(mounts)[:, None] @ mounts[None]
    translation = delta[..., :3, 3].norm(dim=-1)
    rotation = se3.so3_log(delta[..., :3, :3]).norm(dim=-1)
    near = (translation < config.agreement[0]) & (rotation < config.agreement[1])
    near |= torch.eye(len(single), dtype=torch.bool, device=near.device)
    support = near.sum(1).double() - 1e-3 * torch.tensor([f.score for f in single]).to(mounts)
    centre = int(support.argmax())
    agree = near[centre]

    chosen = [e for e, a in zip(episodes, agree.tolist(), strict=True) if a]
    focal = torch.stack(
        [f.intrinsics[0, 0] for f, a in zip(single, agree.tolist(), strict=True) if a]
    )
    start_k = single[centre].intrinsics.clone()
    start_k[:2, :2] *= focal.median() / start_k[0, 0]
    # The distortion starts where the consensus episode's fit left it
    lens = single[centre].lens
    chosen = [(q, replace(t, lens=lens)) for q, t in chosen]
    pooled = refine(
        robot,
        chosen,
        camera,
        reference,
        mounts[centre],
        start_k,
        config,
        intrinsics[0, 0],
        focal_range,
    )
    return SetupFit(tuple(single), agree, pooled)


def pool(
    episodes: list[tuple[Tensor[[int, int]], Tracks]],
) -> tuple[Tensor[[int, int]], Tracks, Tensor[[int]]]:
    """Concatenate episodes: joint positions, tracks with distinct ids and view numbers, and
    the episode of each timestep."""

    qpos = torch.cat([q for q, _ in episodes])
    device = qpos.device
    parts, offset, first = [], 0, 0
    for q, tracks in episodes:
        parts.append(replace(tracks, track=tracks.track + offset, view=tracks.view + first))
        offset += int(tracks.track.max()) + 1
        first += len(q)
    episode = torch.cat(
        [torch.full((len(q),), i, device=device) for i, (q, _) in enumerate(episodes)]
    )
    return (
        qpos,
        replace(
            parts[0],
            view_camera=torch.zeros(len(qpos), dtype=torch.long, device=device),
            view_time=torch.arange(len(qpos), device=device),
            track=torch.cat([t.track for t in parts]),
            view=torch.cat([t.view for t in parts]),
            pixel=torch.cat([t.pixel for t in parts]),
            weight=torch.cat([t.weight for t in parts]),
            moving=torch.cat([t.moving for t in parts]),
        ),
        episode,
    )


def shifted(episodes: list[tuple[Tensor[[int, int]], Tracks]], shift: int) -> Tensor[[int, int]]:
    """Concatenated joint positions, each episode's frame t taken from t + shift (clamped)."""

    parts = []
    for q, _ in episodes:
        index = (torch.arange(len(q), device=q.device) + shift).clamp(0, len(q) - 1)
        parts.append(q[index])
    return torch.cat(parts)
