"""Setup calibration of a whole rig, in stages robust to coupled unknowns.

1. Each moving camera's mount from its own video (`calibrate.fit_setup`: per-episode fits,
   consensus, pooled fit). A mount does not depend on any body above it on its chain (a base
   error moves the camera's whole trajectory rigidly, and the static scene with it), so it is
   solved first and frozen. Mounts the priors already know are kept.
2. Per episode: KLT tracks of every camera, merged into multi-camera tracks through RoMa v2
   links between cameras at the same instants.
3. Static cameras: PnP on points triangulated from the frozen moving cameras, per episode.
4. One bundle over all episodes: free bodies (arm bases, at their scope) and static cameras
   (at theirs) free; moving mounts frozen.
5. The same bundle per group of episodes: their agreement is the check on dataset-scope
   unknowns (residuals alone do not reveal a poorly determined one).

Ref: docs/architecture.md, "Solving coupled unknowns in stages".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch

from r4r.geometry import se3

from .bundle import BundleConfig, BundleResult, bundle_adjust
from .calibrate import FitConfig, fit_setup, video_tracks
from .cross import cross_matches
from .initialize import locate_camera, triangulate
from .kinematics import forward_kinematics
from .klt import klt_tracks
from .merge import merge_tracks
from .tracks import Tracks

if TYPE_CHECKING:
    from torch import Tensor

    from r4r.matching import RoMaV2

    from .robot import Robot

# A static camera needs this many points seen by moving cameras (≥ 3 observations each)
MIN_POINTS = 50
MIN_OBSERVATIONS = 3


@dataclass(frozen=True, slots=True)
class RigEpisode:
    qpos: Tensor[[int, int]]
    frames: dict[str, Tensor[[int, 3, int, int]]]  # uint8 videos, one per camera


@dataclass(frozen=True, slots=True, kw_only=True)
class RigConfig:
    fit: FitConfig = field(default_factory=FitConfig)
    # RoMa links between every pair of cameras at every `link_every`-th timestep
    link_every: int = 10
    bundle: BundleConfig = field(
        default_factory=lambda: BundleConfig(focal_std=0.2, attach_links=True, iterations=1200)
    )
    groups: int = 2


@dataclass(frozen=True, slots=True)
class RigFit:
    # `body_from_camera` per camera (moving: its mount; static: its first episode's pose)
    mounts: Tensor[[int, 4, 4]]
    intrinsics: Tensor[[int, 3, 3]]
    # Free bodies' corrected `world_from_body` at the first timestep of every episode [E, B]
    bodies: Tensor[[int, int, 4, 4]]
    # Static cameras' poses per episode [E, C] (moving cameras: their mount)
    camera_poses: Tensor[[int, int, 4, 4]]
    # The free bodies' corrections (description's offset → fitted: `parent_from_body @ this`),
    # from all episodes [B], and per group of episodes [G, B]
    body_corrections: Tensor[[int, 4, 4]]
    group_corrections: Tensor[[int, int, 4, 4]]
    median_error: float
    inlier_fraction: float


def calibrate_rig(
    robot: Robot,
    episodes: list[RigEpisode],
    names: tuple[str, ...],
    intrinsics: Tensor[[int, 3, 3]],
    known_focal: tuple[bool, ...],
    known_mount: tuple[bool, ...],
    model: RoMaV2,
    config: RigConfig,
) -> RigFit:
    """Calibrate the free mounts, free bodies and static cameras of `robot` (see module doc).

    `intrinsics` are the focal priors (or guesses where `known_focal` is False); cameras on
    the world body are static, the others moving. Moving cameras with `known_mount` keep the
    description's mount (no search).
    """

    index = [robot.camera(n) for n in names]
    static = [robot.camera_body[i] == 0 for i in index]
    mounts, intrinsics = _moving_mounts(
        robot, episodes, names, intrinsics, known_focal, known_mount, config
    )
    merged = [_multi_camera_tracks(e, names, intrinsics, model, config) for e in episodes]
    starts, intrinsics = _static_starts(
        robot, episodes, merged, names, mounts, intrinsics, known_focal
    )

    # 4-5. Bundles: all episodes, then groups; moving mounts frozen
    frozen = replace(
        robot,
        free_cameras={
            n: f for n, f in robot.free_cameras.items() if n in names and static[names.index(n)]
        },
        body_from_camera=robot.body_from_camera.to(mounts).clone(),
    )
    for c, i in enumerate(index):
        frozen.body_from_camera[i] = starts[0, c]
    everything = _bundle(frozen, episodes, merged, names, intrinsics, starts, config)
    groups = [list(range(g, len(episodes), config.groups)) for g in range(config.groups)]
    corrections = []
    for group in filter(None, groups):
        fit = _bundle(
            frozen,
            [episodes[i] for i in group],
            [merged[i] for i in group],
            names,
            intrinsics,
            starts[group],
            config,
        )
        corrections.append(_body_corrections(frozen, episodes[group[0]].qpos, fit.world_from_body))
    first = torch.cumsum(torch.tensor([0] + [len(e.qpos) for e in episodes[:-1]]), 0).tolist()
    free = sorted(robot.body(n) for n in robot.free_bodies)
    camera_poses = everything.mounts_at[:, first].transpose(0, 1)
    return RigFit(
        mounts=everything.body_from_camera,
        intrinsics=everything.intrinsics,
        bodies=everything.world_from_body[first][:, free],
        body_corrections=_body_corrections(frozen, episodes[0].qpos, everything.world_from_body),
        camera_poses=camera_poses,
        group_corrections=torch.stack(corrections),
        median_error=everything.median_error,
        inlier_fraction=everything.inlier_fraction,
    )


def _moving_mounts(
    robot: Robot,
    episodes: list[RigEpisode],
    names: tuple[str, ...],
    intrinsics: Tensor[[int, 3, 3]],
    known_focal: tuple[bool, ...],
    known_mount: tuple[bool, ...],
    config: RigConfig,
) -> tuple[Tensor[[int, 4, 4]], Tensor[[int, 3, 3]]]:
    """Stage 1: each free moving camera's mount (and focal) from its own video alone."""

    index = [robot.camera(n) for n in names]
    mounts = robot.body_from_camera[index].to(episodes[0].qpos).clone()
    intrinsics = intrinsics.clone()
    for c, name in enumerate(names):
        body = robot.camera_body[index[c]]
        if body == 0 or known_mount[c] or name not in robot.free_cameras:
            continue
        data = [
            (e.qpos, video_tracks(e.frames[name], intrinsics[c], config.fit.klt)) for e in episodes
        ]
        alone = replace(robot, free_cameras={name: robot.free_cameras[name]}, free_bodies={})
        focal_range = 0.0 if known_focal[c] else None
        setup = fit_setup(alone, data, name, (body,), intrinsics[c], config.fit, focal_range)
        mounts[c], intrinsics[c] = setup.fit.body_from_camera, setup.fit.intrinsics
    return mounts, intrinsics


def _multi_camera_tracks(
    episode: RigEpisode,
    names: tuple[str, ...],
    intrinsics: Tensor[[int, 3, 3]],
    model: RoMaV2,
    config: RigConfig,
) -> Tracks:
    """Stage 2: KLT tracks of every camera, merged through RoMa links at the same instants."""

    device = episode.qpos.device
    steps = len(episode.qpos)
    view_camera = torch.arange(len(names), device=device).repeat_interleave(steps)
    view_time = torch.arange(steps, device=device).repeat(len(names))
    videos = [episode.frames[n] for n in names]
    tracks = klt_tracks(videos, view_camera, view_time, intrinsics, config.fit.klt)
    pairs = [
        (a, t, b, t)
        for t in range(0, steps, config.link_every)
        for a in range(len(names))
        for b in range(a + 1, len(names))
    ]
    found = cross_matches(model, videos, pairs)
    return merge_tracks(
        tracks,
        found.camera_a * steps + found.time_a,
        found.pixel_a.to(tracks.pixel),
        found.camera_b * steps + found.time_b,
        found.pixel_b.to(tracks.pixel),
    )


def _static_starts(
    robot: Robot,
    episodes: list[RigEpisode],
    merged: list[Tracks],
    names: tuple[str, ...],
    mounts: Tensor[[int, 4, 4]],
    intrinsics: Tensor[[int, 3, 3]],
    known_focal: tuple[bool, ...],
) -> tuple[Tensor[[int, int, 4, 4]], Tensor[[int, 3, 3]]]:
    """Stage 3: each static camera's pose per episode (and focal, from the first episode where
    it is found, when unknown) by PnP on points the moving cameras triangulate. Episodes where
    PnP fails keep the description's pose."""

    device = mounts.device
    index = [robot.camera(n) for n in names]
    body = torch.tensor([robot.camera_body[i] for i in index], device=device)
    static = body == 0
    starts = mounts[None].repeat(len(episodes), 1, 1, 1)  # [E, C, 4, 4]
    focal = intrinsics[:, 0, 0].clone()
    found = [False] * len(names)
    for i, (e, tracks) in enumerate(zip(episodes, merged, strict=True)):
        world = forward_kinematics(robot, e.qpos)
        cam = tracks.view_camera
        world_from_view = world[tracks.view_time, body[cam]] @ mounts[cam]
        observed = tracks.view_camera[tracks.view]
        points = triangulate(tracks, world_from_view, ~static[observed])
        for c in static.nonzero()[:, 0].tolist():
            seen = (observed == c) & (points.count[tracks.track] >= MIN_OBSERVATIONS)
            if int(seen.sum()) < MIN_POINTS:
                continue
            size = tuple(e.frames[names[c]].shape[-2:])
            prior = float(focal[c]) if known_focal[c] or found[c] else None
            start = locate_camera(points.point[tracks.track[seen]], tracks.pixel[seen], size, prior)
            if start is not None:
                starts[i, c] = start.world_from_camera
                focal[c], found[c] = start.focal, True
    intrinsics = intrinsics.clone()
    intrinsics[:, :2, :2] *= (focal / intrinsics[:, 0, 0])[:, None, None]
    return starts, intrinsics


def _bundle(
    robot: Robot,
    episodes: list[RigEpisode],
    merged: list[Tracks],
    names: tuple[str, ...],
    intrinsics: Tensor[[int, 3, 3]],
    starts: Tensor[[int, int, 4, 4]],
    config: RigConfig,
) -> BundleResult:
    """One bundle over `episodes`: their tracks pooled (distinct ids and timesteps)."""

    steps = [len(e.qpos) for e in episodes]
    total = sum(steps)
    device = episodes[0].qpos.device
    track_offset, time_offset = 0, 0
    track, view, pixel, weight, moving = [], [], [], [], []
    for tracks, n in zip(merged, steps, strict=True):
        cam, time = tracks.view_camera[tracks.view], tracks.view_time[tracks.view]
        track.append(tracks.track + track_offset)
        view.append(cam * total + time + time_offset)
        pixel.append(tracks.pixel)
        weight.append(tracks.weight)
        moving.append(tracks.moving)
        track_offset += int(tracks.track.max()) + 1
        time_offset += n
    pooled = Tracks(
        view_camera=torch.arange(len(names), device=device).repeat_interleave(total),
        view_time=torch.arange(total, device=device).repeat(len(names)),
        track=torch.cat(track),
        view=torch.cat(view),
        pixel=torch.cat(pixel),
        weight=torch.cat(weight),
        moving=torch.cat(moving),
        intrinsics=intrinsics,
    )
    episode = torch.cat([torch.full((n,), i, device=device) for i, n in enumerate(steps)])
    start_at = torch.cat(
        [starts[i][:, None].expand(-1, n, 4, 4) for i, n in enumerate(steps)], 1
    )  # [C, T, 4, 4]
    qpos = torch.cat([e.qpos for e in episodes])
    return bundle_adjust(
        robot, qpos, pooled, names, starts[0], config.bundle, episode=episode, start_at=start_at
    )


def _body_corrections(
    robot: Robot, qpos: Tensor[[int, int]], corrected: Tensor[[int, int, 4, 4]]
) -> Tensor[[int, 4, 4]]:
    """Each free body's correction at the first timestep (description → fitted): since a
    correction moves the body's subtree as `parent_from_body @ E`, `E` is the nominal pose's
    inverse times the corrected one."""

    free = sorted(robot.body(n) for n in robot.free_bodies)
    nominal = forward_kinematics(robot, qpos[:1])[0]
    return se3.inverse(nominal[free]) @ corrected[0, free]


def spread(corrections: Tensor[[int, int, 4, 4]]) -> list[tuple[float, float]]:
    """Per free body: largest disagreement between groups (cm, degrees)."""

    out = []
    for b in range(corrections.shape[1]):
        poses = corrections[:, b]
        delta = se3.inverse(poses)[:, None] @ poses[None]
        out.append(
            (
                float(delta[..., :3, 3].norm(dim=-1).max()) * 100,
                math.degrees(float(se3.so3_log(delta[..., :3, :3]).norm(dim=-1).max())),
            )
        )
    return out
