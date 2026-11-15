"""One static camera of a new station (setup), calibrated on the CPU from a neighbour station.

A dataset recorded in many sessions (openneodata arx5: ~420 stations) often changes only its
third-person camera between them: the arms, the wrist mounts and the second arm's base stay.
Instead of a full `r4r rig` per station, the camera's pose and focal are fitted with
everything else held at a calibrated neighbour station's values (the previous one in episode
order: its camera is the start, so the solve stays local):

1. Bursts of frames from a few episodes of the station (`light.burst_steps`), decoded alone;
   KLT within each burst, SIFT links between the static camera and the wrist cameras at a few
   moments of each burst (`light.burst_links`, `light.burst_tracks`), the episodes' bursts
   side by side as one sequence (the camera is fixed over the station).
2. Levenberg–Marquardt on the camera's 6-DoF pose (dataset scope, Gaussian prior around the
   neighbour's pose, `ranges`) and its focal (log-normal prior, `focal_std`): tracks on the
   robot pin it through the held wrist mounts and arm kinematics (`light.light_solve`).

A static camera's focal and distance trade off (a far camera with a long focal sees the robot
alike): the focal prior keeps it near the neighbour's unless the tracks say otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch
from einops import repeat

from . import light
from .robot import Freedom

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from torch import Tensor

    from r4r.types import Mat3x3, Mat4x4

    from r4r.matching import RoMaV2

    from .lens import Lens
    from .robot import Robot
    from .tracks import Tracks
    from .stages import RigEpisode


@dataclass(frozen=True, slots=True, kw_only=True)
class StationConfig:
    light: light.LightConfig = field(default_factory=lambda: light.LightConfig(bursts=4))
    # Prior std of the camera's departure from the neighbour's pose (m, degrees) and of its
    # log focal
    position_range: float = 0.3
    orientation_range_deg: float = 15.0
    focal_std: float = 0.3
    # RoMa matches kept per pair of frames when a matcher is given
    dense: int = 400


@dataclass(frozen=True, slots=True)
class StationFit:
    """The camera's `base_from_camera` and focal (pixels), the median reprojection error of the
    tracks after the solve and how many observations it is over."""

    pose: Tensor[[*Mat4x4]]
    focal: float
    error: light.FitCheck
    # Whether PnP gave the start (else the neighbour's pose)
    placed: bool


def samples(episodes: Sequence[int], count: int) -> list[int]:
    """`count` episodes spread evenly over a station's (sorted) `episodes`."""

    if len(episodes) <= count:
        return list(episodes)
    return [episodes[round(i * (len(episodes) - 1) / (count - 1))] for i in range(count)]


def calibrate_station(
    root: Path,
    numbers: Sequence[int],
    robot: Robot[int, int, int],
    episodes: Sequence[RigEpisode[int, int]],
    names: tuple[str, ...],
    intrinsics: Tensor[[int, *Mat3x3]],
    lens: Lens[int] | None,
    camera: str,
    stride: int,
    config: StationConfig,
    matcher: RoMaV2 | None = None,
) -> StationFit:
    """Fit `camera` (static) of the station of episodes `numbers` (`episodes`, loaded with the
    neighbour's values as the station's: `robot.body_from_camera` is the start) on bursts of
    those episodes; every other value held (module doc)."""

    from r4r.data.lerobot import keyframes

    videos: dict[str, list[Tensor]] = {n: [] for n in names}
    sizes, qpos = [], []
    for number, episode in zip(numbers, episodes, strict=True):
        bursts = light.burst_steps(len(episode.qpos), config.light)
        steps = [t for b in bursts for t in b]
        frames = keyframes(root, number, [t * stride for t in steps], names)
        for n in names:
            videos[n].append(frames[n])
        sizes += [len(b) for b in bursts]
        qpos.append(episode.qpos[steps])
    stacked = [torch.cat(videos[n]) for n in names]
    static = [robot.camera_body[robot.camera(n)] == 0 for n in names]
    links = light.burst_links(stacked, sizes, intrinsics, lens, static, config.light)
    if matcher is not None:
        links = _with_dense(links, matcher, stacked, sizes, names.index(camera), config)
    tracks = light.burst_tracks(stacked, sizes, intrinsics, lens, links, config.light)
    joints = torch.cat(qpos)
    mounts = robot.body_from_camera[[robot.camera(n) for n in names]].to(joints)
    c = names.index(camera)
    intrinsics = intrinsics.clone()
    start = _pnp_start(robot, joints, stacked, tracks, names, mounts, intrinsics, camera)
    if start is not None:
        mounts[c] = start[0]
        intrinsics[c, :2, :2] *= start[1] / intrinsics[c, 0, 0]
        tracks = replace(tracks, intrinsics=intrinsics)
    tracks = light.informative(tracks, torch.tensor(static), False, config.light)
    freedom = Freedom(
        config.position_range, math.radians(config.orientation_range_deg), "dataset"
    )
    solving = replace(light.held_robot(robot, ()), free_cameras={camera: freedom})
    at = repeat(mounts, "c i j -> c t i j", t=len(joints))
    focal_std = torch.zeros(len(names), dtype=joints.dtype)
    focal_std[c] = config.focal_std
    rig, fit = light.light_solve(
        solving, joints, tracks, names, at, None, config.light, focal_std
    )
    with torch.no_grad():
        index = torch.tensor([c])
        pose = rig.mounts_at(index, torch.zeros_like(index))[0]
        focal = float(rig.intrinsics()[c, 0, 0])
    return StationFit(pose.detach(), focal, fit, start is not None)


def _pnp_start(
    robot: Robot[int, int, int],
    joints: Tensor[[int, int]],
    frames: Sequence[Tensor],
    tracks: Tracks[int, int, int],
    names: tuple[str, ...],
    mounts: Tensor[[int, *Mat4x4]],
    intrinsics: Tensor[[int, *Mat3x3]],
    camera: str,
) -> tuple[Tensor[[*Mat4x4]], float] | None:
    """The camera's pose and focal by PnP on the points the wrist cameras triangulate and the
    links see in it, its focal searched (`stages.static_starts`); None when too few."""

    from .stages import RigEpisode, static_starts

    searched = replace(robot, free_cameras={camera: Freedom(math.inf, math.inf, "dataset")})
    episode = RigEpisode(joints, dict(zip(names, frames, strict=True)))
    known = tuple(n != camera for n in names)
    starts, found = static_starts(
        searched, [episode], [tracks], names, mounts, intrinsics, known, known
    )
    c = names.index(camera)
    if bool(torch.equal(starts[0, c], mounts[c])):
        return None
    return starts[0, c], float(found[c, 0, 0])


def _with_dense(
    links: light.Links[int],
    matcher: RoMaV2,
    frames: Sequence[Tensor],
    sizes: Sequence[int],
    camera: int,
    config: StationConfig,
) -> light.Links[int]:
    """`links` and RoMa v2 matches between `camera` and every other camera at each burst's
    middle frame (certain ones only, `cross.cross_matches`): a few dozen pairs per station."""

    from .cross import cross_matches

    count = len(frames[0])
    first = [sum(sizes[:i]) for i in range(len(sizes))]
    moments = [start + size // 2 for start, size in zip(first, sizes, strict=True)]
    others = [c for c in range(len(frames)) if c != camera]
    pairs = [(o, k, camera, k) for k in moments for o in others]
    found = cross_matches(matcher, list(frames), pairs, per_pair=config.dense)
    a = found.camera_a.cpu() * count + found.time_a.cpu()
    b = found.camera_b.cpu() * count + found.time_b.cpu()
    like = links.pixel_a
    return light.Links(
        torch.cat((links.view_a, a)),
        torch.cat((links.pixel_a, found.pixel_a.cpu().to(like))),
        torch.cat((links.view_b, b)),
        torch.cat((links.pixel_b, found.pixel_b.cpu().to(like))),
    )
