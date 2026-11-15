"""One static camera of a new station (setup), calibrated from a neighbour station, mostly CPU.

A dataset recorded in many sessions (openneodata arx5: ~420 stations) often changes only its
third-person camera between them: the arms, the wrist mounts and the second arm's base stay.
Instead of a full `r4r rig` per station, the camera's pose and focal are fitted with
everything else held at a calibrated neighbour station's values:

1. Bursts of frames from a few episodes of the station (`light.burst_steps`), decoded alone,
   side by side as one sequence (the camera is fixed over the station); KLT within each burst;
   links between the camera and the wrist cameras at the bursts' middle frames: SIFT, and
   RoMa v2 when a matcher is given (a few dozen frame pairs per station).
2. Starts: seeds from calibrated stations (the nearest one on each side of the robot: a third
   camera facing the robot or behind the arms, `side`), all solved, the passing solve with
   the lowest check kept; if none passes, PnP on the points the wrist cameras triangulate
   (`stages.static_starts`, focal searched), then PnP on robot surface points transferred
   from a neighbour's frames (`transfer`, given by the caller).
3. From each start, Levenberg–Marquardt on the camera's pose (Gaussian prior around the start,
   wide `position_range` / `orientation_range_deg`) and focal (log-normal, `focal_std`), the
   wrist mounts and arm kinematics held (`light.light_solve`).
4. The fit check: the median epipolar distance of the links under the solved poses
   (`light.check`), below `flag` to pass; a station none passes keeps its best solve, flagged
   for reading its overlay.

A static camera's focal and distance trade off (a far camera with a long focal sees the robot
alike): links to the wrist cameras, which move, are what separates them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch
from einops import rearrange, repeat

from r4r.geometry import se3

from . import light
from .initialize import locate_camera
from .kinematics import forward_kinematics
from .robot import Freedom

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.matching import RoMaV2
    from r4r.types import Mat3x3, Mat4x4

    from .lens import Lens
    from .robot import Robot
    from .stages import RigEpisode
    from .tracks import Tracks
    from .transfer import Transferred


@dataclass(frozen=True, slots=True, kw_only=True)
class StationConfig:
    light: light.LightConfig = field(default_factory=lambda: light.LightConfig(bursts=4))
    # Prior std of the camera's departure from its start (m, degrees: wide, the start may be a
    # neighbour's) and of its log focal
    position_range: float = 0.3
    orientation_range_deg: float = 15.0
    focal_std: float = 0.3
    # RoMa matches kept per pair of frames when a matcher is given
    dense: int = 400
    # A solve whose links' median epipolar distance exceeds this (pixels) fails the check
    flag: float = 4.0


@dataclass(frozen=True, slots=True)
class StationData[C: IntVar]:
    """A station's bursts: the frames of every camera (side by side), the burst sizes, the
    joint positions at each frame, the links and the tracks."""

    frames: list[Tensor]
    sizes: list[int]
    qpos: Tensor[[int, int]]
    links: light.Links[int]
    tracks: Tracks[int, int, C]


@dataclass(frozen=True, slots=True)
class StationFit:
    """The camera's `base_from_camera` and focal (pixels), the start it was solved from
    (`neighbour`, `pnp`, `picks`), the fit check after the solve and whether it failed."""

    pose: Tensor[[*Mat4x4]]
    focal: float
    start: str
    check: light.FitCheck
    flagged: bool


def samples(episodes: Sequence[int], count: int) -> list[int]:
    """`count` episodes spread evenly over a station's (sorted) `episodes`."""

    if len(episodes) <= count:
        return list(episodes)
    return [episodes[round(i * (len(episodes) - 1) / (count - 1))] for i in range(count)]


def station_data(
    root: Path,
    numbers: Sequence[int],
    episodes: Sequence[RigEpisode[int, int]],
    names: tuple[str, ...],
    static: Sequence[bool],
    camera: str,
    intrinsics: Tensor[[int, *Mat3x3]],
    lens: Lens[int] | None,
    stride: int,
    config: StationConfig,
    matcher: RoMaV2 | None = None,
) -> StationData[int]:
    """Bursts of the station's episodes `numbers` (module doc, step 1)."""

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
    links = light.burst_links(stacked, sizes, intrinsics, lens, list(static), config.light)
    if matcher is not None:
        links = _with_dense(links, matcher, stacked, sizes, names.index(camera), config)
    tracks = light.burst_tracks(stacked, sizes, intrinsics, lens, links, config.light)
    return StationData(stacked, sizes, torch.cat(qpos), links, tracks)


def calibrate_station(
    data: StationData[int],
    robot: Robot[int, int, int],
    names: tuple[str, ...],
    intrinsics: Tensor[[int, *Mat3x3]],
    lens: Lens[int] | None,
    camera: str,
    config: StationConfig,
    seeds: Sequence[tuple[str, Tensor[[*Mat4x4]], float]],
    picks: Callable[[], tuple[Transferred[int, int], Tensor[[int, int]]] | None] | None = None,
) -> StationFit:
    """Fit `camera` (static) from every seed (name, pose, focal: calibrated stations, e.g. the
    nearest one placed in front of the robot and the nearest behind it) and keep the solve with
    the lowest check among those that pass; if none passes, PnP on wrist-triangulated points,
    then `picks()` (points transferred into some of the station's frames and the joint
    positions there), the first that passes; else the lowest check, flagged."""

    c = names.index(camera)
    solved = [
        _solve(data, robot, names, intrinsics, lens, camera, config, (pose, focal), name)
        for name, pose, focal in seeds
    ]
    passing = [f for f in solved if not f.flagged]
    if passing:
        return min(passing, key=lambda f: f.check.median_px)
    starts: list[tuple[str, Callable[[], tuple[Tensor, float] | None]]] = [
        ("pnp", lambda: _pnp_start(robot, data, names, intrinsics, camera)),
    ]
    if picks is not None:
        starts.append(("picks", lambda: _picks_start(robot, picks(), intrinsics[c], data)))
    for name, start in starts:
        found = start()
        if found is None:
            continue
        fit = _solve(data, robot, names, intrinsics, lens, camera, config, found, name)
        if not fit.flagged:
            return fit
        solved.append(fit)
    if not solved:
        raise ValueError(f"{camera}: no start (no seed, PnP and picks found none)")
    return min(solved, key=lambda f: f.check.median_px)


def side(pose: Tensor[[*Mat4x4]]) -> str:
    """Where a static camera sits: `facing` the robot (in front of the base, x > 0) or
    `behind` it."""

    return "facing" if float(pose[0, 3]) > 0 else "behind"


def _solve(
    data: StationData[int],
    robot: Robot[int, int, int],
    names: tuple[str, ...],
    intrinsics: Tensor[[int, *Mat3x3]],
    lens: Lens[int] | None,
    camera: str,
    config: StationConfig,
    start: tuple[Tensor[[*Mat4x4]], float],
    name: str,
) -> StationFit:
    """LM from `start` (pose, focal), then the fit check (module doc, steps 3-4)."""

    c = names.index(camera)
    joints = data.qpos
    mounts = robot.body_from_camera[[robot.camera(n) for n in names]].to(joints).clone()
    mounts[c] = start[0]
    k = intrinsics.clone()
    k[c, :2, :2] *= start[1] / k[c, 0, 0]
    static = torch.tensor([robot.camera_body[robot.camera(n)] == 0 for n in names])
    tracks = light.informative(replace(data.tracks, intrinsics=k), static, False, config.light)
    freedom = Freedom(config.position_range, math.radians(config.orientation_range_deg), "dataset")
    solving = replace(light.held_robot(robot, ()), free_cameras={camera: freedom})
    at = repeat(mounts, "c i j -> c t i j", t=len(joints))
    focal_std = torch.zeros(len(names), dtype=joints.dtype)
    focal_std[c] = config.focal_std
    rig, _ = light.light_solve(solving, joints, tracks, names, at, None, config.light, focal_std)
    with torch.no_grad():
        index = torch.tensor([c])
        pose = rig.mounts_at(index, torch.zeros_like(index))[0].detach()
        k = rig.intrinsics().detach()
    mounts[c] = pose
    check = _check(robot, data, names, mounts, k, lens)
    flagged = not check.observations or check.median_px > config.flag
    return StationFit(pose, float(k[c, 0, 0]), name, check, flagged)


def _check(
    robot: Robot[int, int, int],
    data: StationData[int],
    names: tuple[str, ...],
    mounts: Tensor[[int, *Mat4x4]],
    intrinsics: Tensor[[int, *Mat3x3]],
    lens: Lens[int] | None,
) -> light.FitCheck:
    """The links' median epipolar distance under the cameras at `mounts` (`light.check`)."""

    world = forward_kinematics(robot, data.qpos)
    bodies = [robot.camera_body[robot.camera(n)] for n in names]
    poses = rearrange(world[:, bodies] @ mounts, "t c i j -> (c t) i j")
    view_camera = torch.arange(len(names)).repeat_interleave(len(data.qpos))
    static = torch.tensor([b == 0 for b in bodies])
    return light.check(data.links, poses, view_camera, intrinsics, lens, static)


def _pnp_start(
    robot: Robot[int, int, int],
    data: StationData[int],
    names: tuple[str, ...],
    intrinsics: Tensor[[int, *Mat3x3]],
    camera: str,
) -> tuple[Tensor[[*Mat4x4]], float] | None:
    """The camera's pose and focal by PnP on the points the wrist cameras triangulate and the
    links see in it, its focal searched (`stages.static_starts`); None when too few."""

    from .stages import RigEpisode, static_starts

    mounts = robot.body_from_camera[[robot.camera(n) for n in names]].to(data.qpos)
    searched = replace(robot, free_cameras={camera: Freedom(math.inf, math.inf, "dataset")})
    episode = RigEpisode(data.qpos, dict(zip(names, data.frames, strict=True)))
    known = tuple(n != camera for n in names)
    starts, found = static_starts(
        searched, [episode], [data.tracks], names, mounts, intrinsics, known, known
    )
    c = names.index(camera)
    if bool(torch.equal(starts[0, c], mounts[c])):
        return None
    return starts[0, c], float(found[c, 0, 0])


def _picks_start(
    robot: Robot[int, int, int],
    found: tuple[Transferred[int, int], Tensor[[int, int]]] | None,
    intrinsics: Tensor[[*Mat3x3]],
    data: StationData[int],
) -> tuple[Tensor[[*Mat4x4]], float] | None:
    """PnP (focal searched) on transferred surface points: `X = W_b(q) · offset` against their
    pixels; None without enough of them."""

    if found is None:
        return None
    moved, qpos = found
    frame, point = moved.kept.nonzero().T
    if len(frame) == 0:
        return None
    world = forward_kinematics(robot, qpos.to(robot.parent_from_body))
    body_pose = world[frame, moved.body[frame, point]]
    offset = moved.offset[frame, point].to(body_pose)
    xyz = se3.transform_points(body_pose, offset[:, None])[:, 0]
    width = int(data.frames[0].shape[-1])
    start = locate_camera(xyz, moved.pixels[frame, point].to(xyz), intrinsics, width, False)
    return None if start is None else (start.world_from_camera, start.focal)


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
