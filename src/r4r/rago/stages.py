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

Robot keypoints (`RigEpisode.keypoints`, see `keypoints.py`) start the cameras the stages
above cannot: their known points (picks) place a static camera with no moving camera to
triangulate for it (PnP, with the triangulated points when there are some), and a moving
camera from its picks pooled over episodes in its link's frame (instead of the search). After
stage 3, cameras with `RigConfig.seeds` look for their robot points where their start projects
them (chains, for the bundles only). In the bundles known points are residuals next to the
tracks, chains are tracks on a known body, and a moving mount with keypoints stays free within
its priors' ranges (unless a free body above it would trade off with it).

Intrinsics recorded per episode (`RigEpisode.intrinsics`) that differ between episodes
(randomising simulators) are each episode's own in every stage: the camera's intrinsics are
the first episode's and each view is rescaled to its episode's (`Tracks.view_scaling`); a
focal correction then scales all episodes alike. Cameras whose intrinsics do not change are
solved with one K, as before.

Mobile bases: an episode's recorded base motion (`RigEpisode.base_motion`, odometry) moves the
frame-scope free body nearest the root (the base) in stage 3 and starts its per-frame
correction in the bundles. Stage 1 needs the base still: it uses each episode's longest span
where the record says the base is parked.

Ref: docs/architecture.md, "Solving coupled unknowns in stages".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch
from einops import rearrange

from r4r.geometry import se3

from .bundle import BundleConfig, BundleResult, bundle_adjust
from .calibrate import FitConfig, fit_setup, video_tracks
from .cross import cross_matches
from .initialize import CameraStart, Points, locate_camera, triangulate
from .keypoints import (
    MIN_KEYPOINTS,
    KeypointConfig,
    Keypoints,
    carried,
    join,
    seeded,
)
from .kinematics import forward_kinematics
from .klt import VideoTracks, klt_tracks, require_tracks, track_video
from .merge import merge_tracks
from .rig import KinematicFreedom
from .tracks import Tracks

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.data.priors import RobotPoint
    from r4r.matching import RoMaV2
    from r4r.types import RGB, Mat3x3, Mat4x4

    from .lens import Lens
    from .robot import Freedom, Robot

# A static camera needs this many points seen by moving cameras (≥ 3 observations each)
MIN_POINTS = 50
MIN_OBSERVATIONS = 3
# Stage 1 on a mobile base: timesteps where the recorded base moves less than this per step
# count as still (m, rad), and a mount is searched on spans of at least MIN_STILL of them
STILL_STEP = (0.002, math.radians(0.1))
# A static camera's PnP start farther than this from the robot base (m) is degenerate
MAX_STATIC_DISTANCE = 10.0
MIN_STILL = 20


class _Tracked:
    """KLT tracks of each (episode, camera) video, computed once and shared by the stages."""

    def __init__(self, episodes: list[RigEpisode[int, int]], config: FitConfig) -> None:
        self.episodes, self.config = episodes, config
        self.done: dict[tuple[int, str], VideoTracks[int]] = {}

    def __call__(self, episode: int, name: str) -> VideoTracks[int]:
        """Tracks of camera `name` in `episode` (its index); raises when there are none."""

        key = (episode, name)
        if key not in self.done:
            found = track_video(self.episodes[episode].frames[name], self.config.klt)
            require_tracks(found.track, name, episode)
            self.done[key] = found
        return self.done[key]


@dataclass(frozen=True, slots=True)
class RigEpisode[T: IntVar, NQ: IntVar]:
    """One episode of `T` timesteps: joint positions `qpos [T, NQ]` and every camera's video."""

    qpos: Tensor[[T, NQ]]
    # uint8 videos, one per camera: T frames (int: stages that read no pixels take a stand-in)
    frames: dict[str, Tensor[[int, *RGB, int, int]]]
    # Mobile base: its pose at each timestep relative to the first, in its own frame (odometry)
    base_motion: Tensor[[T, *Mat4x4]] | None = None
    # Recorded start poses (base frame, OpenCV axes) of static cameras in this episode, e.g. a
    # dataset's own per-episode calibration (per-episode table): used instead of PnP. For a
    # moving camera: its mount on its link in this episode (e.g. a simulator randomising it)
    poses: dict[str, Tensor[[*Mat4x4]]] = field(default_factory=dict)
    # Recorded intrinsics of cameras in this episode (e.g. a simulator redrawing focals)
    intrinsics: dict[str, Tensor[[*Mat3x3]]] = field(default_factory=dict)
    # Known robot points seen by the cameras (camera index in the rig's names, timestep here)
    keypoints: Keypoints[int] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class RigConfig:
    fit: FitConfig = field(default_factory=FitConfig)
    # RoMa links between every pair of cameras at every `link_every`-th timestep
    link_every: int = 10
    bundle: BundleConfig = field(
        default_factory=lambda: BundleConfig(focal_std=0.2, attach_links=True, iterations=1200)
    )
    groups: int = 2
    # Joint-encoder and video/joint time offsets refined in the bundles (None: held)
    kinematic: KinematicFreedom | None = None
    # Robot points each camera looks for where its start projects them, and how
    seeds: dict[str, tuple[RobotPoint, ...]] = field(default_factory=dict)
    keypoints: KeypointConfig = field(default_factory=KeypointConfig)


@dataclass(frozen=True, slots=True)
class RigFit[C: IntVar, L: IntVar, J: IntVar]:
    """A rig's calibration: `C` cameras on a robot of `L` bodies and `J` joints, over `E`
    episodes (`T` timesteps in all), `F` free bodies and `G` groups of episodes."""

    # `body_from_camera` per camera (moving: its mount; static: its first episode's pose)
    mounts: Tensor[[C, *Mat4x4]]
    intrinsics: Tensor[[C, *Mat3x3]]
    # Per episode [E, C]: `intrinsics` rescaled to each episode's recorded ones (all equal to
    # `intrinsics` unless a camera's recorded intrinsics change between episodes)
    episode_intrinsics: Tensor[[int, C, *Mat3x3]]
    # Distortion per camera (None: ideal pinholes)
    lens: Lens[C] | None
    # Free bodies' corrected `world_from_body` at the first timestep of every episode [E, F]
    bodies: Tensor[[int, int, *Mat4x4]]
    # Static cameras' poses per episode [E, C] (moving cameras: their mount)
    camera_poses: Tensor[[int, C, *Mat4x4]]
    # The free bodies' corrections (description's offset → fitted: `parent_from_body @ this`),
    # from all episodes [F], and per group of episodes [G, F]
    body_corrections: Tensor[[int, *Mat4x4]]
    group_corrections: Tensor[[int, int, *Mat4x4]]
    median_error: float
    inlier_fraction: float
    # Per camera: (agreeing, searched) episodes of its stage-1 mount search (None: not searched)
    mount_agreement: tuple[tuple[int, int] | None, ...]
    # The solution at every timestep (episodes concatenated): `world_from_body [T, L]` with the
    # free bodies' corrections, and each camera's `body_from_camera [C, T]` (static: world pose)
    world_from_body: Tensor[[int, L, *Mat4x4]]
    mounts_at: Tensor[[C, int, *Mat4x4]]
    # Mobile base, per episode: how far the fitted base departs from its recorded motion
    # (median cm, max cm, max degrees); empty without a record
    base_departure: tuple[tuple[float, float, float], ...]
    # Kinematic refinements (zero where held): joint offsets [J] and time offset (timesteps),
    # from all episodes and per group ([G, J], [G])
    joint_offset: Tensor[[J]]
    time_offset: float
    group_joint_offsets: Tensor[[int, J]]
    group_time_offsets: tuple[float, ...]
    # Median reprojection error of the robot keypoints (pixels; NaN without keypoints)
    keypoint_error: float = math.nan


def calibrate_rig[L: IntVar, J: IntVar, C: IntVar](
    robot: Robot[L, J, int],
    episodes: list[RigEpisode[int, int]],
    names: tuple[str, ...],
    intrinsics: Tensor[[C, *Mat3x3]],
    known_focal: tuple[bool, ...],
    known_mount: tuple[bool, ...],
    model: RoMaV2,
    config: RigConfig,
    lens: Lens[C] | None = None,
) -> RigFit[C, L, J]:
    """Calibrate the free mounts, free bodies and static cameras of `robot` (see module doc).

    `intrinsics` are the focal priors (or guesses where `known_focal` is False); cameras on
    the world body are static, the others moving. Moving cameras with `known_mount` keep the
    description's mount (no search); static ones start from it (no PnP), as do those with a
    recorded pose for the episode (`RigEpisode.poses`). `lens`: the cameras' distortion
    priors (None: ideal pinholes), refined where their std allows. `intrinsics` are the first
    episode's where episodes record their own (`RigEpisode.intrinsics`).
    """

    tracked = _Tracked(episodes, config.fit)
    scaling = episode_scaling(episodes, names)
    mounts, intrinsics, lens, agreement = _moving_mounts(
        robot, episodes, names, intrinsics, known_focal, known_mount, config, tracked, lens
    )
    merged = [
        _multi_camera_tracks(
            e,
            names,
            intrinsics,
            model,
            config,
            [tracked(i, n) for n in names],
            lens,
            None if scaling is None else scaling[i],
        )
        for i, e in enumerate(episodes)
    ]
    starts, intrinsics = _static_starts(
        robot, episodes, merged, names, mounts, intrinsics, known_focal, known_mount
    )
    starts = _recorded_mounts(robot, episodes, names, starts)
    if config.seeds:
        episodes = _seeded_episodes(robot, episodes, names, starts, intrinsics, lens, config)

    # 4-5. Bundles: all episodes, then groups. Known focals stay fixed; those found in stage 3
    # are refined
    frozen = _freeze_moving_mounts(robot, names, starts[0], _with_keypoints(episodes, names))
    everything = _bundle(frozen, episodes, merged, names, intrinsics, starts, config, known_focal)
    group_fits = _group_bundles(
        frozen, episodes, merged, names, intrinsics, starts, config, known_focal
    )
    first = torch.cumsum(torch.tensor([0] + [len(e.qpos) for e in episodes[:-1]]), 0).tolist()
    free = sorted(robot.body(n) for n in robot.free_bodies)
    overall = _body_corrections(frozen, episodes[0].qpos, everything.world_from_body)
    corrections = [
        _body_corrections(frozen, episodes[group[0]].qpos, fit.world_from_body)
        for group, fit in group_fits
    ]
    fits = [fit for _, fit in group_fits] or [everything]
    return RigFit(
        mounts=everything.body_from_camera,
        intrinsics=everything.intrinsics,
        episode_intrinsics=everything.intrinsics.expand(len(episodes), -1, 3, 3)
        if scaling is None
        else scaling.to(everything.intrinsics) @ everything.intrinsics,
        lens=everything.lens,
        bodies=everything.world_from_body[first][:, free],
        body_corrections=overall,
        camera_poses=rearrange(everything.mounts_at[:, first], "c e i j -> e c i j"),
        group_corrections=torch.stack(corrections) if corrections else overall[None],
        median_error=everything.median_error,
        inlier_fraction=everything.inlier_fraction,
        mount_agreement=agreement,
        world_from_body=everything.world_from_body,
        mounts_at=everything.mounts_at,
        base_departure=_base_departure(frozen, episodes, everything.world_from_body),
        joint_offset=everything.joint_offset,
        time_offset=everything.time_offset,
        group_joint_offsets=torch.stack([f.joint_offset for f in fits]),
        group_time_offsets=tuple(f.time_offset for f in fits),
        keypoint_error=everything.keypoint_error,
    )


def _with_keypoints(episodes: list[RigEpisode[int, int]], names: tuple[str, ...]) -> set[str]:
    """Cameras with robot keypoints in some episode."""

    seen = set()
    for e in episodes:
        if e.keypoints is not None:
            seen |= {names[c] for c in torch.unique(e.keypoints.camera).tolist()}
    return seen


def _seeded_episodes(
    robot: Robot[int, int, int],
    episodes: list[RigEpisode[int, int]],
    names: tuple[str, ...],
    starts: Tensor[[int, int, *Mat4x4]],
    intrinsics: Tensor[[int, *Mat3x3]],
    lens: Lens[int] | None,
    config: RigConfig,
) -> list[RigEpisode[int, int]]:
    """`episodes` with chains seeded where each seeded camera's start (`starts [E, C]`: static
    pose, or mount) projects its robot points, added to their keypoints."""

    scaling = episode_scaling(episodes, names)
    out = []
    for i, e in enumerate(episodes):
        world = moved_kinematics(robot, e)
        parts = [e.keypoints]
        for name, points in config.seeds.items():
            c = names.index(name)
            body = robot.camera_body[robot.camera(name)]
            k = intrinsics[c] if scaling is None else scaling[i, c].to(intrinsics) @ intrinsics[c]
            parts.append(
                seeded(
                    world,
                    world[:, body] @ starts[i, c],
                    k,
                    None if lens is None else lens.select(torch.tensor([c])),
                    [(robot.body(p.body), p.offset) for p in points],
                    e.frames[name],
                    c,
                    config.keypoints,
                )
            )
        found = join(parts)
        out.append(replace(e, keypoints=None if found is None else found.to(e.qpos)))
    return out


def _recorded_mounts[E: IntVar, C: IntVar](
    robot: Robot[int, int, int],
    episodes: list[RigEpisode[int, int]],
    names: tuple[str, ...],
    starts: Tensor[[E, C, *Mat4x4]],
) -> Tensor[[E, C, *Mat4x4]]:
    """`starts` with the moving cameras' recorded per-episode mounts (`RigEpisode.poses`).

    Raises when they differ between episodes for a camera the bundles freeze (dataset scope,
    or held: `_freeze_moving_mounts` keeps the first episode's mount for all)."""

    for c, name in enumerate(names):
        if robot.camera_body[robot.camera(name)] == 0:
            continue
        for i, e in enumerate(episodes):
            if name in e.poses:
                starts[i, c] = e.poses[name].to(starts)
        freedom = robot.free_cameras.get(name)
        frozen = freedom is None or freedom.scope == "dataset"
        if frozen and bool((starts[:, c] != starts[0, c]).any()):
            raise ValueError(
                f"{name}: recorded mounts differ between episodes, but the mount is held or "
                "dataset scope (one for all episodes); give it episode scope in the priors"
            )
    return starts


def _freeze_moving_mounts[L: IntVar, J: IntVar, C: IntVar](
    robot: Robot[L, J, C],
    names: tuple[str, ...],
    mounts: Tensor[[int, *Mat4x4]],
    seen: set[str] | frozenset[str] = frozenset(),
) -> Robot[L, J, C]:
    """`robot` with `names`' cameras at `mounts`, free only where the bundles refine them: static
    cameras, moving ones whose mount changes between episodes (randomising simulators:
    episode scope, refined per episode within range), and moving ones that see robot keypoints
    (`seen`) with no free body above them (it would trade off with the mount). Other moving
    mounts are frozen."""

    static = [robot.camera_body[robot.camera(n)] == 0 for n in names]
    bodies = [robot.body(b) for b, f in robot.free_bodies.items() if f.scope != "frame"]

    def refined(name: str, freedom: Freedom) -> bool:
        if static[names.index(name)] or freedom.scope != "dataset":
            return True
        carrier = robot.camera_body[robot.camera(name)]
        return name in seen and not any(_under(robot, carrier, b) for b in bodies)

    frozen = replace(
        robot,
        free_cameras={n: f for n, f in robot.free_cameras.items() if n in names and refined(n, f)},
        body_from_camera=robot.body_from_camera.to(mounts).clone(),
    )
    for c, name in enumerate(names):
        frozen.body_from_camera[robot.camera(name)] = mounts[c]
    return frozen


def _group_bundles(
    robot: Robot[int, int, int],
    episodes: list[RigEpisode[int, int]],
    merged: list[Tracks[int, int, int]],
    names: tuple[str, ...],
    intrinsics: Tensor[[int, *Mat3x3]],
    starts: Tensor[[int, int, *Mat4x4]],
    config: RigConfig,
    known_focal: tuple[bool, ...],
) -> list[tuple[list[int], BundleResult[int, int, int, int]]]:
    """Stage 5: the bundle per group of episodes (every `groups`-th), the agreement check on
    dataset-scope unknowns. Fewer than two non-empty groups (0 or 1 asked, or one episode):
    none, since one group would only repeat the bundle over all episodes."""

    groups = [list(range(g, len(episodes), config.groups)) for g in range(config.groups)]
    groups = [g for g in groups if g]
    if len(groups) < 2:
        return []
    return [
        (
            group,
            _bundle(
                robot,
                [episodes[i] for i in group],
                [merged[i] for i in group],
                names,
                intrinsics,
                starts[group],
                config,
                known_focal,
            ),
        )
        for group in groups
    ]


def _moving_mounts[C: IntVar](
    robot: Robot[int, int, int],
    episodes: list[RigEpisode[int, int]],
    names: tuple[str, ...],
    intrinsics: Tensor[[C, *Mat3x3]],
    known_focal: tuple[bool, ...],
    known_mount: tuple[bool, ...],
    config: RigConfig,
    tracked: _Tracked | None = None,
    lens: Lens[C] | None = None,
) -> tuple[
    Tensor[[C, *Mat4x4]], Tensor[[C, *Mat3x3]], Lens[C] | None, tuple[tuple[int, int] | None, ...]
]:
    """Stage 1: each free moving camera's mount (and focal, distortion) from its own video, or
    from its robot keypoints when it has enough (`_keypoint_mount`)."""

    index = [robot.camera(n) for n in names]
    mounts = robot.body_from_camera[index].to(episodes[0].qpos).clone()
    intrinsics = intrinsics.clone()
    agreement: list[tuple[int, int] | None] = [None] * len(names)
    scaling = episode_scaling(episodes, names)
    for c, name in enumerate(names):
        body = robot.camera_body[index[c]]
        if body == 0 or known_mount[c] or name not in robot.free_cameras:
            continue
        own = None if lens is None else lens.select(torch.tensor([c], device=mounts.device))
        width = int(episodes[0].frames[name].shape[-1])
        found = _keypoint_mount(robot, episodes, c, body, intrinsics[c], width, known_focal[c], own)
        if found is not None:
            mounts[c], focal = found.world_from_camera, found.focal
            intrinsics[c, :2, :2] *= focal / intrinsics[c, 0, 0]
            continue
        own = None if lens is None else lens.select(torch.tensor([c], device=mounts.device))
        data = _still_videos(episodes, c, name, intrinsics[c], config, tracked, own, scaling)
        if not data:
            raise ValueError(
                f"{name}: no span of {MIN_STILL} timesteps with the base still in any episode;"
                " give this camera's mount in the priors"
            )
        alone = replace(robot, free_cameras={name: robot.free_cameras[name]}, free_bodies={})
        focal_range = 0.0 if known_focal[c] else None
        setup = fit_setup(alone, data, name, (body,), intrinsics[c], config.fit, focal_range)
        mounts[c], intrinsics[c] = setup.fit.body_from_camera, setup.fit.intrinsics
        if lens is not None and setup.fit.lens is not None:
            lens = lens.with_distortion(c, setup.fit.lens.distortion[0])
        agreement[c] = (int(setup.agree.sum()), len(data))
    return mounts, intrinsics, lens, tuple(agreement)


def _keypoint_mount(
    robot: Robot[int, int, int],
    episodes: list[RigEpisode[int, int]],
    camera: int,
    body: int,
    intrinsics: Tensor[[*Mat3x3]],
    width: int,
    known_focal: bool,
    lens: Lens[1] | None,
) -> CameraStart | None:
    """A moving camera's mount on `body` by PnP on its robot keypoints of all episodes, carried
    into `body`'s frame (where the mount is constant); None without enough of them. Its
    `world_from_camera` is the mount `body_from_camera`."""

    points, pixels = [], []
    for e in episodes:
        if e.keypoints is None:
            continue
        mine = e.keypoints.known()
        mine = mine.where(mine.camera == camera)
        points.append(carried(moved_kinematics(robot, e), mine, body))
        pixels.append(mine.pixel)
    if not points or sum(len(p) for p in points) < MIN_KEYPOINTS:
        return None
    return locate_camera(
        torch.cat(points),
        torch.cat(pixels).to(points[0]),
        intrinsics,
        width,
        known_focal,
        None if lens is None else lens.distortion[0],
        None if lens is None else lens.fisheye[0],
    )


def _still_videos(
    episodes: list[RigEpisode[int, int]],
    camera: int,
    name: str,
    intrinsics: Tensor[[*Mat3x3]],
    config: RigConfig,
    tracked: _Tracked | None,
    lens: Lens[1] | None,
    scaling: Tensor[[int, int, *Mat3x3]] | None,
) -> list[tuple[Tensor[[int, int]], Tracks[int, int, 1]]]:
    """Per episode with a long enough still base span: its joint positions and the camera's
    KLT tracks over that span (`lens`, `scaling [E, C]`: the camera's own)."""

    data = []
    for i, e in enumerate(episodes):
        a, b = still_span(e.base_motion)
        if b - a < MIN_STILL:
            continue
        whole = a == 0 and b >= len(e.qpos)  # the whole video: its tracks are shared
        found = tracked(i, name) if whole and tracked is not None else None
        frames = e.frames[name][a:b]
        tracks = video_tracks(frames, intrinsics, config.fit.klt, found, lens=lens)
        require_tracks(tracks.track, name, i)
        if scaling is not None:
            tracks = replace(tracks, view_scaling=scaling[i, camera].expand(len(frames), 3, 3))
        data.append((e.qpos[a:b], tracks))
    return data


def _multi_camera_tracks[C: IntVar](
    episode: RigEpisode[int, int],
    names: tuple[str, ...],
    intrinsics: Tensor[[C, *Mat3x3]],
    model: RoMaV2,
    config: RigConfig,
    found: list[VideoTracks[int]] | None = None,
    lens: Lens[C] | None = None,
    scaling: Tensor[[C, *Mat3x3]] | None = None,
) -> Tracks[int, int, C]:
    """Stage 2: KLT tracks of every camera, merged through RoMa links at the same instants.
    `scaling [C]`: the episode's own intrinsics relative to `intrinsics` (`episode_scaling`)."""

    device = episode.qpos.device
    steps = len(episode.qpos)
    view_camera = torch.arange(len(names), device=device).repeat_interleave(steps)
    view_time = torch.arange(steps, device=device).repeat(len(names))
    videos = [episode.frames[n] for n in names]
    tracks = klt_tracks(
        videos, view_camera, view_time, intrinsics, config.fit.klt, found, lens=lens
    )
    if scaling is not None:
        tracks = replace(tracks, view_scaling=scaling[view_camera])
    pairs = [
        (a, t, b, t)
        for t in range(0, steps, config.link_every)
        for a in range(len(names))
        for b in range(a + 1, len(names))
    ]
    if not pairs:  # one camera: nothing to link
        return tracks
    links = cross_matches(model, videos, pairs)
    return merge_tracks(
        tracks,
        links.camera_a * steps + links.time_a,
        links.pixel_a.to(tracks.pixel),
        links.camera_b * steps + links.time_b,
        links.pixel_b.to(tracks.pixel),
    )


def _static_starts[C: IntVar](
    robot: Robot[int, int, int],
    episodes: list[RigEpisode[int, int]],
    merged: list[Tracks[int, int, C]],
    names: tuple[str, ...],
    mounts: Tensor[[C, *Mat4x4]],
    intrinsics: Tensor[[C, *Mat3x3]],
    known_focal: tuple[bool, ...],
    known_mount: tuple[bool, ...] | None = None,
) -> tuple[Tensor[[int, C, *Mat4x4]], Tensor[[C, *Mat3x3]]]:
    """Stage 3: each free static camera's pose per episode (and focal, from the first episode
    where it is found, when unknown) by PnP on points the moving cameras triangulate and on the
    episode's known robot points. Episodes where PnP fails keep the description's pose. A pose
    recorded for the episode (`RigEpisode.poses`) is used whatever the priors say; otherwise
    cameras the priors fix keep theirs, and cameras whose pose the priors give (`known_mount`)
    start from it: the bundle refines them within the priors' ranges (PnP on robot-attached
    tracks can land a known camera 10 cm / 6 deg off and the bundle then keeps it there); their
    known robot points may still move them by PnP within the gate. A dataset-scope camera
    starts every episode from the consensus of its episodes' PnP poses (`_consensus`): one
    episode with few links can land metres away."""

    device = mounts.device
    index = [robot.camera(n) for n in names]
    body = torch.tensor([robot.camera_body[i] for i in index], device=device)
    static = body == 0
    starts = mounts[None].repeat(len(episodes), 1, 1, 1)  # [E, C, 4, 4]
    support = torch.zeros(len(episodes), len(names), device=device)  # PnP inliers
    focal = intrinsics[:, 0, 0].clone()
    found = [False] * len(names)
    known_pose = known_mount or (False,) * len(names)
    ranged = [n in robot.free_cameras for n in names]
    for i, (e, tracks) in enumerate(zip(episodes, merged, strict=True)):
        world = moved_kinematics(robot, e)
        cam = tracks.view_camera
        world_from_view = world[tracks.view_time, body[cam]] @ mounts[cam]
        observed = tracks.view_camera[tracks.view]
        points = triangulate(tracks, world_from_view, ~static[observed])
        for c in static.nonzero()[:, 0].tolist():
            # The episode's own calibration comes first, even where the priors give a pose
            recorded = e.poses.get(names[c])
            if recorded is not None:
                starts[i, c] = recorded.to(starts)
                support[i, c] = 1.0
                continue
            if not ranged[c]:
                continue
            seen = (observed == c) & (points.count[tracks.track] >= MIN_OBSERVATIONS)
            if known_pose[c] or int(seen.sum()) < MIN_POINTS:
                seen = torch.zeros_like(seen)
            pairs = _pairs(tracks, points, seen, world, e.keypoints, c)
            if pairs is None:
                continue
            width = int(e.frames[names[c]].shape[-1])
            # This episode's K: the camera's at the current focal (fx, fy scaled together),
            # through its view scaling; `scale` its focal over the camera's (1 unless recorded
            # ones change)
            camera_k = intrinsics[c].clone()
            camera_k[:2, :2] *= focal[c] / intrinsics[c, 0, 0]
            view_k = tracks.view_intrinsics(camera_k.expand(len(names), 3, 3))
            k = view_k[int((cam == c).nonzero()[0, 0])]
            scale = float(k[0, 0] / camera_k[0, 0])
            known = known_focal[c] or found[c]
            freedom = robot.free_cameras[names[c]]
            start = _checked_pnp(pairs, tracks.lens, c, k, width, known, mounts[c], freedom)
            if start is not None:
                starts[i, c] = start.world_from_camera
                support[i, c] = start.inliers
                focal[c], found[c] = start.focal / scale, True
    for c, name in enumerate(names):
        # A known pose moved by no keypoint PnP keeps its (recorded or prior) starts
        moved = not known_pose[c] or found[c]
        if bool(static[c]) and ranged[c] and moved and robot.free_cameras[name].scope == "dataset":
            starts[:, c] = _consensus(starts[:, c], support[:, c]).clone()
    intrinsics = intrinsics.clone()
    intrinsics[:, :2, :2] *= (focal / intrinsics[:, 0, 0])[:, None, None]
    return starts, intrinsics


def _pairs[M: IntVar](
    tracks: Tracks[M, int, int],
    points: Points[int],
    seen: Tensor[[M]],
    world: Tensor[[int, int, *Mat4x4]],
    keypoints: Keypoints[int] | None,
    camera: int,
) -> tuple[Tensor[[int, 3]], Tensor[[int, 2]]] | None:
    """3D–2D pairs for a static camera's PnP: its `seen` observations of triangulated `points`
    and its robot keypoints (world points by forward kinematics); None when there are fewer
    than `MIN_POINTS` of the first and `MIN_KEYPOINTS` of the second."""

    xyz, uv = [points.point[tracks.track[seen]]], [tracks.pixel[seen]]
    mine = None if keypoints is None else keypoints.known()
    mine = None if mine is None else mine.where(mine.camera == camera)
    if mine is not None and len(mine.time) >= MIN_KEYPOINTS:
        xyz.append(carried(world, mine, 0).to(xyz[0]))
        uv.append(mine.pixel.to(uv[0]))
    pixel = torch.cat(uv)
    if len(pixel) == 0:
        return None
    return torch.cat(xyz), pixel


def _checked_pnp(
    pairs: tuple[Tensor[[int, 3]], Tensor[[int, 2]]],
    lens: Lens[int] | None,
    camera: int,
    intrinsics: Tensor[[*Mat3x3]],
    width: int,
    known_focal: bool,
    prior: Tensor[[*Mat4x4]],
    freedom: Freedom,
) -> CameraStart | None:
    """PnP start of static `camera` (seen through `intrinsics`, its focal searched unless
    `known_focal`) from 3D–2D `pairs`; None when PnP fails, is degenerate (NaN) or lands
    outside `_within_prior` (keep the prior)."""

    start = locate_camera(
        *pairs,
        intrinsics,
        width,
        known_focal,
        None if lens is None else lens.distortion[camera],
        None if lens is None else lens.fisheye[camera],
    )
    if start is None or not (
        bool(torch.isfinite(start.world_from_camera).all())
        and _within_prior(start.world_from_camera, prior, freedom)
    ):
        return None
    return start


def _within_prior(pose: Tensor[[*Mat4x4]], prior: Tensor[[*Mat4x4]], freedom: Freedom) -> bool:
    """Whether `pose` lies within twice the priors' range of the prior pose (when the priors
    give no pose: within `MAX_STATIC_DISTANCE` of the robot base, a degenerate PnP otherwise)."""

    if float(pose[:3, 3].norm()) > MAX_STATIC_DISTANCE:
        return False
    if math.isinf(freedom.translation_std):
        return True
    delta = se3.inverse(prior) @ pose
    angle = float(se3.so3_log(delta[:3, :3]).norm())
    return (
        float(delta[:3, 3].norm()) <= 2 * freedom.translation_std
        and angle <= 2 * freedom.rotation_std
    )


def _consensus[E: IntVar](
    poses: Tensor[[E, *Mat4x4]],
    support: Tensor[[E]],
    tolerance: tuple[float, float] = (0.1, math.radians(5.0)),
) -> Tensor[[*Mat4x4]]:
    """The pose (among per-episode estimates) whose neighbours within `tolerance` (m, rad)
    carry the most support (PnP inliers): `argmax_i Σ_j [Δ_ij near] s_j`, `Δ_ij = P_i⁻¹ P_j`;
    episodes without an estimate have none."""

    delta = se3.inverse(poses)[:, None] @ poses[None]
    near = (delta[..., :3, 3].norm(dim=-1) < tolerance[0]) & (
        se3.so3_log(delta[..., :3, :3]).norm(dim=-1) < tolerance[1]
    )
    return poses[int((near.to(support) @ support).argmax())]


def _bundle[C: IntVar](
    robot: Robot[int, int, int],
    episodes: list[RigEpisode[int, int]],
    merged: list[Tracks[int, int, C]],
    names: tuple[str, ...],
    intrinsics: Tensor[[C, *Mat3x3]],
    starts: Tensor[[int, C, *Mat4x4]],
    config: RigConfig,
    known_focal: tuple[bool, ...] | None = None,
) -> BundleResult[C, int, int, int]:
    """One bundle over `episodes`: their tracks pooled (distinct ids and timesteps)."""

    steps = [len(e.qpos) for e in episodes]
    total = sum(steps)
    device = episodes[0].qpos.device
    track_offset, time_offset = 0, 0
    track, view, pixel, weight, moving = [], [], [], [], []
    scaling = None  # per pooled view (camera-major), when episodes' intrinsics differ
    if any(t.view_scaling is not None for t in merged):
        scaling = torch.eye(3, dtype=intrinsics.dtype, device=device).repeat(
            len(names), total, 1, 1
        )
    for tracks, n in zip(merged, steps, strict=True):
        if scaling is not None and tracks.view_scaling is not None:
            at = tracks.view_camera * total + tracks.view_time + time_offset
            scaling.view(-1, 3, 3)[at] = tracks.view_scaling.to(scaling)
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
        lens=merged[0].lens,
        view_scaling=None if scaling is None else scaling.flatten(0, 1),
    )
    episode = torch.cat([torch.full((n,), i, device=device) for i, n in enumerate(steps)])
    start_at = torch.cat(
        [starts[i][:, None].expand(-1, n, 4, 4) for i, n in enumerate(steps)], 1
    )  # [C, T, 4, 4]
    qpos = torch.cat([e.qpos for e in episodes])
    first = [sum(steps[:i]) for i in range(len(steps))]
    keypoints = join(
        [
            None if e.keypoints is None else e.keypoints.shifted(f)
            for e, f in zip(episodes, first, strict=True)
        ]
    )
    return bundle_adjust(
        robot,
        qpos,
        pooled,
        names,
        starts[0],
        config.bundle,
        episode=episode,
        start_at=start_at,
        bodies_at=_bodies_at(robot, episodes),
        known_focal=known_focal,
        kinematic=config.kinematic,
        keypoints=keypoints,
    )


def episode_scaling(
    episodes: list[RigEpisode[int, int]], names: tuple[str, ...]
) -> Tensor[[int, int, *Mat3x3]] | None:
    """Per episode and camera [E, C]: `S_ec = K_ec · K_0c⁻¹`, its recorded intrinsics over the
    first episode's, for cameras recorded in every episode with intrinsics that change
    (identity elsewhere); None when no camera's change."""

    qpos = episodes[0].qpos
    eye = torch.eye(3, dtype=qpos.dtype, device=qpos.device)
    scaling = eye.repeat(len(episodes), len(names), 1, 1)
    varies = False
    for c, name in enumerate(names):
        if not all(name in e.intrinsics for e in episodes):
            continue
        recorded = torch.stack([e.intrinsics[name] for e in episodes]).to(eye)
        if bool((recorded != recorded[0]).any()):
            scaling[:, c] = recorded @ torch.linalg.inv(recorded[0])
            varies = True
    return scaling if varies else None


def still_span[T: IntVar](motion: Tensor[[T, *Mat4x4]] | None) -> tuple[int, int]:
    """The longest span (first, last + 1) where the recorded base motion stays still: every
    step `M_t⁻¹ M_t+1` below `STILL_STEP` (m, rad). No record: the whole episode."""

    if motion is None:
        return 0, 1 << 62
    step = se3.inverse(motion[:-1]) @ motion[1:]
    still = (step[:, :3, 3].norm(dim=-1) < STILL_STEP[0]) & (
        se3.so3_log(step[:, :3, :3]).norm(dim=-1) < STILL_STEP[1]
    )
    best, start = (0, 1), 0
    for t, ok in enumerate([*still.tolist(), False]):
        if not ok:
            if t + 1 - start > best[1] - best[0]:
                best = (start, t + 1)
            start = t + 1
    return best


def _base_departure[L: IntVar](
    robot: Robot[L, int, int],
    episodes: list[RigEpisode[int, int]],
    fitted: Tensor[[int, L, *Mat4x4]],
) -> tuple[tuple[float, float, float], ...]:
    """Per episode with a record: the fitted base against the recorded motion."""

    base = _moving_base(robot)
    if base is None:
        return ()
    out, start = [], 0
    for e in episodes:
        n = len(e.qpos)
        if e.base_motion is not None:
            recorded = moved_kinematics(robot, e)[:, base]
            delta = se3.inverse(recorded) @ fitted[start : start + n, base]
            cm = delta[:, :3, 3].norm(dim=-1) * 100
            deg = se3.so3_log(delta[:, :3, :3]).norm(dim=-1).rad2deg()
            out.append((float(cm.median()), float(cm.max()), float(deg.max())))
        start += n
    return tuple(out)


def _moving_base(robot: Robot[int, int, int]) -> int | None:
    """The body a recorded base motion moves: the frame-scope free body nearest the root."""

    frame = [robot.body(n) for n, f in robot.free_bodies.items() if f.scope == "frame"]
    return min(frame) if frame else None


def _bodies_at(
    robot: Robot[int, int, int], episodes: list[RigEpisode[int, int]]
) -> Tensor[[int, int, *Mat4x4]] | None:
    """Starting corrections of the free bodies [B, T] (all episodes): the base's recorded
    motion, identity elsewhere; None without any record."""

    base = _moving_base(robot)
    if base is None or all(e.base_motion is None for e in episodes):
        return None
    qpos = episodes[0].qpos
    eye = torch.eye(4, dtype=qpos.dtype, device=qpos.device)
    free = sorted(robot.body(n) for n in robot.free_bodies)
    total = sum(len(e.qpos) for e in episodes)
    out = eye.repeat(len(free), total, 1, 1)
    out[free.index(base)] = torch.cat(
        [
            eye.expand(len(e.qpos), 4, 4) if e.base_motion is None else e.base_motion.to(qpos)
            for e in episodes
        ]
    )
    return out


def moved_kinematics[L: IntVar, T: IntVar, NQ: IntVar](
    robot: Robot[L, int, int], episode: RigEpisode[T, NQ]
) -> Tensor[[T, L, *Mat4x4]]:
    """`world_from_body [T, L]` with the episode's recorded base motion applied to the base's
    subtree, as the bundle's free-body correction does: `W_p · P_b · M_t · J_b` (W_p the parent's
    world pose, P_b the base's offset, M_t the recorded motion, J_b its joints)."""

    world = forward_kinematics(robot, episode.qpos)
    base = _moving_base(robot)
    if base is None or episode.base_motion is None:
        return world
    parent = robot.body_parent[base]
    offset = robot.parent_from_body[base].to(world)
    local = offset @ episode.base_motion.to(world) @ se3.inverse(offset)
    change = world[:, parent] @ local @ se3.inverse(world[:, parent])
    below = torch.tensor([_under(robot, b, base) for b in range(len(robot.body_names))])
    return torch.where(below.to(world.device)[None, :, None, None], change[:, None] @ world, world)


def _under(robot: Robot[int, int, int], body: int, ancestor: int) -> bool:
    """Whether `body` is `ancestor` or below it in the kinematic tree."""

    while body > 0 and body != ancestor:
        body = robot.body_parent[body]
    return body == ancestor


def _body_corrections[L: IntVar](
    robot: Robot[L, int, int], qpos: Tensor[[int, int]], corrected: Tensor[[int, L, *Mat4x4]]
) -> Tensor[[int, *Mat4x4]]:
    """Each free body's correction at the first timestep (description → fitted): since a
    correction moves the body's subtree as `parent_from_body @ E`, `E = W_nominal⁻¹ · W_fitted`."""

    free = sorted(robot.body(n) for n in robot.free_bodies)
    nominal = forward_kinematics(robot, qpos[:1])[0]
    return se3.inverse(nominal[free]) @ corrected[0, free]


def spread(corrections: Tensor[[int, int, *Mat4x4]]) -> list[tuple[float, float]]:
    """Per free body (`corrections [G, F]`): largest disagreement between groups,
    `max_ij ‖t(E_i⁻¹ E_j)‖` (cm) and `max_ij ‖log R(E_i⁻¹ E_j)‖` (degrees)."""

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
