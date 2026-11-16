"""`r4r`: one subcommand per tool and pipeline stage.

Datasets are named by their delivery root, absolute or relative to `DELIVERY`
(e.g. `agibot-world/datasets/g2-cohort-01`).
"""

from __future__ import annotations

import functools
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import tyro

if TYPE_CHECKING:
    import pyarrow as pa
    from torch import Tensor

    from r4r.data.episode import CameraStream, Episode
    from r4r.data.priors import CameraPrior, DatasetPriors, JsonObject
    from r4r.matching import RoMaV2
    from r4r.rago.calibrate import FitConfig
    from r4r.rago.lens import Lens
    from r4r.rago.rig import KinematicFreedom
    from r4r.rago.robot import BodyMount, CameraMount, Robot
    from r4r.rago.stages import RigConfig, RigEpisode, RigFit
    from r4r.rago.transfer import Reference
    from r4r.types import Mat3x3, Mat4x4
    from r4r.viz.overlay import SavedSolution, Solution

DELIVERY = Path("/e/data1/datasets/playground/mmlaion/guybrush/delivery")


def dataset_root(name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else DELIVERY / path


def pinhole_guess(focal: float, aspect: float, width: int, height: int) -> list[list[float]]:
    """`K = [[f, 0, (W − 1)/2], [0, a·f, (H − 1)/2], [0, 0, 1]]`: a focal guess `f` (pixels),
    pixel aspect `a = fy / fx`, the principal point at the image centre."""

    return [[focal, 0, (width - 1) / 2], [0, focal * aspect, (height - 1) / 2], [0, 0, 1]]


# Ranges written for a fitted mount without a measured spread (m, deg): indicative
INDICATIVE_RANGES = (0.005, 1.0)


def range_fields(metres: float, degrees: float) -> dict[str, float]:
    """Priors ranges for a fitted pose from the spread of independent fits, floored at 2 mm
    and 0.2 deg (what a fit can resolve)."""

    return {
        "position_range": round(max(metres, 0.002), 4),
        "orientation_range_deg": round(max(degrees, 0.2), 3),
    }


def pose_fields(pose: Tensor[[*Mat4x4]]) -> dict[str, list[float]]:
    """A pose as priors fields: `position` (m) and `orientation` (quaternion w, x, y, z)."""

    from r4r.geometry import se3

    return {
        "position": [round(v, 5) for v in pose[:3, 3].tolist()],
        "orientation": [round(v, 6) for v in se3.matrix_to_quaternion(pose[:3, :3]).tolist()],
    }


@dataclass(frozen=True, slots=True)
class Inspect:
    """Cameras, joint mapping and how the robot's bodies move in one episode."""

    dataset: tyro.conf.Positional[str]
    episode: int = 0
    stride: int = 1

    def run(self) -> None:
        from r4r.data.lerobot import load_episode
        from r4r.inspect import inspect_episode

        root = dataset_root(self.dataset)
        episode = load_episode(root, self.episode, self.stride, decode=())
        print(inspect_episode(episode, root).text())


@dataclass(frozen=True, slots=True)
class Links:
    """Evidence about which body each camera rides on (still share, rotation residuals)."""

    dataset: tyro.conf.Positional[str]
    episode: int = 0
    cameras: tuple[str, ...] = ()  # default: all
    # Focal length guess in pixels (searched ×0.2..1.5); default 0.7 × image width
    focal: float | None = None
    top: int = 4

    def run(self) -> None:
        import torch

        from r4r.data.lerobot import load_episode
        from r4r.rago.kinematics import forward_kinematics
        from r4r.rago.klt import KLTConfig, require_tracks, track_video
        from r4r.rago.links import body_evidence
        from r4r.rago.robot import load_robot, qpos_from_state

        root = dataset_root(self.dataset)
        episode = load_episode(root, self.episode, decode=self.cameras or None)
        robot = load_robot(episode.robot).to("cpu", torch.float64)
        world_from_body = forward_kinematics(
            robot, qpos_from_state(robot, episode.state, episode.joints)
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for camera in episode.cameras:
            if self.cameras and camera.name not in self.cameras:
                continue
            height, width = camera.frames.shape[-2:]
            focal = self.focal or 0.7 * width
            k = torch.tensor(
                pinhole_guess(focal, 1.0, width, height), dtype=torch.float64, device=device
            )
            found = track_video(camera.frames, KLTConfig())
            require_tracks(found.track, camera.name, self.episode)
            evidence = body_evidence(found, world_from_body, k)
            groups = "; ".join(
                f"{residual:.2f}° "
                + ", ".join(robot.body_names[b] for b in group[:3])
                + (f" (+{len(group) - 3})" if len(group) > 3 else "")
                for group, residual in zip(
                    evidence.groups[: self.top], evidence.residual, strict=False
                )
            )
            declared = (
                f"declared {'moving' if camera.moving else 'static'} on {camera.mount or '?'}"
            )
            print(f"{camera.name}: still share {evidence.still_share:.2f} ({declared})\n  {groups}")


@dataclass(frozen=True, slots=True)
class Fit:
    """Fit a moving camera's body (among candidates), mount and focal length.

    Several episodes are one setup: fitted one by one, their consensus kept, then pooled.
    """

    dataset: tyro.conf.Positional[str]
    camera: tyro.conf.Positional[str]
    episodes: tuple[int, ...] = (0,)
    # Candidate bodies (default: the camera's prior or declared link)
    bodies: tuple[str, ...] = ()
    # Focal length guess in pixels when the priors give none (searched ×0.2..1.5);
    # default 0.7 × image width
    focal: float | None = None
    # Video/joint-state offsets to test (frames), on top of the priors' state_shift
    shifts: tuple[int, ...] = (0,)
    # Episodes agreeing within (metres, degrees) of the consensus are pooled; large values
    # pool every episode
    agreement: tuple[float, float] | None = None
    # Store the result in the camera's priors (mount, focal, ranges from episode agreement)
    write: bool = False

    def _config(self) -> FitConfig:
        import math
        from dataclasses import replace

        from r4r.rago.calibrate import FitConfig

        config = replace(FitConfig(), shifts=self.shifts)
        if self.agreement is not None:
            config = replace(config, agreement=(self.agreement[0], math.radians(self.agreement[1])))
        return config

    def run(self) -> None:
        import math

        import torch

        from r4r.data.lerobot import load_episode
        from r4r.data.priors import write_camera
        from r4r.geometry import se3
        from r4r.rago.calibrate import (
            MountFit,
            fit_moving_camera,
            fit_setup,
            refine,
            video_tracks,
        )
        from r4r.rago.klt import require_tracks
        from r4r.rago.lens import lens_from_priors
        from r4r.rago.robot import CameraMount, body_mounts, load_robot, qpos_from_state

        root = dataset_root(self.dataset)
        loaded = [load_episode(root, e, decode=(self.camera,)) for e in self.episodes]
        stream = next(c for c in loaded[0].cameras if c.name == self.camera)
        bodies = self.bodies or ((stream.mount,) if stream.mount else ())
        assert bodies, f"{self.camera}: no candidate body (give --bodies)"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mount = CameraMount.from_prior(self.camera, stream.prior, bodies[0], moving=True)
        free = body_mounts(loaded[0].priors)
        robot = load_robot(loaded[0].robot, cameras=(mount,), bodies=free).to(device, torch.float64)
        height, width = stream.frames.shape[-2:]
        focal = stream.prior.focal or self.focal or 0.7 * width
        k = torch.tensor(
            pinhole_guess(focal, stream.prior.aspect, width, height),
            dtype=torch.float64,
            device=device,
        )
        if stream.intrinsics is not None:  # known for this episode (per-episode table)
            k = stream.intrinsics.to(device)
        config = self._config()
        lens = lens_from_priors((stream.prior,), k)
        episodes = []
        for number, e in zip(self.episodes, loaded, strict=True):
            frames = next(c for c in e.cameras if c.name == self.camera).frames
            qpos = qpos_from_state(robot, e.state.to(device), e.joints)
            tracks = video_tracks(frames, k, config.klt, lens=lens)
            require_tracks(tracks.track, self.camera, number)
            episodes.append((qpos, tracks))
        ids = tuple(robot.body(b) for b in bodies)
        focal_range = stream.prior.focal_range if stream.prior.focal else None
        if stream.intrinsics is not None:
            focal_range = 0.0

        def describe(fit: MountFit) -> str:
            t = [round(v, 4) for v in fit.body_from_camera[:3, 3].tolist()]
            axis = [round(v, 3) for v in fit.body_from_camera[:3, 2].tolist()]
            quat = [
                round(v, 4) for v in se3.matrix_to_quaternion(fit.body_from_camera[:3, :3]).tolist()
            ]
            return (
                f"on {robot.body_names[fit.body]}: position {t} m, optical axis {axis}, "
                f"quaternion {quat}, focal {float(fit.intrinsics[0, 0]):.1f} px; "
                f"score {fit.score:.3f}, median {fit.median_error:.2f} px, "
                f"inliers {fit.inlier_fraction:.0%}"
            )

        spread = INDICATIVE_RANGES
        if mount.body_from_camera is not None:
            # Known mount (priors): refine within its ranges, no search
            start = robot.body_from_camera[robot.camera(self.camera)]
            body = robot.camera_body[robot.camera(self.camera)]
            fit = refine(robot, episodes, self.camera, body, start, k, config, k[0, 0], focal_range)
        elif len(episodes) == 1:
            fit = fit_moving_camera(robot, episodes, self.camera, ids, k, config, focal_range)
            for b, score in zip(bodies, fit.body_scores.tolist(), strict=True):
                print(f"  search score on {b}: {score:.3f}")
        else:
            setup = fit_setup(robot, episodes, self.camera, ids, k, config, focal_range)
            for e, single, agree in zip(
                self.episodes, setup.episodes, setup.agree.tolist(), strict=True
            ):
                print(f"  episode {e} {'agrees' if agree else 'DISAGREES'}: {describe(single)}")
            fit = setup.fit
            deviation = [
                se3.inverse(fit.body_from_camera) @ single.body_from_camera
                for single, agree in zip(setup.episodes, setup.agree.tolist(), strict=True)
                if agree and single.body == fit.body
            ]
            if len(deviation) > 1:
                d = torch.stack(deviation)
                spread = (
                    float(d[:, :3, 3].norm(dim=-1).max()),
                    math.degrees(float(se3.so3_log(d[:, :3, :3]).norm(dim=-1).max())),
                )
        print(f"{self.camera} {describe(fit)}")
        if len(self.shifts) > 1:
            scores = ", ".join(
                f"{s:+d}: {v:.3f}"
                for s, v in zip(self.shifts, fit.shift_scores.tolist(), strict=True)
            )
            print(
                f"  state shift scores ({scores}): best {fit.shift:+d} -> set state_shift in priors"
            )
        if self.write:
            values = {
                "mount": "moving",
                "link": robot.body_names[fit.body],
                **pose_fields(fit.body_from_camera.cpu()),
                **range_fields(*spread),
                "focal": round(float(fit.intrinsics[0, 0]), 2),
                "focal_range": focal_range if focal_range is not None else 0.0,
                **distortion_values(fit.lens, 0, stream.prior),
                "fitted": {
                    "episodes": list(self.episodes),
                    "median_px": round(fit.median_error, 3),
                    "score": round(fit.score, 4),
                },
            }
            refusal = write_refusal(fit.median_error)
            print(
                f"not written: {refusal}"
                if refusal
                else f"wrote {write_camera(root, self.camera, values, *write_target(loaded))}"
            )


def focal_ranges(streams: tuple[CameraStream[int, int, int], ...]) -> tuple[float, ...]:
    """Per camera, how far the bundles may refine a focal the priors give (relative std): its
    `focal_range`; 0 (held) for focals recorded per episode or given without a range."""

    return tuple(
        0.0 if c.intrinsics is not None else float(c.prior.focal_range or 0.0) for c in streams
    )


def write_refusal(median_error: float) -> str | None:
    """Why a fitted mount must not be written to the priors (None: it may be). A NaN median
    means no reprojection measured the fit (e.g. a head camera still on the parked spans the
    search uses): writing would store an unchecked mount with tightened ranges."""

    import math

    if not math.isfinite(median_error):
        return "no reprojection error measured (median NaN): the mount is not determined"
    return None


@dataclass(frozen=True, slots=True)
class _RigInputs:
    root: Path
    loaded: list[Episode[int, int]]
    streams: tuple[CameraStream[int, int, int], ...]
    names: tuple[str, ...]
    robot: Robot[int, int, int]
    device: str
    intrinsics: Tensor[[int, *Mat3x3]]
    episodes: list[RigEpisode]
    known_focal: tuple[bool, ...]
    known_mount: tuple[bool, ...]
    lens: Lens | None


def rig_config(
    every: int,
    groups: int,
    kinematic: KinematicFreedom | None,
    ranges: tuple[float, ...] | None,
    picks_weight: float = 10.0,
) -> RigConfig:
    """`r4r rig`'s stage settings: RoMa links every `every` timesteps, `groups` agreement
    groups, the kinematic freedom and focal ranges, and the picks' weight in the bundles."""

    from dataclasses import replace

    from r4r.rago.stages import RigConfig

    config = RigConfig()
    return replace(
        config,
        link_every=every,
        groups=groups,
        kinematic=kinematic,
        focal_ranges=ranges,
        bundle=replace(config.bundle, keypoint_weight=picks_weight),
    )


def _prepare_rig(
    dataset: str,
    numbers: tuple[int, ...],
    stride: int,
    focal: float,
    decode: tuple[str, ...] | None = None,
    device: str | None = None,
    picks: Path | None = None,
) -> _RigInputs:
    """Load episodes of one station and build what `calibrate_rig` needs from the priors (and
    robot keypoints from a `picks` file, if given).

    `decode`: the cameras whose frames are read (None: all); `device`: default CUDA if any.
    """

    import torch

    from r4r.data.lerobot import load_episode
    from r4r.data.priors import read_picks
    from r4r.rago.keypoints import picked
    from r4r.rago.kinematics import corrected_qpos
    from r4r.rago.lens import lens_from_priors
    from r4r.rago.robot import CameraMount, body_mounts, qpos_from_state
    from r4r.rago.stages import RigEpisode

    root = dataset_root(dataset)
    loaded = [load_episode(root, e, stride=stride, decode=decode) for e in numbers]
    streams = loaded[0].cameras
    names = tuple(c.name for c in streams)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mounts = tuple(CameraMount.from_prior(c.name, c.prior, c.mount, c.moving) for c in streams)
    bodies = body_mounts(loaded[0].priors)
    robot = _robot(loaded[0].robot, mounts, bodies).to(device, torch.float64)
    groups = {e.priors.group for e in loaded}
    if len(groups) > 1:
        raise SystemExit(
            f"episodes come from several priors groups {sorted(map(str, groups))}: each group "
            "has its own robot and cameras (pick episodes of one)"
        )
    stations = {e.station for e in loaded}
    if len(stations) > 1:
        raise SystemExit(
            f"episodes come from several stations {sorted(map(str, stations))}: a rig is "
            "calibrated per station (pick episodes of one)"
        )
    for name in names:
        own = {e.camera_stations.get(name) for e in loaded}
        if len(own) > 1:
            raise SystemExit(
                f"episodes come from several {name} stations {sorted(map(str, own))}: a rig "
                "is calibrated per station (pick episodes of one)"
            )
    k = []
    for c in streams:
        height, width = c.size
        if c.intrinsics is not None:  # known for this episode (per-episode table)
            k.append(c.intrinsics.tolist())
            continue
        k.append(pinhole_guess(c.prior.focal or focal * width, c.prior.aspect, width, height))
    intrinsics = torch.tensor(k, dtype=torch.float64, device=device)
    points, clicks = loaded[0].priors.robot_points, ()
    if picks is not None:
        named, clicks = read_picks(picks)
        points = {**points, **named}
    keypoints = [
        picked([p for p in clicks if p.episode == number], points, robot, names, stride)
        for number in numbers
    ]
    fitted = loaded[0].priors.kinematics
    offsets, delay = ({}, 0.0) if fitted is None else (fitted.offsets, fitted.time_offset / stride)
    episodes = [
        RigEpisode(
            corrected_qpos(
                robot, qpos_from_state(robot, e.state.to(device), e.joints), offsets, delay
            ),
            {c.name: c.frames for c in e.cameras if c.name in names},
            None if e.base_motion is None else e.base_motion.to(device),
            {c.name: c.pose.to(device) for c in e.cameras if c.pose is not None},
            {
                c.name: c.intrinsics.to(device)
                for c in e.cameras
                if c.name in names and c.intrinsics is not None
            },
            None if found is None else found.to(intrinsics),
        )
        for e, found in zip(loaded, keypoints, strict=True)
    ]
    return _RigInputs(
        root=root,
        loaded=loaded,
        streams=streams,
        names=names,
        robot=robot,
        device=device,
        intrinsics=intrinsics,
        episodes=episodes,
        known_focal=tuple(c.intrinsics is not None or c.prior.focal is not None for c in streams),
        known_mount=tuple(m.body_from_camera is not None for m in mounts),
        lens=lens_from_priors([c.prior for c in streams], intrinsics),
    )


@functools.cache
def _robot(
    path: Path, mounts: tuple[CameraMount, ...], bodies: tuple[BodyMount, ...]
) -> Robot[int, int, int]:
    """A description compiled once per process for each set of camera and body mounts."""

    from r4r.rago.robot import load_robot

    return load_robot(path, cameras=mounts, bodies=bodies)


@dataclass(frozen=True, slots=True)
class Rig:
    """Calibrate a whole rig in stages: moving mounts, then free bodies (arm bases) and static
    cameras over pooled episodes, checked by agreement between groups of episodes.

    Cameras, links, focals and free bodies come from the priors (see
    docs/dataset-onboarding.md, step 4).
    """

    dataset: tyro.conf.Positional[str]
    episodes: tuple[int, ...] = (0, 1, 2, 3)
    # Groups for the agreement check (episodes dealt round-robin)
    groups: int = 2
    # Frame stride when decoding (1: every frame)
    stride: int = 1
    # RoMa links between cameras every this many (strided) timesteps
    every: int = 10
    # Focal guess (× image width) for cameras the priors give none
    focal: float = 0.7
    # Store fitted mounts and focals in the camera priors
    write: bool = False
    # Save the whole solution (per-timestep camera and body poses, intrinsics) to this file,
    # e.g. for `research/overlay_video.py`
    save: Path | None = None
    # Robot points clicked in a few frames (`r4r.data.priors.read_picks`): they start and pin
    # cameras the scene cannot (a static-only rig, a head camera, a wrist seeing the other arm)
    picks: Path | None = None
    # Weight of a pick relative to a track observation in the bundles: a handful of picks
    # against ~10^5 track residuals barely moves the solve at the default (robomind2 ark)
    picks_weight: float = 10.0

    def run(self) -> None:
        from dataclasses import replace

        import torch

        from r4r.matching import RoMaV2
        from r4r.rago.stages import calibrate_rig, spread

        rig = _prepare_rig(self.dataset, self.episodes, self.stride, self.focal, picks=self.picks)
        loaded, streams, names, robot = rig.loaded, rig.streams, rig.names, rig.robot
        model = RoMaV2.from_checkpoint(device=rig.device)
        config = replace(
            rig_config(
                self.every,
                self.groups,
                kinematic_freedom(loaded[0].priors, robot, self.stride),
                focal_ranges(streams),
                self.picks_weight,
            ),
            refined=frozenset(c.name for c in streams if c.moving and c.prior.refine),
        )
        fit = calibrate_rig(
            robot,
            rig.episodes,
            names,
            rig.intrinsics,
            rig.known_focal,
            rig.known_mount,
            model,
            config,
            rig.lens,
        )

        free = sorted(robot.free_bodies, key=robot.body)
        disagreement = spread(fit.group_corrections)
        _print_rig(fit, streams, robot, self.episodes, free, disagreement)
        if config.kinematic is not None:
            _print_kinematics(fit, robot, self.stride)
        if self.save is not None:
            solution: SavedSolution = {
                "dataset": self.dataset,
                "episodes": list(self.episodes),
                "stride": self.stride,
                "names": names,
                "lengths": [len(e.qpos) for e in rig.episodes],
                "intrinsics": fit.intrinsics.cpu(),
                "episode_intrinsics": fit.episode_intrinsics.detach().cpu(),
                "world_from_body": fit.world_from_body.detach().float().cpu(),
                "mounts_at": fit.mounts_at.detach().cpu(),
                "camera_body": [robot.camera_body[robot.camera(n)] for n in names],
                "median_error": fit.median_error,
            }
            if fit.lens is not None:
                solution["distortion"] = fit.lens.distortion.detach().cpu()
                solution["fisheye"] = fit.lens.fisheye.cpu()
            torch.save(solution, self.save)
            print(f"saved {self.save}")
        if self.write:
            _write_rig(fit, rig, free, disagreement, self.episodes, self.groups)
            if config.kinematic is not None:
                _write_kinematics(fit, rig, self.stride, self.episodes)


def _write_rig(
    fit: RigFit,
    rig: _RigInputs,
    free: list[str],
    disagreement: list[tuple[float, float]],
    episodes: tuple[int, ...],
    groups: int,
) -> None:
    """Store a rig's fitted focals, lenses, static poses, unknown mounts and dataset-scope body
    corrections in the priors, noting the `episodes` and agreement `groups` they come from."""

    import torch

    from r4r.data.priors import write_body, write_camera
    from r4r.geometry import se3

    robot, streams = rig.robot, rig.streams
    target = write_target(rig.loaded)
    for c, stream in enumerate(streams):
        values: dict[str, object] = {
            "focal": round(float(fit.intrinsics[c, 0, 0]), 2),
            "focal_range": stream.prior.focal_range if rig.known_focal[c] else 0.0,
            **distortion_values(fit.lens, c, stream.prior),
        }
        if _focal_varies(fit, c):  # recorded per episode: no single focal to write
            del values["focal"], values["focal_range"]
        prior = robot.free_cameras.get(stream.name)
        if not stream.moving and prior is not None and prior.scope == "dataset":
            # A static camera fixed for the dataset: its pose in the base frame
            known = stream.prior.position is not None
            values |= {
                "mount": "static",
                **pose_fields(fit.camera_poses[0, c].cpu()),
                "position_range": stream.prior.position_range if known else 0.02,
                "orientation_range_deg": stream.prior.orientation_range_deg if known else 2.0,
            }
        if stream.moving and not rig.known_mount[c]:
            values |= {
                "mount": "moving",
                "link": robot.body_names[robot.camera_body[robot.camera(stream.name)]],
                **pose_fields(fit.mounts[c].cpu()),
                # No per-group mounts to measure a spread from: the indicative ranges
                **range_fields(*INDICATIVE_RANGES),
            }
        elif stream.moving and stream.prior.refine:  # a known mount the bundles refined
            values |= {"mount": "moving", **pose_fields(fit.mounts[c].cpu())}
        own = rig.loaded[0].camera_stations.get(stream.name)  # one per rig (`_prepare_rig`)
        where = target if own is None else (own, None)
        print(f"wrote {write_camera(rig.root, stream.name, values, *where)}")
    checked = len(fit.group_corrections) > 1
    for b, (name, (cm, deg)) in enumerate(zip(free, disagreement, strict=True)):
        if robot.free_bodies[name].scope != "dataset":
            continue
        prior = rig.loaded[0].priors.bodies[name]
        # The range left: how much independent groups disagree; without two groups to compare,
        # the priors' ranges stand
        ranges = (
            range_fields(cm / 100, deg)
            if checked
            else {
                "position_range": prior.position_range,
                "orientation_range_deg": prior.orientation_range_deg,
            }
        )
        # The rig started from the priors' correction (its kinematics apply it): the bundle's
        # correction is on top of it, so the whole correction is their composition
        correction = fit.body_corrections[b].cpu()
        if prior.position is not None and prior.orientation is not None:
            start = se3.from_rotation_translation(
                se3.quaternion_to_matrix(torch.tensor(prior.orientation, dtype=correction.dtype)),
                torch.tensor(prior.position, dtype=correction.dtype),
            )
            correction = start @ correction
        body = {
            **pose_fields(correction),
            **ranges,
            "fitted": {"episodes": list(episodes), "groups": groups},
        }
        print(f"wrote {write_body(rig.root, name, body, *target)}")
    _write_odometry(fit, rig, episodes)


def _write_kinematics(fit: RigFit, rig: _RigInputs, stride: int, episodes: tuple[int, ...]) -> None:
    """Store the bundle's joint and time offsets in the priors (`kinematics.offsets`,
    `time_offset`), on top of those already applied to the episodes' joint positions."""

    import json

    from r4r.data.priors import priors_path

    spec = rig.loaded[0].priors.kinematics
    assert spec is not None
    offsets = dict(spec.offsets)
    for j, name in enumerate(rig.robot.joint_names):
        value = float(fit.joint_offset[j])
        if value:
            offsets[name] = round(offsets.get(name, 0.0) + value, 6)
    path = priors_path(rig.root)
    raw = json.loads(path.read_text())
    target = kinematics_target(raw, rig.loaded[0].priors.group)
    target["kinematics"] |= {
        "offsets": offsets,
        "time_offset": round(spec.time_offset + fit.time_offset * stride, 4),
        "fitted": {"episodes": list(episodes)},
    }
    path.write_text(json.dumps(raw, indent=2) + "\n")
    print(f"wrote {path}")


def kinematics_target(raw: JsonObject, group: str | None) -> JsonObject:
    """The raw priors block whose `kinematics` a rig's fit goes into: the dataset's, or a
    priors group's (a dataset mixing robots), started from the dataset's when the group has
    none (either may be absent)."""

    if group is None:
        raw.setdefault("kinematics", {})
        return raw
    target = raw["groups"][group]
    target["kinematics"] = dict(target.get("kinematics") or raw.get("kinematics", {}))
    return target


def _write_odometry(fit: RigFit, rig: _RigInputs, episodes: tuple[int, ...]) -> None:
    """Fit the recorded base motion's dataset-level correction to the bundle's free base
    (`odometry.fit_gain`) and store it, on top of any correction already applied."""

    from r4r.data.priors import write_odometry_correction
    from r4r.rago.odometry import fit_gain
    from r4r.rago.stages import solved_base_motion

    spec = rig.loaded[0].priors.base_odometry
    if spec is None:
        return
    solved = solved_base_motion(rig.robot, rig.episodes, fit.world_from_body.detach())
    if not solved:
        return
    motions = [e.base_motion for e in rig.episodes if e.base_motion is not None]
    gain = fit_gain([m.cpu() for m in motions], [s.cpu() for s in solved], 1 / rig.loaded[0].fps)
    if spec.correction is not None:
        gain = spec.correction.then(gain)
    values = {
        "scale": round(gain.scale, 5),
        "heading_gain": round(gain.heading_gain, 5),
        "yaw_rate": round(gain.yaw_rate, 6),
        "fitted": {"episodes": list(episodes)},
    }
    print(f"wrote {write_odometry_correction(rig.root, values)}")


def _focal_varies(fit: RigFit, camera: int) -> bool:
    """Whether the camera's intrinsics differ between the solved episodes."""

    k = fit.episode_intrinsics[:, camera]
    return bool((k != k[:1]).any())


def station_of(loaded: list[Episode[int, int]]) -> str | None:
    """Where fitted values go: the station all the episodes share (the episodes table's
    `station`), else the dataset level (None: no station, or episodes of several, e.g. a wrist
    mount fitted across setups). The first station written starts the priors' `stations`."""

    stations = {e.station for e in loaded}
    return stations.pop() if len(stations) == 1 else None


def write_target(loaded: list[Episode[int, int]]) -> tuple[str | None, str | None]:
    """Where fitted values go, as (station, group): the episodes' station (`station_of`), else
    their priors group (a dataset mixing robots), else the dataset level."""

    station = station_of(loaded)
    return station, None if station is not None else loaded[0].priors.group


def distortion_values(
    lens: Lens | None, camera: int, prior: CameraPrior
) -> dict[str, list[float] | float]:
    """Priors fields of a camera's fitted distortion (none for an ideal pinhole): held from
    now on unless its prior gave the coefficients with a range."""

    if lens is None or not (bool(lens.fisheye[camera]) or bool(lens.distortion[camera].any())):
        return {}
    size = 4 if bool(lens.fisheye[camera]) else 3
    known = prior.distortion is not None
    return {
        "distortion": [round(v, 6) for v in lens.distortion[camera, :size].tolist()],
        "distortion_range": (prior.distortion_range or 0.0) if known else 0.0,
    }


def kinematic_freedom(
    priors: DatasetPriors, robot: Robot[int, int, int], stride: int
) -> KinematicFreedom | None:
    """The priors' `kinematics` block as a `KinematicFreedom` for `robot` (None if absent):
    hinge offsets in degrees, slide offsets in millimetres, the time offset in frames at the
    dataset's rate (converted to strided timesteps)."""

    import math

    from r4r.rago.rig import KinematicFreedom
    from r4r.rago.robot import HINGE

    spec = priors.kinematics
    if spec is None:
        return None
    only = set(robot.joint_names if spec.joints is None else spec.joints)
    unknown = only - set(robot.joint_names)
    if unknown:
        raise ValueError(f"priors kinematics.joints: {sorted(unknown)} are not robot joints")
    stds = []
    for name, kind in zip(robot.joint_names, robot.joint_kind, strict=True):
        if name not in only:
            stds.append(0.0)
        elif kind == HINGE:
            stds.append(math.radians(spec.joint_offset_deg))
        else:
            stds.append(spec.joint_offset_mm / 1000)
    return KinematicFreedom(joint_std=tuple(stds), time_std=spec.time_offset_frames / stride)


def _print_kinematics(fit: RigFit, robot: Robot[int, int, int], stride: int) -> None:
    """Fitted joint and time offsets, with each group's value (their spread is the check)."""

    import math

    from r4r.rago.robot import HINGE

    groups = fit.group_time_offsets
    print(
        f"time offset: {fit.time_offset * stride:+.2f} frames (groups "
        + ", ".join(f"{t * stride:+.2f}" for t in groups)
        + ") on top of state_shift"
    )
    for j, name in enumerate(robot.joint_names):
        value = float(fit.joint_offset[j])
        if value == 0.0:
            continue
        scale, unit = (math.degrees(1), "deg") if robot.joint_kind[j] == HINGE else (1000, "mm")
        each = ", ".join(f"{float(g[j]) * scale:+.2f}" for g in fit.group_joint_offsets)
        print(f"joint offset {name}: {value * scale:+.2f} {unit} (groups {each})")


def _print_rig(
    fit: RigFit,
    streams: tuple[CameraStream[int, int, int], ...],
    robot: Robot[int, int, int],
    episodes: tuple[int, ...],
    free: list[str],
    disagreement: list[tuple[float, float]],
) -> None:
    """What `r4r rig` found, per camera, mobile-base episode and free body."""

    import math

    from r4r.geometry import se3

    if math.isnan(fit.median_error):
        print("no track constrains the rig (a static-only rig is placed by robot keypoints)")
    else:
        print(f"median {fit.median_error:.2f} px, inliers {fit.inlier_fraction:.0%}")
    if not math.isnan(fit.keypoint_error):
        print(f"robot keypoints: median {fit.keypoint_error:.2f} px")
    for c, agree in enumerate(fit.mount_agreement):
        if agree is not None:
            print(f"{streams[c].name}: mount search, {agree[0]}/{agree[1]} episodes agree")
    for e, (median, worst, deg) in zip(episodes, fit.base_departure, strict=False):
        print(
            f"episode {e}: base departs from its record by {median:.1f} cm median, "
            f"{worst:.1f} cm / {deg:.2f} deg at most"
        )
    for c, stream in enumerate(streams):
        pose = fit.mounts[c].cpu()
        where = robot.body_names[robot.camera_body[robot.camera(stream.name)]]
        lens = distortion_values(fit.lens, c, stream.prior).get("distortion")
        focal = fit.episode_intrinsics[:, c, 0, 0]
        print(
            f"{stream.name} on {where}: position "
            f"{[round(v, 4) for v in pose[:3, 3].tolist()]} m, "
            f"focal {float(fit.intrinsics[c, 0, 0]):.1f} px"
            + (
                f" (per episode {float(focal.min()):.1f}-{float(focal.max()):.1f})"
                if _focal_varies(fit, c)
                else ""
            )
            + ("" if lens is None else f", {stream.prior.lens} distortion {lens}")
        )
    for b, (name, (cm, deg)) in enumerate(zip(free, disagreement, strict=True)):
        if robot.free_bodies[name].scope == "frame":
            continue  # held at each episode's first frame: see the departure above
        offset = fit.body_corrections[b]
        # Groups check dataset-scope bodies only (an episode-scope one differs by design)
        checked = robot.free_bodies[name].scope == "dataset" and len(fit.group_corrections) > 1
        print(
            f"{name}: correction {[round(v, 4) for v in offset[:3, 3].tolist()]} m, "
            f"{math.degrees(float(se3.so3_log(offset[:3, :3]).norm())):.2f} deg"
            + (f"; groups disagree by {cm:.1f} cm / {deg:.2f} deg" if checked else "")
        )


@dataclass(frozen=True, slots=True)
class Picks:
    """Robot keypoints of a static camera transferred from calibrated frames by RoMa v2: a picks
    file for `r4r rig --picks`, without clicks (`r4r.rago.transfer`).

    References are episodes where the camera's pose is known: recorded for the episode
    (per-episode table, pose column) or held by the priors.
    """

    dataset: tyro.conf.Positional[str]
    output: tyro.conf.Positional[Path]
    camera: str
    # Episodes to place the camera in (targets)
    episodes: tuple[int, ...]
    # Calibrated episodes to transfer from (of `reference_dataset`, default this dataset)
    references: tuple[int, ...]
    reference_dataset: str | None = None
    # Bodies whose surface is sampled (default: all but the world); leave out parts whose
    # geometry is not the real robot's (a different gripper)
    bodies: tuple[str, ...] = ()
    # Target frames per episode, spread over its joint configurations
    frames: int = 16
    # Surface points per reference frame
    points: int = 96
    # Decoding strides of the target and reference videos
    stride: int = 1
    reference_stride: int = 3
    certainty: float = 0.1
    # Forward-backward tolerance (reference pixels)
    cycle: float = 3.0
    # Reference crop margin around its points (fraction; inf: whole frames)
    margin: float = 0.5

    def run(self) -> None:
        import json

        import torch

        from r4r.data.lerobot import load_episode
        from r4r.rago.robot import body_mounts, compile_model, qpos_from_state
        from r4r.rago.transfer import TransferConfig, as_picks, spread_frames, transfer

        config = TransferConfig(
            certainty=self.certainty,
            cycle=self.cycle,
            margin=self.margin,
            frames=self.frames,
            points=self.points,
        )
        source = dataset_root(self.reference_dataset or self.dataset)
        first = load_episode(source, self.references[0], decode=())
        mounts = body_mounts(first.priors)
        robot = _robot(first.robot, (), mounts).to("cpu", torch.float64)
        model = compile_model(first.robot, None, (), mounts, visual=True)
        bodies = {robot.body(b) for b in self.bodies} or set(range(1, len(robot.body_names)))
        reference = _transfer_reference(
            source, self.references, self.camera, self.reference_stride, robot
        )
        matcher = _matcher("cuda" if torch.cuda.is_available() else "cpu")
        root = dataset_root(self.dataset)
        points, picks = {}, []
        for number in self.episodes:
            episode = load_episode(root, number, stride=self.stride, decode=(self.camera,))
            stream = next(c for c in episode.cameras if c.name == self.camera)
            qpos = qpos_from_state(robot, episode.state, episode.joints)
            chosen = spread_frames(qpos, config.frames)
            moved = transfer(
                matcher,
                model,
                robot,
                reference,
                stream.frames[chosen],
                qpos[chosen],
                bodies,
                config,
            )
            named, found = as_picks(
                moved, robot.body_names, number, self.camera, (chosen * self.stride).tolist()
            )
            points |= named
            picks += found
            print(f"episode {number}: {len(named)} points in {len(found)} of {len(chosen)} frames")
        raw = {
            "points": {n: {"body": p.body, "offset": list(p.offset)} for n, p in points.items()},
            "picks": [
                {"episode": p.episode, "camera": p.camera, "frame": p.frame, "pixels": p.pixels}
                for p in picks
            ],
        }
        self.output.write_text(json.dumps(raw))
        print(f"wrote {self.output} ({len(points)} points in {len(picks)} frames)")


def _transfer_reference(
    root: Path, numbers: tuple[int, ...], name: str, stride: int, robot: Robot[int, int, int]
) -> Reference[int, int]:
    """Frames of camera `name` in calibrated episodes: its pose recorded for the episode, else
    the priors'; its intrinsics recorded, else from the priors' focal."""

    import torch

    from r4r.data.lerobot import load_episode
    from r4r.geometry import se3
    from r4r.rago.robot import qpos_from_state
    from r4r.rago.transfer import Reference

    frames, qpos, poses, intrinsics = [], [], [], []
    for number in numbers:
        episode = load_episode(root, number, stride=stride, decode=(name,))
        stream = next(c for c in episode.cameras if c.name == name)
        prior = stream.prior
        if stream.moving or prior.distortion:
            raise SystemExit(f"{name}: references are static ideal-pinhole cameras")
        pose = stream.pose
        if pose is None and prior.position is not None and prior.orientation is not None:
            rotation = se3.quaternion_to_matrix(
                torch.tensor(prior.orientation, dtype=torch.float64)
            )
            pose = se3.from_rotation_translation(rotation, torch.tensor(prior.position).double())
        if pose is None or (stream.intrinsics is None and prior.focal is None):
            raise SystemExit(f"{name} in episode {number}: no known pose or focal to transfer from")
        height, width = stream.size
        k = stream.intrinsics
        if k is None:
            k = torch.tensor(
                pinhole_guess(prior.focal or 0.0, prior.aspect, width, height), dtype=torch.float64
            )
        steps = len(stream.frames)
        frames.append(stream.frames)
        qpos.append(qpos_from_state(robot, episode.state, episode.joints))
        poses.append(pose.double().expand(steps, 4, 4))
        intrinsics.append(k.double().expand(steps, 3, 3))
    return Reference(torch.cat(frames), torch.cat(qpos), torch.cat(poses), torch.cat(intrinsics))


@dataclass(frozen=True, slots=True)
class Sheet:
    """Contact sheet: camera frames next to the robot skeleton (top, side, front views)."""

    dataset: tyro.conf.Positional[str]
    output: tyro.conf.Positional[Path]
    episode: int = 0
    moments: int = 4

    def run(self) -> None:
        import cv2
        import torch

        from r4r.data.lerobot import load_episode
        from r4r.rago.robot import load_robot
        from r4r.viz.sheet import contact_sheet

        episode = load_episode(dataset_root(self.dataset), self.episode)
        robot = load_robot(episode.robot).to("cpu", torch.float64)
        image = contact_sheet(episode, robot, self.moments)
        cv2.imwrite(str(self.output), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"wrote {self.output} ({image.shape[1]}x{image.shape[0]})")


@dataclass(frozen=True, slots=True)
class Export:
    """The robot description (MJCF) with every camera whose mount the priors know placed."""

    dataset: tyro.conf.Positional[str]
    output: tyro.conf.Positional[Path]

    def run(self) -> None:
        from r4r.data.lerobot import load_episode
        from r4r.rago.robot import CameraMount, body_mounts, describe

        episode = load_episode(dataset_root(self.dataset), 0, decode=())
        mounts = tuple(
            CameraMount.from_prior(c.name, c.prior, c.mount, c.moving)
            for c in episode.cameras
            if c.prior.position is not None and c.prior.orientation is not None
        )
        self.output.write_text(describe(episode.robot, mounts, body_mounts(episode.priors)))
        print(f"wrote {self.output} ({len(mounts)} cameras placed)")


@dataclass(frozen=True, slots=True)
class Overlay:
    """The robot (description meshes, solved kinematics) drawn over every camera with a solution
    saved by `r4r rig --save`: a video (`.mp4`) of one solved episode, or a still grid (`.png`)
    of a few moments of every solved episode. The visual check of a calibration."""

    solution: tyro.conf.Positional[Path]
    output: tyro.conf.Positional[Path]
    # Video: which solved episode (index among them), every n-th timestep
    episode: int = 0
    every: int = 1
    # Still: moments per solved episode
    moments: int = 3
    height: int = 360

    def run(self) -> None:
        import cv2
        import numpy as np
        import torch

        from r4r.data.lerobot import load_episode
        from r4r.rago.robot import CameraMount, body_mounts, compile_model
        from r4r.viz.overlay import overlay_frames, write_video

        solution: SavedSolution = torch.load(self.solution, weights_only=False)
        root = dataset_root(solution["dataset"])
        names, model, rows = solution["names"], None, []
        still = self.output.suffix.lower() == ".png"
        indices = range(len(solution["episodes"])) if still else [self.episode]
        fps = 1.0
        for index in indices:
            number = solution["episodes"][index]
            episode = load_episode(root, number, stride=solution["stride"], decode=names)
            if model is None:
                mounts = tuple(
                    CameraMount.from_prior(c.name, c.prior, c.mount, c.moving)
                    for c in episode.cameras
                )
                model = compile_model(
                    episode.robot, None, mounts, body_mounts(episode.priors), visual=True
                )
            videos = {
                c.name: c.frames.permute(0, 2, 3, 1).numpy()
                for c in episode.cameras
                if c.name in names
            }
            steps = solution["lengths"][index]
            if still:
                times = np.linspace(0, steps - 1, self.moments).round().astype(int).tolist()
            else:
                times = list(range(0, steps, self.every))
            label = (
                f"{solution['dataset']}  episode {number}  fit median "
                f"{solution['median_error']:.2f} px"
            )
            frames = overlay_frames(solution, videos, model, index, times, self.height, label)
            if not still:
                fps = episode.fps / self.every
                count = write_video(frames, self.output, fps)
                print(f"wrote {self.output} ({count} frames, {fps:.1f} fps)")
                return
            rows.append(np.vstack(list(frames)))
        image = np.vstack(rows)
        cv2.imwrite(str(self.output), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"wrote {self.output} ({image.shape[1]}x{image.shape[0]})")


@dataclass(frozen=True, slots=True)
class Preview:
    """The robot drawn over every camera from the priors alone (no solve): the description at
    the recorded joint states, each camera where the priors (or the description) put it, with
    the priors' focal (else `--focal` × width). The agent's first tool on a new dataset: place
    cameras, check joints, signs, timing and focal by eye, then calibrate. `.png`: a grid of
    moments; `.mp4`: the episode."""

    dataset: tyro.conf.Positional[str]
    output: tyro.conf.Positional[Path]
    episode: int = 0
    moments: int = 4
    stride: int = 1
    focal: float = 0.7
    height: int = 360

    def run(self) -> None:
        import cv2
        import numpy as np
        import torch

        from r4r.data.lerobot import load_episode
        from r4r.rago.kinematics import corrected_qpos, forward_kinematics
        from r4r.rago.lens import lens_from_priors
        from r4r.rago.robot import (
            CameraMount,
            body_mounts,
            compile_model,
            load_robot,
            qpos_from_state,
        )
        from r4r.viz.overlay import overlay_frames, write_video

        episode = load_episode(dataset_root(self.dataset), self.episode, stride=self.stride)
        streams = episode.cameras
        mounts = tuple(CameraMount.from_prior(c.name, c.prior, c.mount, c.moving) for c in streams)
        bodies = body_mounts(episode.priors)
        robot = load_robot(episode.robot, cameras=mounts, bodies=bodies).to("cpu", torch.float64)
        # The priors' fitted kinematics (joint offsets, lag), as every solve applies them
        fitted = episode.priors.kinematics
        qpos = qpos_from_state(robot, episode.state, episode.joints)
        if fitted is not None:
            qpos = corrected_qpos(robot, qpos, fitted.offsets, fitted.time_offset / self.stride)
        names = tuple(c.name for c in streams)
        index = [robot.camera(n) for n in names]
        k = []
        for c in streams:
            height, width = c.frames.shape[-2:]
            if c.intrinsics is not None:
                k.append(c.intrinsics.tolist())
                continue
            k.append(
                pinhole_guess(c.prior.focal or self.focal * width, c.prior.aspect, width, height)
            )
        steps = len(qpos)
        placed = robot.body_from_camera[index].clone()
        for c, stream in enumerate(streams):  # a static camera's recorded pose this episode
            if stream.pose is not None:
                placed[c] = stream.pose.to(placed)
        solution: Solution = {
            "dataset": self.dataset,
            "names": names,
            "lengths": [steps],
            "world_from_body": forward_kinematics(robot, qpos),
            "mounts_at": placed[:, None].expand(-1, steps, 4, 4),
            "camera_body": [robot.camera_body[i] for i in index],
            "intrinsics": torch.tensor(k, dtype=torch.float64),
        }
        lens = lens_from_priors([c.prior for c in streams], solution["intrinsics"])
        if lens is not None:
            solution["distortion"], solution["fisheye"] = lens.distortion, lens.fisheye
        model = compile_model(episode.robot, None, mounts, bodies, visual=True)
        videos = {c.name: c.frames.permute(0, 2, 3, 1).numpy() for c in streams}
        label = f"{self.dataset}  episode {self.episode}  PRIORS ONLY (no solve)"
        still = self.output.suffix.lower() == ".png"
        times = (
            np.linspace(0, steps - 1, self.moments).round().astype(int).tolist()
            if still
            else list(range(steps))
        )
        frames = overlay_frames(solution, videos, model, 0, times, self.height, label)
        if still:
            image = np.vstack(list(frames))
            cv2.imwrite(str(self.output), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print(f"wrote {self.output} ({image.shape[1]}x{image.shape[0]})")
        else:
            count = write_video(frames, self.output, episode.fps)
            print(f"wrote {self.output} ({count} frames)")


@dataclass(frozen=True, slots=True)
class Focal:
    """A camera's focal length from ARROW (gen2 adapter): a draft when nothing is recorded."""

    dataset: tyro.conf.Positional[str]
    camera: tyro.conf.Positional[str]
    episodes: tuple[int, ...] = (0, 1, 2)
    # Frames per episode, spread over it
    frames: int = 64
    adapter: Path = Path("/e/scratch/reformo/blanchon1/r4r-cache/arrow_gen2.safetensors")

    def run(self) -> None:
        import torch

        from r4r.arrow.adapter import load_adapter
        from r4r.arrow.focal import self_drafted_focal
        from r4r.arrow.models.arrow import ARROW
        from r4r.data.lerobot import load_episode

        model = ARROW.from_pretrained(device="cuda")
        encoder = load_adapter(model, self.adapter)
        root = dataset_root(self.dataset)
        estimates = []
        for e in self.episodes:
            cameras = load_episode(root, e, decode=(self.camera,)).cameras
            stream = next(c for c in cameras if c.name == self.camera)
            pick = torch.linspace(0, len(stream.frames) - 1, min(self.frames, len(stream.frames)))
            first, drafted = self_drafted_focal(model, encoder, stream.frames[pick.round().long()])
            estimates.append(drafted)
            known = stream.intrinsics
            recorded = "" if known is None else f", recorded {float(known[0, 0]):.1f}"
            print(f"episode {e}: fx {drafted:.1f} px (no draft {first:.1f}{recorded})")
        width = int(stream.frames.shape[-1])
        pooled = float(torch.tensor(estimates).median())
        spread = (max(estimates) - min(estimates)) / pooled
        print(
            f"{self.camera}: fx {pooled:.1f} px = {pooled / width:.3f} × width {width};"
            f" spread over episodes {spread:.1%}"
        )


@dataclass(frozen=True, slots=True)
class Reconstruct:
    """Every episode with the dataset's frozen priors (after `r4r rig --write`), by the path its
    unknowns allow (`r4r.plan`): fast (every value held or recorded: the priors and forward
    kinematics, no video read), light (episode-scope unknowns fitted on a few keyframes, CPU:
    `r4r.rago.light`) or full (something never calibrated: the whole rig per episode).

    Writes `<out>/<family>/<dataset>/shard-<k>.parquet`, one per `--shard-size` episodes of the
    selection: one row per frame in the columns of docs/output-spec.md (`r4r.output`: camera
    poses, intrinsics and distortion, end effectors, base, aligned state, in the episode
    frame); per episode (schema metadata `r4r.episodes`, JSON) its route, fit check and errors,
    or the error that stopped it; the dataset's layout (`r4r.layout`: feature info). Shards
    already written are skipped. `r4r merge` turns them into a LeRobot dataset. `--plan` only
    counts the routes and estimates the cost (`r4r.plan`)."""

    dataset: tyro.conf.Positional[str]
    out: tyro.conf.Positional[Path] = Path("reconstruction")
    # Episodes to process (default: every episode of the dataset)
    episodes: tuple[int, ...] = ()
    # Episodes per output file
    shard_size: int = 1000
    # This worker's part, `i/n`: the shards k with k mod n = i (e.g. a Slurm array task)
    shard: str = "0/1"
    # Solve every episode in full, also those the priors determine
    solve: bool = False
    # The fit check on keyframes: `auto` for real (not simulated) episodes holding an
    # episode-scope camera at its record
    check: Literal["auto", "on", "off"] = "auto"
    # Count each route and estimate CPU / GPU hours, without reconstructing
    plan: bool = False
    # Episodes whose setup leaves a camera uncalibrated: `solve` the whole rig (full path), or
    # write that camera as NaN (`nan`) and the rest by their own route (an episode whose robot is
    # not calibrated, a body or the odometry, then fails with the reason: no full solve)
    uncalibrated: Literal["solve", "nan"] = "solve"
    # Every `stride`-th frame (1 for `r4r merge`: every video frame)
    stride: int = 1
    # RoMa links between cameras every this many (strided) timesteps (full path)
    every: int = 10
    # Focal guess (× image width) for cameras the priors give none (full path)
    focal: float = 0.7

    def run(self) -> None:
        import time

        from r4r.data.lerobot import episode_count
        from r4r.reconstruction import limit_threads, shard_path, shards, write_shard

        root = dataset_root(self.dataset)
        if self.plan:
            from r4r.plan import dataset_plan, describe

            print(describe([dataset_plan(root, self.episodes or None)]))
            return
        threads = limit_threads()
        if threads is not None:
            print(f"{threads} CPU threads (set OMP_NUM_THREADS to choose)")
        selection = list(self.episodes) or list(range(episode_count(root)))
        worker, workers = (int(v) for v in self.shard.split("/"))
        if not 0 <= worker < workers:
            raise SystemExit(f"--shard {self.shard}: expected i/n with 0 <= i < n")
        target = self.out / root.parent.parent.name / root.name
        target.mkdir(parents=True, exist_ok=True)
        size = self.shard_size
        for k in shards(selection, size, worker, workers):
            path = shard_path(target, k)
            if path.exists():
                print(f"shard {k}: done, skipped ({path})")
                continue
            start = time.perf_counter()
            tables, summary, layout = [], [], None
            for number in selection[k * size : (k + 1) * size]:
                began = time.perf_counter()
                try:
                    table, record, layout = self._episode(number)
                except Exception as error:  # one bad episode must not stop the others
                    record: dict[str, object] = {
                        "episode": number,
                        "error": f"{type(error).__name__}: {error}",
                    }
                    print(f"episode {number}: failed: {record['error']}")
                else:
                    tables.append(table)
                record["seconds"] = round(time.perf_counter() - began, 3)
                summary.append(record)
            write_shard(path, tables, summary, layout)
            routes = Counter(str(r.get("route", "failed")) for r in summary)
            flagged = sum(bool(r.get("flagged")) for r in summary)
            print(
                f"shard {k}: {len(summary)} episodes ({dict(routes)}, {flagged} flagged) -> "
                f"{path} ({time.perf_counter() - start:.1f} s)"
            )

    def _episode(self, number: int) -> tuple[pa.Table, dict[str, object], str]:
        """One episode's frame table, summary and the dataset's layout (JSON), by its route
        (class doc)."""

        import torch

        from r4r.data.lerobot import simulated
        from r4r.plan import episode_unknowns
        from r4r.rago.light import LightConfig
        from r4r.rago.stages import moved_kinematics
        from r4r.reconstruction import held_poses, keyframe_episode, uncalibrated_cameras

        rig = _prepare_rig(self.dataset, (number,), self.stride, self.focal, (), "cpu")
        episode = rig.episodes[0]
        priors = rig.loaded[0].priors
        moving = {s.name for s in rig.streams if s.moving}
        unknowns = episode_unknowns(
            priors, rig.names, moving, set(episode.poses), set(episode.intrinsics)
        )
        summary: dict[str, object] = {"episode": number, "route": unknowns.route}
        lacking = uncalibrated_cameras(unknowns, rig.names)
        if self.solve or (unknowns.route == "full" and self.uncalibrated == "solve"):
            return self._solved(number, summary | {"missing": list(unknowns.missing)})
        if lacking is None:  # `nan`: a body or the odometry is not calibrated, not a camera
            reasons = "; ".join((*unknowns.setup, *unknowns.missing))
            raise ValueError(f"not calibrated (no camera alone): {reasons}")
        # Cameras not calibrated for this episode are written as NaN; the others take the
        # route their own unknowns allow
        lacking = lacking or ()
        index = [c for c, n in enumerate(rig.names) if n not in lacking]
        names = tuple(rig.names[c] for c in index)
        if lacking:
            unknowns = episode_unknowns(
                priors, names, moving, set(episode.poses), set(episode.intrinsics)
            )
            summary |= {"route": unknowns.route, "uncalibrated": list(lacking)}
        check = self.check == "on" or (self.check == "auto" and simulated(rig.root) is False)
        checked = unknowns.recorded if check else ()
        poses = held_poses(rig.robot, episode, rig.names)
        world = moved_kinematics(rig.robot, episode)
        if unknowns.free or checked:
            found_poses, world, found = keyframe_episode(
                rig.root,
                number,
                rig.robot,
                episode,
                names,
                rig.intrinsics[index],
                None if rig.lens is None else rig.lens.select(torch.tensor(index)),
                self.stride,
                unknowns.free,
                checked,
                LightConfig(),
            )
            poses[index] = found_poses
            summary |= found
            if found.get("flagged"):
                summary["route"] = "light"
        light = set(unknowns.free) | (set(checked) if summary.get("flagged") else set())
        if light - set(names):  # a body solved (a re-placed robot base): every camera with it
            light |= set(names)
        summary["cameras"] = {
            n: "not calibrated"
            if n in lacking
            else "light solve"
            if n in light
            else "recorded"
            if n in episode.poses
            else "held calibration"
            for n in rig.names
        }
        missing = tuple(c for c, n in enumerate(rig.names) if n in lacking)
        return self._written(
            rig, number, episode.qpos, poses, world, rig.intrinsics, rig.lens, summary, missing
        )

    def _written(
        self,
        rig: _RigInputs,
        number: int,
        qpos: Tensor[[int, int]],
        poses: Tensor[[int, int, *Mat4x4]],
        world: Tensor[[int, int, *Mat4x4]],
        intrinsics: Tensor[[int, *Mat3x3]],
        lens: Lens | None,
        summary: dict[str, object],
        uncalibrated: tuple[int, ...] = (),
    ) -> tuple[pa.Table, dict[str, object], str]:
        """The episode's frame table in the spec's columns (`r4r.output`); `uncalibrated`
        cameras (indices) are written as NaN."""

        from dataclasses import replace

        from r4r.output import frame_table
        from r4r.reconstruction import dataset_layout, episode_output

        loaded = rig.loaded[0]
        layout = dataset_layout(rig.root, rig.robot, loaded.priors, rig.streams, loaded.joints)
        values = episode_output(
            rig.root,
            number,
            layout,
            rig.robot,
            loaded.joints,
            qpos,
            self.stride,
            poses,
            world,
            intrinsics,
            lens,
        )
        values = replace(values, uncalibrated=uncalibrated)
        return frame_table(layout, values), summary, layout.to_json()

    def _solved(
        self, number: int, summary: dict[str, object]
    ) -> tuple[pa.Table, dict[str, object], str]:
        """The full path: the whole rig on this episode (free values within their ranges)."""

        import time
        from dataclasses import replace

        import torch

        from r4r.rago.kinematics import corrected_qpos
        from r4r.rago.stages import RigConfig, calibrate_rig

        start = time.perf_counter()
        rig = _prepare_rig(self.dataset, (number,), self.stride, self.focal)
        kinematic = kinematic_freedom(rig.loaded[0].priors, rig.robot, self.stride)
        unknown = [
            n
            for n, s, k in zip(rig.names, rig.streams, rig.known_mount, strict=True)
            if s.moving and not k
        ]
        if unknown:
            print(f"warning: mounts of {unknown} are not in the priors: searched per episode")
        config = replace(RigConfig(), link_every=self.every, groups=0, kinematic=kinematic)
        fit = calibrate_rig(
            rig.robot,
            rig.episodes,
            rig.names,
            rig.intrinsics,
            rig.known_focal,
            rig.known_mount,
            _matcher(rig.device),
            config,
            rig.lens,
        )
        world = fit.world_from_body.detach().cpu().double()
        mounts = fit.mounts_at.detach().cpu().double()
        bodies = [rig.robot.camera_body[rig.robot.camera(n)] for n in rig.names]
        poses = torch.stack([world[:, b] @ mounts[c] for c, b in enumerate(bodies)])
        # The joint state as the bundle saw it: its joint and time offsets on top
        offsets = {
            name: float(v)
            for name, v in zip(rig.robot.joint_names, fit.joint_offset.tolist(), strict=True)
            if v
        }
        robot = rig.robot.to("cpu")
        qpos = corrected_qpos(robot, rig.episodes[0].qpos.cpu(), offsets, fit.time_offset)
        print(
            f"episode {number}: median {fit.median_error:.2f} px, inliers "
            f"{fit.inlier_fraction:.0%} ({time.perf_counter() - start:.0f} s)"
        )
        summary |= {
            "route": "full",
            "median_px": round(fit.median_error, 4),
            "inliers": round(fit.inlier_fraction, 4),
            "cameras": dict.fromkeys(rig.names, "full solve"),
        }
        cpu = replace(rig, robot=robot)
        # One episode: its own intrinsics
        return self._written(
            cpu, number, qpos, poses, world, fit.episode_intrinsics[0], fit.lens, summary
        )


@dataclass(frozen=True, slots=True)
class Stations:
    """Setups (stations) of a dataset from two thumbnails per episode of a camera that is not on
    the robot (`r4r.data.setups`, CPU): runs of episodes with the camera in one place. Prints
    the runs; `--write` sets each episode's `station` in the episodes table (a run takes the
    label most of its labelled episodes carry, else `auto-<first episode>`), or with
    `--per-camera` the camera's own setup key `station.<camera>` (runs named
    `<camera>-auto-<first episode>`): for static cameras re-placed independently of each other.
    `--group` keeps a priors group's episodes only (a dataset mixing robots: setups within each
    robot)."""

    dataset: tyro.conf.Positional[str]
    camera: tyro.conf.Positional[str]
    # First and last episode (inclusive; default: every episode)
    first: int = 0
    last: int | None = None
    # Only the episodes of this priors group (`groups`; `-` for those of no group)
    group: str | None = None
    # Write the camera's own setup key (`station.<camera>`) instead of the common `station`
    per_camera: bool = False
    write: bool = False

    def run(self) -> None:
        import time

        from r4r.data.lerobot import episode_count, episode_group
        from r4r.data.priors import episode_table_path
        from r4r.data.setups import SetupConfig, labelled, setups, thumbnails, write_stations

        root = dataset_root(self.dataset)
        last = episode_count(root) - 1 if self.last is None else self.last
        episodes = list(range(self.first, last + 1))
        if self.group is not None:
            wanted = None if self.group == "-" else self.group
            episodes = [e for e in episodes if episode_group(root, e) == wanted]
        start = time.perf_counter()
        found = thumbnails(root, episodes, self.camera)
        runs = setups(found, SetupConfig())
        took = time.perf_counter() - start
        print(f"{len(runs)} runs in {len(episodes)} episodes ({took:.0f} s)")
        path = episode_table_path(root)
        column = f"station.{self.camera}" if self.per_camera else "station"
        known = {}
        if path.exists():
            import pyarrow.parquet as pq

            if column in pq.read_schema(path).names:
                table = pq.read_table(path, columns=["episode", column]).to_pylist()
                known = {int(r["episode"]): str(r[column]) for r in table if r[column]}
        prefix = f"{self.camera}-" if self.per_camera else ""
        stations, relabelled = labelled(runs, known, episodes, prefix)
        for first, end in runs:
            count = sum(first <= e <= end for e in stations)
            print(f"{first}-{end}: {stations[first]} ({count} episodes)")
        if relabelled:
            print(f"labelled episodes another run claims: {relabelled}")
        if self.write:
            write_stations(path, stations, relabelled, self.camera if self.per_camera else None)
            print(f"wrote {path} ({column})")


@dataclass(frozen=True, slots=True)
class Station:
    """Calibrate one static camera per station from its neighbour station
    (`r4r.rago.station`): for each station of the episodes table (`r4r stations --write`)
    whose camera has no pose and focal yet, in episode order, the previous calibrated station's
    values are the start and everything else is held; starts from PnP and from transferred
    robot points follow if the fit check fails. RoMa links and transfer run on a GPU when
    there is one (~20 GPU s per station), the rest on the CPU. `--write` stores the station's
    camera (and the neighbour's other values) under `stations.<name>`; failed checks are
    flagged in the station's `fitted` record for reading its overlay. When the episodes table
    gives the camera its own setup key (`station.<camera>`, `r4r stations --per-camera`), its
    stations are those keys and only the camera's values are written (`--only` implied)."""

    dataset: tyro.conf.Positional[str]
    camera: tyro.conf.Positional[str]
    # Only these stations (default: every uncalibrated one with at least 10 episodes)
    stations: tuple[str, ...] = ()
    # Episodes per station used (spread over it)
    samples: int = 3
    # Stop after this many stations (0: all)
    limit: int = 0
    # Prior std of the camera's log focal around its start (0 holds the neighbour's focal: one
    # camera model in every station, whose focal a full rig measured)
    focal_std: float = 0.3
    # Store each station's camera under `stations.<name>`
    write: bool = False
    # Write only this camera's values to each station: its other static cameras are placed by
    # their own `r4r station` pass (several cameras re-placed per setup)
    only: bool = False
    # Fit again the stations a previous run flagged (default: skip them, read their overlays)
    retry: bool = False
    # Seeds from up to this many calibrated stations, nearest first, each placement at least
    # 5 cm / 5 deg from the seeds already taken (0: the nearest on each side of the robot); for
    # a camera moved between a few placements
    seeds: int = 0

    def run(self) -> None:
        import copy

        import pyarrow.parquet as pq
        import torch

        from r4r.data.lerobot import _root, episode_group
        from r4r.data.priors import episode_table_path
        from r4r.plan import MIN_SETUP
        from r4r.rago.station import (
            StationConfig,
            by_side,
            calibrate_station,
            distinct,
            samples,
            station_data,
        )

        root = dataset_root(self.dataset)
        priors = _root(root).priors
        path = episode_table_path(root)
        own = f"station.{self.camera}" in pq.read_schema(path).names
        column = f"station.{self.camera}" if own else "station"
        only = self.only or own
        table = pq.read_table(path, columns=["episode", column])
        runs: dict[str, list[int]] = {}
        for episode, name in zip(
            table["episode"].to_pylist(), table[column].to_pylist(), strict=True
        ):
            if name is not None and episode not in runs.setdefault(name, []):
                runs[name].append(episode)
        order = sorted(runs, key=lambda n: min(runs[n]))
        done = [n for n in order if _placed(priors.stations.get(n, {}), self.camera)]
        if not done:
            raise SystemExit(f"no station has {self.camera} calibrated: rig one first")
        tried = {  # stations whose camera a previous run fitted and flagged
            n
            for n in order
            if priors.stations.get(n, {}).get("cameras", {}).get(self.camera, {}).get("fitted")
        }
        todo = [
            n
            for n in (self.stations or order)
            if n not in done
            and len(runs[n]) >= MIN_SETUP
            and (self.retry or self.stations or n not in tried)
        ][: self.limit or None]
        matcher = _matcher("cuda") if torch.cuda.is_available() else None
        config = StationConfig(focal_std=self.focal_std)
        # A dataset mixing robots: neighbours come from the station's own priors group
        group = {n: episode_group(root, min(runs[n])) for n in order}
        for name in todo:
            # The nearest calibrated station in episode order gives the held values; the
            # nearest on each side of the robot (facing it, behind it) a seed for the camera
            kin = [n for n in done if group[n] == group[name]]
            if not kin:
                print(f"{name}: no calibrated station in its group {group[name]}, skipped")
                continue
            nearest = sorted(kin, key=lambda n: abs(min(runs[n]) - min(runs[name])))
            neighbour = nearest[0]
            placed = [(n, *_station_camera(priors.stations[n], self.camera)) for n in nearest]
            seeds = distinct(placed, self.seeds) if self.seeds else by_side(placed)
            block = copy.deepcopy(priors.stations[neighbour])
            if own:  # the camera's own setup: only its entry is read from the block
                block = {"cameras": {self.camera: block["cameras"][self.camera]}}
            values = block["cameras"][self.camera]
            values |= {
                "scope": "dataset",
                "position_range": config.position_range,
                "orientation_range_deg": config.orientation_range_deg,
            }
            priors.stations[name] = block
            numbers = tuple(samples(sorted(runs[name]), self.samples))
            rig = _prepare_rig(self.dataset, numbers, 1, 0.7, (), "cpu")
            # The cameras placed there (other static cameras with their own setups may not be
            # calibrated yet: left out, not held at a missing pose)
            use = [
                i
                for i, s in enumerate(rig.streams)
                if s.name == self.camera
                or s.moving
                or s.pose is not None
                or s.prior.position is not None
            ]
            names = tuple(rig.names[i] for i in use)
            index = torch.tensor(use)
            intrinsics = rig.intrinsics[index]
            lens = None if rig.lens is None else rig.lens.select(index)
            static = [rig.robot.camera_body[rig.robot.camera(n)] == 0 for n in names]
            data = station_data(
                root,
                numbers,
                rig.episodes,
                names,
                static,
                self.camera,
                intrinsics,
                lens,
                1,
                config,
                matcher,
            )
            fit = calibrate_station(
                data, rig.robot, names, intrinsics, lens, self.camera, config, seeds
            )
            print(
                f"{name} ({len(runs[name])} episodes), held values of {neighbour}: "
                f"from {fit.start}, "
                f"focal {fit.focal:.0f} px, check {fit.check.median_px:.2f} px"
                + (" FLAGGED: read its overlay" if fit.flagged else "")
            )
            values |= {
                **pose_fields(fit.pose.cpu()),
                "focal": round(fit.focal, 2),
                "focal_range": 0.0,
                "position_range": 0.02,
                "orientation_range_deg": 2.0,
                "fitted": {
                    "episodes": list(numbers),
                    "from": f"r4r station from {neighbour} ({fit.start})",
                    "check_px": round(fit.check.median_px, 3),
                    "flagged": fit.flagged,
                },
            }
            if self.write:
                _write_station(root, name, {"cameras": {self.camera: values}} if only else block)
            if not fit.flagged:
                done.append(name)
                done.sort(key=lambda n: min(runs[n]))


@dataclass(frozen=True, slots=True)
class Silhouette:
    """Calibrate one static camera per setup from the robot's silhouette
    (`r4r.rago.silhouette`, CPU, ~40 s per setup): for a camera re-placed between setups that
    scene matches and neighbour seeds cannot place. Per setup of the episodes table (the
    camera's own key `station.<camera>` when present, else `station`), `samples` episodes fit
    the camera's pose (the focal is held when the priors give one) to background-subtraction
    masks, from the setup's own prior and up to `seeds` calibrated setups; `held` other
    episodes score it. A setup whose score is under the config's pass values is flagged and
    written as a start only (no ranges: `--uncalibrated nan` leaves it NaN). `--write` stores
    the camera alone under `stations.<name>`."""

    dataset: tyro.conf.Positional[str]
    camera: tyro.conf.Positional[str]
    # Only these setups (default: every one with at least 10 episodes and no calibrated camera)
    stations: tuple[str, ...] = ()
    samples: int = 4
    held: int = 2
    # Calibrated setups used as extra starts (distinct placements, nearest in episode order)
    seeds: int = 3
    # Fit the focal too (default: only when the priors give none)
    free_focal: bool = False
    # Stop after this many setups (0: all)
    limit: int = 0
    # Fit again the setups a previous run flagged
    retry: bool = False
    write: bool = False
    # Write the frames a segmentation model should mask, per setup and episode, to this JSON
    # and stop (`python -m r4r.rago.segment` masks them on a GPU)
    select: Path | None = None
    # Robot masks of a segmentation model: `<setup>.npz` files in this folder
    # (`r4r.rago.segment`), used instead of background subtraction
    masks: Path | None = None

    def run(self) -> None:
        import json
        import time

        import pyarrow.parquet as pq

        from r4r.data.lerobot import _root
        from r4r.data.priors import episode_table_path
        from r4r.plan import MIN_SETUP
        from r4r.rago import segment, silhouette
        from r4r.rago.station import distinct, samples

        root = dataset_root(self.dataset)
        priors = _root(root).priors
        path = episode_table_path(root)
        column = f"station.{self.camera}"
        if column not in pq.read_schema(path).names:
            column = "station"
        table = pq.read_table(path, columns=["episode", column])
        runs: dict[str, list[int]] = {}
        for episode, name in zip(
            table["episode"].to_pylist(), table[column].to_pylist(), strict=True
        ):
            if name is not None and episode not in runs.setdefault(name, []):
                runs[name].append(episode)
        order = sorted(runs, key=lambda n: min(runs[n]))
        done = [n for n in order if _placed(priors.stations.get(n, {}), self.camera)]
        tried = {
            n
            for n in order
            if priors.stations.get(n, {}).get("cameras", {}).get(self.camera, {}).get("fitted")
        }
        todo = [
            n
            for n in (self.stations or order)
            if self.stations
            or (n not in done and len(runs[n]) >= MIN_SETUP and (self.retry or n not in tried))
        ][: self.limit or None]
        config = silhouette.SilhouetteConfig()
        chosen: dict[str, dict[str, list[int]]] = {}
        for name in todo:
            start = time.perf_counter()
            numbers = samples(sorted(runs[name]), self.samples + self.held)
            held = numbers[1::3][: self.held] if len(numbers) > self.samples else []
            if self.select is not None:
                chosen[name] = {
                    str(e): frames
                    for e in numbers
                    if (frames := silhouette.segment_frames(root, e, config))
                }
                self.select.write_text(json.dumps(chosen))
                print(f"{name}: frames selected")
                continue
            masks = {} if self.masks is None else segment.read(self.masks / f"{name}.npz")
            loaded = {
                e: silhouette.load(root, e, self.camera, config, masks.get(e))
                for e in numbers
                if self.masks is None or e in masks
            }
            fitting = [x for e, x in loaded.items() if x and x.frames.masks and e not in held]
            scoring = [x for e, x in loaded.items() if x and x.frames.masks and e in held]
            if not fitting:
                print(f"{name}: no usable episode, skipped")
                continue
            pose, focal, focal_range = fitting[0].prior
            if focal is None:
                raise SystemExit(
                    f"{self.camera}: no focal in the priors (give one, then --free-focal)"
                )
            nearest = sorted(done, key=lambda n: abs(min(runs[n]) - min(runs[name])))
            placed = [(n, *_station_camera(priors.stations[n], self.camera)) for n in nearest]
            starts = ([] if pose is None else [(pose, focal)]) + [
                (p, f) for _, p, f in distinct(placed, self.seeds)
            ]
            free = self.free_focal or focal_range is None
            fit = silhouette.fit(
                silhouette.stacked([x.frames for x in fitting], config), starts, free, config
            )
            checked = scoring or fitting
            found = silhouette.scored(
                [x.frames for x in checked], fit.pose, fit.focal, checked[0].geoms, config
            )
            flagged = found.precision < config.pass_precision or found.iou < config.pass_iou
            took = time.perf_counter() - start
            print(
                f"{name} ({len(runs[name])} episodes): precision {found.precision:.2f}, "
                f"IoU {found.iou:.2f} on {found.frames} {'held-out' if scoring else 'fitted'} "
                f"frames, focal {fit.focal:.0f} px, {took:.0f} s"
                + (" FLAGGED: read its overlay" if flagged else "")
            )
            values = {
                **pose_fields(fit.pose),
                "focal": round(fit.focal, 2),
                "focal_range": 0.0,
                # A flagged camera is only a start: its setup stays uncalibrated
                **(
                    {"position_range": 1.0, "orientation_range_deg": 180.0}
                    if flagged
                    else {"position_range": 0.02, "orientation_range_deg": 2.0}
                ),
                "fitted": {
                    "episodes": [e for e in numbers if e not in held],
                    "held": held,
                    "from": "r4r silhouette",
                    "precision": round(found.precision, 3),
                    "iou": round(found.iou, 3),
                    "flagged": flagged,
                },
            }
            if self.write:
                _write_station(root, name, {"cameras": {self.camera: values}})
            if not flagged:
                done.append(name)
                priors.stations.setdefault(name, {}).setdefault("cameras", {})[self.camera] = values


def _placed(block: JsonObject, camera: str) -> bool:
    """Whether a station block gives `camera` a pose and a focal (and was not flagged)."""

    values = block.get("cameras", {}).get(camera, {})
    return (
        values.get("position") is not None
        and values.get("focal") is not None
        and not values.get("fitted", {}).get("flagged")
    )


def _station_camera(block: JsonObject, camera: str) -> tuple[Tensor[[*Mat4x4]], float]:
    """A calibrated station's camera: its `base_from_camera` and focal."""

    import torch

    from r4r.geometry import se3

    values = block["cameras"][camera]
    rotation = se3.quaternion_to_matrix(torch.tensor(values["orientation"], dtype=torch.float64))
    position = torch.tensor(values["position"], dtype=torch.float64)
    return se3.from_rotation_translation(rotation, position), float(values["focal"])


def _write_station(root: Path, name: str, block: JsonObject) -> None:
    """Store a whole station block (cameras, bodies) under `stations.<name>`."""

    from r4r.data.priors import write_body, write_camera

    for camera, values in block.get("cameras", {}).items():
        print(f"wrote {write_camera(root, camera, values, name)} station {name}: {camera}")
    for body, values in block.get("bodies", {}).items():
        print(f"wrote {write_body(root, body, values, name)} station {name}: {body}")


@dataclass(frozen=True, slots=True)
class Merge:
    """A LeRobot v3 dataset from `r4r reconstruct` shards (`r4r.merge`, docs/output-spec.md):
    by default a twin `<dest>/<dataset>--r4r` (index and r4r columns, `meta/episodes` linking
    back with `source/episode_index`, videos linked from the main root); `--full` a merged
    copy `<dest>/<dataset>` (the main rows with r4r's columns). Writes `meta/r4r.json`."""

    dataset: tyro.conf.Positional[str]
    reconstruction: tyro.conf.Positional[Path]
    dest: tyro.conf.Positional[Path]
    full: bool = False
    # Frames per data file (a row group per episode)
    rows_per_file: int = 2_000_000

    def run(self) -> None:
        from r4r.merge import MergeConfig, merge

        config = MergeConfig(full=self.full, rows_per_file=self.rows_per_file)
        print(f"wrote {merge(dataset_root(self.dataset), self.reconstruction, self.dest, config)}")


@functools.cache
def _matcher(device: str) -> RoMaV2:
    """RoMa v2, loaded once per process and device."""

    from r4r.matching import RoMaV2

    return RoMaV2.from_checkpoint(device=device)


type Command = (
    Annotated[Inspect, tyro.conf.subcommand("inspect")]
    | Annotated[Links, tyro.conf.subcommand("links")]
    | Annotated[Fit, tyro.conf.subcommand("fit")]
    | Annotated[Rig, tyro.conf.subcommand("rig")]
    | Annotated[Picks, tyro.conf.subcommand("picks")]
    | Annotated[Sheet, tyro.conf.subcommand("sheet")]
    | Annotated[Export, tyro.conf.subcommand("export")]
    | Annotated[Overlay, tyro.conf.subcommand("overlay")]
    | Annotated[Preview, tyro.conf.subcommand("preview")]
    | Annotated[Focal, tyro.conf.subcommand("focal")]
    | Annotated[Reconstruct, tyro.conf.subcommand("reconstruct")]
    | Annotated[Stations, tyro.conf.subcommand("stations")]
    | Annotated[Station, tyro.conf.subcommand("station")]
    | Annotated[Silhouette, tyro.conf.subcommand("silhouette")]
    | Annotated[Merge, tyro.conf.subcommand("merge")]
)


def main() -> None:
    tyro.cli(Command).run()
