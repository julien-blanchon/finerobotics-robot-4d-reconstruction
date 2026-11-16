"""Every episode of a dataset at scale: which episodes need no solve, their camera poses from the
priors and forward kinematics, and the sharded pose tables `r4r reconstruct` writes.

An episode needs no solve when its priors leave nothing free (`r4r.plan`: every value of
dataset or station scope calibrated, every camera's pose recorded or held, a moving base on its
calibrated record). Then

    world_from_camera_t = W_b(q_t) · body_from_camera

with `W_b` the forward kinematics of the camera's body `b` (moved by the recorded base motion,
if any; the world body for a static camera): no video is decoded and nothing is tracked.

Each episode is written in the spec's columns (`r4r.output`, docs/output-spec.md): one row per
frame. Episodes are written in shards, `shard-<k>.parquet` holding the `k`-th run of `size`
consecutive episodes of the selection: one file per shard keeps the file count (inodes) small,
an existing shard is done (a run resumes where it stopped), and shard `k` belongs to worker
`k mod n` of `n` (e.g. a Slurm array task).
"""

from __future__ import annotations

import json
import math
import os
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from einops import rearrange, repeat

from r4r.output import LAYOUT_KEY, CameraInfo, EpisodeOutput, Layout
from r4r.rago.stages import moved_kinematics

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping, Sequence
    from pathlib import Path

    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.data.episode import CameraStream, JointMap
    from r4r.data.priors import DatasetPriors
    from r4r.plan import Unknowns
    from r4r.rago.lens import Lens
    from r4r.rago.light import LightConfig
    from r4r.rago.robot import Robot
    from r4r.rago.stages import RigEpisode
    from r4r.types import Mat3x3, Mat4x4

# Per-episode summary of a shard, in its schema metadata (JSON list of records)
EPISODES_KEY = b"r4r.episodes"

# CPU threads of one reconstruct process when the user sets none: torch and OpenCV otherwise
# take every core, and on a shared login node the light route then ran 10 episodes in over
# 16 min instead of 150-170 s with 8 threads (docs/verification.md)
DEFAULT_THREADS = 8
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def default_threads(environ: Mapping[str, str], cpus: int | None) -> int | None:
    """Threads for one reconstruct process: `DEFAULT_THREADS` (at most the `cpus` available),
    or None when the user chose a count (`OMP_NUM_THREADS` or a sibling in `environ`)."""

    if any(environ.get(name) for name in THREAD_VARIABLES):
        return None
    return max(1, min(DEFAULT_THREADS, cpus or 1))


def limit_threads(environ: MutableMapping[str, str] | None = None) -> int | None:
    """Cap torch's and OpenCV's thread pools at `default_threads` unless the user set a count
    (then nothing changes); also exported to `environ` for libraries started later. Returns
    the count set, or None."""

    import cv2

    environ = os.environ if environ is None else environ
    count = default_threads(environ, os.process_cpu_count())
    if count is None:
        return None
    for name in THREAD_VARIABLES:
        environ[name] = str(count)
    torch.set_num_threads(count)
    cv2.setNumThreads(count)
    return count


def shards(episodes: Sequence[int], size: int, worker: int = 0, workers: int = 1) -> list[int]:
    """The shards of `episodes` (runs of `size`) that belong to `worker` of `workers`."""

    count = -(-len(episodes) // size)  # ceil
    return [k for k in range(count) if k % workers == worker]


def shard_path(target: Path, shard: int) -> Path:
    return target / f"shard-{shard:06d}.parquet"


def held_mounts(
    robot: Robot[int, int, int], episode: RigEpisode[int, int], names: tuple[str, ...]
) -> Tensor[[int, *Mat4x4]]:
    """Each camera's held `body_from_camera` (a static camera's pose in the base frame): its
    record for the episode, else the priors'."""

    mounts = []
    for name in names:
        if name in episode.poses:  # recorded for this episode
            mounts.append(episode.poses[name].to(robot.body_from_camera))
        else:
            mounts.append(robot.body_from_camera[robot.camera(name)])
    return torch.stack(mounts)


def held_poses[T: IntVar](
    robot: Robot[int, int, int], episode: RigEpisode[T, int], names: tuple[str, ...]
) -> Tensor[[int, T, *Mat4x4]]:
    """Each camera's `world_from_camera` at every timestep with every value held (the episode
    takes the fast path, `r4r.plan`): `W_b(q_t) · body_from_camera` (`held_mounts`)."""

    world = moved_kinematics(robot, episode)
    bodies = [robot.camera_body[robot.camera(n)] for n in names]
    return rearrange(world[:, bodies] @ held_mounts(robot, episode, names), "t c i j -> c t i j")


def keyframe_episode[C: IntVar, T: IntVar](
    root: Path,
    number: int,
    robot: Robot[int, int, int],
    episode: RigEpisode[T, int],
    names: tuple[str, ...],
    intrinsics: Tensor[[C, *Mat3x3]],
    lens: Lens[C] | None,
    stride: int,
    free: tuple[str, ...],
    checked: tuple[str, ...],
    config: LightConfig,
) -> tuple[Tensor[[C, T, *Mat4x4]], Tensor[[T, int, *Mat4x4]], dict[str, object]]:
    """An episode from a few bursts of frames (`rago.light`): with `checked` cameras (episode
    scope, held at their record), the fit check first (epipolar distance of the verified
    matches under the held poses), which frees them when it flags the episode; then the light
    solve of what is `free` (nothing: the held poses). Returns every camera's
    `world_from_camera`, every body's `world_from_body` and the summary entries (check and
    solve errors, pixels)."""

    from r4r.data.lerobot import keyframes
    from r4r.rago import light

    bursts = light.burst_steps(len(episode.qpos), config)
    steps = [t for burst in bursts for t in burst]
    sizes = [len(b) for b in bursts]
    frames = keyframes(root, number, [t * stride for t in steps], names)
    videos = [frames[n] for n in names]
    static = torch.tensor([robot.camera_body[robot.camera(n)] == 0 for n in names])
    links = light.burst_links(videos, sizes, intrinsics, lens, static.tolist(), config)
    held = held_poses(robot, episode, names)
    summary: dict[str, object] = {"links": len(links.view_a)}
    view_camera = torch.arange(len(names)).repeat_interleave(len(steps))

    def checked_at(poses: Tensor[[C, T, *Mat4x4]]) -> light.FitCheck:
        at_views = rearrange(poses[:, steps], "c k i j -> (c k) i j")
        return light.check(links, at_views, view_camera, intrinsics, lens, static)

    if checked:
        fit = checked_at(held)
        summary["check_px"] = _rounded(fit.median_px)
        if fit.flagged(config):
            summary["flagged"] = True
            free = (*free, *checked)
    if not free:
        return held, moved_kinematics(robot, episode), summary
    alone = any(n in free and not s for n, s in zip(names, static.tolist(), strict=True))
    tracks = light.burst_tracks(videos, sizes, intrinsics, lens, links, config)
    tracks = light.informative(tracks, static, alone, config)
    mounts = held_mounts(robot, episode, names)
    base = moving_base(robot)
    motion = episode.base_motion
    solving = light.held_robot(robot, free)
    at = repeat(mounts, "c i j -> c t i j", t=len(steps))
    bodies = light.motion_at(solving, base, _at(motion, steps))
    rig, fit = light.light_solve(solving, episode.qpos[steps], tracks, names, at, bodies, config)
    summary |= {"free": list(free), "light_px": _rounded(fit.median_px)}
    poses, world = light.extended(
        rig,
        solving,
        episode.qpos,
        repeat(mounts, "c i j -> c t i j", t=len(episode.qpos)),
        light.motion_at(solving, base, motion),
    )
    if checked:
        summary["check_after_px"] = _rounded(checked_at(poses).median_px)
    return poses, world, summary


def _rounded(value: float) -> float | None:
    """A value for the JSON summary: rounded, None for NaN."""

    return None if math.isnan(value) else round(value, 3)


def _at(motion: Tensor[[int, *Mat4x4]] | None, steps: list[int]) -> Tensor[[int, *Mat4x4]] | None:
    return None if motion is None else motion[steps]


def moving_base(robot: Robot[int, int, int]) -> str | None:
    """The frame-scope free body nearest the root (a mobile base), if any."""

    frame = [robot.body(n) for n, f in robot.free_bodies.items() if f.scope == "frame"]
    return robot.body_names[min(frame)] if frame else None


def write_shard(
    path: Path, tables: list[pa.Table], episodes: list[dict[str, object]], layout: str | None
) -> None:
    """Write a shard's frame tables as one file, atomically (a partial file is never left as
    done), with the per-episode summary `episodes` and the dataset's `layout` (JSON,
    `r4r.output.Layout.to_json`) in its metadata."""

    empty = pa.table({"episode_index": pa.array([], pa.int64())})
    table = pa.concat_tables(tables) if tables else empty
    metadata = {EPISODES_KEY: json.dumps(episodes).encode()}
    if layout is not None:
        metadata[LAYOUT_KEY] = layout.encode()
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    pq.write_table(table.replace_schema_metadata(metadata), partial)
    partial.rename(path)


def read_summary(path: Path) -> list[dict[str, object]]:
    """The per-episode summary of a written shard."""

    metadata = pq.read_schema(path).metadata or {}
    return json.loads(metadata.get(EPISODES_KEY, b"[]"))


def _mount(root: Path, name: str, moving: bool) -> str:
    """`static` or `moving`, or `per station` when the priors' stations give the stream another
    mount than the dataset level (streams swapped between setups, e.g. robomind2 ark)."""

    from r4r.data.priors import load_priors

    priors = load_priors(root)
    own = priors.raw.get("cameras", {}).get(name, {}).get("mount")
    mounts = {own} | {
        s.get("cameras", {}).get(name, {}).get("mount") for s in priors.stations.values()
    }
    mounts.discard(None)
    if len(mounts) > 1:
        return "per station"
    return "moving" if moving else "static"


def dataset_layout(
    root: Path,
    robot: Robot[int, int, int],
    priors: DatasetPriors,
    streams: Sequence[CameraStream[int, int, int]],
    joints: tuple[JointMap, ...],
) -> Layout:
    """The columns every episode of the dataset gets (`r4r.output.Layout`)."""

    from r4r.data.lerobot import joint_action
    from r4r.effectors import effectors
    from r4r.output import CameraInfo, aligned_joints, state_mapping

    names = aligned_joints(robot.joint_names, joints)
    fitted = priors.kinematics
    cameras = tuple(
        CameraInfo(s.name, s.prior.lens, _mount(root, s.name, s.moving), s.role) for s in streams
    )
    return Layout(
        cameras=cameras,
        ignored=ignored_cameras(root),
        effectors=effectors(robot, priors),
        joints=names,
        mobile=priors.base == "mobile" or moving_base(robot) is not None,
        base_pose=priors.base != "mobile" or moving_base(robot) is not None,
        action=joint_action(root),
        mapping=state_mapping(names, joints, {} if fitted is None else fitted.offsets),
        lag={
            "state_shift": float(priors.state_shift),
            "time_offset": 0.0 if fitted is None else fitted.time_offset,
        },
    )


def ignored_cameras(root: Path) -> tuple[CameraInfo, ...]:
    """The dataset's video cameras (not depth maps) that the priors ignore, with what their
    feature info states (`camera.mount`, `camera.role`; lens from the priors)."""

    from r4r.data.lerobot import _root
    from r4r.data.priors import CameraPrior

    opened = _root(root)
    out = []
    for key in opened.tables.video_keys:
        info = opened.tables.features[key].get("info", {}) or {}
        name = key.removeprefix("observation.images.")
        prior = opened.priors.cameras.get(name, CameraPrior())
        if prior.ignore and not info.get("is_depth_map", False):
            mount = str(info.get("camera.mount") or "unknown")
            out.append(CameraInfo(name, prior.lens, mount, str(info.get("camera.role") or "")))
    return tuple(out)


def episode_output[T: IntVar](
    root: Path,
    number: int,
    layout: Layout,
    robot: Robot[int, int, int],
    joints: tuple[JointMap, ...],
    qpos: Tensor[[T, int]],
    stride: int,
    world_from_camera: Tensor[[int, T, *Mat4x4]],
    world_from_body: Tensor[[T, int, *Mat4x4]],
    intrinsics: Tensor[[int, *Mat3x3]],
    lens: Lens[int] | None,
) -> EpisodeOutput:
    """An episode's values for `frame_table`: the solution (camera and body poses), the
    corrected joint positions `qpos` (`state_aligned`), the action through the same joint map
    (no lag correction) and the end effectors' recorded actuation."""

    from r4r.data.lerobot import episode_columns
    from r4r.output import select_joints
    from r4r.rago.robot import qpos_from_state

    def aligned(values: Tensor[[T, int]]) -> Tensor[[T, int]]:
        return select_joints(values, robot.joint_names, robot.joint_qpos, layout.joints)

    wanted = [e.actuation.feature for e in layout.effectors if e.actuation is not None]
    wanted += ["action"] if layout.action else []
    recorded = episode_columns(root, number, wanted, stride) if wanted else {}
    action = None
    if layout.action:
        action = aligned(qpos_from_state(robot, recorded["action"].to(qpos), joints))
    steps = len(qpos)
    frames = torch.arange(steps) * stride
    count = len(intrinsics)
    return EpisodeOutput(
        episode=number,
        frames=frames,
        timestamps=frames.double() / episode_fps(root),
        world_from_camera=world_from_camera.detach().cpu(),
        intrinsics=intrinsics.detach().cpu(),
        distortion=torch.zeros(count, 4, dtype=torch.float64)
        if lens is None
        else lens.distortion.detach().cpu().double(),
        world_from_body=world_from_body.detach().cpu(),
        base=None if moving_base(robot) is None else robot.body(str(moving_base(robot))),
        state=aligned(qpos).detach().cpu(),
        action=None if action is None else action.detach().cpu(),
        actuation={
            e.arm: recorded[e.actuation.feature][:, e.actuation.index]
            for e in layout.effectors
            if e.actuation is not None
        },
    )


def episode_fps(root: Path) -> float:
    """The dataset's frame rate (frames per second)."""

    from r4r.data.lerobot import frame_rate

    return frame_rate(root)


def uncalibrated_cameras(unknowns: Unknowns, names: tuple[str, ...]) -> tuple[str, ...] | None:
    """The cameras whose own unknowns (setup not calibrated, no start) send an episode to the
    full path; None when an unknown of something else does (a body, the odometry)."""

    named = {reason.partition(": ")[0] for reason in (*unknowns.setup, *unknowns.missing)}
    if not named <= set(names):
        return None
    return tuple(n for n in names if n in named)
