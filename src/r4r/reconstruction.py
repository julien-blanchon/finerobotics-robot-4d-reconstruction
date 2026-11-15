"""Every episode of a dataset at scale: which episodes need no solve, their camera poses from the
priors and forward kinematics, and the sharded pose tables `r4r reconstruct` writes.

An episode needs no solve when its priors leave nothing free (`r4r.plan`: every value of
dataset or station scope calibrated, every camera's pose recorded or held, a moving base on its
calibrated record). Then

    world_from_camera_t = W_b(q_t) · body_from_camera

with `W_b` the forward kinematics of the camera's body `b` (moved by the recorded base motion,
if any; the world body for a static camera): no video is decoded and nothing is tracked.

Episodes are written in shards, `shard-<k>.parquet` holding the `k`-th run of `size`
consecutive episodes of the selection: one file per shard keeps the file count (inodes) small,
an existing shard is done (a run resumes where it stopped), and shard `k` belongs to worker
`k mod n` of `n` (e.g. a Slurm array task).
"""

from __future__ import annotations

import json
import math
import os
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from einops import rearrange, repeat

from r4r.rago.stages import moved_kinematics

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.rago.lens import Lens
    from r4r.rago.light import LightConfig
    from r4r.rago.robot import Robot
    from r4r.rago.stages import RigEpisode
    from r4r.types import Mat3x3, Mat4x4

# Per-episode summary of a shard, in its schema metadata (JSON list of records)
EPISODES_KEY = b"r4r.episodes"


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
) -> tuple[Tensor[[C, T, *Mat4x4]], dict[str, object]]:
    """An episode from a few bursts of frames (`rago.light`): with `checked` cameras (episode
    scope, held at their record), the fit check first (epipolar distance of the verified
    matches under the held poses), which frees them when it flags the episode; then the light
    solve of what is `free` (nothing: the held poses). Returns every camera's
    `world_from_camera` and the summary entries (check and solve errors, pixels)."""

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
    if checked:
        view_camera = torch.arange(len(names)).repeat_interleave(len(steps))
        at_views = rearrange(held[:, steps], "c k i j -> (c k) i j")
        fit = light.check(links, at_views, view_camera, intrinsics, lens, static)
        summary["check_px"] = _rounded(fit.median_px)
        if fit.flagged(config):
            summary["flagged"] = True
            free = (*free, *checked)
    if not free:
        return held, summary
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
    poses = light.extended(
        rig,
        solving,
        episode.qpos,
        repeat(mounts, "c i j -> c t i j", t=len(episode.qpos)),
        light.motion_at(solving, base, motion),
    )
    return poses, summary


def _rounded(value: float) -> float | None:
    """A value for the JSON summary: rounded, None for NaN."""

    return None if math.isnan(value) else round(value, 3)


def _at(motion: Tensor[[int, *Mat4x4]] | None, steps: list[int]) -> Tensor[[int, *Mat4x4]] | None:
    return None if motion is None else motion[steps]


def moving_base(robot: Robot[int, int, int]) -> str | None:
    """The frame-scope free body nearest the root (a mobile base), if any."""

    frame = [robot.body(n) for n, f in robot.free_bodies.items() if f.scope == "frame"]
    return robot.body_names[min(frame)] if frame else None


def pose_table[C: IntVar, T: IntVar](
    episode: int,
    stride: int,
    names: tuple[str, ...],
    world_from_camera: Tensor[[C, T, *Mat4x4]],
    intrinsics: Tensor[[C, *Mat3x3]],
) -> pa.Table:
    """One row per camera and frame (camera-major): `episode, frame, camera`,
    `world_from_camera` (16 values, row-major) and `fx, fy, cx, cy`."""

    cameras, steps = world_from_camera.shape[:2]
    rows = cameras * steps
    poses = rearrange(world_from_camera.detach().cpu().double(), "c t i j -> (c t i j)")
    k = intrinsics.detach().cpu().double()
    offsets = pa.array(np.arange(0, 16 * rows + 1, 16, dtype=np.int32))

    def per_row(values: Tensor[[C]]) -> np.ndarray:
        return repeat(values, "c -> (c t)", t=steps).numpy()

    return pa.table(
        {
            "episode": pa.array(np.full(rows, episode, dtype=np.int64)),
            "frame": pa.array(np.tile(np.arange(steps, dtype=np.int64) * stride, cameras)),
            "camera": pa.array(np.repeat(np.array(names, dtype=object), steps), pa.string()),
            "world_from_camera": pa.ListArray.from_arrays(offsets, pa.array(poses.numpy())),
            "fx": per_row(k[:, 0, 0]),
            "fy": per_row(k[:, 1, 1]),
            "cx": per_row(k[:, 0, 2]),
            "cy": per_row(k[:, 1, 2]),
        }
    )


def write_shard(path: Path, tables: list[pa.Table], episodes: list[dict[str, object]]) -> None:
    """Write a shard's pose tables as one file, atomically (a partial file is never left as
    done), with the per-episode summary `episodes` in its metadata."""

    table = pa.concat_tables(tables) if tables else _empty()
    metadata = {EPISODES_KEY: json.dumps(episodes).encode()}
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    pq.write_table(table.replace_schema_metadata(metadata), partial)
    partial.rename(path)


def read_summary(path: Path) -> list[dict[str, object]]:
    """The per-episode summary of a written shard."""

    metadata = pq.read_schema(path).metadata or {}
    return json.loads(metadata.get(EPISODES_KEY, b"[]"))


def _empty() -> pa.Table:
    """A pose table without rows (a shard whose episodes all failed)."""

    return pose_table(0, 1, (), torch.zeros(0, 0, 4, 4), torch.zeros(0, 3, 3))
