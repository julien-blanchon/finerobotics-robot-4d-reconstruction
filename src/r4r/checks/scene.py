"""An episode as the checks see it: each camera's pose, intrinsics and lens per frame, and every
robot body's pose per frame, in one frame (the episode frame of `docs/output-spec.md`).

Cameras come from a reconstruction (a shard of `r4r reconstruct`, or the arrays the reconstruct
loop holds); bodies from forward kinematics with the priors' corrected kinematics, as the
reconstruction drew them. Surfaces are the description's visual meshes (`r4r.viz.overlay`).
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    import mujoco
    import pyarrow as pa

    from r4r.viz.overlay import Surface

# Triangles kept per geom (a random subset beyond): enough to keep the silhouette closed
MAX_FACES = 4000


@dataclass(frozen=True, slots=True)
class View:
    """One camera over an episode: `camera_from_world [T, 4, 4]`, `intrinsics [T, 3, 3]`,
    `distortion [T, 4]` (OpenCV radial k1 k2 k3 0, or fisheye k1..k4), image `size` (w, h)."""

    name: str
    role: str  # scene (static or head) or wrist
    fisheye: bool
    size: tuple[int, int]
    camera_from_world: np.ndarray
    intrinsics: np.ndarray
    distortion: np.ndarray

    def calibrated(self, t: int) -> bool:
        return bool(np.isfinite(self.camera_from_world[t]).all())


@dataclass(frozen=True, slots=True)
class Scene:
    """An episode's views, body poses `world_from_body [T, L, 4, 4]` and the robot surfaces
    (body index into the poses, vertices in the body frame, triangles)."""

    dataset: str
    episode: int
    frames: np.ndarray  # source frame index per timestep [T]
    views: dict[str, View]
    world_from_body: np.ndarray
    surfaces: list[Surface]


def _matrix(pose: np.ndarray) -> np.ndarray:
    """4×4 from the spec's `x, y, z, qx, qy, qz, qw` rows [T, 7] -> [T, 4, 4]."""

    x, y, z, w = pose[:, 3], pose[:, 4], pose[:, 5], pose[:, 6]
    out = np.zeros((len(pose), 4, 4))
    out[:, 0, :3] = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], 1)
    out[:, 1, :3] = np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], 1)
    out[:, 2, :3] = np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1)
    out[:, :3, 3] = pose[:, :3]
    out[:, 3, 3] = 1.0
    return out


def _column(table: pa.Table, name: str) -> np.ndarray:
    return np.array(table.column(name).to_pylist(), dtype=np.float64)


def _inverse(poses: np.ndarray) -> np.ndarray:
    """Inverse rigid transforms [T, 4, 4], NaN rows (uncalibrated) kept NaN."""

    out = np.full_like(poses, np.nan)
    finite = np.isfinite(poses).all(axis=(1, 2))
    out[finite] = np.linalg.inv(poses[finite])
    return out


def views_from_table(part: pa.Table, layout: dict) -> dict[str, View]:
    """Every calibrated camera of one episode's rows of a shard (spec columns)."""

    out = {}
    for key, info in layout["cameras"].items():
        name = key.removeprefix("observation.images.")
        prefix = f"observation.camera.{name}"
        if not info.get("camera.calibrated", True) or f"{prefix}.pose" not in part.column_names:
            continue
        k = _column(part, f"{prefix}.intrinsics")
        matrix = np.zeros((len(k), 3, 3))
        matrix[:, 0, 0], matrix[:, 1, 1] = k[:, 0], k[:, 1]
        matrix[:, 0, 2], matrix[:, 1, 2], matrix[:, 2, 2] = k[:, 2], k[:, 3], 1.0
        if not np.isfinite(k).any():
            continue
        # cx, cy are pixel centres of the delivered video: (W − 1)/2, (H − 1)/2
        size = (round(2 * np.nanmedian(k[:, 2]) + 1), round(2 * np.nanmedian(k[:, 3]) + 1))
        out[name] = View(
            name=name,
            role="wrist" if info.get("camera.role") == "wrist" else "scene",
            fisheye=info.get("camera.lens") == "fisheye",
            size=size,
            camera_from_world=_inverse(_matrix(_column(part, f"{prefix}.pose"))),
            intrinsics=matrix,
            distortion=_column(part, f"{prefix}.distortion"),
        )
    return out


@functools.cache
def _surfaces(path: Path, mounts: tuple, bodies: tuple, names: tuple[str, ...]) -> list[Surface]:
    """The description's visual surfaces, their bodies as indices into `names` (the kinematic
    model's bodies), at most `MAX_FACES` triangles per geom."""

    from r4r.rago.robot import compile_model
    from r4r.viz.overlay import surfaces

    model: mujoco.MjModel = compile_model(path, None, mounts, bodies, visual=True)
    out = []
    rng = np.random.default_rng(0)
    for body, vertices, faces in surfaces(model):
        name = model.body(body).name
        if name not in names:
            continue
        if len(faces) > MAX_FACES:
            faces = faces[rng.choice(len(faces), MAX_FACES, replace=False)]
        out.append((names.index(name), vertices, faces))
    return out


def body_poses(dataset: str, number: int) -> tuple[np.ndarray, list[Surface]]:
    """`world_from_body [T, L, 4, 4]` of an episode in its episode frame (forward kinematics
    with the priors' corrected kinematics and recorded base motion) and its surfaces."""

    from r4r.cli import _prepare_rig
    from r4r.geometry import se3
    from r4r.rago.robot import CameraMount, body_mounts
    from r4r.rago.stages import moved_kinematics
    from r4r.reconstruction import moving_base

    rig = _prepare_rig(dataset, (number,), 1, 0.7, (), "cpu")
    world = moved_kinematics(rig.robot, rig.episodes[0])
    base = moving_base(rig.robot)
    if base is not None:
        world = se3.inverse(world[0, rig.robot.body(base)]) @ world
    loaded = rig.loaded[0]
    mounts = tuple(CameraMount.from_prior(c.name, c.prior, c.mount, c.moving) for c in rig.streams)
    found = _surfaces(loaded.robot, mounts, body_mounts(loaded.priors), rig.robot.body_names)
    return world.detach().cpu().double().numpy(), found


def scenes_from_shard(shard: Path, episodes: set[int] | None = None) -> list[Scene]:
    """The scenes of a shard's episodes (all, or those in `episodes`)."""

    import pyarrow.parquet as pq

    from r4r.output import LAYOUT_KEY

    table = pq.read_table(shard)
    layout = json.loads(table.schema.metadata[LAYOUT_KEY])
    dataset = _dataset_of(shard)
    index = np.array(table.column("episode_index").to_pylist())
    out = []
    for number in dict.fromkeys(index.tolist()):
        if episodes is not None and number not in episodes:
            continue
        part = table.take(np.flatnonzero(index == number))
        world, found = body_poses(dataset, number)
        out.append(
            Scene(
                dataset=dataset,
                episode=int(number),
                frames=np.array(part.column("frame_index").to_pylist()),
                views=views_from_table(part, layout),
                world_from_body=world,
                surfaces=found,
            )
        )
    return out


def _dataset_of(shard: Path) -> str:
    """`family/datasets/name` from `<out>/<family>/<name>/shard-k.parquet`."""

    return f"{shard.parent.parent.name}/datasets/{shard.parent.name}"
