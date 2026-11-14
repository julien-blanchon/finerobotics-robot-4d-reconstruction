"""Load one episode of a LeRobot v3 dataset (finerobotics layout) into an `Episode`.

The only module that imports LeRobot (optional `lerobot` group). finerobotics roots ship the
robot description in `meta/robot/` and `meta/robot.json`, which maps state columns to
description joints and may name camera links; camera features declare their role, mount
(static/dynamic) and side. The dataset's priors (`priors.py`) override all of these.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import torch
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames

from ..geometry import se3
from .episode import CameraStream, Episode, JointMap
from .priors import PRIORS, CameraPrior, DatasetPriors, load_episode_record, load_priors

IMAGE_PREFIX = "observation.images."


def load_episode(
    root: Path, episode: int, stride: int = 1, decode: tuple[str, ...] | None = None
) -> Episode:
    """Every `stride`-th timestep of `episode` from the dataset root `root`.

    Only the cameras in `decode` get frames (all when None); the others come with none.
    """

    repo_id = f"local/{root.name}"
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    rows = LeRobotDataset(repo_id, root=root, episodes=[episode], video_backend="pyav")
    every = rows.hf_dataset.with_format("torch")
    rows = every[::stride]
    timestamps = rows["timestamp"].tolist()

    robot = json.loads((root / "meta" / "robot.json").read_text())
    record = load_episode_record(root, episode)
    # A station's own mounts and bodies (priors `stations`) on top of the dataset's
    priors = load_priors(root).for_station(None if record is None else record.station)
    links = {camera["key"]: camera["urdf_link"] for camera in robot.get("cameras", [])}
    cameras = []
    for key in meta.video_keys:
        if meta.features[key].get("info", {}).get("is_depth_map", False):
            continue
        name = key.removeprefix(IMAGE_PREFIX)
        frames = None if decode is None or name in decode else torch.empty(0, 3, 0, 0)
        prior = priors.cameras.get(name, CameraPrior())
        stream = _camera(meta, root, episode, key, timestamps, links.get(key), prior, frames)
        if record is not None and name in record.intrinsics:
            fx, fy, cx, cy = record.intrinsics[name]
            k = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float64)
            stream = replace(stream, intrinsics=k)
        if record is not None and name in record.poses and not stream.moving:
            x, y, z, *q = record.poses[name]
            rotation = se3.quaternion_to_matrix(torch.tensor(q, dtype=torch.float64))
            position = torch.tensor([x, y, z], dtype=torch.float64)
            stream = replace(stream, pose=se3.from_rotation_translation(rotation, position))
        cameras.append(stream)
    state = rows[robot["state"]["feature"]].double()
    if priors.state_shift:
        # Frame t's state from t + shift (a known video/joint-state offset, in frames at the
        # dataset's rate whatever the stride), clamped
        full = every[:][robot["state"]["feature"]].double()
        index = torch.arange(0, len(full), stride)[: len(state)] + priors.state_shift
        state = full[index.clamp(0, len(full) - 1)]
    return Episode(
        name=f"{root.name}/episode_{episode:06d}",
        fps=meta.fps / stride,
        state=state,
        joints=priors.joint_map(_joints(robot, priors.replaced)),
        robot=_description(root, robot, priors),
        cameras=tuple(cameras),
        priors=priors,
        station=None if record is None else record.station,
        base_motion=None if priors.base_odometry is None else _base_motion(priors, rows),
    )


def _description(root: Path, robot: dict, priors: DatasetPriors) -> Path:
    """The robot description: the corrected one of the priors, else the delivered one."""

    if priors.description:
        return PRIORS / priors.description
    return root / "meta" / robot["urdf"]


def _base_motion(priors: DatasetPriors, rows) -> torch.Tensor:  # noqa: ANN001
    """The base's recorded motion (see `r4r.rago.odometry`) at the loaded timesteps."""

    from r4r.rago import odometry

    spec = priors.base_odometry
    assert spec is not None
    values = rows[spec.feature].double()[:, list(spec.columns)]
    if spec.kind == "pose":
        return odometry.from_poses(values[:, 0], values[:, 1], values[:, 2])
    if spec.kind == "velocity":
        return odometry.from_velocity(values, rows["timestamp"].double())
    assert spec.radius is not None and spec.half_track is not None, "caster: radius, half_track"
    geometry = odometry.CasterGeometry(spec.radius, spec.half_track, spec.offset, spec.signs)
    steering = values[:, 2] if values.shape[1] > 2 else None
    return odometry.from_steps(odometry.caster_steps(values[:, :2], steering, geometry))


def _camera(
    meta: LeRobotDatasetMetadata,
    root: Path,
    episode: int,
    key: str,
    timestamps: list[float],
    link: str | None,
    prior: CameraPrior,
    frames: torch.Tensor | None,
) -> CameraStream:
    info = meta.features[key].get("info", {})
    if frames is None:
        assert meta.episodes is not None, f"{root}: episode metadata missing"
        start = meta.episodes[episode][f"videos/{key}/from_timestamp"]
        path = root / meta.get_video_file_path(episode, key)
        frames = decode_video_frames(
            path, [start + t for t in timestamps], 1 / meta.fps / 2, "pyav", return_uint8=True
        )
    moving = info.get("camera.mount") == "dynamic"
    if prior.mount is not None:
        moving = prior.mount == "moving"
    # Metadata links are unverified; "world" or a frame name is not a mount
    declared = link or info.get("camera.extrinsics_frame")
    mount = prior.link or (declared if moving and declared != "world" else None)
    return CameraStream(
        name=key.removeprefix(IMAGE_PREFIX),
        frames=frames,
        role=info.get("camera.role", "other"),
        moving=moving,
        mount=mount,
        side=info.get("camera.side"),
        prior=prior,
    )


def _joints(robot: dict, replaced: frozenset[str] = frozenset()) -> tuple[JointMap, ...]:
    """Arm joints and gripper joints of every arm in `robot.json`, in the description's units.

    A joint's `joint_ranges` entry (column range → joint range) applies to any joint; without
    one, degrees are converted to radians. `normalised_percent` has no fixed mapping (it
    depends on each robot's calibration): such joints need a mapping in the dataset's priors.
    Joints in `replaced` (set by the priors) are skipped.
    """

    state = robot["state"]
    signs = state.get("joint_signs", {})
    ranges = state.get("joint_ranges", {})
    joints = []
    for arm in state["arms"]:
        for joint in arm["joints"]:
            name = joint["urdf_joint"]
            if name in replaced:
                continue
            sign = float(signs.get(name, 1.0))
            unit = joint.get("unit", state.get("joint_unit", "rad"))
            if name in ranges:
                source, target = tuple(ranges[name]["from"]), tuple(ranges[name]["to"])
            elif unit == "deg":
                source, target = (0.0, 180.0), (0.0, math.pi)
            elif unit == "rad":
                source = target = None
            else:
                raise ValueError(f"{name}: state unit {unit!r} needs a mapping in the priors")
            joints.append(JointMap(name, joint["index"], sign, source, target))
        for gripper in arm.get("gripper", []):
            # No range: the state is already in the joint's units
            source, target = (tuple(gripper[k]) if gripper[k] else None for k in ("from", "to"))
            joints.append(JointMap(gripper["urdf_joint"], gripper["index"], 1.0, source, target))
    return tuple(joints)
