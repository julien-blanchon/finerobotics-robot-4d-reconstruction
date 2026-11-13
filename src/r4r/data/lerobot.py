"""Load one episode of a LeRobot v3 dataset (finerobotics layout) into an `Episode`.

The only module that imports LeRobot (optional `lerobot` group). finerobotics roots ship the
robot description in `meta/robot/` and `meta/robot.json`, which maps state columns to
description joints and may name camera links; camera features declare their role, mount
(static/dynamic) and side. The dataset's priors (`priors.py`) override all of these.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames

from .episode import CameraStream, Episode, JointMap
from .priors import CameraPrior, load_priors

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
    rows = rows.hf_dataset.with_format("torch")[::stride]
    timestamps = rows["timestamp"].tolist()

    robot = json.loads((root / "meta" / "robot.json").read_text())
    priors = load_priors(root)
    links = {camera["key"]: camera["urdf_link"] for camera in robot.get("cameras", [])}
    cameras = []
    for key in meta.video_keys:
        if meta.features[key].get("info", {}).get("is_depth_map", False):
            continue
        name = key.removeprefix(IMAGE_PREFIX)
        frames = None if decode is None or name in decode else torch.empty(0, 3, 0, 0)
        prior = priors.cameras.get(name, CameraPrior())
        cameras.append(_camera(meta, root, episode, key, timestamps, links.get(key), prior, frames))
    return Episode(
        name=f"{root.name}/episode_{episode:06d}",
        fps=meta.fps / stride,
        state=rows[robot["state"]["feature"]].double(),
        joints=priors.joint_map(_joints(robot)),
        robot=root / "meta" / robot["urdf"],
        cameras=tuple(cameras),
        priors=priors,
    )


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


def _joints(robot: dict) -> tuple[JointMap, ...]:
    """Arm joints and gripper joints of every arm in `robot.json`."""

    signs = robot["state"].get("joint_signs", {})
    joints = []
    for arm in robot["state"]["arms"]:
        for joint in arm["joints"]:
            name = joint["urdf_joint"]
            joints.append(JointMap(name, joint["index"], float(signs.get(name, 1.0))))
        for gripper in arm.get("gripper", []):
            # No range: the state is already in the joint's units
            source, target = (tuple(gripper[k]) if gripper[k] else None for k in ("from", "to"))
            joints.append(JointMap(gripper["urdf_joint"], gripper["index"], 1.0, source, target))
    return tuple(joints)
