"""Load one episode of a LeRobot v3 dataset (finerobotics layout) into an `Episode`.

The only module that imports LeRobot (optional `lerobot` group). finerobotics roots ship the
robot description in `meta/robot/` and `meta/robot.json`, which maps state columns to
description joints and names each wrist camera's link; camera features declare their
role, mount (static/dynamic) and side.
"""

from __future__ import annotations

import json
from pathlib import Path

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames

from .episode import CameraStream, Episode, JointMap

IMAGE_PREFIX = "observation.images."


def load_episode(root: Path, episode: int, stride: int = 1) -> Episode:
    """Every `stride`-th timestep of `episode` from the dataset root `root`."""

    repo_id = f"local/{root.name}"
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    rows = LeRobotDataset(repo_id, root=root, episodes=[episode], video_backend="pyav")
    rows = rows.hf_dataset.with_format("torch")[::stride]
    timestamps = rows["timestamp"].tolist()

    robot = json.loads((root / "meta" / "robot.json").read_text())
    links = {camera["key"]: camera["urdf_link"] for camera in robot.get("cameras", [])}
    cameras = tuple(
        _camera(meta, root, episode, key, timestamps, links.get(key))
        for key in meta.video_keys
        if not meta.features[key].get("info", {}).get("is_depth_map", False)
    )
    return Episode(
        name=f"{root.name}/episode_{episode:06d}",
        fps=meta.fps / stride,
        state=rows[robot["state"]["feature"]].double(),
        joints=_joints(robot),
        robot=root / "meta" / robot["urdf"],
        cameras=cameras,
    )


def _camera(
    meta: LeRobotDatasetMetadata,
    root: Path,
    episode: int,
    key: str,
    timestamps: list[float],
    link: str | None,
) -> CameraStream:
    info = meta.features[key].get("info", {})
    assert meta.episodes is not None, f"{root}: episode metadata missing"
    start = meta.episodes[episode][f"videos/{key}/from_timestamp"]
    path = root / meta.get_video_file_path(episode, key)
    frames = decode_video_frames(
        path, [start + t for t in timestamps], 1 / meta.fps / 2, "pyav", return_uint8=True
    )
    dynamic = info.get("camera.mount") == "dynamic"
    mount = link or (info.get("camera.extrinsics_frame") if dynamic else None)
    assert not dynamic or mount, f"{key}: moving camera without a declared link"
    return CameraStream(
        name=key.removeprefix(IMAGE_PREFIX),
        frames=frames,
        role=info.get("camera.role", "other"),
        mount=mount,
        side=info.get("camera.side"),
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
            source, target = (tuple(gripper[k]) for k in ("from", "to"))
            joints.append(JointMap(gripper["urdf_joint"], gripper["index"], 1.0, source, target))
    return tuple(joints)
