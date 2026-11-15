"""Cameras recorded in a world frame, expressed in the robot's base frame at load time.

Some simulators record every camera's pose per frame in their scene ("world") frame but not
the robot's base pose in it (molmobot's houses). One camera rigid on a known link (a wrist)
ties the two frames: between frames, that link's motion from forward kinematics and the
camera's recorded motion are related by hand-eye (`A X = X B`), which gives the mount `X`, then
`world_from_base = world_from_camera · X⁻¹ · base_from_link⁻¹` at every frame (constant for a
fixed base; its spread checks the record). Every camera of the episode is then placed in the
base frame (static) or on its link (moving), at the episode's middle frame. The recorded poses
are taken synchronous with the joint-state rows of the same frames (no `state_shift`).
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, cast

import torch

from ..geometry import alignment, se3

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from r4r.data import episode as checked_episode
    from r4r.types import Mat4x4

    from .episode import JointMap
    from .priors import PoseColumn


def world_from_camera(values: Tensor[[int, int]], source: PoseColumn) -> Tensor[[int, *Mat4x4]]:
    """Per-frame `world_from_camera` (OpenCV axes) from a recorded column of flattened 3×4 or
    4×4 matrices, or of x, y, z, qw, qx, qy, qz rows (`world_to_camera` or `world_from_camera`
    layout); NaN where a row is not a finite rigid transform (some records end with zero or
    infinite rows)."""

    values = values.double()
    if values.shape[1] == 7:
        q = values[:, 3:]
        rotation = se3.quaternion_to_matrix(q / q.norm(dim=1, keepdim=True))
        matrix = se3.from_rotation_translation(rotation, values[:, :3])
    else:
        rows = values.shape[1] // 4
        matrix = torch.zeros(len(values), 4, 4, dtype=torch.float64)
        matrix[:, :rows] = values.reshape(len(values), rows, 4)
        matrix[:, 3, 3] = 1.0
    rotation = matrix[:, :3, :3].nan_to_num(0.0, 0.0, 0.0)
    eye = torch.eye(3, dtype=torch.float64)
    valid = matrix.flatten(1).isfinite().all(1) & (
        (rotation @ rotation.transpose(1, 2) - eye).abs().flatten(1).amax(1) < 1e-3
    )
    pose = torch.full_like(matrix, float("nan"))
    if source.layout == "world_to_camera":
        pose[valid] = se3.inverse(matrix[valid])
    else:
        pose[valid] = matrix[valid]
    return pose


@functools.cache
def _robot(description: Path):  # noqa: ANN202 (r4r.rago is imported lazily)
    from r4r.rago.robot import load_robot

    return load_robot(description).to("cpu", torch.float64)


def base_frame_poses(
    poses: dict[str, Tensor[[int, *Mat4x4]]],
    links: dict[str, str | None],
    anchor: str,
    state: Tensor[[int, int]],
    joints: tuple[JointMap, ...],
    description: Path,
    frames: int = 12,
) -> dict[str, Tensor[[*Mat4x4]]]:
    """Each camera's pose at the episode's middle frame: in the base frame (`links[c]` None) or
    on its link, from per-frame `world_from_camera` `poses`, the anchor camera on its link and
    the joint states (one row per frame, unshifted). Uses frames where every camera's record is
    valid; `frames` of them, evenly spread, for the hand-eye."""

    from r4r.rago.kinematics import forward_kinematics
    from r4r.rago.robot import qpos_from_state

    robot = _robot(description)
    link = links[anchor]
    assert link is not None, f"recorded cameras: the anchor {anchor} needs a link"
    valid = torch.stack([p.flatten(1).isfinite().all(1) for p in poses.values()]).all(0)
    index = valid.nonzero()[:, 0]
    if len(index) < 3:
        raise ValueError(f"recorded cameras: {len(index)} frames with every camera recorded")
    middle = int(index[len(index) // 2])
    sample = index[torch.linspace(0, len(index) - 1, min(frames, len(index))).long()]
    # The same JointMap seen through r4r.rago's absolute import (one class at run time)
    mapping = cast("tuple[checked_episode.JointMap, ...]", joints)
    world = forward_kinematics(robot, qpos_from_state(robot, state[sample].double(), mapping))
    body = robot.body(link)
    base_from_link = world[:, body]
    camera = poses[anchor][sample]
    pairs = torch.arange(len(sample) - 1)
    mount, _ = alignment.hand_eye(
        se3.inverse(base_from_link[pairs]) @ base_from_link[pairs + 1],
        se3.inverse(camera[pairs]) @ camera[pairs + 1],
        torch.zeros(len(pairs), dtype=torch.long),
    )
    world_from_base = alignment.mean_transform(
        camera @ se3.inverse(mount) @ se3.inverse(base_from_link)
    )
    at_middle = forward_kinematics(robot, qpos_from_state(robot, state[[middle]].double(), mapping))
    base_from_world = se3.inverse(world_from_base)
    placed = {}
    for name, pose in poses.items():
        base_from_camera = base_from_world @ pose[middle]
        on = links[name]
        if on is not None:
            base_from_camera = se3.inverse(at_middle[0, robot.body(on)]) @ base_from_camera
        placed[name] = base_from_camera
    return placed
