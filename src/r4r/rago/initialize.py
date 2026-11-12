"""Closed-form initial calibration from ARROW's relative poses and the robot's kinematics.

1. The free camera on a moving body with the most views anchors everything: hand-eye
   (`A X = X B(s)`) on its motions gives its mount X and ARROW's metric scale s.
2. Where ARROW's reference frame sits in the world follows from that camera's views:
   `world_from_reference = world_from_body(t) · X · (s-scaled reference_from_view)⁻¹`, averaged.
3. Every other free camera, static or moving, is placed through the reference frame and its
   mount averaged over its views. Cameras without a prior keep their description mount.

Ref: Calib3R arXiv:2509.08813 §III-C (the hand-eye formulation); Park & Martin 1994.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from r4r.geometry import alignment, se3

from .kinematics import forward_kinematics
from .rig import MOTION_THRESHOLD

if TYPE_CHECKING:
    from torch import Tensor

    from .observations import Observations
    from .robot import Robot


def initialize(
    robot: Robot, qpos: Tensor[[int, int]], observations: Observations, names: tuple[str, ...]
) -> tuple[Tensor[[int, 4, 4]], float]:
    """Initial `body_from_camera` per camera and metres per ARROW unit."""

    obs = observations
    world_from_body = forward_kinematics(robot, qpos)
    body = [robot.camera_body[robot.camera(name)] for name in names]
    mounts = robot.body_from_camera[[robot.camera(name) for name in names]].clone()

    def views(camera: int) -> Tensor[[int]]:
        index = (obs.view_camera == camera).nonzero()[:, 0]
        return index[obs.view_time[index].argsort()]

    def moving(camera: int) -> bool:
        poses = world_from_body[:, body[camera]]
        return bool((poses - poses[:1]).abs().amax() > MOTION_THRESHOLD)

    free = [name in robot.free_cameras for name in names]
    anchors = [c for c in range(len(names)) if free[c] and moving(c)]
    assert anchors, "need a free camera on a moving body to recover scale (e.g. a wrist camera)"
    anchor = max(anchors, key=lambda c: len(views(c)))

    # 1. Hand-eye on the anchor camera's motions relative to its first view
    index = views(anchor)
    link = world_from_body[obs.view_time[index], body[anchor]]
    seen = obs.reference_from_view[index]
    robot_motion = se3.inverse(link[:1]) @ link[1:]
    camera_motion = se3.inverse(seen[:1]) @ seen[1:]
    mounts[anchor], scale = alignment.hand_eye(robot_motion, camera_motion)

    # 2. World pose of ARROW's reference frame
    def metric(reference_from_view: Tensor[[int, 4, 4]]) -> Tensor[[int, 4, 4]]:
        scaled = reference_from_view.clone()
        scaled[:, :3, 3] = scaled[:, :3, 3] * scale
        return scaled

    world_from_reference = alignment.mean_transform(
        link @ mounts[anchor] @ se3.inverse(metric(seen))
    )

    # 3. Every other free camera through the reference frame
    for camera in range(len(names)):
        if camera == anchor or not free[camera]:
            continue
        index = views(camera)
        world_from_view = world_from_reference @ metric(obs.reference_from_view[index])
        link = world_from_body[obs.view_time[index], body[camera]]
        mounts[camera] = alignment.mean_transform(se3.inverse(link) @ world_from_view)
    return mounts, float(scale)
