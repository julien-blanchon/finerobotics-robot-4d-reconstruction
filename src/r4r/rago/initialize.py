"""Closed-form initial calibration from ARROW's relative poses and the robot's kinematics.

1. The free camera on a moving body with the most views anchors everything: hand-eye
   (`A X = X B(s_e)`) on its motions, relative to its first view in each episode, gives its
   mount X and every episode's metric scale s_e.
2. Where each episode's ARROW reference frame sits in the world follows from that camera's views:
   `world_from_reference = world_from_body(t) · X · (s-scaled reference_from_view)⁻¹`, averaged.
3. Every other free camera, static or moving, is placed through its episode's reference frame
   and its mount averaged over all its views. Cameras without a prior keep their description mount.

Ref: Calib3R arXiv:2509.08813 §III-C (the hand-eye formulation); Park & Martin 1994.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from r4r.geometry import alignment, se3

from .kinematics import forward_kinematics
from .rig import MOTION_THRESHOLD

if TYPE_CHECKING:
    from torch import Tensor

    from .observations import Observations
    from .robot import Robot


def initialize(
    robot: Robot, qpos: Tensor[[int, int]], observations: Observations, names: tuple[str, ...]
) -> tuple[Tensor[[int, 4, 4]], Tensor[[int]]]:
    """Initial `body_from_camera` per camera and metres per ARROW unit per episode."""

    obs = observations
    world_from_body = forward_kinematics(robot, qpos)
    body = [robot.camera_body[robot.camera(name)] for name in names]
    mounts = robot.body_from_camera[[robot.camera(name) for name in names]].clone()

    def views(camera: int, episode: int) -> Tensor[[int]]:
        index = ((obs.view_camera == camera) & (obs.view_episode == episode)).nonzero()[:, 0]
        return index[obs.view_time[index].argsort()]

    def moving(camera: int) -> bool:
        poses = world_from_body[:, body[camera]]
        return bool((poses - poses[:1]).abs().amax() > MOTION_THRESHOLD)

    free = [name in robot.free_cameras for name in names]
    anchors = [c for c in range(len(names)) if free[c] and moving(c)]
    assert anchors, "need a free camera on a moving body to recover scale (e.g. a wrist camera)"
    anchor = max(anchors, key=lambda c: int((obs.view_camera == c).sum()))
    episodes = range(obs.episodes)

    # 1. Hand-eye on the anchor's motions relative to its first view, all episodes at once
    robot_motion, camera_motion, group = [], [], []
    for episode in episodes:
        index = views(anchor, episode)
        link = world_from_body[obs.view_time[index], body[anchor]]
        seen = obs.reference_from_view[index]
        robot_motion.append(se3.inverse(link[:1]) @ link[1:])
        camera_motion.append(se3.inverse(seen[:1]) @ seen[1:])
        group.append(torch.full((len(index) - 1,), episode, device=index.device))
    mounts[anchor], scales = alignment.hand_eye(
        torch.cat(robot_motion), torch.cat(camera_motion), torch.cat(group)
    )

    # 2. World pose of each episode's ARROW reference frame
    def metric(reference_from_view: Tensor[[int, 4, 4]], episode: int) -> Tensor[[int, 4, 4]]:
        scaled = reference_from_view.clone()
        scaled[:, :3, 3] = scaled[:, :3, 3] * scales[episode]
        return scaled

    world_from_reference = []
    for episode in episodes:
        index = views(anchor, episode)
        link = world_from_body[obs.view_time[index], body[anchor]]
        seen = metric(obs.reference_from_view[index], episode)
        world_from_reference.append(
            alignment.mean_transform(link @ mounts[anchor] @ se3.inverse(seen))
        )

    # 3. Every other free camera through its episodes' reference frames
    for camera in range(len(names)):
        if camera == anchor or not free[camera]:
            continue
        candidates = []
        for episode in episodes:
            index = views(camera, episode)
            world_from_view = world_from_reference[episode] @ metric(
                obs.reference_from_view[index], episode
            )
            link = world_from_body[obs.view_time[index], body[camera]]
            candidates.append(se3.inverse(link) @ world_from_view)
        mounts[camera] = alignment.mean_transform(torch.cat(candidates))
    return mounts, scales
