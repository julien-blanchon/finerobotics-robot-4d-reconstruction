"""Robust initial calibration: hand-eye on many small ARROW subsets, then a consensus.

Each subset is one small ARROW encoding with its own relative poses and scale. On the anchor
camera (a free camera on a moving body) every well-conditioned subset gives a mount candidate
by hand-eye, scored by how consistent it is with the kinematics. Bad subsets (confusing views,
ARROW failures) disagree with the rest, so a medoid under a robust distance, refined over its
inliers, ignores them. Other cameras are placed through each subset's reference frame once the
anchor mount is fixed, and get the same consensus.

Ref: Calib3R arXiv:2509.08813 §III-C (hand-eye with scale); Park & Martin 1994.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from r4r.geometry import alignment, se3

from .kinematics import forward_kinematics
from .rig import MOTION_THRESHOLD

if TYPE_CHECKING:
    from torch import Tensor

    from .robot import Robot
    from .tracks import SubsetPoses

# Hand-eye needs rotation about ≥ 2 axes: second singular value of the motions' rotation logs
MIN_ROTATION_SPREAD = 0.1  # rad
# Robust distance between mount candidates: 1 unit = this translation or this rotation
DISTANCE_TRANSLATION = 0.02  # m
DISTANCE_ROTATION = math.radians(2.0)
INLIER_DISTANCE = 2.0


def consensus_initialize(
    robot: Robot,
    qpos: Tensor[[int, int]],
    view_camera: Tensor[[int]],
    view_time: Tensor[[int]],
    subsets: list[SubsetPoses],
    names: tuple[str, ...],
    anchor_mount: Tensor[[4, 4]] | None = None,
) -> Tensor[[int, 4, 4]]:
    """Initial `body_from_camera` per camera (free cameras; others keep their description).

    `anchor_mount`: the anchor camera's mount when already known (e.g. from its own video
    tracks); the other cameras are then placed through the subsets around it.
    """

    world_from_body = forward_kinematics(robot, qpos)
    body = [robot.camera_body[robot.camera(name)] for name in names]
    mounts = robot.body_from_camera[[robot.camera(name) for name in names]].clone()
    free = [name in robot.free_cameras for name in names]

    def moving(c: int) -> bool:
        poses = world_from_body[:, body[c]]
        return bool((poses - poses[:1]).abs().amax() > MOTION_THRESHOLD)

    anchors = [c for c in range(len(names)) if free[c] and moving(c)]
    assert anchors, "need a free camera on a moving body (e.g. a wrist or head camera)"
    anchor = max(anchors, key=lambda c: int((view_camera == c).sum()))

    def link(views: Tensor[[int]]) -> Tensor[[int, 4, 4]]:
        return world_from_body[
            view_time[views], torch.tensor(body, device=views.device)[view_camera[views]]
        ]

    # 1. Anchor mount candidates, one per well-conditioned subset
    candidates = []
    for subset in subsets if anchor_mount is None else ():
        mine = view_camera[subset.views] == anchor
        if mine.sum() < 3:
            continue
        a_link, seen = link(subset.views[mine]), subset.reference_from_view[mine]
        robot_motion = se3.inverse(a_link[:1]) @ a_link[1:]
        if torch.linalg.svdvals(se3.so3_log(robot_motion[:, :3, :3]))[1] < MIN_ROTATION_SPREAD:
            continue
        camera_motion = se3.inverse(seen[:1]) @ seen[1:]
        group = torch.zeros(len(robot_motion), dtype=torch.long, device=robot_motion.device)
        mount, _ = alignment.hand_eye(robot_motion, camera_motion.to(robot_motion), group)
        candidates.append(mount)
    if anchor_mount is None:
        assert candidates, "no subset with enough robot rotation for hand-eye"
        mounts[anchor] = robust_mean(torch.stack(candidates))
    else:
        mounts[anchor] = anchor_mount.to(mounts)

    # 2. Per subset: metric scale and world pose of its reference frame, given the anchor mount
    placed = []
    for subset in subsets:
        mine = view_camera[subset.views] == anchor
        if mine.sum() < 2:
            continue
        world_from_view = link(subset.views[mine]) @ mounts[anchor]
        seen = subset.reference_from_view[mine].to(world_from_view)
        scale = _scale(world_from_view, seen)
        scaled = subset.reference_from_view.to(world_from_view).clone()
        scaled[:, :3, 3] *= scale
        world_from_reference = alignment.mean_transform(world_from_view @ se3.inverse(scaled[mine]))
        placed.append((subset.views, world_from_reference @ scaled))

    # 3. Other free cameras through the placed subsets
    for c in range(len(names)):
        if c == anchor or not free[c]:
            continue
        candidates = [
            se3.inverse(link(views[view_camera[views] == c])) @ poses[view_camera[views] == c]
            for views, poses in placed
            if (view_camera[views] == c).any()
        ]
        mounts[c] = robust_mean(torch.cat(candidates))
    return mounts


def robust_mean(candidates: Tensor[[int, 4, 4]]) -> Tensor[[4, 4]]:
    """Medoid under a truncated distance, then the mean of its inliers."""

    relative = se3.inverse(candidates)[:, None] @ candidates[None]
    rotation = se3.so3_log(relative[..., :3, :3]).norm(dim=-1) / DISTANCE_ROTATION
    translation = (candidates[:, None, :3, 3] - candidates[None, :, :3, 3]).norm(dim=-1)
    distance = rotation + translation / DISTANCE_TRANSLATION
    medoid = distance.clamp_max(10 * INLIER_DISTANCE).sum(1).argmin()
    inliers = distance[medoid] < INLIER_DISTANCE
    return alignment.mean_transform(candidates[inliers])


def _scale(world_from_view: Tensor[[int, 4, 4]], seen: Tensor[[int, 4, 4]]) -> Tensor[[]]:
    """Metres per ARROW unit: least squares on the camera displacements from the first view."""

    metric = (se3.inverse(world_from_view[:1]) @ world_from_view)[:, :3, 3]
    arrow = (se3.inverse(seen[:1]) @ seen)[:, :3, 3]
    return (metric * arrow).sum() / (arrow * arrow).sum().clamp_min(1e-12)
