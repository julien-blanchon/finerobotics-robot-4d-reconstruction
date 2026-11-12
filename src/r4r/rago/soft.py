"""Soft constraint (Calib3R): free per-view poses tied to the robot by an AX = XB penalty.

Every view has its own pose. For consecutive views of a camera, the camera's motion
`B = pose_i⁻¹ pose_{i+1}` must match its mount body's motion `A` seen through the mount `X`:
`‖A X − X B‖_F`. Poses are metric here (ARROW points are scaled before use), so Calib3R's
per-camera λ is the rig's single scale. Static cameras get `A = I`: they must not move.

Ref: Calib3R arXiv:2509.08813, Eqs. 6–9 and 12.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from einops import rearrange
from torch import nn

from r4r.geometry import alignment, se3

from .losses import cauchy

if TYPE_CHECKING:
    from torch import Tensor

    from .observations import Observations
    from .rig import Rig


class Soft(nn.Module):
    def __init__(self, rig: Rig, observations: Observations, cauchy_scale: float) -> None:
        super().__init__()
        self.rig = rig
        self.cauchy_scale = cauchy_scale
        obs = observations
        with torch.no_grad():
            initial = rig.camera_poses(obs.view_camera, obs.view_time)
        self.register_buffer("initial_pose", initial)
        self.view_delta = nn.Parameter(initial.new_zeros(len(initial), 6))

        # Consecutive (in time) pairs of views of the same camera within one episode
        order = torch.argsort(obs.view_camera * (obs.view_time.max() + 1) + obs.view_time)
        first, second = order[:-1], order[1:]
        same = (obs.view_camera[first] == obs.view_camera[second]) & (
            obs.view_episode[first] == obs.view_episode[second]
        )
        self.register_buffer("first", order[:-1][same])
        self.register_buffer("second", order[1:][same])
        # Weigh each robot motion like the correspondences of one view pair
        self.matches_per_motion = len(obs.view_a) / max(int(same.sum()), 1)

    def forward(self, observations: Observations) -> tuple[Tensor[[int, 4, 4]], Tensor[[]]]:
        """(world_from_view, penalty)."""

        obs = observations
        poses = self.initial_pose @ se3.se3_exp(self.view_delta)

        body = self.rig.body_poses(obs.view_camera, obs.view_time)
        body_motion = se3.inverse(body[self.first]) @ body[self.second]
        camera_motion = se3.inverse(poses[self.first]) @ poses[self.second]
        mount = self.rig.mounts()[obs.view_camera[self.first]]
        residual = body_motion @ mount - mount @ camera_motion
        error = rearrange(residual, "k i j -> k (i j)").norm(dim=-1)

        # A data term like the scene terms: Cauchy cost per robot motion, summed (the solver
        # divides penalties by the number of correspondences)
        calibration = cauchy(error, self.cauchy_scale).sum() * self.matches_per_motion
        return poses, self.rig.prior() + calibration

    def calibrated_mounts(self, observations: Observations) -> Tensor[[int, 4, 4]]:
        """Mounts of moving cameras from the AX = XB term; static ones through the gauge.

        The free poses live in their own world frame. Moving cameras tie it to the robot's:
        `robot_from_free = world_from_body · X · pose⁻¹`, averaged over their views. A static
        camera's mount (unconstrained by AX = XB, where A = I) is then read off its poses.
        """

        obs = observations
        poses, _ = self(obs)
        body = self.rig.body_poses(obs.view_camera, obs.view_time)
        mounts = self.rig.mounts().clone()
        moving = self.rig.moving_cameras()
        dynamic = moving[obs.view_camera]
        robot_from_free = alignment.mean_transform(
            body[dynamic] @ mounts[obs.view_camera[dynamic]] @ se3.inverse(poses[dynamic])
        )
        for camera in (~moving & self.rig.free).nonzero()[:, 0]:
            views = obs.view_camera == camera
            mounts[camera] = alignment.mean_transform(
                se3.inverse(body[views]) @ robot_from_free @ poses[views]
            )
        return mounts
