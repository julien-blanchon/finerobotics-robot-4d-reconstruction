"""The robot and its cameras with their unknowns: mounts, focal lengths, ARROW's metric scales.

A camera is rigidly mounted on a body of the kinematic tree (a wrist camera on the hand, an
exterior camera on the world body). Its pose at time t is `world_from_body(q_t) · body_from_camera`.
Free mounts (declared with priors in the robot description) are optimised as
`initial · exp(δ)`; the others stay at their description pose. Each camera's focal length is
corrected by a factor `exp(φ)` with a log-normal prior (std `focal_std`): wide when the initial
intrinsics come from ARROW, tight when the dataset knows them. Each episode (one ARROW encoding)
has its own scale, metres per ARROW unit.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

from r4r.geometry import se3

from .kinematics import forward_kinematics

# A body "moves" if its pose changes by more than this over the episode (m or rad)
MOTION_THRESHOLD = 1e-3

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    from .robot import Robot


class Rig(nn.Module):
    """Cameras `names` on `robot`, joint trajectory `qpos [T, Q]` (all episodes concatenated)."""

    def __init__(
        self,
        robot: Robot,
        qpos: Tensor[[int, int]],
        names: tuple[str, ...],
        initial_mounts: Tensor[[int, 4, 4]],
        initial_scales: Tensor[[int]],
        initial_intrinsics: Tensor[[int, 3, 3]],
        focal_std: float,
    ) -> None:
        super().__init__()
        assert not robot.free_bodies, "free bodies (e.g. a second arm's base) are not supported yet"
        index = [robot.camera(name) for name in names]
        self.names = names
        self.camera_body = [robot.camera_body[i] for i in index]

        # Joint values are data: FK is computed once
        self.register_buffer("world_from_body", forward_kinematics(robot, qpos))
        self.register_buffer(
            "camera_body_index", torch.tensor(self.camera_body, device=qpos.device)
        )
        self.register_buffer("initial_mount", initial_mounts)
        self.register_buffer("default_mount", robot.body_from_camera[index])

        priors = [robot.free_cameras.get(name) for name in names]
        self.register_buffer(
            "free", torch.tensor([p is not None for p in priors], device=qpos.device)
        )
        stds = [p or (math.inf, math.inf) for p in priors]
        self.register_buffer("prior_std", qpos.new_tensor(stds))

        self.mount_delta = nn.Parameter(qpos.new_zeros(len(names), 6))
        self.log_scale = nn.Parameter(initial_scales.to(qpos).log())
        self.register_buffer("initial_intrinsics", initial_intrinsics.to(qpos))
        self.log_focal = nn.Parameter(qpos.new_zeros(len(names)))
        self.focal_std = focal_std

    def scales(self) -> Tensor[[int]]:
        """Metres per ARROW unit, per episode."""

        return self.log_scale.exp()

    def intrinsics(self) -> Tensor[[int, 3, 3]]:
        """Per camera: initial intrinsics with corrected focal lengths (square pixels kept)."""

        factor = torch.ones_like(self.initial_intrinsics)
        factor[:, [0, 1], [0, 1]] = self.log_focal.exp()[:, None]
        return self.initial_intrinsics * factor

    def mounts(self) -> Tensor[[int, 4, 4]]:
        """`body_from_camera` per camera."""

        delta = torch.where(self.free[:, None], self.mount_delta, 0.0)
        return self.initial_mount @ se3.se3_exp(delta)

    def body_poses[V: IntVar](self, camera: Tensor[[V]], time: Tensor[[V]]) -> Tensor[[V, 4, 4]]:
        """`world_from_body` of each view's mount body."""

        return self.world_from_body[time, self.camera_body_index[camera]]

    def camera_poses[V: IntVar](self, camera: Tensor[[V]], time: Tensor[[V]]) -> Tensor[[V, 4, 4]]:
        """`world_from_camera` of each view, from kinematics and the mounts."""

        return self.body_poses(camera, time) @ self.mounts()[camera]

    def moving_cameras(self) -> Tensor[[int]]:
        """Which cameras sit on a body that moves during the episode."""

        body = self.world_from_body[:, self.camera_body]
        return (body - body[:1]).abs().flatten(-2).amax(-1).amax(0) > MOTION_THRESHOLD

    def prior(self) -> Tensor[[]]:
        """Gaussian prior of the free mounts around their description pose."""

        deviation = se3.se3_log(se3.inverse(self.default_mount) @ self.mounts())
        translation, rotation = deviation[:, :3], deviation[:, 3:]
        cost = (translation**2).sum(-1) / self.prior_std[:, 0] ** 2
        cost = cost + (rotation**2).sum(-1) / self.prior_std[:, 1] ** 2
        focal = (self.log_focal**2).sum() / self.focal_std**2
        return torch.where(self.free, cost, 0.0).sum() + focal
