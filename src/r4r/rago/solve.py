"""Calibrate camera mounts and scale: closed-form initialisation, then gradient refinement.

`method` selects one of three formulations under evaluation (STYLE §5 exception): they share
the observations, the rig and the scene losses, and differ only in how view poses are made.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from .hard import Hard
from .hybrid import Hybrid
from .initialize import initialize
from .losses import scene_losses
from .rig import Rig
from .soft import Soft

if TYPE_CHECKING:
    from torch import Tensor

    from .observations import Observations
    from .robot import Robot

type Method = Literal["soft", "hard", "hybrid"]


@dataclass(frozen=True, slots=True, kw_only=True)
class SolveConfig:
    method: Method = "hybrid"
    iterations: int = 800
    learning_rate: float = 3e-3
    weight_3d: float = 1.0
    weight_2d: float = 1.0
    # Cauchy scales: 3D in metres, 2D in pixels / focal length
    cauchy_3d: float = 0.01
    cauchy_2d: float = 0.005
    # hybrid: per-view deviation from kinematics (m, rad)
    view_translation_std: float = 0.01
    view_rotation_std: float = 0.01
    # soft: Cauchy scale of the AX = XB residual (m)
    cauchy_calibration: float = 0.01


@dataclass(frozen=True, slots=True)
class Calibration:
    """Calibrated mounts, scale (metres per ARROW unit) and kinematic view poses."""

    names: tuple[str, ...]
    body_from_camera: Tensor[[int, 4, 4]]
    initial_body_from_camera: Tensor[[int, 4, 4]]
    scale: float
    initial_scale: float
    world_from_view: Tensor[[int, 4, 4]]
    # Per iteration: total, 3D, 2D, penalty
    history: Tensor[[int, 4]]


def calibrate(
    robot: Robot,
    qpos: Tensor[[int, int]],
    observations: Observations,
    names: tuple[str, ...],
    config: SolveConfig,
) -> Calibration:
    """Initialise in closed form, then refine with `config.method`."""

    obs = observations.to(qpos.device, qpos.dtype)
    initial_mounts, initial_scale = initialize(robot, qpos, obs, names)
    rig = Rig(robot, qpos, names, initial_mounts, initial_scale)
    method = {
        "hard": lambda: Hard(rig),
        "hybrid": lambda: Hybrid(
            rig, len(obs.view_time), config.view_translation_std, config.view_rotation_std
        ),
        "soft": lambda: Soft(rig, obs, config.cauchy_calibration),
    }[config.method]()

    optimizer = torch.optim.Adam(method.parameters(), lr=config.learning_rate)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 0.5 * (1 + math.cos(math.pi * step / config.iterations))
    )
    history = []
    for _ in range(config.iterations):
        poses, penalty = method(obs)
        loss_3d, loss_2d = scene_losses(poses, rig.scale, obs, config.cauchy_3d, config.cauchy_2d)
        # Scene terms are means over correspondences; priors weigh like one correspondence each,
        # as in a sum of residuals plus a prior
        penalty = penalty / len(obs.view_a)
        loss = config.weight_3d * loss_3d + config.weight_2d * loss_2d + penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        history.append(torch.stack((loss, loss_3d, loss_2d, penalty)).detach())

    with torch.no_grad():
        return Calibration(
            names=names,
            body_from_camera=method.calibrated_mounts(obs),
            initial_body_from_camera=initial_mounts,
            scale=float(rig.scale),
            initial_scale=initial_scale,
            world_from_view=rig.camera_poses(obs.view_camera, obs.view_time),
            history=torch.stack(history),
        )
