"""A mobile base's motion over an episode, from what the dataset records, as a starting
trajectory for its per-frame pose (`bundle_adjust(..., bodies_at=...)`).

Three kinds of record, declared per dataset in the priors (`base_odometry`):

- `pose`: planar x, y, θ of the base in some world frame (odometry or a simulator's state);
- `velocity`: the body-frame planar velocity vx, vy, ω (commanded or measured);
- `caster`: wheel encoder angles of a differential-drive pair, optionally on a steered unit
  whose axis is `offset` ahead of the wheels (an active caster, e.g. Toyota HSR; a plain
  differential drive is offset 0 and no steering).

All give `base_motion [T, 4, 4]`: the base's pose at each timestep relative to the first, in
the base's own frame (planar, z up), the correction a frame-scope free base starts from.

Active caster: with wheel rates ω_l, ω_r, radius r, half track d, the unit moves forward
v = r (ω_l + ω_r) / 2 and turns ω_u = r (ω_r − ω_l) / (2d). The steering axis, `offset` ahead
of the wheels, then moves (v, offset · ω_u) in the unit's frame, i.e. R(φ) (v, offset · ω_u) in
the body's (φ: steering angle), and the body turns ω_u − φ̇.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from r4r.geometry import se3

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.types import WXYZ, XY, XYZ, Mat4x4, XYTheta


@dataclass(frozen=True, slots=True)
class CasterGeometry:
    radius: float  # wheel radius (m)
    half_track: float  # half the distance between the wheels (m)
    offset: float = 0.0  # steering axis ahead of the wheel axle (m); 0: plain differential drive
    # Sign of each recorded angle: left wheel, right wheel, steering (encoder conventions)
    signs: tuple[float, float, float] = (1.0, 1.0, 1.0)


def from_poses[T: IntVar](
    x: Tensor[[T]], y: Tensor[[T]], theta: Tensor[[T]]
) -> Tensor[[T, *Mat4x4]]:
    """Base motion from planar poses in any fixed world frame: `M_t = P_0⁻¹ P_t`,
    `P_t = [R_z(θ_t), (x_t, y_t, 0); 0, 1]` (metres, radians)."""

    pose = torch.eye(4, dtype=x.dtype, device=x.device).repeat(len(x), 1, 1)
    c, s = theta.cos(), theta.sin()
    pose[:, 0, 0], pose[:, 0, 1], pose[:, 1, 0], pose[:, 1, 1] = c, -s, s, c
    pose[:, 0, 3], pose[:, 1, 3] = x, y
    return se3.inverse(pose[:1]) @ pose


def from_poses_3d[T: IntVar](
    position: Tensor[[T, *XYZ]], quaternion: Tensor[[T, *WXYZ]]
) -> Tensor[[T, *Mat4x4]]:
    """Base motion from full poses (position, quaternion w x y z) in any fixed world frame: a
    recorded root pose (simulator free joint, a legged robot's state estimator).
    `M_t = P_0⁻¹ P_t` as in `from_poses`."""

    rotation = se3.quaternion_to_matrix(quaternion / quaternion.norm(dim=-1, keepdim=True))
    pose = se3.from_rotation_translation(rotation, position)
    return se3.inverse(pose[:1]) @ pose


def from_steps[T: IntVar](step: Tensor[[T, *XYTheta]]) -> Tensor[[T, *Mat4x4]]:
    """Base motion from per-step body-frame displacements (dx, dy, dθ) between consecutive
    timesteps (row t: from t − 1 to t; row 0 ignored), each a constant-twist arc:
    `M_t = M_{t−1} · exp(dx, dy, 0, 0, 0, dθ)`."""

    zero = torch.zeros_like(step[:, 0])
    twist = torch.stack((step[:, 0], step[:, 1], zero, zero, zero, step[:, 2]), -1)
    increments = se3.se3_exp(twist)
    increments[0] = torch.eye(4, dtype=step.dtype, device=step.device)
    poses = [increments[0]]
    for t in range(1, len(step)):
        poses.append(poses[-1] @ increments[t])
    return torch.stack(poses)


def from_velocity[T: IntVar](
    velocity: Tensor[[T, *XYTheta]], time: Tensor[[T]]
) -> Tensor[[T, *Mat4x4]]:
    """Base motion from body-frame planar velocities (vx, vy, ω) at timestamps `time` (s):
    trapezoidal steps `Δ_t = ½ (u_{t−1} + u_t) (t_t − t_{t−1})`, integrated by `from_steps`."""

    dt = torch.diff(time, prepend=time[:1])
    mean = torch.cat((velocity[:1], 0.5 * (velocity[1:] + velocity[:-1])))
    return from_steps(mean * dt[:, None])


def swerve_twist[T: IntVar, NWheel: IntVar](
    steering: Tensor[[T, NWheel]], speed: Tensor[[T, NWheel]], wheels: Tensor[[NWheel, *XY]]
) -> Tensor[[T, *XYTheta]]:
    """Body-frame twist (vx, vy, ω) of a swerve drive per timestep, by least squares over its
    wheels: wheel i at (x_i, y_i), steered to θ_i and rolling at s_i (m/s) moves at
    s_i (cos θ_i, sin θ_i) = (vx − ω y_i, vy + ω x_i)."""

    x, y = wheels[:, 0], wheels[:, 1]
    one, zero = torch.ones_like(x), torch.zeros_like(x)
    design = torch.stack(
        (torch.stack((one, zero, -y), -1), torch.stack((zero, one, x), -1)), 1
    ).reshape(-1, 3)  # [2W, 3]: rows (vx, vy, ω) per wheel, x then y
    target = torch.stack((speed * torch.cos(steering), speed * torch.sin(steering)), -1)
    return target.reshape(len(target), -1) @ torch.linalg.pinv(design).T


def caster_steps[T: IntVar](
    wheels: Tensor[[T, 2]], steering: Tensor[[T]] | None, geometry: CasterGeometry
) -> Tensor[[T, *XYTheta]]:
    """Per-step body-frame displacements (dx, dy, dθ) from wheel angles (left, right) and the
    steering angle (None: no steering, a plain differential drive)."""

    left, right, steer = geometry.signs
    angle = wheels * torch.tensor([left, right], dtype=wheels.dtype, device=wheels.device)
    turn = torch.diff(angle, dim=0, prepend=angle[:1])
    forward = geometry.radius * turn.mean(-1)
    unit_turn = geometry.radius * (turn[:, 1] - turn[:, 0]) / (2 * geometry.half_track)
    phi = torch.zeros_like(forward) if steering is None else steer * steering
    # Steering angle at mid-step; the body turns by the unit's turn minus the steering change
    mid = torch.cat((phi[:1], 0.5 * (phi[1:] + phi[:-1])))
    sideways = geometry.offset * unit_turn
    dx = mid.cos() * forward - mid.sin() * sideways
    dy = mid.sin() * forward + mid.cos() * sideways
    dtheta = unit_turn - torch.diff(phi, prepend=phi[:1])
    return torch.stack((dx, dy, dtheta), -1)


Kind = Literal["pose", "velocity", "caster"]
