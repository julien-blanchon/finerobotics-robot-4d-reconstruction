"""Rigid transforms as 4×4 matrices, with exp/log maps for tangent-space optimisation.

Transforms are named `a_from_b`: `x_a = a_from_b @ x_b`. A tangent vector is
`xi = (v, ω)`: translational part first, rotation (axis-angle) second, and
`exp(xi)` is the group element reached by the constant twist `xi` in unit time.
Quaternions are `(w, x, y, z)` and appear only at I/O boundaries.

Ref: Sola et al., A micro Lie theory for state estimation in robotics, arXiv:1812.01537.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from shape_extensions import IntTuple, IntVar
    from torch import Tensor

# Below this squared angle the closed forms are replaced by their Taylor expansions.
SMALL_ANGLE_SQ = 1e-6


def skew[Bs: IntTuple](v: Tensor[[*Bs, 3]]) -> Tensor[[*Bs, 3, 3]]:
    """Cross-product matrix: `skew(a) @ b == a × b`."""

    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    rows = (
        torch.stack((zero, -z, y), -1),
        torch.stack((z, zero, -x), -1),
        torch.stack((-y, x, zero), -1),
    )
    return torch.stack(rows, -2)


def so3_exp[Bs: IntTuple](omega: Tensor[[*Bs, 3]]) -> Tensor[[*Bs, 3, 3]]:
    """Rodrigues: `R = I + a [ω]× + b [ω]×²`, a = sin θ/θ, b = (1 − cos θ)/θ²."""

    a, b, _ = _rotation_coefficients(omega)
    k = skew(omega)
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device)
    return eye + a * k + b * (k @ k)


def so3_log[Bs: IntTuple](rotation: Tensor[[*Bs, 3, 3]]) -> Tensor[[*Bs, 3]]:
    """Axis-angle of a rotation, through a quaternion so that θ → π stays well conditioned:
    `θ = 2 atan2(‖q_xyz‖, q_w)`, `ω = θ · q_xyz / ‖q_xyz‖` (radians)."""

    q = matrix_to_quaternion(rotation)

    # Pick the hemisphere with w ≥ 0 so that θ ∈ [0, π]
    q = torch.where(q[..., :1] < 0, -q, q)
    w, xyz = q[..., :1], q[..., 1:]
    sin_half = xyz.norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sin_half, w)

    # θ / sin(θ/2) → 2 (1 + θ²/24) as θ → 0
    small = sin_half < SMALL_ANGLE_SQ
    scale = torch.where(small, 2 + angle**2 / 12, angle / sin_half.clamp_min(SMALL_ANGLE_SQ))
    return scale * xyz


def se3_exp[Bs: IntTuple](xi: Tensor[[*Bs, 6]]) -> Tensor[[*Bs, 4, 4]]:
    """Exponential map: `T = [R, V v; 0, 1]`, `V = I + b [ω]× + c [ω]×²`."""

    v, omega = xi[..., :3], xi[..., 3:]
    a, b, c = _rotation_coefficients(omega)
    k = skew(omega)
    kk = k @ k
    eye = torch.eye(3, dtype=xi.dtype, device=xi.device)
    rotation = eye + a * k + b * kk
    translation = ((eye + b * k + c * kk) @ v[..., None])[..., 0]
    return from_rotation_translation(rotation, translation)


def se3_log[Bs: IntTuple](transform: Tensor[[*Bs, 4, 4]]) -> Tensor[[*Bs, 6]]:
    """Logarithm map, the inverse of `se3_exp`: `ω = log R`, `v = V⁻¹ t`."""

    rotation, translation = transform[..., :3, :3], transform[..., :3, 3]
    omega = so3_log(rotation)
    theta_sq = (omega * omega).sum(-1, keepdim=True)[..., None]
    a, b, _ = _rotation_coefficients(omega)

    # V⁻¹ = I − ½ [ω]× + d [ω]×², d = (1 − a / 2b) / θ² → 1/12 + θ²/720
    small = theta_sq < SMALL_ANGLE_SQ
    d = torch.where(
        small,
        1 / 12 + theta_sq / 720,
        (1 - a / (2 * b.clamp_min(SMALL_ANGLE_SQ))) / theta_sq.clamp_min(SMALL_ANGLE_SQ),
    )
    k = skew(omega)
    eye = torch.eye(3, dtype=transform.dtype, device=transform.device)
    v = ((eye - 0.5 * k + d * (k @ k)) @ translation[..., None])[..., 0]
    return torch.cat((v, omega), -1)


def from_rotation_translation[Bs: IntTuple](
    rotation: Tensor[[*Bs, 3, 3]], translation: Tensor[[*Bs, 3]]
) -> Tensor[[*Bs, 4, 4]]:
    """Assemble `[R, t; 0, 1]`."""

    top = torch.cat((rotation, translation[..., None]), -1)
    bottom = torch.zeros_like(top[..., :1, :])
    bottom[..., 3] = 1
    return torch.cat((top, bottom), -2)


def inverse[Bs: IntTuple](transform: Tensor[[*Bs, 4, 4]]) -> Tensor[[*Bs, 4, 4]]:
    """Closed-form rigid inverse `[Rᵀ, −Rᵀ t; 0, 1]`."""

    rotation_t = transform[..., :3, :3].mT
    translation = -(rotation_t @ transform[..., :3, 3:])[..., 0]
    return from_rotation_translation(rotation_t, translation)


def transform_points[Bs: IntTuple, N: IntVar](
    a_from_b: Tensor[[*Bs, 4, 4]], points_b: Tensor[[*Bs, N, 3]]
) -> Tensor[[*Bs, N, 3]]:
    """Map points from frame b to frame a: `x_a = R x_b + t`."""

    return points_b @ a_from_b[..., :3, :3].mT + a_from_b[..., None, :3, 3]


def quaternion_to_matrix[Bs: IntTuple](q: Tensor[[*Bs, 4]]) -> Tensor[[*Bs, 3, 3]]:
    """Rotation matrix of a `(w, x, y, z)` quaternion (normalised first)."""

    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    rows = (
        torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
        torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
        torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1),
    )
    return torch.stack(rows, -2)


def matrix_to_quaternion[Bs: IntTuple](rotation: Tensor[[*Bs, 3, 3]]) -> Tensor[[*Bs, 4]]:
    """`(w, x, y, z)` of a rotation matrix, branch-free (Shepperd's method).

    Each of the four candidate formulas divides by a different component; we keep the one
    whose divisor is largest, which is always ≥ 1/2 and therefore well conditioned.
    """

    m = rotation
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    divisors = torch.stack(
        (1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22), -1
    )
    q_abs = divisors.clamp_min(0).sqrt()
    d21, d12 = m[..., 2, 1], m[..., 1, 2]
    d02, d20 = m[..., 0, 2], m[..., 2, 0]
    d10, d01 = m[..., 1, 0], m[..., 0, 1]

    # Row k is 2 q_k · q, written using only the entries that are stable for that k
    candidates = torch.stack(
        (
            torch.stack((q_abs[..., 0] ** 2, d21 - d12, d02 - d20, d10 - d01), -1),
            torch.stack((d21 - d12, q_abs[..., 1] ** 2, d10 + d01, d02 + d20), -1),
            torch.stack((d02 - d20, d10 + d01, q_abs[..., 2] ** 2, d12 + d21), -1),
            torch.stack((d10 - d01, d20 + d02, d21 + d12, q_abs[..., 3] ** 2), -1),
        ),
        -2,
    )
    candidates = candidates / (2 * q_abs[..., None].clamp_min(0.1))
    best = q_abs.argmax(-1, keepdim=True)[..., None].expand_as(candidates[..., :1, :])
    return F.normalize(candidates.gather(-2, best)[..., 0, :], dim=-1)


def _rotation_coefficients[Bs: IntTuple](
    omega: Tensor[[*Bs, 3]],
) -> tuple[Tensor[[*Bs, 1, 1]], Tensor[[*Bs, 1, 1]], Tensor[[*Bs, 1, 1]]]:
    """`(sin θ/θ, (1 − cos θ)/θ², (θ − sin θ)/θ³)` with Taylor fallbacks near θ = 0."""

    theta_sq = (omega * omega).sum(-1)[..., None, None]
    small = theta_sq < SMALL_ANGLE_SQ
    theta_sq_safe = theta_sq.clamp_min(SMALL_ANGLE_SQ)
    theta = theta_sq_safe.sqrt()
    a = torch.where(small, 1 - theta_sq / 6, theta.sin() / theta)
    b = torch.where(small, 0.5 - theta_sq / 24, (1 - theta.cos()) / theta_sq_safe)
    c = torch.where(small, 1 / 6 - theta_sq / 120, (theta - theta.sin()) / (theta_sq_safe * theta))
    return a, b, c
