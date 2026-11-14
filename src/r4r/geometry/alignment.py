"""Closed-form alignment: weighted rigid Procrustes and hand-eye calibration (AX = XB) with scale.

Ref: Arun et al. 1987 / Umeyama 1991 (Procrustes); Park & Martin 1994 (AX = XB on SE(3)).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from einops import rearrange

from . import se3

if TYPE_CHECKING:
    from shape_extensions import IntTuple, IntVar
    from torch import Tensor

    from r4r.types import Mat4x4


def procrustes[Bs: IntTuple, N: IntVar](
    source: Tensor[[*Bs, N, 3]], target: Tensor[[*Bs, N, 3]], weight: Tensor[[*Bs, N]]
) -> Tensor[[*Bs, 4, 4]]:
    """`target_from_source` minimising `Σ w ‖target − (R source + t)‖²` (fp64 SVD).

    With weighted means `μ_s`, `μ_t` and cross-covariance `Σ = Σ w (s − μ_s)(t − μ_t)ᵀ`:
    `R = closest_rotation(Σ)`, `t = μ_t − R μ_s`. Rigid (no scale); units of the inputs.
    """

    w = (weight / weight.sum(-1, keepdim=True).clamp_min(1e-12))[..., None].double()
    source, target = source.double(), target.double()
    source_mean, target_mean = (w * source).sum(-2), (w * target).sum(-2)
    # Annotated: pyrefly loses the variadic batch through `.mT @` (known gap)
    covariance: Tensor[[*Bs, 3, 3]] = (w * (source - source_mean[..., None, :])).mT @ (
        target - target_mean[..., None, :]
    )
    rotation = closest_rotation(covariance)
    translation: Tensor[[*Bs, 3]] = target_mean - (rotation @ source_mean[..., None])[..., 0]
    return se3.from_rotation_translation(rotation, translation).to(weight.dtype)


def hand_eye[M: IntVar](
    robot_motion: Tensor[[M, *Mat4x4]], camera_motion: Tensor[[M, *Mat4x4]], group: Tensor[[M]]
) -> tuple[Tensor[[*Mat4x4]], Tensor[[int]]]:
    """Solve `A_i X = X B_i(s_g)`: the mount `X` and one scale `s_g` per motion group.

    `A_i` are metric motions of the mount link, `B_i` the same motions seen by the camera in
    unknown units (`B_i(s)` scales their translation); motions of group `g` (e.g. one episode,
    one encoding) share the scale `s_g`. Rotation first (Park & Martin: align the rotation axes
    `log R_B → log R_A`), then translation and scales jointly by linear least squares:
    `(R_A − I) t_X − s_g R_X t_B = −t_A`. Needs ≥ 2 motions with non-parallel rotation axes.
    The scales (one per group, `max(group) + 1` of them) convert camera units to metres.
    """

    a, b = robot_motion.double(), camera_motion.double()
    alpha, beta = se3.so3_log(a[:, :3, :3]), se3.so3_log(b[:, :3, :3])
    rotation = closest_rotation(beta.mT @ alpha)

    groups = int(group.max()) + 1
    eye = torch.eye(3, dtype=a.dtype, device=a.device)
    design = a.new_zeros(len(a), 3, 3 + groups)
    design[:, :, :3] = a[:, :3, :3] - eye
    design[torch.arange(len(a)), :, 3 + group] = -(rotation @ b[:, :3, 3:])[..., 0]
    solution = torch.linalg.lstsq(
        rearrange(design, "m i u -> (m i) u"), rearrange(-a[:, :3, 3], "m i -> (m i) 1")
    ).solution[:, 0]
    mount = se3.from_rotation_translation(rotation, solution[:3])
    return mount.to(robot_motion.dtype), solution[3:].to(robot_motion.dtype)


def mean_transform[N: IntVar](transforms: Tensor[[N, *Mat4x4]]) -> Tensor[[*Mat4x4]]:
    """Chordal mean: `R = argmin_R Σ ‖R − R_i‖²_F` (the rotation closest to `Σ R_i`), and the
    mean translation."""

    rotation = closest_rotation(transforms[:, :3, :3].mT.sum(0))
    return se3.from_rotation_translation(rotation, transforms[:, :3, 3].mean(0))


def closest_rotation[Bs: IntTuple](covariance: Tensor[[*Bs, 3, 3]]) -> Tensor[[*Bs, 3, 3]]:
    """`R = V diag(1, 1, det) Uᵀ` for `covariance = U S Vᵀ`: maximises `tr(R · covariance)`."""

    u, _, vh = torch.linalg.svd(covariance)
    sign = torch.linalg.det(vh.mT @ u.mT).sign()
    correction = torch.ones_like(covariance[..., 0])
    correction[..., 2] = sign
    return vh.mT @ (correction[..., None] * u.mT)
