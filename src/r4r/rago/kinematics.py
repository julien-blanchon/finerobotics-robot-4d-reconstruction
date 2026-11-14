"""Forward kinematics in torch, batched over time and differentiable in the body offsets.

Matches MuJoCo's `mj_kinematics`: a body's pose is its parent's pose, then its fixed offset,
then its joints in order. A hinge rotates about an axis through its anchor; a slide translates
along its axis. Joint values are measured from the model's reference configuration (`qpos0`):

    W_b(q) = W_parent(b) · P_b · Π_j M_j(q_j − q0_j)
    hinge: M = [R, p − R p; 0, 1], R = exp([a]× θ)       slide: M = [I, a d; 0, 1]

with `P_b` the body's offset (`parent_from_body`), `a` the unit axis and `p` the anchor in
the body frame; θ in radians, d and every translation in metres.

Gradients flow into `parent_from_body`, which is how RAGO optimises free body offsets
(e.g. the second arm's base). Joint values are data and are never optimised.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from r4r.geometry import se3

from .robot import HINGE, Robot

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.types import Mat4x4


def forward_kinematics[T: IntVar, L: IntVar, Q: IntVar](
    robot: Robot[L, int, int],
    qpos: Tensor[[T, Q]],
    parent_from_body: Tensor[[L, *Mat4x4]] | None = None,
) -> Tensor[[T, L, *Mat4x4]]:
    """`world_from_body` for every body at every timestep."""

    offsets = robot.parent_from_body if parent_from_body is None else parent_from_body
    qpos = apply_couplings(robot, qpos)

    # Joint motions, all at once: [T, J, 4, 4]
    value = qpos[:, list(robot.joint_qpos)] - robot.joint_reference
    is_hinge = torch.tensor([k == HINGE for k in robot.joint_kind], device=qpos.device)[:, None]
    tangent = torch.cat(
        (
            torch.where(is_hinge, 0.0, robot.joint_axis * value[..., None]),
            torch.where(is_hinge, robot.joint_axis * value[..., None], 0.0),
        ),
        -1,
    )
    about_origin = se3.se3_exp(tangent)

    # A hinge rotates about its anchor p, not the body origin: [R, t + p − R p]
    rotation = about_origin[..., :3, :3]
    anchor = robot.joint_anchor
    shift = anchor - (rotation @ anchor[..., None])[..., 0]
    motion = se3.from_rotation_translation(rotation, about_origin[..., :3, 3] + shift)

    # Each body's local transform: fixed offset, then its joints in order
    local = list(offsets.expand(qpos.shape[0], -1, -1, -1).unbind(1))
    for joint, body in enumerate(robot.joint_body):
        local[body] = local[body] @ motion[:, joint]

    # Walk the tree root-first (MuJoCo order guarantees parents come first)
    world = [local[0]]
    for body in range(1, len(local)):
        world.append(world[robot.body_parent[body]] @ local[body])
    return torch.stack(world, 1)


def apply_couplings[T: IntVar, Q: IntVar](
    robot: Robot[int, int, int], qpos: Tensor[[T, Q]]
) -> Tensor[[T, Q]]:
    """Fill joints driven by an equality (URDF mimic) from their source joint:
    `q_target = c0 + c1 · q_source`."""

    if not robot.coupling:
        return qpos
    qpos = qpos.clone()
    for target, source, offset, gain in robot.coupling:
        qpos[:, target] = offset + gain * qpos[:, source]
    return qpos
