"""Inverse kinematics: joint trajectories from recorded end-effector poses.

Datasets that record only the end effector's pose (Open X-Embodiment conversions, BridgeData,
simulated palm poses) have no joint state for forward kinematics. Per frame, damped least
squares on the geometric Jacobian moves the chain's joints until the body's frame meets the
target; each frame starts from the previous frame's solution, so the trajectory stays on one
branch, and the redundancy (a 7-DoF arm, a free tool axis) is pulled toward a rest pose in the
Jacobian's null space, so it moves smoothly and never drifts:

    e = (p* − p, w · log(R* Rᵀ)),   J⁺ = Jᵀ (J Jᵀ + λ² I)⁻¹
    Δq = J⁺ e + (I − J⁺ J) · k (q_rest − q),   q ← clamp(q + Δq, limits)

`p` in metres, rotations in radians, `w` (m/rad) weighs rotation against position. A tool axis
`a` about which the pose is free (a cylindrical pusher, a yaw nobody recorded) drops that
component: the rotation error is `a × a*` (the turn of the tool's axis onto the target's) and
the Jacobian's rotation rows are projected by `I − a aᵀ`.

Episodes are solved together (a batch), frame after frame; frames past an episode's end are
masked.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from r4r.geometry import se3

from .kinematics import forward_kinematics
from .robot import HINGE

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.types import XYZ, Mat4x4

    from .robot import Robot


@dataclass(frozen=True, slots=True, kw_only=True)
class IkConfig:
    """Solver settings: rotation weight (m/rad), damping, null-space gain, iterations for an
    episode's first frame (from the rest pose) and for every following frame (warm)."""

    rotation_weight: float = 0.2
    damping: float = 1e-3
    null_gain: float = 0.05
    first_iterations: int = 100
    iterations: int = 12
    # Final iterations of a frame without the null-space pull
    polish: int = 4
    # Largest change of one joint in one step (rad or m): long first moves stay stable
    max_step: float = 0.2
    # Weighted error (m) under which a frame needs no further iterations
    tolerance: float = 1e-5
    # At most this many extra iterations; a frame stops once its error shrinks by less than
    # `stall` (relative) in one
    extra_iterations: int = 30
    stall: float = 0.01


@dataclass(frozen=True, slots=True)
class Chain[J: IntVar]:
    """What one arm's IK moves: the description's joints (qpos addresses), their limits and
    rest values, the body whose frame meets the targets and the tool frame on it."""

    joints: tuple[str, ...]
    qpos: tuple[int, ...]
    lower: Tensor[[J]]
    upper: Tensor[[J]]
    rest: Tensor[[J]]
    body: int
    body_from_tool: Tensor[[*Mat4x4]]
    # Tool-frame axis about which the target's rotation is free (None: full pose)
    free_axis: Tensor[[*XYZ]] | None = None


@dataclass(frozen=True, slots=True)
class IkSolution[B: IntVar, T: IntVar, J: IntVar]:
    """Joint values per episode and frame, and the remaining error: position (m), rotation
    (rad, the free axis left out). Frames past an episode's end hold its last values."""

    joints: Tensor[[B, T, J]]
    position_error: Tensor[[B, T]]
    rotation_error: Tensor[[B, T]]


def solve_ik[B: IntVar, T: IntVar, J: IntVar](
    robot: Robot[int, int, int],
    chain: Chain[J],
    world_from_target: Tensor[[B, T, *Mat4x4]],
    lengths: Tensor[[B]],
    config: IkConfig,
    qpos: Tensor[[int]] | None = None,
) -> IkSolution[B, T, J]:
    """Joint trajectories of `chain` putting its tool frame on `world_from_target` (frames
    `t ≥ lengths[b]` ignored). Joints off the chain stay at `qpos` (default zeros)."""

    batch, frames = world_from_target.shape[:2]
    base = robot.joint_reference.new_zeros(robot.nq) if qpos is None else qpos
    q = chain.rest.expand(batch, -1).clone()
    out = q.new_zeros(batch, frames, len(chain.joints))
    position_error = q.new_zeros(batch, frames)
    rotation_error = q.new_zeros(batch, frames)
    for t in range(frames):
        active = (t < lengths)[:, None]
        steps = config.first_iterations if t == 0 else config.iterations
        for i in range(steps):
            target = world_from_target[:, t]
            error, jacobian = _error_jacobian(robot, chain, base, q, target, config)
            # The last iterations only meet the target (the pull would hold it back a little)
            gain = config.null_gain if i < steps - config.polish else 0.0
            q = torch.where(active, _step(q, error, jacobian, chain, config, gain), q)
        # Frames still off the target (a long move between frames) get more iterations, each
        # until its error stops shrinking (an unreachable frame must not hold the batch)
        moving = active
        previous = torch.full_like(q[:, :1], torch.inf)
        for _ in range(config.extra_iterations):
            target = world_from_target[:, t]
            error, jacobian = _error_jacobian(robot, chain, base, q, target, config)
            size = error.norm(dim=1, keepdim=True)
            moving = moving & (size > config.tolerance) & (size < (1 - config.stall) * previous)
            if not moving.any():
                break
            previous = size
            q = torch.where(moving, _step(q, error, jacobian, chain, config, 0.0), q)
        error, _ = _error_jacobian(robot, chain, base, q, world_from_target[:, t], config)
        out[:, t] = q
        position_error[:, t] = error[:, :3].norm(dim=1)
        rotation_error[:, t] = error[:, 3:].norm(dim=1) / config.rotation_weight
    return IkSolution(out, position_error, rotation_error)


def tool_poses[T: IntVar, J: IntVar](
    robot: Robot[int, int, int],
    chain: Chain[J],
    joints: Tensor[[T, J]],
    qpos: Tensor[[int]] | None = None,
) -> Tensor[[T, *Mat4x4]]:
    """`world_from_tool` at the chain's joint values (forward kinematics, for checks)."""

    full = _full_qpos(robot, chain, joints, qpos)
    return forward_kinematics(robot, full)[:, chain.body] @ chain.body_from_tool


def branch(robot: Robot[int, int, int], body: str) -> Robot[int, int, int]:
    """The description cut down to `body` and its ancestors (forward kinematics of a long
    tree, e.g. two arms with dexterous hands, costs its bodies; IK needs one branch). Joint
    qpos addresses and `nq` are kept, so a full qpos vector still applies."""

    path = [robot.body(body)]
    while path[-1] != 0:
        path.append(robot.body_parent[path[-1]])
    path.reverse()
    index = {old: new for new, old in enumerate(path)}
    joints = [j for j, b in enumerate(robot.joint_body) if b in index]
    assert not any(target in {robot.joint_qpos[j] for j in joints} for target, *_ in robot.coupling)
    return replace(
        robot,
        body_names=tuple(robot.body_names[b] for b in path),
        body_parent=tuple(index[robot.body_parent[b]] if b else 0 for b in path),
        parent_from_body=robot.parent_from_body[path],
        joint_names=tuple(robot.joint_names[j] for j in joints),
        joint_body=tuple(index[robot.joint_body[j]] for j in joints),
        joint_kind=tuple(robot.joint_kind[j] for j in joints),
        joint_qpos=tuple(robot.joint_qpos[j] for j in joints),
        joint_axis=robot.joint_axis[joints],
        joint_anchor=robot.joint_anchor[joints],
        joint_reference=robot.joint_reference[joints],
        coupling=(),
        camera_names=(),
        camera_body=(),
        body_from_camera=robot.body_from_camera[:0],
        free_cameras={},
        free_bodies={},
    )


def _full_qpos[B: IntVar, J: IntVar](
    robot: Robot[int, int, int],
    chain: Chain[J],
    joints: Tensor[[B, J]],
    qpos: Tensor[[int]] | None,
) -> Tensor[[B, int]]:
    base = robot.joint_reference.new_zeros(robot.nq) if qpos is None else qpos
    full = base.expand(len(joints), -1).clone()
    full[:, list(chain.qpos)] = joints
    return full


def _error_jacobian[B: IntVar, J: IntVar](
    robot: Robot[int, int, int],
    chain: Chain[J],
    qpos: Tensor[[int]],
    q: Tensor[[B, J]],
    world_from_target: Tensor[[B, *Mat4x4]],
    config: IkConfig,
) -> tuple[Tensor[[B, 6]], Tensor[[B, 6, J]]]:
    """The weighted pose error `(p* − p, w · log(R* Rᵀ))` and its geometric Jacobian."""

    world_from_body = forward_kinematics(robot, _full_qpos(robot, chain, q, qpos))
    tool = world_from_body[:, chain.body] @ chain.body_from_tool
    position = tool[:, :3, 3]
    rotation_error = se3.so3_log(world_from_target[:, :3, :3] @ tool[:, :3, :3].transpose(1, 2))
    columns = []
    for name in chain.joints:
        joint = robot.joint_names.index(name)
        pose = world_from_body[:, robot.joint_body[joint]]
        axis = pose[:, :3, :3] @ robot.joint_axis[joint]
        if robot.joint_kind[joint] == HINGE:
            anchor = (pose[:, :3, :3] @ robot.joint_anchor[joint]) + pose[:, :3, 3]
            columns.append(torch.cat((torch.cross(axis, position - anchor, dim=1), axis), 1))
        else:
            columns.append(torch.cat((axis, torch.zeros_like(axis)), 1))
    jacobian = torch.stack(columns, 2)
    weight = config.rotation_weight
    error = torch.cat((world_from_target[:, :3, 3] - position, weight * rotation_error), 1)
    jacobian = torch.cat((jacobian[:, :3], weight * jacobian[:, 3:]), 1)
    if chain.free_axis is not None:
        # Only align the free axis: the error is the rotation `a × a*` turning the tool's
        # axis onto the target's, and rotation about the axis drops out of the Jacobian
        axis = world_from_target[:, :3, :3] @ chain.free_axis
        current = tool[:, :3, :3] @ chain.free_axis
        keep = torch.eye(3, dtype=q.dtype, device=q.device) - axis[:, :, None] * axis[:, None, :]
        turn = weight * torch.cross(current, axis, dim=1)
        error = torch.cat((error[:, :3], turn), 1)
        jacobian = torch.cat((jacobian[:, :3], keep @ jacobian[:, 3:]), 1)
    return error, jacobian


def _step[B: IntVar, J: IntVar](
    q: Tensor[[B, J]],
    error: Tensor[[B, 6]],
    jacobian: Tensor[[B, 6, J]],
    chain: Chain[J],
    config: IkConfig,
    gain: float,
) -> Tensor[[B, J]]:
    """One damped least-squares step with the null-space pull toward the rest pose."""

    eye = torch.eye(6, dtype=q.dtype, device=q.device)
    inverse = jacobian.transpose(1, 2) @ torch.linalg.inv(
        jacobian @ jacobian.transpose(1, 2) + config.damping**2 * eye
    )
    task = (inverse @ error[..., None])[..., 0]
    null = torch.eye(q.shape[1], dtype=q.dtype, device=q.device) - inverse @ jacobian
    pull = (null @ (gain * (chain.rest - q))[..., None])[..., 0]
    step = task + pull
    step = step * (config.max_step / step.abs().amax(1, keepdim=True).clamp(min=config.max_step))
    return torch.minimum(torch.maximum(q + step, chain.lower), chain.upper)
