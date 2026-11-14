"""The robot and its cameras with their unknowns: mounts, body offsets and focal lengths.

A camera is rigidly mounted on a body of the kinematic tree (a wrist camera on the hand, an
exterior camera on the world body). Its pose at time t is `world_from_body(q_t) · body_from_camera`.
Each camera's focal length is corrected by a factor `exp(φ)` with a log-normal prior (std
`focal_std`, residual `φ / σ_f`): wide when the initial intrinsics are a guess, tight when the
dataset knows them.
Distortion coefficients (`lens.Lens`) are corrected additively, with a Gaussian prior per
coefficient (`Lens.std`, 0: held).

Unknowns of the description (declared with priors in it, see `robot.Freedom`) are optimised
as corrections `exp(δ)` with a scope:
- a free camera mount: `body_from_camera = initial · exp(δ)`;
- a free body (a second arm's base, a mobile base): its offset from its parent becomes
  `parent_from_body · exp(δ)`, which moves its whole subtree.
A correction is one value for the whole data (`dataset`), one per episode (`episode`: e.g. a
static camera re-placed between episodes), or one per frame (`frame`: e.g. a driving base). The
prior is Gaussian around the description for dataset and episode scopes; per frame it is a
random walk (each frame near the previous one), the first frame of each episode held at the
description (the episode's world frame is where it starts). Each prior is a residual vector
whose squares sum to the negative log-prior, `δ = (v, ω)` scaled to `(v / σ_t, ω / σ_r)`:

    dataset, episode:  δ = log(E_ref⁻¹ E)
    frame:             δ = log((S_{f−1}⁻¹ S_f)⁻¹ · E_{f−1}⁻¹ E_f)   (S: starting trajectory)

The kinematics themselves may be refined (`KinematicFreedom`, dataset scope): an offset per
joint (encoder zeros: `q + o`) and a time offset τ between video and joint states (the joint
values of frame t read at t + τ, linearly interpolated within the episode):

    q(t + τ) = (1 − α) q_i + α q_{i+1},   i = ⌊t + τ⌋,  α = t + τ − i   (clamped to the episode)

with priors `o / σ_o` (rad or m) and `τ / σ_τ` (timesteps). Forward kinematics is then
recomputed at each evaluation; otherwise it is computed once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from torch import nn

from r4r.geometry import se3

from .kinematics import forward_kinematics

# A body "moves" if its pose changes by more than this over the episode (m or rad)
MOTION_THRESHOLD = 1e-3


@dataclass(frozen=True, slots=True)
class KinematicFreedom:
    """Dataset-scope refinements of the kinematics, as prior stds (0: held)."""

    joint_std: tuple[float, ...] = ()  # per joint of the robot (rad for hinges, m for slides)
    time_std: float = 0.0  # timesteps


if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.types import K1K2K3K4, Mat3x3, Mat4x4, Scalar, VxVyVzWxWyWz

    from .lens import Lens
    from .robot import Freedom, Robot


class Corrections[T: IntVar](nn.Module):
    """`exp(δ)` corrections of several elements, each with its scope (rows per element).

    `row [elements, T]` says which row applies to each element at each timestep. Row counts
    depend on the scopes and the episodes: they stay `int`.
    """

    row: Tensor[[int, T]]
    owner: Tensor[[int]]  # per row: its element
    std: Tensor[[int, 2]]  # per row: (translation std m, rotation std rad)
    previous: Tensor[[int]]  # per row: the walk's previous row, −1 where none
    active: Tensor[[int]]  # per row: free and not held
    start: Tensor[[int, *Mat4x4]]  # per row: its starting value

    def __init__(
        self, freedoms: list[Freedom | None], episode: Tensor[[T]], dtype: torch.dtype
    ) -> None:
        super().__init__()
        steps, device = len(episode), episode.device
        groups = {
            "dataset": torch.zeros(steps, dtype=torch.long, device=device),
            "episode": episode,
            "frame": torch.arange(steps, device=device),
        }
        rows, owner, stds, free, previous, held, count = [], [], [], [], [], [], 0
        for element, freedom in enumerate(freedoms):
            scope = freedom.scope if freedom is not None else "dataset"
            group = groups[scope]
            size = int(group.max()) + 1
            rows.append(group + count)
            owner.append(torch.full((size,), element, device=device))
            std = (freedom.translation_std, freedom.rotation_std) if freedom else (math.inf,) * 2
            stds.append(torch.tensor(std, dtype=dtype, device=device).expand(size, 2))
            free.append(torch.full((size,), freedom is not None, device=device))
            start = torch.zeros(size, dtype=torch.bool, device=device)
            back = torch.full((size,), -1, device=device)
            if scope == "frame":
                # Random walk from each episode's first frame, held at the description
                start[0] = True
                start[1:] = episode[1:] != episode[:-1]
                back = torch.where(start, -1, torch.arange(size, device=device) + count - 1)
            previous.append(back)
            held.append(start)
            count += size
        none = torch.zeros(0, dtype=torch.long, device=device)
        self.register_buffer("row", torch.stack(rows) if rows else none.reshape(0, steps))
        self.register_buffer("owner", torch.cat(owner) if owner else none)
        self.register_buffer(
            "std", torch.cat(stds) if stds else torch.zeros(0, 2, dtype=dtype, device=device)
        )
        self.register_buffer("previous", torch.cat(previous) if previous else none)
        active = torch.cat(free) & ~torch.cat(held) if free else none.bool()
        self.register_buffer("active", active)
        self.delta = nn.Parameter(torch.zeros(count, 6, dtype=dtype, device=device))
        # Each row's starting value: a frame-scope walk is costed by how far each step departs
        # from the starting trajectory's step (e.g. odometry), not from standing still
        eye = torch.eye(4, dtype=dtype, device=device)
        self.register_buffer("start", eye.repeat(count, 1, 1))
        self.free_elements = [f is not None for f in freedoms]

    def values(self) -> Tensor[[int, *VxVyVzWxWyWz]]:
        """δ per row (zero where not free, or held)."""

        return torch.where(self.active[:, None], self.delta, 0.0)

    def prior(self, element_from_reference: Tensor[[int, *Mat4x4]]) -> Scalar:
        """Gaussian cost of each row's deviation: `Σ ‖δ‖²` over `prior_residuals`."""

        return (self.prior_residuals(element_from_reference) ** 2).sum()

    def prior_residuals(
        self, element_from_reference: Tensor[[int, *Mat4x4]]
    ) -> Tensor[[int, *VxVyVzWxWyWz]]:
        """Each row's deviation over its std: from the reference (the description) for dataset
        and episode scopes; for frame scope, of its step from the previous frame against the
        starting trajectory's step. Zero for rows that are not free.

        `element_from_reference [rows, 4, 4]`: each row's corrected pose relative to its
        reference.
        """

        pose = element_from_reference
        walk = self.previous >= 0
        previous = self.previous.clamp_min(0)
        step = se3.inverse(pose[previous]) @ pose
        expected = se3.inverse(self.start[previous]) @ self.start
        walked = se3.inverse(expected) @ step
        relative = torch.where(walk[:, None, None], walked, pose)
        deviation = se3.se3_log(relative)
        scaled = torch.cat(
            (deviation[:, :3] / self.std[:, :1], deviation[:, 3:] / self.std[:, 1:]), -1
        )
        return torch.where(self.active[:, None], scaled, 0.0)


class Rig[T: IntVar, L: IntVar, C: IntVar](nn.Module):
    """Cameras `names` on `robot`, joint trajectory `qpos [T, Q]` (all episodes concatenated,
    `episode [T]` saying which episode each timestep belongs to)."""

    kinematics: Tensor[[T, L, *Mat4x4]]  # forward kinematics of `qpos` (no refinements)
    qpos: Tensor[[T, int]]
    camera_body_index: Tensor[[C]]
    initial_mount: Tensor[[C, *Mat4x4]]
    default_mount: Tensor[[C, *Mat4x4]]  # the description's
    reference_mount: Tensor[[int, *Mat4x4]]  # per camera row: its prior centre
    subtree: Tensor[[int, L]]  # per free body: the bodies it moves
    free_offset: Tensor[[int, *Mat4x4]]
    initial_intrinsics: Tensor[[C, *Mat3x3]]
    pixel_unit: Tensor[[C]]
    focal_std: Tensor[[C]]
    joint_std: Tensor[[int]]
    time_std: Tensor[[1]]
    joint_index: Tensor[[int]]
    episode_first: Tensor[[T]]
    episode_last: Tensor[[T]]
    distortion_std: Tensor[[C, *K1K2K3K4]]
    cameras: Corrections[T]
    bodies: Corrections[T]
    robot: Robot[L, int, int] | None  # kept only when forward kinematics is re-evaluated
    initial_lens: Lens[C] | None

    def __init__(
        self,
        robot: Robot[L, int, int],
        qpos: Tensor[[T, int]],
        names: tuple[str, ...],
        initial_mounts: Tensor[[C, *Mat4x4]],
        initial_intrinsics: Tensor[[C, *Mat3x3]],
        focal_std: float | Tensor[[C]],
        pixel_unit: Tensor[[C]] | None = None,
        episode: Tensor[[T]] | None = None,
        start_at: Tensor[[C, T, *Mat4x4]] | None = None,
        bodies_at: Tensor[[int, T, *Mat4x4]] | None = None,
        kinematic: KinematicFreedom | None = None,
        lens: Lens[C] | None = None,
    ) -> None:
        """`start_at [C, T]`, when given: each camera's starting mount at each timestep (for
        episode-scope cameras placed differently in each episode). `bodies_at [B, T]`: each
        free body's starting correction at each timestep (free bodies in body order), e.g. a
        mobile base's odometry since the episode's first frame. `lens`: the cameras'
        distortion (None: ideal pinholes)."""

        super().__init__()
        index = [robot.camera(name) for name in names]
        self.names = names
        self.camera_body = [robot.camera_body[i] for i in index]
        device = qpos.device
        if episode is None:
            episode = torch.zeros(len(qpos), dtype=torch.long, device=device)
        _, episode = torch.unique(episode, return_inverse=True)

        # Joint values are data: forward kinematics is computed once (unless refined)
        self.register_buffer("kinematics", forward_kinematics(robot, qpos))
        self._init_kinematic(robot, qpos, episode, kinematic)
        self.register_buffer("camera_body_index", torch.tensor(self.camera_body, device=device))
        self.register_buffer("initial_mount", initial_mounts)
        self.register_buffer("default_mount", robot.body_from_camera[index].to(qpos))
        self.cameras = Corrections([robot.free_cameras.get(n) for n in names], episode, qpos.dtype)
        self._init_camera_starts(robot, names, initial_mounts, start_at)

        # Free bodies, root first: each correction moves its subtree
        free = sorted(robot.body(n) for n in robot.free_bodies)
        self.free_parent = [robot.body_parent[b] for b in free]
        subtree = torch.zeros(len(free), len(robot.body_names), dtype=torch.bool)
        for i, body in enumerate(free):
            for b in range(len(robot.body_names)):
                node = b
                while node > 0 and node != body:
                    node = robot.body_parent[node]
                subtree[i, b] = node == body
        # A free mount and a free body above it on the same chain trade off against each other
        # (a base offset is absorbed by the mount): such unknowns are solved in stages
        # A frame-scope body (a driving base) is held at each episode's first frame: it does
        # not trade off against a mount there
        scopes = [robot.free_bodies[robot.body_names[b]].scope for b in free]
        for i, body in enumerate(free):
            coupled = scopes[i] != "frame" and [
                n
                for n, b, f in zip(names, self.camera_body, self.cameras.free_elements, strict=True)
                if f and bool(subtree[i, b])
            ]
            if coupled:
                raise ValueError(
                    f"free body {robot.body_names[body]!r} is above free camera mounts {coupled}:"
                    " solve the mounts first (each camera alone), freeze them, then the body"
                )
        self.register_buffer("subtree", subtree.to(device))
        self.register_buffer("free_offset", robot.parent_from_body[free].to(qpos))
        freedoms = [robot.free_bodies[robot.body_names[b]] for b in free]
        self.bodies = Corrections(list(freedoms), episode, qpos.dtype)
        if bodies_at is not None:
            with torch.no_grad():
                for b in range(len(free)):
                    rows = self.bodies.row[b]
                    first = torch.zeros(len(self.bodies.delta), dtype=torch.long, device=device)
                    # The first timestep of each row (scatter of the smallest time per row)
                    time = torch.arange(len(rows), device=device)
                    first.scatter_reduce_(0, rows, time, reduce="amin", include_self=False)
                    used = torch.unique(rows)
                    self.bodies.delta[used] = se3.se3_log(bodies_at[b, first[used]])
                    self.bodies.start[used] = bodies_at[b, first[used]]

        self.register_buffer("initial_intrinsics", initial_intrinsics.to(qpos))
        # Pixels per unit of reprojection error, a constant per camera (default: initial focal)
        unit = initial_intrinsics[:, 0, 0] if pixel_unit is None else pixel_unit
        self.register_buffer("pixel_unit", unit.to(qpos))
        self.log_focal = nn.Parameter(qpos.new_zeros(len(names)))
        # Per camera; 0: the focal is known and held
        std = torch.as_tensor(focal_std, dtype=qpos.dtype, device=device).expand(len(names))
        self.register_buffer("focal_std", std.clone())
        self._init_lens(lens, qpos, len(names))

    def _init_lens(self, lens: Lens[C] | None, qpos: Tensor[[T, int]], cameras: int) -> None:
        if lens is None:
            self.initial_lens = self.distortion = None
            return
        self.initial_lens = initial = lens.to(qpos)
        self.distortion = nn.Parameter(qpos.new_zeros(cameras, lens.std.shape[1]))
        self.register_buffer("distortion_std", initial.std.clone())

    def _init_kinematic(
        self,
        robot: Robot[L, int, int],
        qpos: Tensor[[T, int]],
        episode: Tensor[[T]],
        kinematic: KinematicFreedom | None,
    ) -> None:
        joints = len(robot.joint_qpos)
        free = kinematic or KinematicFreedom()
        std = free.joint_std or (0.0,) * joints
        assert len(std) == joints, f"joint_std: {len(std)} values for {joints} joints"
        self.register_buffer("joint_std", torch.tensor(std, dtype=qpos.dtype, device=qpos.device))
        self.register_buffer(
            "time_std", torch.tensor([free.time_std], dtype=qpos.dtype, device=qpos.device)
        )
        self.joint_offset = nn.Parameter(qpos.new_zeros(joints))
        self.time_offset = nn.Parameter(qpos.new_zeros(1))
        self.refines = bool((self.joint_std > 0).any() or (self.time_std > 0).any())
        self.robot = robot if self.refines else None
        self.register_buffer("qpos", qpos)
        self.register_buffer(
            "joint_index", torch.tensor(robot.joint_qpos, dtype=torch.long, device=qpos.device)
        )
        # Each timestep's episode bounds (interpolation stays within its episode)
        steps = torch.arange(len(qpos), device=qpos.device)
        first = torch.zeros(int(episode.max()) + 1, dtype=torch.long, device=qpos.device)
        first = first.scatter_reduce(0, episode, steps, "amin", include_self=False)
        last = torch.zeros_like(first).scatter_reduce(0, episode, steps, "amax", include_self=False)
        self.register_buffer("episode_first", first[episode])
        self.register_buffer("episode_last", last[episode])

    def joint_kinematics(self) -> Tensor[[T, L, *Mat4x4]]:
        """`world_from_body [T, L]` from the joints (with the kinematic refinements)."""

        if self.robot is None:  # nothing refined
            return self.kinematics
        time = torch.where(self.time_std > 0, self.time_offset, 0.0)
        at = torch.arange(len(self.qpos), device=self.qpos.device, dtype=self.qpos.dtype) + time
        at = torch.minimum(torch.maximum(at, self.episode_first.to(at)), self.episode_last.to(at))
        below = at.floor().long()
        above = torch.minimum(below + 1, self.episode_last)
        fraction = (at - below.to(at))[:, None]
        qpos = self.qpos[below] * (1 - fraction) + self.qpos[above] * fraction
        offset = torch.where(self.joint_std > 0, self.joint_offset, 0.0)
        qpos = qpos.index_add(1, self.joint_index, offset[None].expand(len(qpos), -1))
        return forward_kinematics(self.robot, qpos)

    def intrinsics(self) -> Tensor[[C, *Mat3x3]]:
        """Per camera: initial intrinsics with corrected focal lengths `f = f₀ · exp(φ)` (square
        pixels kept)."""

        factor = torch.ones_like(self.initial_intrinsics)
        log_focal = torch.where(self.focal_std > 0, self.log_focal, 0.0)
        factor[:, [0, 1], [0, 1]] = log_focal.exp()[:, None]
        return self.initial_intrinsics * factor

    def lens(self) -> Lens[C] | None:
        """Per camera: initial distortion with its corrections (held coefficients kept)."""

        if self.initial_lens is None or self.distortion is None:
            return None
        lens = self.initial_lens
        correction = torch.where(lens.std > 0, self.distortion, 0.0)
        return replace(lens, distortion=lens.distortion + correction)

    def world_from_body(self) -> Tensor[[T, L, *Mat4x4]]:
        """`world_from_body [T, L]`: the kinematics with the free bodies' corrections.

        A free body b with parent p: `W_b' = W_p · P_b · E · J_b = C · W_b` with
        `C = W_p · P_b · E · P_b⁻¹ · W_p⁻¹`, applied to b's whole subtree.
        """

        world = self.joint_kinematics()
        if not self.free_parent:
            return world
        correction = se3.se3_exp(self.bodies.values())  # [rows, 4, 4]
        for i, parent in enumerate(self.free_parent):
            offset = self.free_offset[i]
            local = offset @ correction[self.bodies.row[i]] @ se3.inverse(offset)  # [T, 4, 4]
            change = world[:, parent] @ local @ se3.inverse(world[:, parent])
            moved = change[:, None] @ world
            world = torch.where(self.subtree[i][None, :, None, None], moved, world)
        return world

    def mounts(self) -> Tensor[[C, *Mat4x4]]:
        """`body_from_camera` per camera (its value at the first timestep)."""

        camera = torch.arange(len(self.names), device=self.kinematics.device)
        return self.mounts_at(camera, torch.zeros_like(camera))

    def mounts_at[V: IntVar](self, camera: Tensor[[V]], time: Tensor[[V]]) -> Tensor[[V, *Mat4x4]]:
        """`body_from_camera` of each view: `initial · exp(δ)`, δ of its scope's row."""

        delta = self.cameras.values()[self.cameras.row[camera, time]]
        return self.initial_mount[camera] @ se3.se3_exp(delta)

    def camera_poses[V: IntVar](
        self,
        camera: Tensor[[V]],
        time: Tensor[[V]],
        world_from_body: Tensor[[T, L, *Mat4x4]] | None = None,
    ) -> Tensor[[V, *Mat4x4]]:
        """`world_from_camera` of each view: `world_from_body(q_t) · body_from_camera`."""

        world = self.world_from_body() if world_from_body is None else world_from_body
        return world[time, self.camera_body_index[camera]] @ self.mounts_at(camera, time)

    def _init_camera_starts(
        self,
        robot: Robot[L, int, int],
        names: tuple[str, ...],
        initial_mounts: Tensor[[C, *Mat4x4]],
        start_at: Tensor[[C, T, *Mat4x4]] | None,
    ) -> None:
        """Each camera row's start and prior centre: the description's mount; for
        episode-scope cameras given per-episode starts, that episode's start (a recorded or
        estimated placement), so ranges bound the departure from it, not from the description."""

        self.register_buffer("reference_mount", self.default_mount[self.cameras.owner].clone())
        if start_at is None:
            return
        # Rows start at the given mounts (relative to the camera's initial mount)
        with torch.no_grad():
            for c, name in enumerate(names):
                freedom = robot.free_cameras.get(name)
                for row in torch.unique(self.cameras.row[c]).tolist():
                    time = int((self.cameras.row[c] == row).nonzero()[0, 0])
                    relative = se3.inverse(initial_mounts[c]) @ start_at[c, time]
                    self.cameras.delta[row] = se3.se3_log(relative)
                    if freedom is not None and freedom.scope == "episode":
                        self.reference_mount[row] = start_at[c, time]

    def prior(self) -> Scalar:
        """Negative log-prior `Σ e²` of the free mounts, bodies, focals, kinematic offsets and
        distortion (`prior_residuals`)."""

        return (self.prior_residuals() ** 2).sum()

    def prior_residuals(self) -> Tensor[[int]]:
        """The priors as residuals (their squares sum to `prior`): camera rows, body rows,
        focals, joint offsets, the time offset, then distortion (zero where held)."""

        owner = self.cameras.owner
        mounts = self.initial_mount[owner] @ se3.se3_exp(self.cameras.values())
        cameras = self.cameras.prior_residuals(se3.inverse(self.reference_mount) @ mounts)
        bodies = self.bodies.prior_residuals(se3.se3_exp(self.bodies.values()))
        free = self.focal_std > 0
        focal = torch.where(free, self.log_focal / self.focal_std.clamp_min(1e-12), 0.0)
        joints = torch.where(
            self.joint_std > 0, self.joint_offset / self.joint_std.clamp_min(1e-12), 0.0
        )
        time = torch.where(
            self.time_std > 0, self.time_offset / self.time_std.clamp_min(1e-12), 0.0
        )
        parts = [cameras.flatten(), bodies.flatten(), focal, joints, time]
        if self.distortion is not None:
            std = self.distortion_std
            parts.append(
                torch.where(std > 0, self.distortion / std.clamp_min(1e-12), 0.0).flatten()
            )
        return torch.cat(parts)
