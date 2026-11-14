"""The robot and its cameras with their unknowns: mounts, body offsets and focal lengths.

A camera is rigidly mounted on a body of the kinematic tree (a wrist camera on the hand, an
exterior camera on the world body). Its pose at time t is `world_from_body(q_t) · body_from_camera`.
Each camera's focal length is corrected by a factor `exp(φ)` with a log-normal prior (std
`focal_std`): wide when the initial intrinsics are a guess, tight when the dataset knows them.

Unknowns of the description (declared with priors in it, see `robot.Freedom`) are optimised
as corrections `exp(δ)` with a scope:
- a free camera mount: `body_from_camera = initial · exp(δ)`;
- a free body (a second arm's base, a mobile base): its offset from its parent becomes
  `parent_from_body · exp(δ)`, which moves its whole subtree.
A correction is one value for the whole data (`dataset`), one per episode (`episode`: e.g. a
static camera re-placed between episodes), or one per frame (`frame`: e.g. a driving base). The
prior is Gaussian around the description for dataset and episode scopes; per frame it is a
random walk (each frame near the previous one), the first frame of each episode held at the
description (the episode's world frame is where it starts).
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

    from .robot import Freedom, Robot


class Corrections(nn.Module):
    """`exp(δ)` corrections of several elements, each with its scope (rows per element).

    `row [elements, T]` says which row applies to each element at each timestep.
    """

    def __init__(
        self, freedoms: list[Freedom | None], episode: Tensor[[int]], dtype: torch.dtype
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

    def values(self) -> Tensor[[int, 6]]:
        """δ per row (zero where not free, or held)."""

        return torch.where(self.active[:, None], self.delta, 0.0)

    def prior(self, element_from_reference: Tensor[[int, 4, 4]]) -> Tensor[[]]:
        """Gaussian cost of each row's deviation: from the reference (the description) for
        dataset and episode scopes; for frame scope, of its step from the previous frame
        against the starting trajectory's step.

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
        cost = (deviation[:, :3] ** 2).sum(-1) / self.std[:, 0] ** 2
        cost = cost + (deviation[:, 3:] ** 2).sum(-1) / self.std[:, 1] ** 2
        return torch.where(self.active, cost, 0.0).sum()


class Rig(nn.Module):
    """Cameras `names` on `robot`, joint trajectory `qpos [T, Q]` (all episodes concatenated,
    `episode [T]` saying which episode each timestep belongs to)."""

    def __init__(
        self,
        robot: Robot,
        qpos: Tensor[[int, int]],
        names: tuple[str, ...],
        initial_mounts: Tensor[[int, 4, 4]],
        initial_intrinsics: Tensor[[int, 3, 3]],
        focal_std: float | Tensor[[int]],
        pixel_unit: Tensor[[int]] | None = None,
        episode: Tensor[[int]] | None = None,
        start_at: Tensor[[int, int, 4, 4]] | None = None,
        bodies_at: Tensor[[int, int, 4, 4]] | None = None,
    ) -> None:
        """`start_at [C, T]`, when given: each camera's starting mount at each timestep (for
        episode-scope cameras placed differently in each episode). `bodies_at [B, T]`: each
        free body's starting correction at each timestep (free bodies in body order), e.g. a
        mobile base's odometry since the episode's first frame."""

        super().__init__()
        index = [robot.camera(name) for name in names]
        self.names = names
        self.camera_body = [robot.camera_body[i] for i in index]
        device = qpos.device
        if episode is None:
            episode = torch.zeros(len(qpos), dtype=torch.long, device=device)
        _, episode = torch.unique(episode, return_inverse=True)

        # Joint values are data: forward kinematics is computed once
        self.register_buffer("kinematics", forward_kinematics(robot, qpos))
        self.register_buffer("camera_body_index", torch.tensor(self.camera_body, device=device))
        self.register_buffer("initial_mount", initial_mounts)
        self.register_buffer("default_mount", robot.body_from_camera[index].to(qpos))
        self.cameras = Corrections([robot.free_cameras.get(n) for n in names], episode, qpos.dtype)
        if start_at is not None:
            # Rows start at the given mounts (relative to the camera's initial mount)
            with torch.no_grad():
                for c in range(len(names)):
                    for row in torch.unique(self.cameras.row[c]).tolist():
                        time = int((self.cameras.row[c] == row).nonzero()[0, 0])
                        relative = se3.inverse(initial_mounts[c]) @ start_at[c, time]
                        self.cameras.delta[row] = se3.se3_log(relative)

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
        for i, body in enumerate(free):
            coupled = [
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

    def intrinsics(self) -> Tensor[[int, 3, 3]]:
        """Per camera: initial intrinsics with corrected focal lengths (square pixels kept)."""

        factor = torch.ones_like(self.initial_intrinsics)
        log_focal = torch.where(self.focal_std > 0, self.log_focal, 0.0)
        factor[:, [0, 1], [0, 1]] = log_focal.exp()[:, None]
        return self.initial_intrinsics * factor

    def world_from_body(self) -> Tensor[[int, int, 4, 4]]:
        """`world_from_body [T, L]`: the kinematics with the free bodies' corrections.

        A free body b with parent p: `W_b' = W_p · P_b · E · J_b = C · W_b` with
        `C = W_p · P_b · E · P_b⁻¹ · W_p⁻¹`, applied to b's whole subtree.
        """

        world = self.kinematics
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

    def mounts(self) -> Tensor[[int, 4, 4]]:
        """`body_from_camera` per camera (its value at the first timestep)."""

        camera = torch.arange(len(self.names), device=self.kinematics.device)
        return self.mounts_at(camera, torch.zeros_like(camera))

    def mounts_at[V: IntVar](self, camera: Tensor[[V]], time: Tensor[[V]]) -> Tensor[[V, 4, 4]]:
        """`body_from_camera` of each view (with its scope's correction)."""

        delta = self.cameras.values()[self.cameras.row[camera, time]]
        return self.initial_mount[camera] @ se3.se3_exp(delta)

    def camera_poses[V: IntVar](
        self,
        camera: Tensor[[V]],
        time: Tensor[[V]],
        world_from_body: Tensor[[int, int, 4, 4]] | None = None,
    ) -> Tensor[[V, 4, 4]]:
        """`world_from_camera` of each view, from kinematics and the mounts."""

        world = self.world_from_body() if world_from_body is None else world_from_body
        return world[time, self.camera_body_index[camera]] @ self.mounts_at(camera, time)

    def moving_cameras(self) -> Tensor[[int]]:
        """Which cameras sit on a body that moves during the episode (joints only)."""

        body = self.kinematics[:, self.camera_body]
        return (body - body[:1]).abs().flatten(-2).amax(-1).amax(0) > MOTION_THRESHOLD

    def prior(self) -> Tensor[[]]:
        """Priors of the free camera mounts (around the description), free bodies and focals."""

        owner = self.cameras.owner
        mounts = self.initial_mount[owner] @ se3.se3_exp(self.cameras.values())
        cameras = self.cameras.prior(se3.inverse(self.default_mount[owner]) @ mounts)
        bodies = self.bodies.prior(se3.se3_exp(self.bodies.values()))
        free = self.focal_std > 0
        focal = (self.log_focal[free] ** 2 / self.focal_std[free] ** 2).sum()
        return cameras + bodies + focal
