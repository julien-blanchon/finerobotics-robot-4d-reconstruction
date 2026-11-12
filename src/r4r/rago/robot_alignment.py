"""The robot as a calibration target: its FK-posed surface must lie on ARROW's geometry.

For every view of a static camera, the robot's surface points (placed in the world by forward
kinematics, metric) are brought into the camera with the view's pose. Each visible robot point
is matched to the nearest point of ARROW's metric point cloud of that view (back-projected depth,
scaled), and their distance is penalised: robust ICP, with matches recomputed every step. This
fixes static cameras and the metric scale without hand-eye and without relying on how well
ARROW localises a moving wrist camera.

Visibility: in front of the camera, inside the image, and the front-most robot point of its
depth-map cell (self-occlusion). Occlusion by other things leaves robot points without a near
ARROW point; the Cauchy loss bounds their influence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from einops import rearrange
from torch import nn

from r4r.geometry import camera, se3

from .losses import cauchy

if TYPE_CHECKING:
    from torch import Tensor

    from .meshes import Surface
    from .observations import Observations
    from .rig import Rig

# A robot point counts as visible if within this (m) of the front-most robot point on its cell
VISIBILITY_SLACK = 0.02


class RobotAlignment(nn.Module):
    def __init__(self, rig: Rig, surface: Surface, observations: Observations) -> None:
        super().__init__()
        obs = observations
        static = ~rig.moving_cameras()[obs.view_camera]
        views = static.nonzero()[:, 0]
        world_from_body = rig.world_from_body[obs.view_time[views]][:, surface.body]
        points = se3.transform_points(world_from_body, surface.points[None, :, None])[..., 0, :]
        self.register_buffer("views", views)
        self.register_buffer("world_points", points)  # (views, P, 3)

        # Native pixel of every depth-map cell, to back-project ARROW's clouds
        rows, columns = obs.depth_map.shape[1:]
        stride = obs.depth_stride
        y, x = torch.meshgrid(
            torch.arange(rows, device=views.device) * stride,
            torch.arange(columns, device=views.device) * stride,
            indexing="ij",
        )
        self.register_buffer("cell_pixels", torch.stack((x, y), -1).reshape(-1, 2).to(points))

    def forward(
        self,
        world_from_view: Tensor[[int, 4, 4]],
        scales: Tensor[[int]],
        intrinsics: Tensor[[int, 3, 3]],
        observations: Observations,
        cauchy_scale: float,
    ) -> Tensor[[]]:
        """Mean Cauchy cost of robot-vs-ARROW depth over visible robot points (0 if none)."""

        obs = observations
        if len(self.views) == 0:
            return world_from_view.new_zeros(())
        views = self.views
        local = se3.transform_points(se3.inverse(world_from_view[views]), self.world_points)
        k = intrinsics[obs.view_camera[views]]
        pixel = camera.project(k, local)
        depth_map = obs.depth_map[views]
        _, rows, columns = depth_map.shape
        stride = obs.depth_stride

        # Inside the lattice and in front of the camera
        lattice = pixel / stride
        inside = (lattice[..., 0] >= 0) & (lattice[..., 0] <= columns - 1)
        inside &= (lattice[..., 1] >= 0) & (lattice[..., 1] <= rows - 1) & (local[..., 2] > 0.05)

        # Self-occlusion: keep points near the front-most robot point of their cell
        with torch.no_grad():
            cell = lattice.round().long()
            index = cell[..., 1].clamp(0, rows - 1) * columns + cell[..., 0].clamp(0, columns - 1)
            z = torch.where(inside, local[..., 2], torch.inf)
            front = z.new_full((len(views), rows * columns), torch.inf)
            front = front.scatter_reduce(1, index, z, "amin")
            visible = inside & (z <= front.gather(1, index) + VISIBILITY_SLACK)

        # ARROW's metric cloud per view, in the camera frame
        metric = rearrange(depth_map, "v h w -> v (h w)") * scales[obs.view_episode[views]][:, None]
        cloud = camera.backproject(k, self.cell_pixels.expand(len(views), -1, -1), metric)

        # Nearest ARROW point for each robot point (matches fixed within a step, as in ICP)
        with torch.no_grad():
            nearest = torch.cdist(local.float(), cloud.float()).argmin(-1)
        target = cloud.gather(1, nearest[..., None].expand(-1, -1, 3))
        cost = cauchy((local - target).norm(dim=-1), cauchy_scale)
        return (cost * visible).sum() / visible.sum().clamp_min(1)
