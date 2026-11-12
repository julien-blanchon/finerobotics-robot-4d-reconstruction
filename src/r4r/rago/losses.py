"""Scene-consistency losses shared by every calibration method.

Each correspondence is one static point seen in two views. Placed in the world through each
view's pose (after scaling ARROW's points to metres), the two copies must coincide (3D term),
and each must reproject onto the other view's pixel (2D term).

Calib3R uses the plain L2 norm. We use a Cauchy loss `ρ(e) = δ² log(1 + e²/δ²)` instead:
quadratic below δ, logarithmic above, so the outliers ARROW inevitably produces (wrong matches,
points behind a camera) have bounded influence. Points behind the target camera are not
reprojected at all.

Units: the 3D term is in metres; the 2D term in pixels divided by the focal length (≈ radians),
so the two are comparable without hand-tuned weights.

Ref: Calib3R arXiv:2509.08813, Eqs. 3–5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from einops import rearrange

from r4r.geometry import camera, se3

# Points closer than this (m) to the target camera's plane are not reprojected
MIN_REPROJECTION_DEPTH = 1e-2

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    from .observations import Observations


def scene_losses[V: IntVar, C: IntVar](
    world_from_view: Tensor[[V, 4, 4]],
    scales: Tensor[[int]],
    intrinsics: Tensor[[C, 3, 3]],
    observations: Observations[V, int, C],
    scale_3d: float,
    scale_2d: float,
) -> tuple[Tensor[[]], Tensor[[]]]:
    """(3D matching loss, 2D reprojection loss): confidence-weighted means of Cauchy losses.

    Points are rebuilt from ARROW depth, the episode's metric scale and the current
    intrinsics. `scale_3d` (m) and `scale_2d` (pixels / focal) are the Cauchy scales δ.
    """

    obs = observations

    # Per-view tables, then per-correspondence rows. Gathering with `embedding` matters: its
    # backward sorts indices, while plain indexing scatters 10⁵ atomic adds into ~10² rows.
    view_scale = scales[obs.view_episode]
    view_intrinsics = intrinsics[obs.view_camera]
    table = torch.cat(
        (
            rearrange(world_from_view, "v i j -> v (i j)"),
            rearrange(view_intrinsics, "v i j -> v (i j)"),
            view_scale[:, None],
        ),
        -1,
    )
    rows_a, rows_b = F.embedding(obs.view_a, table), F.embedding(obs.view_b, table)

    def unpack[K: IntVar](
        rows: Tensor[[K, 26]],
    ) -> tuple[Tensor[[K, 4, 4]], Tensor[[K, 3, 3]], Tensor[[K]]]:
        pose = rearrange(rows[:, :16], "k (i j) -> k i j", i=4)
        return pose, rearrange(rows[:, 16:25], "k (i j) -> k i j", i=3), rows[:, 25]

    pose_a, k_a, scale_a = unpack(rows_a)
    pose_b, k_b, scale_b = unpack(rows_b)

    def rebuild[K: IntVar](
        k: Tensor[[K, 3, 3]], pixel: Tensor[[K, 2]], depth: Tensor[[K]]
    ) -> Tensor[[K, 3]]:
        return camera.backproject(k, pixel[:, None], depth[:, None])[:, 0]

    world_a = _apply(pose_a, rebuild(k_a, obs.pixel_a, scale_a * obs.depth_a))
    world_b = _apply(pose_b, rebuild(k_b, obs.pixel_b, scale_b * obs.depth_b))
    weight = obs.weight / obs.weight.sum()

    loss_3d = (weight * cauchy((world_a - world_b).norm(dim=-1), scale_3d)).sum()

    def reprojection[K: IntVar](
        pose: Tensor[[K, 4, 4]], k: Tensor[[K, 3, 3]], world: Tensor[[K, 3]], pixel: Tensor[[K, 2]]
    ) -> Tensor[[K]]:
        local = _apply(se3.inverse(pose), world)
        error = (camera.project(k, local[:, None])[:, 0] - pixel).norm(dim=-1)
        in_front = local[:, 2] > MIN_REPROJECTION_DEPTH
        return torch.where(in_front, cauchy(error / k[:, 0, 0], scale_2d), 0.0)

    loss_2d = (
        weight
        * (
            reprojection(pose_b, k_b, world_a, obs.pixel_b)
            + reprojection(pose_a, k_a, world_b, obs.pixel_a)
        )
    ).sum()
    return loss_3d, loss_2d


def cauchy[K: IntVar](error: Tensor[[K]], scale: float) -> Tensor[[K]]:
    """`δ² log(1 + e²/δ²)`, normalised by δ² so values stay O(1) for inliers."""

    return torch.log1p((error / scale) ** 2)


def _apply[K: IntVar](transform: Tensor[[K, 4, 4]], points: Tensor[[K, 3]]) -> Tensor[[K, 3]]:
    """Per-correspondence transform: `points[k] ← transform[k] · points[k]`."""

    return se3.transform_points(transform, points[:, None])[:, 0]
