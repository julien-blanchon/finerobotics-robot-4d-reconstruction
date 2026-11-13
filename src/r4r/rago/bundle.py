"""Kinematic bundle adjustment: camera poses from forward kinematics, points from triangulation.

Every view's pose is `world_from_body(q_t) · mount` (hard constraint, metric by construction).
Each track's 3D point is triangulated in closed form from its observations' rays under the
current poses (weighted least squares on the rays: no point parameters, differentiable), then
reprojected into every view that saw it. We minimise a robust reprojection error over the free
mounts and the focal lengths.

A static track lives in the world frame. A moving track (a point on the robot, or on a grasped
object) is rigid with some link: expressed in that link's frame its point is constant, so the
same triangulation runs on the rays carried into the link frame by forward kinematics. Each
moving track is attached to the link that explains it best, re-decided as the calibration
improves; tracks no link explains (people, pushed objects) are dropped. The moving robot is
then a calibration target that sweeps a large volume: it fixes exterior cameras that only see
a far, flat scene (where distance and focal length trade off).

ARROW contributes only where points appear in the images; scale comes from the robot's metric
motion, the field of view from the known motion.

Ref: Triggs et al., Bundle Adjustment — A Modern Synthesis, 1999 (variable projection);
     Calib3R arXiv:2509.08813 (joint hand-eye + reconstruction).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from r4r.geometry import camera, se3

from .losses import cauchy
from .rig import MOTION_THRESHOLD, Rig

if TYPE_CHECKING:
    from torch import Tensor

    from .robot import Robot
    from .tracks import Tracks

# Observations closer than this (m) to their camera's plane are not reprojected
MIN_DEPTH = 0.05
# Ridge on the triangulation normal equations (weight), against parallel rays, towards a point
# DEFAULT_DEPTH (m) along the track's mean ray
RIDGE = 1e-6
DEFAULT_DEPTH = 1.0
# The world body: frame of static tracks
WORLD = 0
# A track needs this much ray spread in its body's frame (1 − ‖mean unit ray‖ ≈ σ²/2: σ ≈ 1°)
MIN_PARALLAX = 1.5e-4
# A body with parallax is preferred over a parallax-free one if its mean robust cost is at most
# this much higher (log(1 + (e/δ)²): 0.7 ≈ one δ of error)
PARALLAX_MARGIN = math.log1p(1.0)
# Extra mean robust cost to attach a track to a link instead of the world (≈ one δ of error)
LINK_PENALTY = math.log1p(1.0)


@dataclass(frozen=True, slots=True, kw_only=True)
class BundleConfig:
    iterations: int = 800
    learning_rate: float = 5e-3
    # Robust scale of the reprojection error, in pixels / initial focal length (0.01 ≈ 2–3 px)
    cauchy: float = 0.01
    # Focal-length prior (log space): ~0.3 from ARROW, ~0.01 when the dataset knows it
    focal_std: float = 0.3
    # Moving tracks: re-attach to links every this many iterations; drop a track whose mean
    # robust cost on its best link exceeds `outlier_cost` (cauchy(e) = log(1 + (e/δ)²))
    use_moving: bool = True
    # Tracks may attach to moving links (the robot as a calibration target); False: the
    # static world only (calibrating moving cameras, where links invite spurious fits)
    attach_links: bool = True
    reassign_every: int = 100
    outlier_cost: float = math.log1p(9.0)  # mean error ≈ 3δ


@dataclass(frozen=True, slots=True)
class BundleResult:
    body_from_camera: Tensor[[int, 4, 4]]
    intrinsics: Tensor[[int, 3, 3]]
    world_from_view: Tensor[[int, 4, 4]]
    # Per iteration: total, reprojection, prior
    history: Tensor[[int, 3]]
    # Median reprojection error at the end (pixels), and the fraction of inlier observations
    median_error: float
    inlier_fraction: float
    # Robust cost over all observations (dropped tracks at their capped static-world fit):
    # comparable between runs from different starts (the data term averages over kept ones)
    score: float
    # The same score per camera (over its own observations)
    camera_score: Tensor[[int]]
    # Per track (dense ids): body it is attached to (WORLD when static), -1 when dropped
    track_body: Tensor[[int]]


@dataclass(frozen=True, slots=True)
class Observations:
    """Observations with dense track ids `0..count-1`, ready for `reprojection`."""

    view: Tensor[[int]]
    pixel: Tensor[[int, 2]]
    weight: Tensor[[int]]
    track: Tensor[[int]]
    count: int


def bundle_adjust(
    robot: Robot,
    qpos: Tensor[[int, int]],
    tracks: Tracks,
    names: tuple[str, ...],
    initial_mounts: Tensor[[int, 4, 4]],
    config: BundleConfig,
    pixel_unit: Tensor[[int]] | None = None,
) -> BundleResult:
    """Refine free mounts and focal lengths by robust reprojection of triangulated tracks.

    Errors are in pixels per `pixel_unit` (per camera; default: its initial focal length).
    Runs to be compared by score must share the unit.
    """

    t = tracks
    keep = ~t.moving | config.use_moving
    _, track = torch.unique(t.track[keep], return_inverse=True)
    obs = Observations(
        t.view[keep], t.pixel[keep].to(qpos), t.weight[keep].to(qpos), track, int(track.max()) + 1
    )
    scales = qpos.new_ones(1)  # unused: the bundle never touches ARROW's depth
    rig = Rig(
        robot,
        qpos,
        names,
        initial_mounts.to(qpos),
        scales,
        t.intrinsics.to(qpos),
        config.focal_std,
        pixel_unit,
    )
    track_moving = torch.zeros(obs.count, dtype=torch.bool, device=qpos.device)
    track_moving[obs.track[t.moving[keep]]] = True
    candidates = moving_bodies(rig) if config.attach_links else []

    def assign() -> Tensor[[int]]:
        with torch.no_grad():
            return attach(rig, t, obs, track_moving, candidates, config.cauchy, config.outlier_cost)

    track_body = assign()
    optimizer = torch.optim.Adam(rig.parameters(), lr=config.learning_rate)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 0.5 * (1 + math.cos(math.pi * step / config.iterations))
    )
    history = []
    for step in range(config.iterations):
        if step and step % config.reassign_every == 0:
            track_body = assign()
        body = track_body[obs.track]
        weight = torch.where(body >= 0, obs.weight, 0.0)
        error, valid, _ = reprojection(rig, t, obs, body.clamp_min(WORLD))
        data = (cauchy(error, config.cauchy) * weight * valid).sum() / (weight * valid).sum()
        prior = rig.prior() / len(t.view_time)
        loss = data + prior
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        history.append(torch.stack((loss, data, prior)).detach())

    with torch.no_grad():
        track_body = assign()
        body = track_body[obs.track]
        error, in_front, _ = reprojection(rig, t, obs, body.clamp_min(WORLD))
        # The score keeps dropped tracks at their (capped) fit to the static world: counting
        # them as outliers would reward mounts that give more tracks parallax (long levers)
        robust = torch.where(in_front, cauchy(error, config.cauchy), config.outlier_cost)
        valid = in_front & (body >= 0)
        unit = rig.pixel_unit[t.view_camera[obs.view]]
        robust = robust.clamp_max(config.outlier_cost) * obs.weight
        score = float(robust.sum() / obs.weight.sum())
        camera = t.view_camera[obs.view]
        per_camera = torch.zeros(len(names), dtype=robust.dtype, device=robust.device)
        weight_sum = torch.zeros_like(per_camera).index_add(0, camera, obs.weight)
        per_camera = per_camera.index_add(0, camera, robust) / weight_sum.clamp_min(1e-12)
        return BundleResult(
            body_from_camera=rig.mounts(),
            intrinsics=rig.intrinsics(),
            world_from_view=rig.camera_poses(t.view_camera, t.view_time),
            history=torch.stack(history),
            median_error=float((error * unit)[valid].median()),
            inlier_fraction=float((error[valid] < 3 * config.cauchy).float().mean()),
            score=score,
            camera_score=per_camera,
            track_body=track_body,
        )


def attach(
    rig: Rig,
    tracks: Tracks,
    obs: Observations,
    track_moving: Tensor[[int]],
    candidates: list[int],
    scale: float,
    outlier_cost: float,
) -> Tensor[[int]]:
    """Body of each track: WORLD if known static; otherwise the body (WORLD or a moving link)
    that explains it best (mean robust cost over its observations), or -1 if none does.

    In the frame of the body a camera rides on, the camera's rays all leave one centre and any
    point fits them: that body "explains" the track while constraining nothing (a background
    point seen by a static camera, the gripper seen by its wrist camera). Such parallax-free
    fits only win when no body with parallax fits about as well; the track is then
    uninformative and dropped.
    """

    track_body = torch.full(track_moving.shape, WORLD, device=track_moving.device)
    if not bool(track_moving.any()):
        return track_body
    candidates = [WORLD, *candidates]
    mine = track_moving[obs.track]
    owner, local = torch.unique(obs.track[mine], return_inverse=True)
    sub = Observations(obs.view[mine], obs.pixel[mine], obs.weight[mine], local, len(owner))
    costs = []
    for body in candidates:
        error, valid, spread = reprojection(rig, tracks, sub, torch.full_like(sub.track, body))
        w = sub.weight * valid
        total = torch.zeros(sub.count, dtype=w.dtype, device=w.device)
        total = total.index_add(0, sub.track, cauchy(error, scale) * w)
        norm = torch.zeros_like(total).index_add(0, sub.track, w)
        cost = torch.where(norm > 0, total / norm.clamp_min(1e-12), math.inf)
        # Most of a scene is static: a link must fit clearly better than the world
        cost = cost + (LINK_PENALTY if body != WORLD else 0.0)
        costs.append(torch.stack((cost, torch.where(spread > MIN_PARALLAX, cost, math.inf))))
    cost_any, cost_parallax = torch.stack(costs).unbind(1)
    best_any = cost_any.min(0).values
    best, chosen = cost_parallax.min(0)
    informative = best <= best_any + PARALLAX_MARGIN
    bodies = torch.tensor(candidates, device=chosen.device)[chosen]
    track_body[owner] = torch.where(informative & (best < outlier_cost), bodies, -1)
    return track_body


def moving_bodies(rig: Rig) -> list[int]:
    """Bodies whose pose changes during the episode."""

    poses = rig.world_from_body
    moved = (poses - poses[:1]).abs().flatten(-2).amax(-1).amax(0) > MOTION_THRESHOLD
    return moved.nonzero()[:, 0].tolist()


def reprojection(
    rig: Rig,
    tracks: Tracks,
    obs: Observations,
    body: Tensor[[int]],
) -> tuple[Tensor[[int]], Tensor[[int]], Tensor[[int]]]:
    """Per observation: reprojection error and whether it is usable; per track: ray spread.

    Each observation's ray is expressed in the frame of `body[m]` at its view's time (the world
    body for static tracks), where its track's point is constant.

    The error is in pixels divided by the camera's pixel unit, a constant (its initial focal
    length by default): dividing by the current focal would reward growing it.
    """

    t = tracks
    world_from_view = rig.camera_poses(t.view_camera, t.view_time)
    intrinsics = rig.intrinsics()

    # Gather per-observation pose and intrinsics through embedding (sorted backward)
    table = torch.cat((world_from_view.flatten(1), intrinsics[t.view_camera].flatten(1)), -1)
    rows = F.embedding(obs.view, table)
    world_pose, k = rows[:, :16].unflatten(-1, (4, 4)), rows[:, 16:].unflatten(-1, (3, 3))
    pose = se3.inverse(rig.world_from_body[t.view_time[obs.view], body]) @ world_pose

    # Rays: centre c and unit direction d in the track's frame
    pixel, weight = obs.pixel, obs.weight
    local = camera.backproject(k, pixel[:, None], torch.ones_like(pixel[:, :1]))[:, 0]
    direction = F.normalize((pose[:, :3, :3] @ local[:, :, None])[..., 0], dim=-1)
    centre = pose[:, :3, 3]

    # Weighted least-squares triangulation per track: Σ w (I − d dᵀ) X = Σ w (I − d dᵀ) c
    eye = torch.eye(3, dtype=pixel.dtype, device=pixel.device)
    normal = weight[:, None, None] * (eye - direction[:, :, None] * direction[:, None, :])
    lhs = torch.zeros(obs.count, 3, 3, dtype=pixel.dtype, device=pixel.device).index_add(
        0, obs.track, normal
    )
    rhs = torch.zeros(obs.count, 3, dtype=pixel.dtype, device=pixel.device).index_add(
        0, obs.track, (normal @ centre[:, :, None])[..., 0]
    )
    # The ridge pulls along-ray uncertainty towards a point on the mean ray (not the origin):
    # a parallax-free track then sits on its rays, and its error is its pixels' scatter
    total = torch.zeros_like(rhs[:, 0]).index_add(0, obs.track, weight).clamp_min(1e-12)
    mean_centre = torch.zeros_like(rhs).index_add(0, obs.track, weight[:, None] * centre)
    mean_ray = torch.zeros_like(rhs).index_add(0, obs.track, weight[:, None] * direction)
    anchor = mean_centre / total[:, None] + DEFAULT_DEPTH * F.normalize(mean_ray, dim=-1)
    point = torch.linalg.solve(lhs + RIDGE * eye, rhs + RIDGE * anchor)

    # Reproject each track's point into each view that saw it
    in_view = se3.transform_points(se3.inverse(pose), point[obs.track][:, None])[:, 0]
    projected = camera.project(k, in_view[:, None])[:, 0]
    unit = rig.pixel_unit[t.view_camera[obs.view]]
    error = (projected - pixel).norm(dim=-1) / unit
    valid = in_view[:, 2] > MIN_DEPTH
    spread = 1 - mean_ray.norm(dim=-1) / total
    return torch.where(valid, error, 0.0), valid, spread
