"""Starting values for unknowns the global search does not cover: static cameras.

A static camera sees points that calibrated (moving) cameras have already triangulated: its
pose follows from 2D–3D correspondences by PnP with RANSAC, over a grid of focal lengths when
its focal is not known (the one with most inliers), then refined by the bundle. A distorted
lens is undistorted first (its pixels mapped to where an ideal pinhole with the same focal
would see them), so PnP itself stays a pinhole solve.

Ref: Lepetit, Moreno-Noguer & Fua, *EPnP*, IJCV 2009 (via OpenCV).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from r4r.geometry import camera, se3

if TYPE_CHECKING:
    from torch import Tensor

    from .tracks import Tracks


RIDGE = 1e-6
REPROJECTION_PX = 3.0
RANSAC_ITERATIONS = 2000
MIN_PNP_POINTS = 6
# Undistorted rays further than this from the optical axis are left out of PnP (rad)
MAX_PNP_ANGLE = math.radians(75)
FOCAL_GRID = np.geomspace(0.3, 3.0, 25)  # × image width


@dataclass(frozen=True, slots=True)
class Points:
    """Least-squares points of tracks and how many observations made each."""

    point: Tensor[[int, 3]]
    count: Tensor[[int]]


def triangulate(tracks: Tracks, world_from_view: Tensor[[int, 4, 4]], use: Tensor[[int]]) -> Points:
    """Points of every track from its observations in `use` (views with known poses)."""

    v = tracks.view[use]
    k = tracks.view_intrinsics(tracks.intrinsics.to(world_from_view))[v]
    pose = world_from_view[v]
    pixel = tracks.pixel[use].to(pose)[:, None]
    distortion = fisheye = None
    if tracks.lens is not None:
        lens = tracks.lens.to(pose).select(tracks.view_camera[v])
        distortion, fisheye = lens.distortion, lens.fisheye
    rays = camera.rays(k, pixel, distortion, fisheye)[:, 0]
    direction = F.normalize((pose[:, :3, :3] @ rays[..., None])[..., 0], dim=-1)
    eye = torch.eye(3, dtype=pose.dtype, device=pose.device)
    projector = eye - direction[:, :, None] * direction[:, None, :]
    count = int(tracks.track.max()) + 1
    track = tracks.track[use]
    lhs = pose.new_zeros(count, 3, 3).index_add(0, track, projector)
    rhs = pose.new_zeros(count, 3).index_add(0, track, (projector @ pose[:, :3, 3, None])[..., 0])
    point = torch.linalg.solve(lhs + RIDGE * eye, rhs)
    seen = torch.zeros(count, dtype=torch.long, device=pose.device).index_add(
        0, track, torch.ones_like(track)
    )
    return Points(point, seen)


@dataclass(frozen=True, slots=True)
class CameraStart:
    world_from_camera: Tensor[[4, 4]]
    focal: float
    inliers: int


def locate_camera(
    points: Tensor[[int, 3]],
    pixels: Tensor[[int, 2]],
    size: tuple[int, int],
    focal: float | None = None,
    distortion: Tensor[[4]] | None = None,
    fisheye: Tensor[[]] | None = None,
) -> CameraStart | None:
    """PnP-RANSAC pose (and focal from a grid when `focal` is None) of a camera; None if no
    solution. `distortion`, `fisheye`: its lens (None: an ideal pinhole)."""

    h, w = size
    best = None
    for f in [focal] if focal is not None else (FOCAL_GRID * w).tolist():
        k = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]])
        world, image = undistort(points, pixels, torch.from_numpy(k), distortion, fisheye)
        if len(world) < MIN_PNP_POINTS:
            continue
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            world,
            image,
            k,
            None,
            reprojectionError=REPROJECTION_PX,
            iterationsCount=RANSAC_ITERATIONS,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if ok and inliers is not None and (best is None or len(inliers) > best[3]):
            best = (f, rvec, tvec, len(inliers))
    if best is None:
        return None
    f, rvec, tvec, count = best
    camera_from_world = np.eye(4)
    camera_from_world[:3, :3] = cv2.Rodrigues(rvec)[0]
    camera_from_world[:3, 3] = tvec[:, 0]
    pose = se3.inverse(torch.tensor(camera_from_world, dtype=points.dtype, device=points.device))
    return CameraStart(pose, float(f), count)


def undistort(
    points: Tensor[[int, 3]],
    pixels: Tensor[[int, 2]],
    intrinsics: Tensor[[3, 3]],
    distortion: Tensor[[4]] | None,
    fisheye: Tensor[[]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Points and their pixels as an ideal pinhole with `intrinsics` would see them (float64
    arrays for OpenCV); rays too far off the axis for a pinhole are left out."""

    world = points.double().cpu().numpy()
    image = pixels.double().cpu()
    if distortion is None or fisheye is None:
        return world, image.numpy()
    k = intrinsics.double()
    ray = camera.rays(k, image, distortion.double().cpu(), fisheye.cpu())
    keep = ray[:, 2] > math.cos(MAX_PNP_ANGLE)
    ideal = camera.project(k, ray[keep])
    return world[keep.numpy()], ideal.numpy()
