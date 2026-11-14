"""Starting values for unknowns the global search does not cover: static cameras.

A static camera sees points that calibrated (moving) cameras have already triangulated: its
pose follows from 2D–3D correspondences by PnP with RANSAC (`p ≃ K (R X + t)`, inliers within
`REPROJECTION_PX`), over a grid of focal lengths `f ∈ [0.3, 3] · W` when its focal is not known
(the one with most inliers), then refined by the bundle. A distorted
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
import numpy.typing as npt
import torch
import torch.nn.functional as F

from r4r.geometry import camera, se3

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.types import K1K2K3K4, UV, XYZ, Mat3x3, Mat4x4, Scalar

    from .tracks import Tracks


RIDGE = 1e-6
REPROJECTION_PX = 3.0
RANSAC_ITERATIONS = 2000
MIN_PNP_POINTS = 6
# Undistorted rays further than this from the optical axis are left out of PnP (rad)
MAX_PNP_ANGLE = math.radians(75)
FOCAL_GRID = np.geomspace(0.3, 3.0, 25)  # × image width


@dataclass(frozen=True, slots=True)
class Points[N: IntVar]:
    """Least-squares points of `N` tracks and how many observations made each."""

    point: Tensor[[N, *XYZ]]
    count: Tensor[[N]]


def triangulate[M: IntVar, V: IntVar, C: IntVar](
    tracks: Tracks[M, V, C], world_from_view: Tensor[[V, *Mat4x4]], use: Tensor[[M]]
) -> Points[int]:
    """Points of every track from its observations in `use` (views with known poses).

    Each observation is a ray from the camera centre `o` along the unit `d`; the point closest
    to all rays in least squares solves `(Σ P + λI) X = Σ P o`, `P = I − d dᵀ` (the projector
    orthogonal to the ray; λ = `RIDGE` keeps unseen tracks finite). Metres.
    """

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
    world_from_camera: Tensor[[*Mat4x4]]
    focal: float  # pixels
    inliers: int


def locate_camera[N: IntVar](
    points: Tensor[[N, *XYZ]],
    pixels: Tensor[[N, *UV]],
    size: tuple[int, int],
    focal: float | None = None,
    distortion: Tensor[[*K1K2K3K4]] | None = None,
    fisheye: Scalar | None = None,
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


def undistort[N: IntVar](
    points: Tensor[[N, *XYZ]],
    pixels: Tensor[[N, *UV]],
    intrinsics: Tensor[[*Mat3x3]],
    distortion: Tensor[[*K1K2K3K4]] | None,
    fisheye: Scalar | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Points and their pixels as an ideal pinhole with `intrinsics` would see them (float64
    arrays for OpenCV): `p' = K π(ray(p))`; rays more than `MAX_PNP_ANGLE` off the axis (where
    a pinhole cannot see) are left out."""

    world = points.double().cpu().numpy()
    image = pixels.double().cpu()
    if distortion is None or fisheye is None:
        return world, image.numpy()
    k = intrinsics.double()
    ray = camera.rays(k, image, distortion.double().cpu(), fisheye.cpu())
    keep = ray[:, 2] > math.cos(MAX_PNP_ANGLE)
    ideal = camera.project(k, ray[keep])
    return world[keep.numpy()], ideal.numpy()
