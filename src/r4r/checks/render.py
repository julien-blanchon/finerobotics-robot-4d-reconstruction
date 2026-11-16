"""The robot as a camera would see it: its silhouette (the union of its projected triangles)
and the silhouette's outline, at a working scale.

`p = f · d(s) · (x, y)/r + c` (`r4r.geometry.camera.project`), with the mesh vertices moved
by `camera_from_world · world_from_body`. Pixels of the delivered video times `scale`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from r4r.checks.scene import Scene, View

NEAR = 0.02  # metres: vertices nearer (or behind) the camera are not drawn
SUBPIXEL = 2  # bits of sub-pixel precision for the polygon fill


def project(view: View, t: int, local: np.ndarray, scale: float) -> np.ndarray:
    """Pixels [N, 2] at `scale` of camera-frame points [N, 3] through the view's lens."""

    k, dist = view.intrinsics[t], view.distortion[t]
    x, y, z = local[:, 0], local[:, 1], local[:, 2]
    r = np.maximum(np.hypot(x, y), 1e-12)
    s = np.arctan2(r, z) if view.fisheye else r / np.maximum(z, 1e-6)
    s2 = s * s
    rho = s * (1 + s2 * (dist[0] + s2 * (dist[1] + s2 * (dist[2] + s2 * dist[3]))))
    factor = rho / r
    u = (x * factor * k[0, 0] + k[0, 2] + 0.5) * scale - 0.5
    v = (y * factor * k[1, 1] + k[1, 2] + 0.5) * scale - 0.5
    return np.stack([u, v], 1)


def _usable(view: View, t: int, local: np.ndarray) -> np.ndarray:
    """Points in front of the camera and inside the lens's valid field: a radial polynomial
    folds back far outside the image (kept within 1.5× the corners' radius)."""

    if view.fisheye:
        return local[:, 2] > -0.5 * np.hypot(local[:, 0], local[:, 1])  # up to ~120° off-axis
    k = view.intrinsics[t]
    w, h = view.size
    corner = np.hypot(w / 2 / k[0, 0], h / 2 / k[1, 1])
    z = np.maximum(local[:, 2], 1e-6)
    return (local[:, 2] > NEAR) & (np.hypot(local[:, 0] / z, local[:, 1] / z) < 1.5 * corner)


def silhouette(
    scene: Scene,
    view: View,
    t: int,
    scale: float,
    shape: tuple[int, int],
    bodies: set[int] | None = None,
) -> np.ndarray:
    """uint8 mask (255 on the robot) of `shape` (h, w) of the view at timestep `t`, the video's
    pixels times `scale`; only `bodies` when given."""

    mask = np.zeros(shape, np.uint8)
    size = max(shape)
    camera_from_world = view.camera_from_world[t]
    polygons = []
    for body, vertices, faces in scene.surfaces:
        if bodies is not None and body not in bodies:
            continue
        pose = camera_from_world @ scene.world_from_body[t, body]
        local = vertices @ pose[:3, :3].T + pose[:3, 3]
        pixel = project(view, t, local, scale)
        ok = _usable(view, t, local)[faces].all(1)
        tri = pixel[faces[ok]]
        near = (np.abs(tri) < 8 * size).all((1, 2))
        polygons.append(np.round(tri[near] * (1 << SUBPIXEL)).astype(np.int32))
    if polygons:
        stacked = np.concatenate(polygons)
        if len(stacked):
            cv2.fillPoly(mask, list(stacked), 255, cv2.LINE_8, shift=SUBPIXEL)
    # Gaps between triangles of a decimated or open mesh: closed, then holes filled
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(np.zeros_like(mask), contours, -1, 255, cv2.FILLED)


def outline(mask: np.ndarray, margin: int = 2) -> np.ndarray:
    """Pixels [N, 2] (x, y) on the silhouette's border, not on the image border (a robot cut
    by the frame edge has no edge there)."""

    border = cv2.subtract(mask, cv2.erode(mask, np.ones((3, 3), np.uint8)))
    border[:margin], border[-margin:], border[:, :margin], border[:, -margin:] = 0, 0, 0, 0
    ys, xs = np.nonzero(border)
    return np.stack([xs, ys], 1)
