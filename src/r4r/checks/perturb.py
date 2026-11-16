"""Synthetic negatives for evaluating checks: a camera moved by a known amount, and the pixel
error that move causes on the drawn robot.

A perturbation acts in the camera frame at every timestep, `camera'_from_world = Δ⁻¹ ·
camera_from_world` (a static camera moved in place, a wrist camera's mount moved), or scales
the focal. Its truth is the median displacement of the robot's visible surface points between
the two cameras, over the given timesteps (`displacement_px`).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from r4r.checks.render import project

if TYPE_CHECKING:
    from r4r.checks.scene import Scene, View

# The evaluation's perturbations: kind (t: translation m, r: rotation rad, f: focal factor)
PERTURBATIONS = (
    *(("t", m / 100, f"t:{m}cm") for m in (1, 2, 5, 10, 20)),
    *(("r", np.radians(d), f"r:{d}deg") for d in (1, 2, 5, 10, 20)),
    ("f", 1.1, "f:+10%"),
    ("f", 0.9, "f:-10%"),
)


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues: `R = I + sin θ K + (1 − cos θ) K²`."""

    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def perturbed(view: View, kind: str, amount: float, rng: np.random.Generator) -> View:
    """`view` moved by `amount` in a random direction (`t`, metres; `r`, radians about a random
    axis through the camera centre) or with its focal times `amount` (`f`)."""

    if kind == "f":
        k = view.intrinsics.copy()
        k[:, 0, 0] *= amount
        k[:, 1, 1] *= amount
        return replace(view, intrinsics=k)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    delta = np.eye(4)
    if kind == "t":
        delta[:3, 3] = amount * direction
    else:
        delta[:3, :3] = _rotation(direction, amount)
    return replace(view, camera_from_world=np.linalg.inv(delta) @ view.camera_from_world)


def displacement_px(scene: Scene, view: View, moved: View, steps: list[int]) -> float:
    """Median pixel displacement of the robot's surface vertices inside the image (in front of
    both cameras) between `view` and `moved`, over `steps`."""

    found = []
    w, h = view.size
    for t in steps:
        if not view.calibrated(t):
            continue
        for body, vertices, _ in scene.surfaces:
            world = scene.world_from_body[t, body]
            pixels = []
            for v in (view, moved):
                pose = v.camera_from_world[t] @ world
                local = vertices[::10] @ pose[:3, :3].T + pose[:3, 3]
                pixels.append((project(v, t, local, 1.0), local[:, 2] > 0.05))
            (a, front_a), (b, front_b) = pixels
            inside = (a[:, 0] >= 0) & (a[:, 0] < w) & (a[:, 1] >= 0) & (a[:, 1] < h)
            keep = inside & front_a & front_b
            found.append(np.linalg.norm(a[keep] - b[keep], axis=1))
    values = np.concatenate(found) if found else np.zeros(0)
    return float(np.median(values)) if len(values) else float("nan")
