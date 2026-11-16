"""Motion agreement: does the image change where the drawn robot moves?

For a camera that stays still between two frames `t` and `t'`, the robot changes the image
where its silhouette appears or disappears: `P = S_t ⊕ S_t'` (`r4r.checks.render`). The
measured change is the frame difference above its typical level,
`D = max(|g_t − g_t'| − median, 0)` (`g` blurred grey levels). Over 2D shifts `d`,

    `A(d) = Σ_p P(p) · D(p + d) / Σ_p P(p)`

summed over frame pairs peaks at `d ≈ 0` for a right camera, and `d* = argmax A` estimates
the error in pixels. Static clutter has `D ≈ 0` and does not compete; people and held objects
moving with the robot do, and only the peak's position is used. Pairs where the camera moves
(head pans, base motion) are skipped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from r4r.checks.edges import NO_RESULT, ShiftConfig, ShiftResult, best_shift, correlate, resized
from r4r.checks.render import silhouette

if TYPE_CHECKING:
    from r4r.checks.scene import Scene, View

STILL_M = 0.005  # a camera moving less than this between the two frames is still (metres)
STILL_RAD = 0.003


def still(view: View, t: int, later: int) -> bool:
    """Whether the camera stays (nearly) in place between timesteps `t` and `later`."""

    change = view.camera_from_world[later] @ np.linalg.inv(view.camera_from_world[t])
    angle = np.arccos(np.clip((np.trace(change[:3, :3]) - 1) / 2, -1, 1))
    return bool(np.linalg.norm(change[:3, 3]) < STILL_M and angle < STILL_RAD)


def change(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Measured change between two RGB frames: `D` (module doc), in grey levels."""

    a, b = (
        cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (5, 5), 0) for x in (first, second)
    )
    difference = cv2.absdiff(a, b).astype(np.float32)
    return np.maximum(difference - float(np.median(difference)), 0.0)


def check(
    scene: Scene,
    view: View,
    pairs: dict[int, tuple[int, np.ndarray, np.ndarray]],
    config: ShiftConfig,
) -> ShiftResult:
    """The motion check of one view over frame pairs (timestep -> (later timestep, RGB frame
    at the timestep, RGB frame at the later one))."""

    reach = round(config.radius * config.width)
    total, used, scale, explained = None, 0, 1.0, []
    for t, (later, first, second) in pairs.items():
        if not (view.calibrated(t) and view.calibrated(later) and still(view, t, later)):
            continue
        a, scale = resized(first, config.width)
        b, _ = resized(second, config.width)
        shape = a.shape[:2]
        masks = (
            silhouette(scene, view, t, scale, shape),
            silhouette(scene, view, later, scale, shape),
        )
        moved = cv2.bitwise_xor(*masks)
        region = cv2.dilate(cv2.bitwise_or(*masks), np.ones((5, 5), np.uint8))
        area = int(np.count_nonzero(moved))
        if area < config.min_pixels:
            continue
        template = (moved > 0).astype(np.float32) / area
        measured = change(a, b)
        agreement = correlate(measured, template, reach, 0.0)
        total = agreement if total is None else total + agreement
        explained.append(float(measured[region > 0].sum() / max(measured.sum(), 1e-9)))
        used += 1
    if total is None:
        return NO_RESULT
    return best_shift(-total / used, reach, scale, used, float(np.mean(explained)))
