"""State/video agreement over time: does the image change when the robot (or the camera on it)
moves, and stay still when it does not?

Per frame pair `(t, t')` of an episode, the measured change is the mean absolute difference of
the blurred grey frames, `m = mean |g_t − g_t'|`. The predicted motion is

- a camera that moves between the frames (wrist, head on a moving body): its own motion,
  `p = θ + ‖Δx‖ / Z` (rotation angle in radians, translation over a typical depth `Z`);
- a still camera: the drawn robot's change, the area of `S_t ⊕ S_t'` (`r4r.checks.motion`).

Over the pairs, the Spearman correlation `ρ(m, p)` is high when the joint state drives the
video, and near zero or negative when they do not match (a state recorded for another episode,
a stream out of step, a wrong camera). The score is `1 − ρ`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from r4r.checks.edges import resized
from r4r.checks.render import silhouette

if TYPE_CHECKING:
    from r4r.checks.scene import Scene, View

DEPTH_M = 0.5  # typical scene depth turning a camera translation into an angle (metres)
WIDTH = 160  # working width (pixels)
MIN_PAIRS = 6


def camera_motion(view: View, t: int, later: int) -> float:
    """`θ + ‖Δx‖ / Z` of the camera between two timesteps (radians)."""

    change = view.camera_from_world[later] @ np.linalg.inv(view.camera_from_world[t])
    angle = np.arccos(np.clip((np.trace(change[:3, :3]) - 1) / 2, -1, 1))
    return float(angle + np.linalg.norm(change[:3, 3]) / DEPTH_M)


def _grey(image: np.ndarray) -> np.ndarray:
    small, _ = resized(image, WIDTH)
    return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.float32)


def _ranks(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)


def check(
    scene: Scene, view: View, pairs: dict[int, tuple[int, np.ndarray, np.ndarray]]
) -> tuple[float, int]:
    """`ρ(m, p)` over the pairs (timestep -> (later timestep, frames)) and the pairs used; NaN
    with fewer than `MIN_PAIRS` or no variation."""

    moving = any(camera_motion(view, t, later) > 0.005 for t, (later, _, _) in pairs.items())
    measured, predicted = [], []
    for t, (later, first, second) in pairs.items():
        if not (view.calibrated(t) and view.calibrated(later)):
            continue
        a, b = _grey(first), _grey(second)
        measured.append(float(np.mean(np.abs(a - b))))
        if moving:
            predicted.append(camera_motion(view, t, later))
        else:
            scale = a.shape[1] / first.shape[1]
            masks = [silhouette(scene, view, s, scale, a.shape) for s in (t, later)]
            predicted.append(float(np.count_nonzero(cv2.bitwise_xor(*masks))))
    m, p = np.array(measured), np.array(predicted)
    if len(m) < MIN_PAIRS or np.ptp(m) == 0 or np.ptp(p) == 0:
        return float("nan"), len(m)
    return float(np.corrcoef(_ranks(m), _ranks(p))[0, 1]), len(m)
