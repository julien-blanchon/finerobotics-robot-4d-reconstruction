"""Edge alignment: does the drawn robot's outline lie on image edges, and how far off is it?

Per sampled frame, the robot's silhouette outline (`r4r.checks.render`) is compared with the
frame's edges (Canny) through the truncated distance transform `D` of the edge map:

    `C(d) = mean_o min(D(o + d), τ)`,  `d` a 2D shift of the outline (pixels),

averaged over frames. A camera that draws the robot on the real one has its minimum at
`d ≈ 0`; `d* = argmin C` estimates the camera's error in pixels and is the proposed 2D
correction. Clutter adds edges everywhere and lowers `C` uniformly, so the minimum's position
(not its value) is the score; `gain = (C(0) − C(d*)) / C(0)` is how much the shift helps.

Ref: Barrow et al., IJCAI 1977 (chamfer matching); Borgefors, PAMI 1988.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from r4r.checks.render import outline, silhouette

if TYPE_CHECKING:
    from r4r.checks.scene import Scene, View

ON_EDGE = 1.5  # an outline pixel this close to an edge (working pixels) is on it


@dataclass(frozen=True, slots=True, kw_only=True)
class ShiftConfig:
    # Working image width (pixels): the robot and the images are compared at this size
    width: int = 320
    # Shifts searched: up to `radius` × width in each direction
    radius: float = 0.35
    # Distances beyond `truncate` working pixels count as `truncate` (occlusions, clutter)
    truncate: float = 6.0
    # Frames with fewer robot pixels (outline or change) are skipped
    min_pixels: int = 40


@dataclass(frozen=True, slots=True)
class ShiftResult:
    """`shift` (du, dv) in video pixels that best aligns the drawn robot with the image, its
    norm `error_px`, how much better the shift fits than none (`gain`, in [0, 1]), how
    distinct the best shift is from the typical one (`confidence`), the frames used."""

    shift: tuple[float, float]
    error_px: float
    gain: float
    confidence: float
    frames: int
    # Share of the image evidence the drawn robot explains with no shift (edges: outline pixels
    # on an edge; motion: image change inside the drawn robot's region)
    support: float = float("nan")


NAN = float("nan")
NO_RESULT = ShiftResult((NAN, NAN), NAN, NAN, NAN, 0)


def resized(image: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    """`image` at `width` pixels wide and the scale applied."""

    scale = width / image.shape[1]
    size = (width, round(image.shape[0] * scale))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def edge_distance(image: np.ndarray, cap: float) -> np.ndarray:
    """Distance (working pixels, at most `cap`) to the nearest Canny edge of an RGB frame."""

    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    median = float(np.median(np.asarray(gray, np.float32)))
    edges = cv2.Canny(gray, 0.66 * median, 1.33 * median)
    return np.minimum(cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3), cap)


def correlate(field: np.ndarray, template: np.ndarray, reach: int, fill: float) -> np.ndarray:
    """`S(d) = Σ_p template(p) · field(p + d)` for every shift `|d| ≤ reach` [2r+1, 2r+1]
    (rows: dv, columns: du); the field is `fill` outside the image."""

    padded = cv2.copyMakeBorder(
        field.astype(np.float32), reach, reach, reach, reach, cv2.BORDER_CONSTANT, value=fill
    )
    return cv2.matchTemplate(padded, template.astype(np.float32), cv2.TM_CCORR)


def best_shift(
    costs: np.ndarray, reach: int, scale: float, frames: int, support: float
) -> ShiftResult:
    """The minimum of a cost surface over shifts [2r+1, 2r+1] (mean over frames)."""

    row, col = np.unravel_index(int(np.argmin(costs)), costs.shape)
    zero, best, typical = (
        float(costs[reach, reach]),
        float(costs[row, col]),
        float(np.median(costs)),
    )
    du, dv = (col - reach) / scale, (row - reach) / scale
    return ShiftResult(
        (float(du), float(dv)),
        float(np.hypot(du, dv)),
        (zero - best) / max(abs(zero), abs(best), 1e-9),
        (typical - best) / max(abs(typical), abs(best), 1e-9),
        frames,
        support,
    )


def check(
    scene: Scene, view: View, images: dict[int, np.ndarray], config: ShiftConfig
) -> ShiftResult:
    """The edge check of one view over the frames `images` (timestep -> RGB frame)."""

    reach = round(config.radius * config.width)
    total, used, scale, on_edge = None, 0, 1.0, []
    for t, image in images.items():
        if not view.calibrated(t):
            continue
        small, scale = resized(image, config.width)
        h, w = small.shape[:2]
        points = outline(silhouette(scene, view, t, scale, (h, w)))
        if len(points) < config.min_pixels:
            continue
        template = np.zeros((h, w), np.float32)
        template[points[:, 1], points[:, 0]] = 1.0 / len(points)
        distance = edge_distance(small, config.truncate)
        cost = correlate(distance, template, reach, config.truncate)
        total = cost if total is None else total + cost
        on_edge.append(float(np.mean(distance[points[:, 1], points[:, 0]] <= ON_EDGE)))
        used += 1
    if total is None:
        return NO_RESULT
    return best_shift(total / used, reach, scale, used, float(np.mean(on_edge)))
