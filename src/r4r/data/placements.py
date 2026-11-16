"""Did a static camera move between two episodes? The static background's pixel shift (CPU).

A diagnostic for deciding whether two setups share a camera placement, not a setup detector
(`r4r.data.setups` finds the runs). If the camera did not move, every point of the scene that
did not move stays at the same pixel. Per episode, SIFT keypoints of the first frame are kept
where the image barely changes until a later frame (`LATER` of the way in: the robot, the
operator and moved objects change). Between two episodes the keypoints are matched (Lowe's
ratio test) and a homography is fitted (RANSAC, `RANSAC_PX`). The shift is the median
displacement of its inliers, in pixels at `REPORT_WIDTH`:

    s_ab = median_{inliers i} ‖x_i^b − x_i^a‖     (∞ with fewer than `MIN_INLIERS` inliers)

An unmoved camera gives under a pixel whatever the lighting or the objects on the table; a few
centimetres or degrees give pixels (robomind2 dual-franka camera_right: 0.6 px between the
rig_c dates 0827 and 0829, 19.7 px for the 4 cm move between 0410 and 0515). Unlike a
thumbnail distance, the value is in pixels, the unit of the calibration's tolerance.

Known failures (docs/dataset-onboarding.md): a camera on the robot frame whose room changes
(HABIT: a new background gives ∞ although the camera did not move on the robot), and plain
walls and tables (robocoin lane02: too few background keypoints, ∞ on most pairs).

Ref: Lowe, SIFT, IJCV 2004; Fischler & Bolles, RANSAC, 1981.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

WIDTH = 384  # frames are matched at this width
REPORT_WIDTH = 512  # shifts are reported in pixels at this width
LATER = 0.6
STILL = 0.35  # standardised change below which a pixel is background (as `setups.STILL`)
KEYPOINTS = 1500
RATIO = 0.8
RANSAC_PX = 3.0
MIN_INLIERS = 12


@dataclass(frozen=True, slots=True)
class Background:
    """An episode's background keypoints `[N, 2]` (pixels at `WIDTH`) and SIFT descriptors
    `[N, 128]`."""

    points: np.ndarray
    descriptors: np.ndarray


def _standard(gray: np.ndarray) -> np.ndarray:
    gray = gray.astype(np.float32)
    return (gray - gray.mean()) / max(float(gray.std()), 1e-6)


def background(first: np.ndarray, later: np.ndarray) -> Background:
    """Background keypoints of one episode from two uint8 RGB frames `[H, W, 3]`, its first
    and a later one."""

    height, width = first.shape[:2]
    size = (WIDTH, max(1, round(height * WIDTH / width)))
    a, b = (
        cv2.cvtColor(cv2.resize(f, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        for f in (first, later)
    )
    change = cv2.GaussianBlur(_standard(a) - _standard(b), (9, 9), 0)
    still = (np.abs(change) < STILL).astype(np.uint8) * 255
    # OpenCV's stubs lack SIFT_create (in the main module since 4.4)
    sift = cv2.SIFT_create(nfeatures=KEYPOINTS)  # type: ignore[pyrefly:missing-attribute]
    found, descriptors = sift.detectAndCompute(a, still)
    points = np.array([k.pt for k in found], np.float32).reshape(-1, 2)
    if descriptors is None:
        descriptors = np.zeros((0, 128), np.float32)
    return Background(points, descriptors)


def backgrounds(root: Path, episodes: Sequence[int], camera: str) -> dict[int, Background]:
    """`background` of each episode from its first and `LATER` frame of `camera`."""

    from .lerobot import IMAGE_PREFIX, _root, seek_frames

    tables = _root(root).tables
    out = {}
    for e in episodes:
        path, start = tables.files(e).videos[IMAGE_PREFIX + camera]
        later = start + LATER * (tables.length(e) - 1) / tables.fps
        frames = seek_frames(path, [start, later], 0.5 / tables.fps, gap=5.0)
        first, last = (f.permute(1, 2, 0).numpy() for f in frames)
        out[e] = background(first, last)
    return out


def shift(a: Background, b: Background) -> tuple[float, int]:
    """Background shift `s_ab` (module doc) in pixels at `REPORT_WIDTH`, ∞ without enough
    consistent matches, and the homography's inlier count."""

    if len(a.descriptors) < 2 or len(b.descriptors) < 2:
        return np.inf, 0
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(a.descriptors, b.descriptors, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    if len(good) < MIN_INLIERS:
        return np.inf, 0
    xa = a.points[[g.queryIdx for g in good]]
    xb = b.points[[g.trainIdx for g in good]]
    homography, inliers = cv2.findHomography(xa, xb, cv2.RANSAC, RANSAC_PX)
    if homography is None:
        return np.inf, 0
    keep = inliers.ravel().astype(bool)
    if keep.sum() < MIN_INLIERS:
        return np.inf, int(keep.sum())
    moved = float(np.median(np.linalg.norm(xb[keep] - xa[keep], axis=1)))
    return moved * REPORT_WIDTH / WIDTH, int(keep.sum())
