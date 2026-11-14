"""Feature tracks along each camera's video: pyramidal Lucas–Kanade (KLT).

ARROW associates points across distant views and cameras, but its image positions are a few
pixels off, and far worse on close-up wrist views (out of its training distribution). Along a
video, consecutive frames differ little and classical KLT tracking is sub-pixel. Corners are
detected where tracks are missing, followed frame to frame, and kept only if tracking back
returns to the start (forward–backward check). Nothing assumes the point is static: the bundle
decides which body (world or link) each track is rigid with.

Ref: Lucas & Kanade 1981; Shi & Tomasi, Good Features to Track, 1994; Kalal et al. 2010
     (forward–backward error).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

    from .lens import Lens
    from .tracks import Tracks


@dataclass(frozen=True, slots=True, kw_only=True)
class KLTConfig:
    # Corners kept alive per frame, and their minimum spacing (pixels)
    features: int = 600
    min_distance: int = 8
    quality: float = 0.01
    window: int = 21
    levels: int = 3
    # Forward–backward agreement (pixels)
    max_error: float = 0.5
    # Drop tracks seen in fewer frames
    min_length: int = 3


@dataclass(frozen=True, slots=True)
class VideoTracks:
    """Observations of one camera's feature tracks: track id, frame index, pixel."""

    track: Tensor[[int]]
    frame: Tensor[[int]]
    pixel: Tensor[[int, 2]]


def track_video(frames: Tensor[[int, 3, int, int]], config: KLTConfig) -> VideoTracks:
    """KLT tracks through `frames` (uint8 RGB, in time order)."""

    gray = [cv2.cvtColor(f.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2GRAY) for f in frames]
    height, width = gray[0].shape
    flow = {
        "winSize": (config.window, config.window),
        "maxLevel": config.levels,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    active = np.zeros((0, 2), np.float32)
    ids = np.zeros(0, np.int64)
    next_id = 0
    tracks, times, pixels = [], [], []
    for t, image in enumerate(gray):
        if t and len(active):
            previous = gray[t - 1]
            ahead, ok, _ = cv2.calcOpticalFlowPyrLK(previous, image, active, active.copy(), **flow)
            back, ok_b, _ = cv2.calcOpticalFlowPyrLK(image, previous, ahead, ahead.copy(), **flow)
            keep = (ok[:, 0] == 1) & (ok_b[:, 0] == 1)
            keep &= np.linalg.norm(back - active, axis=1) < config.max_error
            keep &= (ahead >= 0).all(1) & (ahead[:, 0] <= width - 1) & (ahead[:, 1] <= height - 1)
            active, ids = ahead[keep], ids[keep]
        if len(active) < config.features:
            mask = np.full((height, width), 255, np.uint8)
            for x, y in active:
                cv2.circle(mask, (int(x), int(y)), config.min_distance, 0, -1)
            found = cv2.goodFeaturesToTrack(
                image,
                config.features - len(active),
                config.quality,
                config.min_distance,
                mask=mask,
            )
            if found is not None:
                found = found[:, 0].astype(np.float32)
                active = np.concatenate((active, found))
                ids = np.concatenate((ids, np.arange(next_id, next_id + len(found))))
                next_id += len(found)
        tracks.append(ids.copy())
        times.append(np.full(len(ids), t))
        pixels.append(active.copy())

    track, frame = np.concatenate(tracks), np.concatenate(times)
    pixel = np.concatenate(pixels).astype(np.float64)
    length = np.bincount(track, minlength=next_id)
    keep = length[track] >= config.min_length
    _, track = np.unique(track[keep], return_inverse=True)
    return VideoTracks(
        torch.from_numpy(track), torch.from_numpy(frame[keep]), torch.from_numpy(pixel[keep])
    )


def klt_tracks(
    frames: list[Tensor[[int, 3, int, int]]],
    view_camera: Tensor[[int]],
    view_time: Tensor[[int]],
    intrinsics: Tensor[[int, 3, 3]],
    config: KLTConfig,
    found: list[VideoTracks] | None = None,
    lens: Lens | None = None,
) -> Tracks:
    """Tracks of every camera's video (frame `t` of camera `c` is view (c, t)); each track is
    rigid with an unknown body, so all are flagged `moving` (the bundle attaches them).
    `found`: each video's tracks when already computed (`track_video`)."""

    from .tracks import Tracks

    lookup = torch.full(
        (len(frames), int(view_time.max()) + 1), -1, dtype=torch.long, device=view_time.device
    )
    lookup[view_camera, view_time] = torch.arange(len(view_time), device=view_time.device)
    tracks, views, pixels = [], [], []
    offset = 0
    for c, video in enumerate(frames):
        one = found[c] if found is not None else track_video(video, config)
        tracks.append(one.track.to(view_time.device) + offset)
        views.append(lookup[c, one.frame.to(view_time.device)])
        pixels.append(one.pixel.to(view_time.device))
        offset += int(one.track.max()) + 1 if len(one.track) else 0
    track, view, pixel = torch.cat(tracks), torch.cat(views), torch.cat(pixels)
    assert bool((view >= 0).all()), "every video frame must be a view"
    ones = torch.ones(len(track), dtype=pixel.dtype, device=pixel.device)
    return Tracks(view_camera, view_time, track, view, pixel, ones, ones > 0, intrinsics, lens)
