"""The checks of one reconstructed episode: frames sampled and decoded once, then the edge and
motion checks of every calibrated scene camera, as rows of the shared result table
(`r4r.checks.eval`).

Wrist cameras are skipped: their drawn gripper stays in place in the image and the real one
often differs from the description (YAM, OmniPicker), so a 2D shift says nothing reliable there
(docs/verification.md, "Automatic per-episode checks").
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from r4r.checks import edges, motion
from r4r.checks.edges import ShiftConfig

if TYPE_CHECKING:
    from pathlib import Path

    from r4r.checks.scene import Scene


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckConfig:
    # Moments sampled per episode, and the gap (seconds) to the second frame of each pair
    moments: int = 8
    gap: float = 0.3
    shift: ShiftConfig = field(default_factory=ShiftConfig)
    # The motion shift counts only when moving the drawn robot improves the agreement by this
    # share (else the evidence is too weak: few or no still pairs, the robot barely moving)
    min_gain: float = 0.1
    # A camera is flagged when its estimated error exceeds this share of the image width
    flag: float = 0.08


def flagged(row: dict, width: int, config: CheckConfig) -> bool:
    """Whether a `camera` row's estimate exceeds the flag (`width`: the image width)."""

    return bool(np.isfinite(row["estimate_px"]) and row["estimate_px"] > config.flag * width)


def moments(steps: int, count: int, gap: int) -> list[int]:
    """`count` timesteps spread over an episode of `steps`, each with `gap` steps after it."""

    starts = np.linspace(0, steps - 1 - gap, count + 2)[1:-1].round().astype(int)
    return sorted({int(s) for s in starts if s >= 0})


def estimate(edge: edges.ShiftResult, moved: edges.ShiftResult, config: CheckConfig) -> float:
    """The camera's error estimate (video pixels): the motion shift when its evidence holds,
    else the edge shift."""

    if moved.frames and moved.gain >= config.min_gain:
        return moved.error_px
    return edge.error_px


def check_episode(scene: Scene, root: Path, fps: float, config: CheckConfig) -> list[dict]:
    """Rows (`method` `edge`, `motion`, and `camera` for the combined estimate) for every
    calibrated scene camera of the episode."""

    from r4r.data.lerobot import keyframes

    views = {n: v for n, v in scene.views.items() if v.role == "scene"}
    if not views:
        return []
    began = time.process_time()
    gap = max(1, round(config.gap * fps))
    starts = moments(len(scene.frames), config.moments, gap)
    steps = sorted(set(starts) | {s + gap for s in starts})
    decoded = keyframes(root, scene.episode, [int(scene.frames[t]) for t in steps], list(views))
    rows = []
    for name, view in views.items():
        frame = {t: decoded[name][i].permute(1, 2, 0).numpy() for i, t in enumerate(steps)}
        edge = edges.check(scene, view, {t: frame[t] for t in starts}, config.shift)
        pairs = {t: (t + gap, frame[t], frame[t + gap]) for t in starts}
        moved = motion.check(scene, view, pairs, config.shift)
        error = estimate(edge, moved, config)
        base = {"dataset": scene.dataset, "episode": scene.episode, "camera": name, "role": "scene"}
        for method, found in (("edge", edge), ("motion", moved)):
            rows.append(
                base
                | {
                    "method": method,
                    "score": found.error_px,
                    "estimate_px": found.error_px,
                    "confidence": found.gain,
                    "shift_u": found.shift[0],
                    "shift_v": found.shift[1],
                }
            )
        rows.append(base | {"method": "camera", "score": error, "estimate_px": error})
    seconds = time.process_time() - began
    return [r | {"seconds": seconds} for r in rows]
