"""The checks of one reconstructed episode, as rows of the shared result table
(`r4r.checks.eval`): frames sampled and decoded once, then

- `sync` for every calibrated camera: does the video move when the state says the robot (or
  the camera) moves (`r4r.checks.sync`)? The episode is flagged when the mean `ρ` over its
  scene cameras (its wrist cameras when it has none) is below `CheckConfig.sync_flag`;
- `edge`, `motion` and their combination `camera` for every calibrated scene camera: the 2D
  shift that puts the drawn robot on the image, an estimate of the camera's pixel error. It is
  reported, not used as a flag: against clicked fingertips it separates cameras above 30 px
  from the rest only with AUC ~0.85 (docs/verification.md, "Automatic per-episode checks").

Wrist cameras get no shift estimate: their drawn gripper stays in place in the image and the
real one often differs from the description (YAM, OmniPicker).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from r4r.checks import edges, motion, sync
from r4r.checks.edges import ShiftConfig

if TYPE_CHECKING:
    from pathlib import Path

    from r4r.checks.scene import Scene


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckConfig:
    # Frame pairs sampled per episode (the shift checks use every other one), and the gap
    # (seconds) to the second frame of each pair
    moments: int = 16
    gap: float = 0.3
    shift: ShiftConfig = field(default_factory=ShiftConfig)
    # The motion shift counts only when moving the drawn robot improves the agreement by this
    # share (else the evidence is too weak: few or no still pairs, the robot barely moving)
    min_gain: float = 0.1
    # An episode whose mean sync ρ is below this is flagged: 5 % of in-sync episodes, 61 % / 78 %
    # of the same episodes with the state 1 s / 3 s late (docs/verification.md)
    sync_flag: float = 0.3


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


def episode_sync(rows: list[dict], config: CheckConfig) -> tuple[float, bool]:
    """An episode's mean sync `ρ` (scene cameras, else wrists) and whether it is flagged."""

    found: dict[str, list[float]] = {}
    for r in rows:
        if r["method"] == "sync" and np.isfinite(r["score"]):
            found.setdefault(r["role"], []).append(1.0 - r["score"])
    values = found.get("scene") or found.get("wrist")
    if not values:
        return float("nan"), False
    rho = float(np.mean(values))
    return rho, rho < config.sync_flag


def check_episode(scene: Scene, root: Path, fps: float, config: CheckConfig) -> list[dict]:
    """Rows (`sync` per calibrated camera; `edge`, `motion` and `camera` per scene camera)."""

    from r4r.data.lerobot import keyframes

    views = dict(scene.views)
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
        pairs = {t: (t + gap, frame[t], frame[t + gap]) for t in starts}
        base = {"dataset": scene.dataset, "episode": scene.episode, "camera": name}
        base |= {"role": view.role}
        rho, used = sync.check(scene, view, pairs)
        rows.append(base | {"method": "sync", "score": 1.0 - rho, "confidence": used / len(starts)})
        if view.role != "scene":
            continue
        half = starts[::2]
        edge = edges.check(scene, view, {t: frame[t] for t in half}, config.shift)
        moved = motion.check(scene, view, {t: pairs[t] for t in half}, config.shift)
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
        error = estimate(edge, moved, config)
        rows.append(base | {"method": "camera", "score": error, "estimate_px": error})
    seconds = time.process_time() - began
    return [r | {"seconds": seconds} for r in rows]
