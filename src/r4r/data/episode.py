"""An episode in our one canonical form: camera streams, robot state, robot description.

Everything dataset-specific is resolved by the loader (`lerobot.py`, with the dataset's priors):
what remains is images, the joint-state trajectory with its mapping to the description's
joints, and for each camera whether it moves and, when known, the link it is mounted on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    from .priors import CameraPrior, DatasetPriors


@dataclass(frozen=True, slots=True)
class JointMap:
    """Description joint `name` ← state column `index`, optionally rescaled linearly.

    `source` → `target` maps a column range onto the joint range (e.g. a gripper command in
    [0, 1] onto finger travel in metres); `sign` flips the column first.
    """

    name: str
    index: int
    sign: float = 1.0
    source: tuple[float, float] | None = None
    target: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class CameraStream[T: IntVar, H: IntVar, W: IntVar]:
    name: str
    frames: Tensor[[T, 3, H, W]]  # uint8 RGB (no frames when not decoded: T = 0)
    role: str  # wrist, exo, ego, other
    moving: bool  # on a moving body (static: fixed in the world)
    mount: str | None  # link it is rigid on, when known
    side: str | None  # left, right, center (metadata, used as a sanity check)
    prior: CameraPrior
    # Known intrinsics of this episode at the video resolution (per-episode table), if any
    intrinsics: Tensor[[3, 3]] | None = None
    # Recorded pose in this episode (OpenCV axes), if any: a static camera's pose in the base
    # frame, a moving camera's mount on its link
    pose: Tensor[[4, 4]] | None = None


@dataclass(frozen=True, slots=True)
class Episode[T: IntVar, S: IntVar]:
    name: str
    fps: float
    state: Tensor[[T, S]]
    joints: tuple[JointMap, ...]
    robot: Path  # URDF or MJCF
    cameras: tuple[CameraStream, ...]
    priors: DatasetPriors
    # Which setup (e.g. recording station) the episode belongs to, when the dataset says
    station: str | None = None
    # A mobile base's recorded motion since the first timestep, in its own frame [T, 4, 4]
    # (from the priors' `base_odometry`), the start of its per-frame pose
    base_motion: Tensor[[T, 4, 4]] | None = None
