"""An episode in our one canonical form: camera streams, robot state, robot description.

Everything dataset-specific is resolved by the loader (`lerobot.py`): what remains is images,
the joint-state trajectory with its mapping to the description's joints, and for each camera
the link it is rigidly mounted on (None for a camera fixed in the world).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor


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
    frames: Tensor[[T, 3, H, W]]  # uint8 RGB
    role: str  # wrist, exo, ego, other
    mount: str | None  # rigid on this link; None: fixed in the world
    side: str | None  # left, right, center (metadata, used as a sanity check)


@dataclass(frozen=True, slots=True)
class Episode[T: IntVar, S: IntVar]:
    name: str
    fps: float
    state: Tensor[[T, S]]
    joints: tuple[JointMap, ...]
    robot: Path  # URDF or MJCF
    cameras: tuple[CameraStream, ...]
