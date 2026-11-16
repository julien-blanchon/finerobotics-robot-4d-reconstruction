"""An episode in our one canonical form: camera streams, robot state, robot description.

Everything dataset-specific is resolved by the loader (`lerobot.py`, with the dataset's priors):
what remains is images, the joint-state trajectory with its mapping to the description's
joints, and for each camera whether it moves and, when known, the link it is mounted on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.types import RGB, Mat3x3, Mat4x4

    from .priors import CameraPrior, DatasetPriors


@dataclass(frozen=True, slots=True)
class JointMap:
    """Description joint `name` ← state column `index`, optionally rescaled linearly:
    `q = t0 + (sign · x − s0) · (t1 − t0) / (s1 − s0)`, `source = (s0, s1)`, `target = (t0, t1)`.

    Maps a column range onto the joint range (e.g. a gripper command in [0, 1] onto finger
    travel in metres); without a range, `q = sign · x`.
    """

    name: str
    index: int
    sign: float = 1.0
    source: tuple[float, float] | None = None
    target: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class CameraStream[T: IntVar, H: IntVar, W: IntVar]:
    name: str
    frames: Tensor[[T, *RGB, H, W]]  # uint8 RGB (no frames when not decoded: T = 0)
    role: str  # wrist, exo, ego, other
    moving: bool  # on a moving body (static: fixed in the world)
    mount: str | None  # link it is rigid on, when known
    side: str | None  # left, right, center (metadata, used as a sanity check)
    prior: CameraPrior
    # Image height and width (pixels), known whether or not the frames are decoded
    size: tuple[int, int]
    # Known intrinsics of this episode at the video resolution (per-episode table), if any
    intrinsics: Tensor[[*Mat3x3]] | None = None
    # Recorded pose in this episode (OpenCV axes), if any: a static camera's pose in the base
    # frame, a moving camera's mount on its link
    pose: Tensor[[*Mat4x4]] | None = None


@dataclass(frozen=True, slots=True)
class Episode[T: IntVar, NState: IntVar]:
    name: str
    fps: float
    state: Tensor[[T, NState]]
    joints: tuple[JointMap, ...]
    robot: Path  # URDF or MJCF
    # Frames per camera: T when decoded, 0 when not (image sizes differ between cameras)
    cameras: tuple[CameraStream[int, int, int], ...]
    priors: DatasetPriors
    # Which setup (e.g. recording station) the episode belongs to, when the dataset says
    station: str | None = None
    # Cameras with their own setup key (episodes table `station.<camera>`): camera → station
    camera_stations: dict[str, str] = field(default_factory=dict)
    # A mobile base's recorded motion since the first timestep, in its own frame [T, 4, 4]
    # (from the priors' `base_odometry`), the start of its per-frame pose
    base_motion: Tensor[[T, *Mat4x4]] | None = None
