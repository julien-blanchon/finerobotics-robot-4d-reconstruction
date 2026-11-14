"""Per-dataset priors: what a human or agent established about a dataset, overriding its metadata.

A dataset's own metadata (joint mapping, camera links, labels) is often incomplete or wrong.
Priors live in this repository as `priors/<family>/<dataset>.json`, one per dataset root, and
take precedence over `meta/robot.json` and the camera feature info:

```json
{
  "base": "fixed",
  "joints": [{"index": 16, "urdf_joint": "idx11_head_joint1", "sign": 1.0}],
  "removed_joints": ["joint8"],
  "state_shift": -1,
  "cameras": {
    "wrist_left": {"mount": "moving", "link": "panda_hand", "position": [0.05, 0.0, 0.04],
                   "orientation": [0.7071, 0.0, 0.0, 0.7071], "position_range": 0.005,
                   "orientation_range_deg": 1.0, "lens": "pinhole",
                   "focal": 182.0, "focal_range": 0.0,
                   "distortion": [-0.28, 0.07], "distortion_range": 0.0}
  },
  "bodies": {"right_arm_base": {"position_range": 0.1, "orientation_range_deg": 10,
                                 "scope": "episode"}},
  "notes": "free text: what was checked and how"
}
```

`description` replaces the delivered robot description by a corrected one kept under
`priors/` (fixes to the URDF/MJCF itself: wrong joints, frames, meshes). `joints` add or
replace (by URDF joint name) entries of the state mapping; `removed_joints`
drops wrong ones; `state_feature` names the state column when the metadata names a wrong one;
`state_shift` fixes a video/joint-state offset (frame t's state is taken
from t + shift; `r4r fit --shifts` measures it). A camera's `position` and `orientation`
(quaternion w, x, y, z of its OpenCV axes) are in the frame of `link` (moving) or of the robot
base (static); ranges bound what fitting may change (small: fixed across episodes, fitted per
episode within them). `scope` says how often a value may change within its ranges: once for
the dataset, per episode (e.g. a static camera re-placed between episodes) or per frame.
`bodies` are elements of the description whose offset from their parent is not exactly known
(a second arm's base, a mobile base). `focal` is the camera's focal length in pixels at the
video resolution, a property of the lens: fixed across episodes unless `focal_range`
(relative) says how far it may vary. `lens` is `pinhole` (radial `distortion`: OpenCV's
k1, k2, k3) or `fisheye` (Kannala–Brandt, OpenCV fisheye's k1..k4); `distortion_range` is the
prior std of each coefficient when fitting refines them (0: held; default 0 when the
coefficients are given, wide for a fisheye without them). `intrinsics` reads a camera's
intrinsics from a column of the dataset itself, per episode (simulators that randomise the
focal): `{"column": "publisher.<name>_intrinsics", "layout": "fx_fy_cx_cy" | "matrix",
"size": [w, h], "resize": "stretch" | "height"}`: `size` is the resolution the values are
given at (default: the video's), `resize` how they map to the video (`stretch`: each axis
scaled; `height`: the same vertical field of view, centred). `ignore` leaves a camera out (one
that rides on no robot body, e.g. worn by a person).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .episode import JointMap

PRIORS = Path(__file__).resolve().parents[3] / "priors"


@dataclass(frozen=True, slots=True)
class IntrinsicsColumn:
    """Where a dataset records a camera's intrinsics (see the module doc)."""

    column: str
    layout: Literal["fx_fy_cx_cy", "matrix"] = "fx_fy_cx_cy"
    size: tuple[int, int] | None = None  # width, height the values are given at
    resize: Literal["stretch", "height"] = "stretch"


@dataclass(frozen=True, slots=True)
class CameraPrior:
    mount: Literal["static", "moving"] | None = None
    link: str | None = None
    position: tuple[float, float, float] | None = None
    orientation: tuple[float, float, float, float] | None = None  # w, x, y, z (OpenCV axes)
    position_range: float = math.inf  # metres
    orientation_range_deg: float = 180.0
    lens: Literal["pinhole", "fisheye"] = "pinhole"
    focal: float | None = None  # pixels at the video resolution
    focal_range: float | None = None  # relative; None: unknown (searched)
    # Pinhole: OpenCV radial k1, k2, k3; fisheye: OpenCV fisheye (Kannala–Brandt) k1..k4.
    # None: zero (an ideal pinhole; for a fisheye, unknown: refined from the equidistant model)
    distortion: tuple[float, ...] | None = None
    # Prior std of each coefficient; None: held, except a fisheye without `distortion` (wide)
    distortion_range: float | None = None
    # How often the mount may change within its ranges: dataset, episode (e.g. a static camera
    # re-placed between episodes) or frame
    scope: Literal["dataset", "episode", "frame"] = "dataset"
    # Intrinsics recorded in the dataset's own columns (per episode), if any
    intrinsics: IntrinsicsColumn | None = None
    # Left out of every command (a camera on no robot body, e.g. worn by a person)
    ignore: bool = False


@dataclass(frozen=True, slots=True)
class BodyPrior:
    """A body whose offset from its parent is not exactly known (an arm base, a mobile base)."""

    position_range: float  # metres
    orientation_range_deg: float
    scope: Literal["dataset", "episode", "frame"] = "dataset"
    # Fitted correction of the body's offset from its parent (applied after the description's
    # offset, in the body's own frame): position (m) and orientation (wxyz). None: not fitted.
    position: tuple[float, float, float] | None = None
    orientation: tuple[float, float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class BaseOdometry:
    """Where a mobile base's motion is recorded (see `r4r.rago.odometry`): `feature` columns
    `columns` holding planar poses (x, y, θ), body-frame velocities (vx, vy, ω), or wheel
    angles (left, right[, steering]) of a caster or differential drive with its geometry."""

    kind: Literal["pose", "velocity", "caster"]
    feature: str
    columns: tuple[int, ...]
    radius: float | None = None
    half_track: float | None = None
    offset: float = 0.0
    signs: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True, slots=True)
class DatasetPriors:
    base: Literal["fixed", "mobile"] = "fixed"
    joints: tuple[JointMap, ...] = ()
    removed_joints: tuple[str, ...] = ()
    state_shift: int = 0
    # The state feature, when the metadata names a wrong one (None: `robot.json`'s)
    state_feature: str | None = None
    cameras: dict[str, CameraPrior] = field(default_factory=dict)
    bodies: dict[str, BodyPrior] = field(default_factory=dict)
    base_odometry: BaseOdometry | None = None
    # A corrected robot description replacing the delivered one: a path under `priors/`
    # (e.g. `descriptions/<embodiment>/robot.xml`, shared by an embodiment's datasets). Meshes
    # may point back into the delivery (absolute `meshdir` / filenames).
    description: str | None = None
    # Kinematic refinement in `r4r rig` (prior stds; absent: held): joint-encoder offsets
    # (`joint_offset_deg` for hinges, `joint_offset_mm` for slides, optionally only `joints`)
    # and a video/joint time offset on top of `state_shift` (`time_offset_frames`)
    kinematics: dict | None = None
    notes: str = ""

    @property
    def replaced(self) -> frozenset[str]:
        """Joints whose metadata mapping these priors replace or remove."""

        return frozenset({j.name for j in self.joints} | set(self.removed_joints))

    def joint_map(self, metadata: tuple[JointMap, ...]) -> tuple[JointMap, ...]:
        """The dataset's joint mapping with these priors applied."""

        return tuple(j for j in metadata if j.name not in self.replaced) + self.joints


def priors_path(root: Path) -> Path:
    """`priors/<family>/<dataset>.json` for a delivery root `<family>/datasets/<dataset>`."""

    return PRIORS / root.parent.parent.name / f"{root.name}.json"


def load_priors(root: Path) -> DatasetPriors:
    """The priors of a dataset root (empty when none were written)."""

    path = priors_path(root)
    if not path.exists():
        return DatasetPriors()
    raw = json.loads(path.read_text())
    joints = tuple(
        JointMap(
            j["urdf_joint"],
            j["index"],
            float(j.get("sign", 1.0)),
            tuple(j["from"]) if j.get("from") else None,
            tuple(j["to"]) if j.get("to") else None,
        )
        for j in raw.get("joints", [])
    )
    cameras = {
        name: CameraPrior(
            mount=c.get("mount"),
            link=c.get("link"),
            position=tuple(c["position"]) if c.get("position") else None,
            position_range=float(c.get("position_range", math.inf)),
            orientation=tuple(c["orientation"]) if c.get("orientation") else None,
            orientation_range_deg=float(c.get("orientation_range_deg", 180.0)),
            lens=c.get("lens", "pinhole"),
            focal=c.get("focal"),
            focal_range=c.get("focal_range"),
            distortion=tuple(c["distortion"]) if c.get("distortion") is not None else None,
            distortion_range=c.get("distortion_range"),
            scope=c.get("scope", "dataset"),
            intrinsics=_intrinsics_column(c.get("intrinsics")),
            ignore=bool(c.get("ignore", False)),
        )
        for name, c in raw.get("cameras", {}).items()
    }
    bodies = {
        name: BodyPrior(
            float(b["position_range"]),
            float(b["orientation_range_deg"]),
            b.get("scope", "dataset"),
            tuple(b["position"]) if b.get("position") else None,
            tuple(b["orientation"]) if b.get("orientation") else None,
        )
        for name, b in raw.get("bodies", {}).items()
    }
    odometry = raw.get("base_odometry")
    if odometry is not None:
        odometry = BaseOdometry(
            odometry["kind"],
            odometry["feature"],
            tuple(odometry["columns"]),
            odometry.get("radius"),
            odometry.get("half_track"),
            float(odometry.get("offset", 0.0)),
            tuple(odometry.get("signs", (1.0, 1.0, 1.0))),
        )
    return DatasetPriors(
        base=raw.get("base", "fixed"),
        base_odometry=odometry,
        bodies=bodies,
        joints=joints,
        removed_joints=tuple(raw.get("removed_joints", [])),
        state_shift=int(raw.get("state_shift", 0)),
        state_feature=raw.get("state_feature"),
        cameras=cameras,
        description=raw.get("description"),
        kinematics=raw.get("kinematics"),
        notes=raw.get("notes", ""),
    )


def _intrinsics_column(raw: dict | None) -> IntrinsicsColumn | None:
    if raw is None:
        return None
    return IntrinsicsColumn(
        raw["column"],
        raw.get("layout", "fx_fy_cx_cy"),
        tuple(raw["size"]) if raw.get("size") else None,
        raw.get("resize", "stretch"),
    )


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """Known per-episode values: the station (setup) and each camera's intrinsics
    `(fx, fy, cx, cy)` at the video resolution."""

    station: str | None
    intrinsics: dict[str, tuple[float, float, float, float]]


def episode_table_path(root: Path) -> Path:
    """`priors/<family>/<dataset>.episodes.parquet` next to the dataset's priors."""

    return priors_path(root).with_suffix(".episodes.parquet")


def load_episode_record(root: Path, episode: int) -> EpisodeRecord | None:
    """One episode's known values; None when the dataset has no table or no row for it.

    The table has one row per (episode, camera): `episode, camera, fx, fy, cx, cy` and an
    optional `station`.
    """

    path = episode_table_path(root)
    if not path.exists():
        return None
    import pyarrow.parquet as pq

    table = pq.read_table(path, filters=[("episode", "=", episode)])
    if table.num_rows == 0:
        return None
    rows = table.to_pylist()
    station = rows[0].get("station")
    return EpisodeRecord(
        None if station is None else str(station),
        {r["camera"]: (r["fx"], r["fy"], r["cx"], r["cy"]) for r in rows},
    )


def write_camera(root: Path, camera: str, values: Mapping[str, object]) -> Path:
    """Set fields of one camera's priors (e.g. a fitted mount), keeping everything else."""

    return _write(root, "cameras", camera, values)


def write_body(root: Path, body: str, values: Mapping[str, object]) -> Path:
    """Set fields of one free body's priors (e.g. a fitted correction), keeping the rest."""

    return _write(root, "bodies", body, values)


def _write(root: Path, kind: str, name: str, values: Mapping[str, object]) -> Path:
    path = priors_path(root)
    raw = json.loads(path.read_text()) if path.exists() else {}
    raw.setdefault(kind, {}).setdefault(name, {}).update(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, indent=2) + "\n")
    return path
