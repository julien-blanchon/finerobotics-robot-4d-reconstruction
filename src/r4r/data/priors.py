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

`state_feature` names the joint-state feature when `meta/robot.json` names a wrong one (a list:
the features' columns concatenated, e.g. leg joints recorded apart from the arms).
`side_table` is a directory of parquet files holding features the delivery dropped (an
absolute path, or relative to `priors/`; `r4r.data.tables.SideTable`), read like the
dataset's own features, e.g. by `state_feature`: `{"side_table":
"/e/scratch/.../panda-omron-mobile.side", "state_feature": ["observation.state",
"upstream.arm_qpos"]}` with `joints` indexing the concatenated columns.
`description` replaces the delivered robot description by a corrected one kept under
`priors/` (fixes to the URDF/MJCF itself: wrong joints, frames, meshes). `joints` add or
replace (by URDF joint name) entries of the state mapping; `removed_joints`
drops wrong ones; `state_shift` fixes a video/joint-state offset (frame t's state is taken from
t + shift; `r4r fit --shifts` measures it). A camera's `position` and `orientation` (quaternion
w, x, y, z of its OpenCV axes) are in the frame of `link` (moving) or of the robot base
(static); ranges bound what fitting may change (small: fixed across episodes, fitted per episode
within them): each is the std σ of a Gaussian prior, a residual `(x − x₀) / σ` (metres, radians
of rotation, relative focal); `orientation_range_deg` ≥ 180 leaves the orientation free. A pose
without both ranges is only a start (free; a warning says so). `scope` says how often a value
may change within its ranges: once for the dataset, per episode (e.g. a static camera re-placed
between episodes) or per frame.
`bodies` are elements of the description whose offset from their parent is not exactly known (a
second arm's base, a mobile base). `focal` is the camera's focal length in pixels at the video
resolution, a property of the lens: fixed across episodes unless `focal_range` (relative) says
how far it may vary; `aspect` (fy / fx, default 1) for non-square pixels.
`lens` is `pinhole` (radial `distortion`: OpenCV's
k1, k2, k3) or `fisheye` (Kannala–Brandt, OpenCV fisheye's k1..k4); `distortion_range` is the
prior std of each coefficient when fitting refines them (0: held; default 0 when the
coefficients are given, wide for a fisheye without them). A static camera whose distortion is
refined needs a known `focal` (an error otherwise: focal, distortion and distance trade
off). `intrinsics` reads a camera's
intrinsics from a column of the dataset itself, per episode (simulators that randomise the
focal): `{"column": "publisher.<name>_intrinsics", "layout": "fx_fy_cx_cy" | "matrix",
"size": [w, h], "resize": "stretch" | "height"}`: `size` is the resolution the values are
given at (default: the video's), `resize` how they map to the video (`stretch`: each axis
scaled; `height`: the same vertical field of view, centred). `pose` reads a camera's recorded
pose per episode from a column of the dataset, its first row (a JSON string; or numbers
x, y, z, qw, qx, qy, qz with no `path`) (simulators that re-place cameras
per episode and record them with the episode, too many episodes for a table):
`{"column": "publisher.episode.extra_info", "path": ["config", "cameras", "name=front"],
"axes": "forward_left_up"}`: `path` walks the JSON (`key=value` picks the list item whose `key`
equals `value`) to an object with `position` (x, y, z) and `orientation` (w, x, y, z) in the
robot base frame (moving camera: on its link); `axes` are the recorded camera axes (`opencv`,
or `forward_left_up`: x forward, y left, z up, Isaac Sim's "world" camera axes). With
`"layout": "world_to_camera"` (or `world_from_camera`) the column holds a flattened 3×4 / 4×4
matrix (or x, y, z, qw, qx, qy, qz) per frame in the simulator's world frame instead, and the
top-level `world_anchor` names
a camera with a `link` and such a column: per episode, hand-eye between that link's forward
kinematics and the anchor's recorded motion gives the base in the world, and every such camera
is placed in the base frame (on its link, moving) at the episode's middle frame
(`r4r.data.recorded`; molmobot records cameras in its houses, not the base). With
`"layout": "base_from_camera"` the per-frame records are already in the robot's base frame, up
to a constant `offset` (x, y, z of the record's origin in the description's base frame): the
episode's pose is the robust median over its frames, on the camera's link when it moves
(AgiBot's jittering hand extrinsics); `start` and `order` (`matrix` or `rotation_translation`)
pick a camera's 12 values out of a feature shared by several cameras. An episodes
table pose takes precedence. `ignore` leaves a camera out (one that rides on no robot body,
e.g. worn by a person). `kinematics` refines the robot's kinematics in `r4r rig` (prior stds;
absent: held): `{"joint_offset_deg": 1.0, "joint_offset_mm": 2.0, "joints": [...],
"time_offset_frames": 1.0}`, encoder offsets of hinges / slides (only `joints`, if given) and a
video/joint time offset on top of `state_shift`; `offsets` (joint name → rad or m) and
`time_offset` (frames) are the values `r4r rig --write` fitted, applied to the joint positions of
every command (`r4r.rago.kinematics.corrected_qpos`).

`ee` defines each arm's end effector (`r4r.effectors`, docs/output-spec.md): `{"right":
{"preset": "franka_hand", "body": "panda_hand", "position": [0, 0, 0.1034], "orientation": [1,
0, 0, 0], "pads": [{"body": "leftfinger$"}, {"body": "rightfinger$"}], "closed": 0.0,
"fingertips": {"thumb": {"body": "R_thumb_distal", "offset": [0.02, 0.01, -0.01]}},
"actuation": {"feature": "observation.suction", "index": 0}}}`: a `preset` gives defaults for a
common gripper or hand, every other field overrides it. `type` is `parallel`, `multi_finger`,
`dexterous`, `suction`, `tool` or `none`; `position`, `orientation` (w, x, y, z) place the `ee`
frame in `body`'s frame; `pads` (parallel grippers: the opening is their distance along the
frame's y, minus `closed`) and `fingertips` name points by a regular expression over the bodies
below `body`. `action` is `joint_absolute` when the dataset's action is an absolute joint target
in the state's columns, when its feature info does not say so.

`robot_points` names points on the robot (`{"tip": {"body": "panda_hand", "offset": [0, 0,
0.105]}}`, metres in the body's frame) for picks, pixels clicked on them (`read_picks`, `r4r rig
--picks`, `r4r.rago.keypoints`); every body's origin is also a point, named after the body.

`groups` give a dataset that mixes robots (one description per group of episodes, e.g.
RoboMIND2 agilex: Piper and ARX X5 arms under one label) its own priors per group: `{"arx":
{"match": {"stat": "observation.state/q99", "columns": [2, 8], "above": 0.1}, "description":
"...", "joints": [...], "cameras": {...}}}`. `match` picks the group's episodes from their
statistics in `meta/episodes` (`stats/<stat>`): an episode belongs to the first group where the
largest of `columns`' values exceeds `above` and/or the smallest lies below `below`; episodes no
group matches keep the dataset-level priors. A group's other fields replace the dataset's
(`cameras` and `bodies` entry by entry, as a station's); `stations` then apply on top
(`for_episode`). Fitted values of a group's episodes are written into its block.

Unknown keys are errors (a typo would otherwise be ignored silently), except records kept for
people and never read: `notes`, `base_odometry_measured` (an alternative odometry record) and
a camera's or body's `fitted` (provenance of a fit).
"""

from __future__ import annotations

import functools
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

from .episode import JointMap

if TYPE_CHECKING:
    import numpy as np
    import pyarrow as pa

PRIORS = Path(__file__).resolve().parents[3] / "priors"

logger = logging.getLogger(__name__)

# Parsed JSON (priors files, `meta/robot.json`): an untyped boundary, read field by field
type JsonObject = dict[str, Any]


@dataclass(frozen=True, slots=True)
class IntrinsicsColumn:
    """Where a dataset records a camera's intrinsics (see the module doc)."""

    column: str
    layout: Literal["fx_fy_cx_cy", "matrix"] = "fx_fy_cx_cy"
    size: tuple[int, int] | None = None  # width, height the values are given at
    resize: Literal["stretch", "height"] = "stretch"


@dataclass(frozen=True, slots=True)
class PoseColumn:
    """Where a dataset records a camera's pose per episode (see the module doc)."""

    column: str
    path: tuple[str, ...]
    axes: Literal["opencv", "forward_left_up"] = "opencv"
    # `record`: the episode's first row, a JSON string or x, y, z, qw, qx, qy, qz, already in
    # the base frame (on the link, moving); `world_to_camera` / `world_from_camera`: a
    # flattened 3×4 or 4×4 matrix per frame in a world frame, placed in the base frame with
    # the priors' `world_anchor`; `base_from_camera`: such a matrix per frame in the robot's
    # base frame, up to a constant `offset` (the record's frame origin in the description's base
    # frame, m): the episode's pose is the robust median over its frames (on the link, moving)
    layout: Literal["record", "world_to_camera", "world_from_camera", "base_from_camera"] = "record"
    # Per frame, the camera's values start at column `start` of the feature (several cameras in
    # one feature); `order`: `matrix` (3×4 or 4×4 row-major) or `rotation_translation` (R
    # row-major, then t: 12 values)
    start: int = 0
    order: Literal["matrix", "rotation_translation"] = "matrix"
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class GroupMatch:
    """Which episodes form a group (see the module doc): those whose `meta/episodes` statistic
    `stats/<stat>` has, over `columns`, a largest value above `above` and/or a smallest value
    below `below`."""

    stat: str
    columns: tuple[int, ...]
    above: float | None = None
    below: float | None = None

    def matches(self, values: np.ndarray) -> np.ndarray:
        """Per episode, whether its statistic `values` `[E, n]` puts it in the group."""

        import numpy as np

        picked = values[:, list(self.columns)]
        inside = np.ones(len(values), dtype=bool)
        if self.above is not None:
            inside &= picked.max(axis=1) > self.above
        if self.below is not None:
            inside &= picked.min(axis=1) < self.below
        return inside


@dataclass(frozen=True, slots=True)
class RobotPoint:
    """A point rigid with body `body`, at `offset` (m) in the body's frame (its origin by
    default): a link origin, the gripper tip, the flange."""

    body: str
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


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
    # Pixel aspect fy / fx (non-square pixels, e.g. a 848-wide sensor resized to 640): `focal`
    # is fx, fy = focal × aspect; focal corrections keep the ratio
    aspect: float = 1.0
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
    # Pose recorded in the dataset's own JSON column (per episode), if any
    pose: PoseColumn | None = None
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
class OdometryGain:
    """A dataset's correction of its recorded base motion, fitted once
    (`r4r.rago.odometry.fit_gain`) and applied to every episode: each step's twist
    `ξ = log(M_{t−1}⁻¹ M_t) = (v, ω)` becomes

        ξ' = (s · v, g · ω + (0, 0, b · Δt))

    `s` scales the distance driven (wheel radius), `g` the turns (track width, heading gain),
    `b` is a constant yaw rate (rad/s, gyro or encoder bias); Δt is the step's duration (s).
    """

    scale: float = 1.0
    heading_gain: float = 1.0
    yaw_rate: float = 0.0

    def then(self, other: OdometryGain) -> OdometryGain:
        """This correction followed by `other` (on the motion this one corrected)."""

        return OdometryGain(
            self.scale * other.scale,
            self.heading_gain * other.heading_gain,
            other.heading_gain * self.yaw_rate + other.yaw_rate,
        )


@dataclass(frozen=True, slots=True)
class BaseOdometry:
    """Where a mobile base's motion is recorded (see `r4r.rago.odometry`): `feature` columns
    `columns` holding planar poses (x, y, θ or x, y, qz, qw), full poses (x, y, z, qw, qx, qy,
    qz: list the columns in that order), body-frame velocities (vx, vy, ω), wheel angles (left,
    right[, steering]) of a caster or differential drive with its geometry, or a swerve drive's
    steering angles (`columns`) and wheel ground speeds (m/s, `speed_columns` of
    `speed_feature`, default `feature`) with each wheel's contact point (x, y) in the base
    frame (`wheels`). `correction` (`OdometryGain`: `scale`, `heading_gain`, `yaw_rate`) is
    applied to every episode's motion when given."""

    kind: Literal["pose", "velocity", "caster", "swerve"]
    feature: str | tuple[str, ...]  # several: their columns concatenated
    columns: tuple[int, ...]
    radius: float | None = None
    half_track: float | None = None
    offset: float = 0.0
    signs: tuple[float, float, float] = (1.0, 1.0, 1.0)
    speed_feature: str | None = None
    speed_columns: tuple[int, ...] = ()
    wheels: tuple[tuple[float, float], ...] = ()
    # Fitted once on the calibration sample (`r4r rig --write`): the motion is then held
    correction: OdometryGain | None = None


@dataclass(frozen=True, slots=True)
class KinematicsPrior:
    """Prior stds of the kinematic refinement in `r4r rig` (see the module doc)."""

    joint_offset_deg: float = 0.0  # hinge encoder offsets
    joint_offset_mm: float = 0.0  # slide encoder offsets
    joints: tuple[str, ...] | None = None  # only these joints (None: all)
    time_offset_frames: float = 0.0  # frames at the dataset's rate
    # Values fitted by `r4r rig --write`, applied to every episode's joint positions: per-joint
    # offsets (rad for hinges, m for slides) and the time offset (frames at the dataset's rate)
    offsets: dict[str, float] = field(default_factory=dict)
    time_offset: float = 0.0


@dataclass(frozen=True, slots=True)
class ActuationColumn:
    """Where a suction cup's or a tool's actuation is recorded: `feature`, column `index`."""

    feature: str
    index: int = 0


type EffectorType = Literal["parallel", "multi_finger", "dexterous", "suction", "tool", "none"]


@dataclass(frozen=True, slots=True)
class EffectorPrior:
    """One arm's end effector (see the module doc); None fields come from the `preset`."""

    preset: str | None = None
    type: EffectorType | None = None
    body: str | None = None
    position: tuple[float, float, float] | None = None
    orientation: tuple[float, float, float, float] | None = None  # w, x, y, z
    pads: tuple[RobotPoint, RobotPoint] | None = None
    closed: float | None = None
    fingertips: dict[str, RobotPoint] | None = None
    actuation: ActuationColumn | None = None


@dataclass(frozen=True, slots=True)
class DatasetPriors:
    base: Literal["fixed", "mobile"] = "fixed"
    joints: tuple[JointMap, ...] = ()
    removed_joints: tuple[str, ...] = ()
    state_shift: int = 0
    cameras: dict[str, CameraPrior] = field(default_factory=dict)
    bodies: dict[str, BodyPrior] = field(default_factory=dict)
    base_odometry: BaseOdometry | None = None
    # A corrected robot description replacing the delivered one: a path under `priors/`
    # (e.g. `descriptions/<embodiment>/robot.xml`, shared by an embodiment's datasets). Meshes
    # may point back into the delivery (absolute `meshdir` / filenames).
    description: str | None = None
    # The feature holding the joint state, when `meta/robot.json` names a wrong one
    state_feature: str | tuple[str, ...] | None = None  # several: concatenated
    # Features kept outside the delivery (`r4r.data.tables.SideTable`): a directory
    side_table: str | None = None
    # Kinematic refinement in `r4r rig` (absent: held)
    kinematics: KinematicsPrior | None = None
    # Named points on the robot (body origins are points without being named here)
    robot_points: dict[str, RobotPoint] = field(default_factory=dict)
    # Per-station overrides (raw priors fields `cameras`, `bodies`, set on top of the
    # dataset's, entry by entry): {station: {"cameras": {...}, "bodies": {...}}}
    stations: dict[str, JsonObject] = field(default_factory=dict)
    # Per-group priors of a dataset mixing robots (raw fields, each with its `match`), and the
    # group these priors are for (None: the dataset level)
    groups: dict[str, JsonObject] = field(default_factory=dict)
    group: str | None = None
    # The camera on a known link that ties cameras recorded in a world frame (pose `layout`
    # world_*) to the robot's base, per episode (`r4r.data.recorded`)
    world_anchor: str | None = None
    # End effectors per arm name (absent: detected from the presets, `r4r.effectors`)
    ee: dict[str, EffectorPrior] = field(default_factory=dict)
    # `joint_absolute`: the action is an absolute joint target in the state's columns
    action: Literal["joint_absolute"] | None = None
    # The raw fields these priors were parsed from (for merging group and station overrides)
    raw: JsonObject = field(default_factory=dict, repr=False, compare=False)
    notes: str = ""

    @property
    def side_directory(self) -> Path | None:
        """The side table's directory (`side_table`, relative to `priors/` unless absolute)."""

        return None if self.side_table is None else PRIORS / self.side_table

    @property
    def replaced(self) -> frozenset[str]:
        """Joints whose metadata mapping these priors replace or remove."""

        return frozenset({j.name for j in self.joints} | set(self.removed_joints))

    def joint_map(self, metadata: tuple[JointMap, ...]) -> tuple[JointMap, ...]:
        """The dataset's joint mapping with these priors applied."""

        return tuple(j for j in metadata if j.name not in self.replaced) + self.joints

    def group_matches(self) -> dict[str, GroupMatch]:
        """Each group's `match`, in file order (the first matching group takes an episode)."""

        return {
            name: GroupMatch(
                block["match"]["stat"],
                tuple(block["match"]["columns"]),
                block["match"].get("above"),
                block["match"].get("below"),
            )
            for name, block in self.groups.items()
        }

    def for_group(self, group: str | None) -> DatasetPriors:
        """These priors with a group's fields applied (the module doc), if any."""

        block = self.groups.get(group) if group is not None else None
        if block is None:
            return self
        merged = {**self.raw, **{k: v for k, v in block.items() if k != "match"}}
        for kind in ("cameras", "bodies"):
            merged[kind] = _merged(self.raw.get(kind, {}), block.get(kind, {}))
        from dataclasses import replace

        return replace(_parsed(merged, f"group {group}"), group=group)

    def for_episode(self, group: str | None, station: str | None) -> DatasetPriors:
        """An episode's priors: its group's (if any), then its station's overrides."""

        return self.for_group(group).for_station(station)

    def for_station(self, station: str | None) -> DatasetPriors:
        """These priors with a station's overrides (cameras, bodies) applied, if any."""

        override = self.stations.get(str(station)) if station is not None else None
        if not override:
            return self
        from dataclasses import replace

        where = f"station {station}"
        cameras = _merged(self.raw.get("cameras", {}), override.get("cameras", {}))
        bodies = _merged(self.raw.get("bodies", {}), override.get("bodies", {}))
        return replace(self, cameras=_cameras(cameras, where), bodies=_bodies(bodies, where))


def priors_path(root: Path) -> Path:
    """`priors/<family>/<dataset>.json` for a delivery root `<family>/datasets/<dataset>`."""

    return PRIORS / root.parent.parent.name / f"{root.name}.json"


def load_priors(root: Path) -> DatasetPriors:
    """The priors of a dataset root (empty when none were written)."""

    path = priors_path(root)
    if not path.exists():
        return DatasetPriors()
    raw = _checked(json.loads(path.read_text()), TOP_KEYS, str(path))
    for group, block in raw.get("groups", {}).items():
        _checked(block, GROUP_KEYS, f"{path}: groups.{group}")
        _checked(block.get("match", {}), MATCH_KEYS, f"{path}: groups.{group}.match")
        if "stat" not in block.get("match", {}) or "columns" not in block["match"]:
            raise ValueError(f"{path}: groups.{group} needs a `match` with `stat` and `columns`")
    priors = _parsed(raw, str(path))
    for group in priors.groups:  # each group's fields parse
        priors.for_group(group)
    return priors


def _parsed(raw: JsonObject, path: str) -> DatasetPriors:
    """Priors from a priors file's (or a merged group's) raw fields; `path` names them in
    errors."""

    joints = tuple(
        JointMap(
            j["urdf_joint"],
            j["index"],
            float(j.get("sign", 1.0)),
            tuple(j["from"]) if j.get("from") else None,
            tuple(j["to"]) if j.get("to") else None,
        )
        for j in (_checked(j, JOINT_KEYS, f"{path}: joints") for j in raw.get("joints", []))
    )
    cameras = _cameras(raw.get("cameras", {}), str(path))
    bodies = _bodies(raw.get("bodies", {}), str(path))
    for station, override in raw.get("stations", {}).items():
        _checked(override, STATION_KEYS, f"{path}: stations.{station}")
    odometry = raw.get("base_odometry")
    if odometry is not None:
        _checked(odometry, _keys(BaseOdometry), f"{path}: base_odometry")
        odometry = BaseOdometry(
            odometry["kind"],
            _feature(odometry["feature"]),
            tuple(odometry["columns"]),
            odometry.get("radius"),
            odometry.get("half_track"),
            float(odometry.get("offset", 0.0)),
            tuple(odometry.get("signs", (1.0, 1.0, 1.0))),
            odometry.get("speed_feature"),
            tuple(odometry.get("speed_columns", ())),
            tuple(tuple(w) for w in odometry.get("wheels", ())),
            _gain(odometry.get("correction"), f"{path}: base_odometry.correction"),
        )
    return DatasetPriors(
        base=raw.get("base", "fixed"),
        base_odometry=odometry,
        bodies=bodies,
        joints=joints,
        removed_joints=tuple(raw.get("removed_joints", [])),
        state_shift=int(raw.get("state_shift", 0)),
        cameras=cameras,
        description=raw.get("description"),
        state_feature=_feature(raw.get("state_feature")),
        side_table=raw.get("side_table"),
        kinematics=_kinematics(raw.get("kinematics"), str(path)),
        robot_points=_robot_points(raw.get("robot_points", {}), str(path)),
        stations=raw.get("stations", {}),
        groups=raw.get("groups", {}),
        world_anchor=raw.get("world_anchor"),
        ee=_effectors(raw.get("ee", {}), str(path)),
        action=raw.get("action"),
        raw=raw,
        notes=raw.get("notes", ""),
    )


@overload
def _feature(value: str | list[str]) -> str | tuple[str, ...]: ...
@overload
def _feature(value: str | list[str] | None) -> str | tuple[str, ...] | None: ...
def _feature(value: str | list[str] | None) -> str | tuple[str, ...] | None:
    return tuple(value) if isinstance(value, list) else value


def _keys(cls: type) -> frozenset[str]:
    """The JSON keys of a priors dataclass: its field names."""

    return frozenset(f.name for f in fields(cls))


TOP_KEYS = _keys(DatasetPriors) - {"raw", "group"} | {"base_odometry_measured"}
JOINT_KEYS = frozenset({"index", "urdf_joint", "sign", "from", "to"})
STATION_KEYS = frozenset({"cameras", "bodies"})
# What a group may set (the robot and its cameras; not stations, odometry or other groups)
GROUP_KEYS = frozenset(
    {
        "match",
        "description",
        "joints",
        "removed_joints",
        "state_shift",
        "state_feature",
        "cameras",
        "bodies",
        "kinematics",
        "robot_points",
        "ee",
        "action",
        "notes",
    }
)
MATCH_KEYS = _keys(GroupMatch)
CAMERA_KEYS = _keys(CameraPrior) | {"fitted"}
BODY_KEYS = _keys(BodyPrior) | {"fitted"}
POSE_KEYS = frozenset({"position", "orientation"})
RANGE_KEYS = frozenset({"position_range", "orientation_range_deg"})


def _checked(raw: JsonObject, allowed: frozenset[str], where: str) -> JsonObject:
    """`raw`, after rejecting keys outside `allowed`."""

    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            f"{where}: unknown priors keys {sorted(unknown)}; known: {sorted(allowed)}"
        )
    return raw


def _kinematics(raw: JsonObject | None, where: str) -> KinematicsPrior | None:
    if not raw:
        return None
    _checked(raw, _keys(KinematicsPrior) | {"fitted"}, f"{where}: kinematics")
    joints = raw.get("joints")
    return KinematicsPrior(
        float(raw.get("joint_offset_deg", 0.0)),
        float(raw.get("joint_offset_mm", 0.0)),
        None if joints is None else tuple(joints),
        float(raw.get("time_offset_frames", 0.0)),
        {str(k): float(v) for k, v in raw.get("offsets", {}).items()},
        float(raw.get("time_offset", 0.0)),
    )


def _point(raw: JsonObject, where: str) -> RobotPoint:
    _checked(raw, _keys(RobotPoint), where)
    return RobotPoint(raw["body"], tuple(raw.get("offset", (0.0, 0.0, 0.0))))


def _effectors(raw: Mapping[str, JsonObject], where: str) -> dict[str, EffectorPrior]:
    out = {}
    for arm, e in raw.items():
        at = f"{where}: ee.{arm}"
        _checked(e, _keys(EffectorPrior), at)
        pads = e.get("pads")
        if pads is not None and len(pads) != 2:
            raise ValueError(f"{at}.pads: two points (one per jaw)")
        tips = e.get("fingertips")
        actuation = e.get("actuation")
        if actuation is not None:
            _checked(actuation, _keys(ActuationColumn), f"{at}.actuation")
        out[arm] = EffectorPrior(
            preset=e.get("preset"),
            type=e.get("type"),
            body=e.get("body"),
            position=tuple(e["position"]) if e.get("position") is not None else None,
            orientation=tuple(e["orientation"]) if e.get("orientation") is not None else None,
            pads=None if pads is None else (_point(pads[0], at), _point(pads[1], at)),
            closed=None if e.get("closed") is None else float(e["closed"]),
            fingertips=None
            if tips is None
            else {n: _point(p, f"{at}.fingertips.{n}") for n, p in tips.items()},
            actuation=None
            if actuation is None
            else ActuationColumn(actuation["feature"], int(actuation.get("index", 0))),
        )
    return out


def _gain(raw: JsonObject | None, where: str) -> OdometryGain | None:
    if raw is None:
        return None
    _checked(raw, _keys(OdometryGain) | {"fitted"}, where)
    return OdometryGain(
        float(raw.get("scale", 1.0)),
        float(raw.get("heading_gain", 1.0)),
        float(raw.get("yaw_rate", 0.0)),
    )


def _robot_points(raw: Mapping[str, JsonObject], where: str) -> dict[str, RobotPoint]:
    for name, point in raw.items():
        _checked(point, _keys(RobotPoint), f"{where}: robot_points.{name}")
    return {
        name: RobotPoint(point["body"], tuple(point.get("offset", (0.0, 0.0, 0.0))))
        for name, point in raw.items()
    }


def _intrinsics_column(raw: JsonObject | None, where: str) -> IntrinsicsColumn | None:
    if raw is None:
        return None
    _checked(raw, _keys(IntrinsicsColumn), f"{where}.intrinsics")
    return IntrinsicsColumn(
        raw["column"],
        raw.get("layout", "fx_fy_cx_cy"),
        tuple(raw["size"]) if raw.get("size") else None,
        raw.get("resize", "stretch"),
    )


def _pose_column(raw: JsonObject | None, where: str) -> PoseColumn | None:
    if raw is None:
        return None
    _checked(raw, _keys(PoseColumn), f"{where}.pose")
    return PoseColumn(
        raw["column"],
        tuple(raw.get("path", ())),
        raw.get("axes", "opencv"),
        raw.get("layout", "record"),
        int(raw.get("start", 0)),
        raw.get("order", "matrix"),
        tuple(raw.get("offset", (0.0, 0.0, 0.0))),
    )


def _cameras(raw: Mapping[str, JsonObject], where: str) -> dict[str, CameraPrior]:
    for name, c in raw.items():
        _checked(c, CAMERA_KEYS, f"{where}: cameras.{name}")
        if set(c) & POSE_KEYS and not set(c) >= RANGE_KEYS:
            logger.warning(
                "%s: cameras.%s gives a pose without both ranges (%s): it is only a start, free",
                where,
                name,
                ", ".join(sorted(RANGE_KEYS)),
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
            aspect=float(c.get("aspect", 1.0)),
            distortion=tuple(c["distortion"]) if c.get("distortion") is not None else None,
            distortion_range=c.get("distortion_range"),
            scope=c.get("scope", "dataset"),
            intrinsics=_intrinsics_column(c.get("intrinsics"), f"{where}: cameras.{name}"),
            pose=_pose_column(c.get("pose"), f"{where}: cameras.{name}"),
            ignore=bool(c.get("ignore", False)),
        )
        for name, c in raw.items()
    }
    for name, camera in cameras.items():
        _check_lens(camera, f"{where}: cameras.{name}")
    return cameras


# Coefficients of each lens model (OpenCV radial k1..k3, Kannala–Brandt k1..k4)
LENS_COEFFICIENTS = {"pinhole": 3, "fisheye": 4}


def _check_lens(camera: CameraPrior, where: str) -> None:
    """Reject lens settings the fit cannot use: unknown models, too many or non-finite
    coefficients, a negative or non-finite `distortion_range`, and a static camera whose
    distortion is refined while its focal is unknown (focal, distortion and distance trade
    off: nothing separates them)."""

    size = LENS_COEFFICIENTS.get(camera.lens)
    if size is None:
        raise ValueError(f"{where}: lens {camera.lens!r}; known: {sorted(LENS_COEFFICIENTS)}")
    given = camera.distortion or ()
    if len(given) > size or not all(math.isfinite(k) for k in given):
        raise ValueError(
            f"{where}: distortion {list(given)}: a {camera.lens} lens takes at most {size} finite"
            " coefficients"
        )
    spread = camera.distortion_range
    if spread is not None and not (math.isfinite(spread) and spread >= 0):
        raise ValueError(f"{where}: distortion_range {spread}: a finite std ≥ 0 (0 holds)")
    # As `rago.lens.lens_from_priors`: a fisheye without coefficients is refined by default
    unknown = camera.lens == "fisheye" and camera.distortion is None
    refined = spread > 0 if spread is not None else unknown
    known_focal = camera.focal is not None or camera.intrinsics is not None
    if camera.mount == "static" and refined and not known_focal:
        raise ValueError(
            f"{where}: a static camera with refined distortion needs a known `focal` (focal,"
            " distortion and distance trade off); give the focal or hold the distortion"
            " (`distortion_range: 0`)"
        )


def _bodies(raw: Mapping[str, JsonObject], where: str) -> dict[str, BodyPrior]:
    for name, b in raw.items():
        _checked(b, BODY_KEYS, f"{where}: bodies.{name}")
    return {
        name: BodyPrior(
            float(b["position_range"]),
            float(b["orientation_range_deg"]),
            b.get("scope", "dataset"),
            tuple(b["position"]) if b.get("position") else None,
            tuple(b["orientation"]) if b.get("orientation") else None,
        )
        for name, b in raw.items()
    }


def _merged(
    base: Mapping[str, JsonObject], override: Mapping[str, JsonObject]
) -> dict[str, JsonObject]:
    """`base` entries with `override`'s fields set on top (entry by entry)."""

    out = {name: dict(values) for name, values in base.items()}
    for name, values in override.items():
        out.setdefault(name, {}).update(values)
    return out


@dataclass(frozen=True, slots=True)
class Pick:
    """Pixels of robot points clicked in one frame (the video's own frame index) of one camera
    in one episode: point name → (x, y), pixel centres at integer coordinates."""

    episode: int
    camera: str
    frame: int
    pixels: dict[str, tuple[float, float]]


def read_picks(path: Path) -> tuple[dict[str, RobotPoint], tuple[Pick, ...]]:
    """A picks file: points it names (as the priors' `robot_points`) and its picks.

    ```json
    {"points": {"tip": {"body": "panda_hand", "offset": [0, 0, 0.105]}},
     "picks": [{"episode": 0, "camera": "front", "frame": 40,
                "pixels": {"tip": [120.5, 88.0], "panda_link4": [101.0, 60.5]}}]}
    ```
    """

    raw = _checked(json.loads(path.read_text()), frozenset({"points", "picks"}), str(path))
    picks = tuple(
        Pick(
            int(p["episode"]),
            p["camera"],
            int(p["frame"]),
            {name: (float(xy[0]), float(xy[1])) for name, xy in p["pixels"].items()},
        )
        for p in (_checked(p, _keys(Pick), f"{path}: picks") for p in raw.get("picks", []))
    )
    return _robot_points(raw.get("points", {}), str(path)), picks


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """Known per-episode values: the station (setup), each camera's intrinsics
    `(fx, fy, cx, cy)` at the video resolution and, for static cameras, a recorded pose
    `(x, y, z, qw, qx, qy, qz)` in the robot base frame (OpenCV axes; e.g. the dataset's own
    per-episode calibration), the start of the rig's refinement."""

    station: str | None
    intrinsics: dict[str, tuple[float, float, float, float]]
    poses: dict[str, tuple[float, ...]] = field(default_factory=dict)


def episode_table_path(root: Path) -> Path:
    """`priors/<family>/<dataset>.episodes.parquet` next to the dataset's priors."""

    return priors_path(root).with_suffix(".episodes.parquet")


def load_episode_record(root: Path, episode: int) -> EpisodeRecord | None:
    """One episode's known values; None when the dataset has no table or no row for it.

    The table has one row per (episode, camera): `episode, camera`, then `fx, fy, cx, cy`,
    an optional `station` and optional pose columns `x, y, z, qw, qx, qy, qz`; null values
    mean unknown (a row may carry only a pose, or only intrinsics).
    """

    path = episode_table_path(root)
    if not path.exists():
        return None
    import numpy as np

    table, episodes = _episode_table(path, path.stat().st_mtime_ns)
    first, last = (int(i) for i in np.searchsorted(episodes, [episode, episode + 1]))
    if first == last:
        return None
    table = table.slice(first, last - first)
    rows = table.to_pylist()
    station = rows[0].get("station")
    pose = ("x", "y", "z", "qw", "qx", "qy", "qz")
    return EpisodeRecord(
        None if station is None else str(station),
        {
            r["camera"]: (r["fx"], r["fy"], r["cx"], r["cy"])
            for r in rows
            if r.get("fx") is not None
        },
        {
            r["camera"]: tuple(float(r[k]) for k in pose)
            for r in rows
            if all(r.get(k) is not None for k in pose)
        },
    )


@functools.lru_cache(maxsize=4)
def _episode_table(path: Path, modified: int) -> tuple[pa.Table, np.ndarray]:
    """A dataset's episodes table sorted by episode, with its `episode` column, read once per
    process (and again only if the file changes: `modified` is its mtime)."""

    import pyarrow.parquet as pq

    del modified  # part of the cache key only
    table = pq.read_table(path).sort_by("episode")
    return table, table["episode"].to_numpy()


def write_camera(
    root: Path,
    camera: str,
    values: Mapping[str, object],
    station: str | None = None,
    group: str | None = None,
) -> Path:
    """Set fields of one camera's priors (e.g. a fitted mount), keeping everything else; under
    `stations.<station>` when a station is given, else under `groups.<group>` when a group is."""

    return _write(root, "cameras", camera, values, station, group)


def write_odometry_correction(root: Path, values: Mapping[str, object]) -> Path:
    """Set the priors' `base_odometry.correction` (a fitted `OdometryGain` and its provenance)."""

    path = priors_path(root)
    raw = json.loads(path.read_text())
    raw["base_odometry"]["correction"] = dict(values)
    path.write_text(json.dumps(raw, indent=2) + "\n")
    return path


def write_body(
    root: Path,
    body: str,
    values: Mapping[str, object],
    station: str | None = None,
    group: str | None = None,
) -> Path:
    """Set fields of one free body's priors (e.g. a fitted correction), keeping the rest; where
    as `write_camera`."""

    return _write(root, "bodies", body, values, station, group)


def _write(
    root: Path,
    kind: str,
    name: str,
    values: Mapping[str, object],
    station: str | None,
    group: str | None,
) -> Path:
    path = priors_path(root)
    raw = json.loads(path.read_text()) if path.exists() else {}
    target = raw
    if station is not None:
        target = raw.setdefault("stations", {}).setdefault(station, {})
    elif group is not None:
        target = raw["groups"][group]
    target.setdefault(kind, {}).setdefault(name, {}).update(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, indent=2) + "\n")
    return path
