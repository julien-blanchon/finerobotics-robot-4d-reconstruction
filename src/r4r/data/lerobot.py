"""Load one episode of a LeRobot v3 dataset (finerobotics layout) into an `Episode`.

The only module that imports LeRobot (optional `lerobot` group), for video decoding only: the
tables are read with pyarrow (`tables.py`), only the rows and columns an episode needs, and a
root's metadata and priors once per process. finerobotics roots ship the robot description in
`meta/robot/` and `meta/robot.json`, which maps state columns to description joints and may
name camera links; camera features declare their role, mount (static/dynamic) and side. The
dataset's priors (`priors.py`) override all of these.
"""

from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from einops import rearrange

from ..geometry import camera, se3
from .episode import CameraStream, Episode, JointMap
from .priors import (
    PRIORS,
    BaseOdometry,
    CameraPrior,
    DatasetPriors,
    EpisodeRecord,
    IntrinsicsColumn,
    JsonObject,
    PoseColumn,
    load_episode_record,
    load_priors,
)
from .tables import Column, EpisodeFiles, LeRobotTables

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np
    from torch import Tensor

    from r4r.data import priors as checked_priors
    from r4r.types import RGB, Mat3x3, Mat4x4

IMAGE_PREFIX = "observation.images."
# OpenCV camera axes (x right, y down, z forward) in x-forward, y-left, z-up camera axes
FORWARD_LEFT_UP_FROM_OPENCV = torch.tensor(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float64
)


@dataclass(frozen=True, slots=True)
class _Root:
    """What every episode of a root shares: its tables, `meta/robot.json` and priors."""

    tables: LeRobotTables
    robot: JsonObject
    priors: DatasetPriors


@functools.cache
def _root(root: Path) -> _Root:
    """A dataset root opened once per process (episodes of a run reuse its reads).

    A root without `meta/robot.json` (kai0) gets an empty one: its priors then name the
    description and every joint.
    """

    path = root / "meta" / "robot.json"
    robot = json.loads(path.read_text()) if path.exists() else {}
    priors = load_priors(root)
    return _Root(LeRobotTables(root, side=priors.side_directory), robot, priors)


@functools.cache
def _groups(root: Path) -> tuple[np.ndarray, list[str | None]]:
    """Each episode's priors group (`priors.groups` and their `match`): the episode indices
    and their groups (None: no group matched), in episode order."""

    import numpy as np

    opened = _root(root)
    matches = opened.priors.group_matches()
    stats = {m.stat for conditions in matches.values() for m in conditions}
    statistics = {s: opened.tables.episode_statistic(s) for s in stats}
    episodes = next(iter(statistics.values()))[0]
    names = np.full(len(episodes), None, dtype=object)
    for name, conditions in matches.items():  # file order: the first matching group takes it
        inside = np.equal(names, None)
        for match in conditions:
            inside &= match.matches(statistics[match.stat][1])
        names[inside] = name
    return episodes, names.tolist()


def episode_group(root: Path, episode: int) -> str | None:
    """The priors group `episode` belongs to (None: the dataset-level priors)."""

    import numpy as np

    if not _root(root).priors.groups:
        return None
    episodes, names = _groups(root)
    i = int(np.searchsorted(episodes, episode))
    return names[i] if i < len(episodes) and episodes[i] == episode else None


@functools.cache
def _episode_priors(
    root: Path,
    group: str | None,
    station: str | None,
    camera_stations: tuple[tuple[str, str], ...] = (),
) -> DatasetPriors:
    """The priors of an episode of `group`, `station` and per-camera stations (camera,
    station) (parsed once per process)."""

    return _root(root).priors.for_episode(group, station, dict(camera_stations))


def episode_count(root: Path) -> int:
    """How many episodes the dataset root holds (`meta/info.json`)."""

    return _root(root).tables.total_episodes


def dataset_features(root: Path) -> JsonObject:
    """The root's features (`meta/info.json`) with those of its priors' side table."""

    return _root(root).tables.features


def camera_names(root: Path) -> tuple[tuple[str, ...], frozenset[str]]:
    """The cameras every command uses (video features that are not depth maps, not ignored by
    the priors) and those of them that move, as `load_episode` decides (dataset-level priors)."""

    opened = _root(root)
    names, moving = [], set()
    for key in opened.tables.video_keys:
        info = opened.tables.features[key].get("info", {})
        name = key.removeprefix(IMAGE_PREFIX)
        prior = opened.priors.cameras.get(name, CameraPrior())
        if info.get("is_depth_map", False) or prior.ignore:
            continue
        names.append(name)
        stream = _camera(key, info, None, prior, torch.empty(0, 3, 0, 0), (0, 0))
        if stream.moving:
            moving.add(name)
    return tuple(names), frozenset(moving)


@functools.cache
def simulated(root: Path) -> bool | None:
    """Whether the dataset root is simulated (its `meta/episodes`; None when unsaid)."""

    return _root(root).tables.simulated()


def load_episode(
    root: Path, episode: int, stride: int = 1, decode: tuple[str, ...] | None = None
) -> Episode[int, int]:
    """Every `stride`-th timestep of `episode` from the dataset root `root`.

    Only the cameras in `decode` get frames (all when None); the others come with none.
    """

    opened = _root(root)
    tables, robot = opened.tables, opened.robot
    record = load_episode_record(root, episode)
    # The episode's group (a dataset mixing robots), then its station's own mounts and bodies
    # (priors `groups`, `stations`) on top of the dataset's, then each camera's own station
    group = episode_group(root, episode)
    own = {} if record is None else record.camera_stations
    priors = _episode_priors(
        root, group, None if record is None else record.station, tuple(sorted(own.items()))
    )
    keys = [
        key
        for key in tables.video_keys
        if not tables.features[key].get("info", {}).get("is_depth_map", False)
        and not priors.cameras.get(key.removeprefix(IMAGE_PREFIX), CameraPrior()).ignore
    ]
    feature = priors.state_feature or _state_feature(robot)
    every = tables.rows(episode, _needed(keys, feature, priors, record))
    timestamps = cast("Tensor", every["timestamp"])[::stride].tolist()

    links = {c["key"]: c["urdf_link"] for c in robot.get("cameras", [])}
    files = tables.files(episode) if decode is None or decode else None
    cameras = []
    for key in keys:
        name = key.removeprefix(IMAGE_PREFIX)
        prior = priors.cameras.get(name, CameraPrior())
        info = tables.features[key].get("info", {})
        height, width = tables.features[key]["shape"][:2]
        frames = torch.empty(0, 3, 0, 0)
        if decode is None or name in decode:
            assert files is not None
            frames = _decode(files, key, timestamps, tables.fps)
            height, width = frames.shape[-2:]
        stream = _camera(key, info, links.get(key), prior, frames, (height, width))
        if record is not None and name in record.intrinsics:
            fx, fy, cx, cy = record.intrinsics[name]
            k = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float64)
            stream = replace(stream, intrinsics=k)
        elif prior.intrinsics is not None:
            recorded = prior.intrinsics.column
            column = _columns(cast("Mapping[str, Tensor]", every), recorded)
            k = _recorded_intrinsics(column, prior.intrinsics, tables.features[key]["shape"][:2])
            stream = replace(stream, intrinsics=k)
        # A static camera's pose in the base frame, or a moving camera's mount on its link
        if record is not None and name in record.poses:
            x, y, z, *q = record.poses[name]
            rotation = se3.quaternion_to_matrix(torch.tensor(q, dtype=torch.float64))
            position = torch.tensor([x, y, z], dtype=torch.float64)
            stream = replace(stream, pose=se3.from_rotation_translation(rotation, position))
        elif prior.pose is not None and prior.pose.layout == "record":
            # The episode's first row: JSON text, or x, y, z, qw, qx, qy, qz
            text = cast("list[str]", every[prior.pose.column])[0]
            stream = replace(stream, pose=recorded_pose(text, prior.pose))
        cameras.append(stream)
    cameras = _world_cameras(cameras, every, priors, record, robot, root, feature)
    state, base_motion = _measurements(every, priors, feature, stride)
    return Episode(
        name=f"{root.name}/episode_{episode:06d}",
        fps=tables.fps / stride,
        state=state,
        joints=priors.joint_map(_joints(robot, priors.replaced)),
        robot=_description(root, robot, priors),
        cameras=tuple(cameras),
        priors=priors,
        station=None if record is None else record.station,
        camera_stations=own,
        base_motion=base_motion,
    )


def _world_cameras(
    cameras: list[CameraStream[int, int, int]],
    every: Mapping[str, Column],
    priors: DatasetPriors,
    record: EpisodeRecord | None,
    robot: JsonObject,
    root: Path,
    feature: str | tuple[str, ...],
) -> list[CameraStream[int, int, int]]:
    """`cameras` with those recorded in a world frame placed in the base frame (or on their
    link) for this episode (`r4r.data.recorded`); an episodes table pose takes precedence."""

    from .recorded import base_frame_poses, base_mounts, world_from_camera

    measured = {
        c.name: world_from_camera(cast("Tensor", every[c.prior.pose.column]), c.prior.pose)
        for c in cameras
        if c.prior.pose is not None
        and c.prior.pose.layout == "base_from_camera"
        and (record is None or c.name not in record.poses)
    }
    if measured:
        state = _columns(cast("Mapping[str, Tensor]", every), feature)
        joints = priors.joint_map(_joints(robot, priors.replaced))
        links = {c.name: c.mount if c.moving else None for c in cameras if c.name in measured}
        offsets = {
            c.name: c.prior.pose.offset for c in cameras if c.prior.pose and c.name in measured
        }
        placed = base_mounts(
            measured, links, offsets, state, joints, _description(root, robot, priors)
        )
        cameras = [replace(c, pose=placed[c.name]) if c.name in placed else c for c in cameras]
    world = {
        c.name: world_from_camera(cast("Tensor", every[c.prior.pose.column]), c.prior.pose)
        for c in cameras
        if c.prior.pose is not None
        and c.prior.pose.layout in ("world_to_camera", "world_from_camera")
        and (record is None or c.name not in record.poses)
    }
    if not world:
        return cameras
    anchor = priors.world_anchor
    if anchor is None or anchor not in world:
        raise ValueError(
            f"{root}: cameras recorded in a world frame need a `world_anchor` among them"
        )
    links = {c.name: c.mount if c.moving else None for c in cameras if c.name in world}
    state = _columns(cast("Mapping[str, Tensor]", every), feature)
    joints = priors.joint_map(_joints(robot, priors.replaced))
    placed = base_frame_poses(
        world, links, anchor, state, joints, _description(root, robot, priors)
    )
    return [replace(c, pose=placed[c.name]) if c.name in placed else c for c in cameras]


def _decode(
    files: EpisodeFiles, key: str, timestamps: list[float], fps: float
) -> Tensor[[int, *RGB, int, int]]:
    """uint8 frames of video `key` at each episode timestamp `t` (seconds), each within half a
    frame period (at the video's own rate, `EpisodeFiles.rates`)."""

    from lerobot.datasets.video_utils import decode_video_frames

    tolerance = files.rates.get(key, 1.0) / fps / 2
    times = files.times(key, timestamps)
    return decode_video_frames(files.videos[key][0], times, tolerance, "pyav", return_uint8=True)


def keyframes(
    root: Path, episode: int, frames: Sequence[int], cameras: Sequence[str]
) -> dict[str, Tensor[[int, *RGB, int, int]]]:
    """uint8 frames of `cameras` at the episode's frame indices `frames` only, each decoded from
    the nearest key frame before it: a few frames cost a few GOPs, not the whole video."""

    tables = _root(root).tables
    files = tables.files(episode)
    timestamps = cast("Tensor", tables.rows(episode, ["timestamp"])["timestamp"])
    out = {}
    for name in cameras:
        key = IMAGE_PREFIX + name
        times = files.times(key, [float(timestamps[f]) for f in frames])
        tolerance = 0.5 * files.rates.get(key, 1.0) / tables.fps
        out[name] = seek_frames(files.videos[key][0], times, tolerance)
    return out


def seek_frames(
    path: Path, times: Sequence[float], tolerance: float, gap: float = 1.0
) -> Tensor[[int, *RGB, int, int]]:
    """uint8 RGB frames of a video file at increasing `times` (seconds): the first frame at or
    after `time − tolerance` for each. Decoding seeks to the key frame before a time, then runs
    forward; a time less than `gap` seconds ahead is reached by decoding on, without a seek."""

    import av
    import numpy as np

    images = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        base = float(stream.time_base or 1.0)
        decoded = None  # the open decoder and the time of its last frame
        for time in times:
            if decoded is None or not decoded[1] < time - tolerance < decoded[1] + gap:
                container.seek(int(max(time - tolerance, 0.0) / base), stream=stream)
                decoded = (container.decode(stream), -math.inf)
            image = None
            for frame in decoded[0]:
                image = frame
                decoded = (decoded[0], frame.pts * base if frame.pts is not None else decoded[1])
                if decoded[1] >= time - tolerance:
                    break
            if image is None:
                raise ValueError(f"{path}: no frame at {time:.3f} s")
            images.append(image.to_ndarray(format="rgb24"))
    return rearrange(torch.from_numpy(np.stack(images)), "n h w c -> n c h w")


def _needed(
    keys: list[str],
    feature: str | tuple[str, ...],
    priors: DatasetPriors,
    record: EpisodeRecord | None,
) -> list[str]:
    """The data columns an episode needs: timestamps, the state, the base's odometry and the
    cameras' recorded intrinsics and poses that no episodes table gives."""

    names = ["timestamp", *((feature,) if isinstance(feature, str) else feature)]
    odometry = priors.base_odometry
    if odometry is not None:
        names += (odometry.feature,) if isinstance(odometry.feature, str) else odometry.feature
        names += [odometry.speed_feature] if odometry.speed_feature else []
    for key in keys:
        name = key.removeprefix(IMAGE_PREFIX)
        prior = priors.cameras.get(name, CameraPrior())
        if prior.intrinsics is not None and (record is None or name not in record.intrinsics):
            names.append(prior.intrinsics.column)
        if prior.pose is not None and (record is None or name not in record.poses):
            names.append(prior.pose.column)
    return list(dict.fromkeys(names))


def _measurements(
    every: Mapping[str, Column], priors: DatasetPriors, feature: str | tuple[str, ...], stride: int
) -> tuple[Tensor[[int, int]], Tensor[[int, *Mat4x4]] | None]:
    """The robot's state and base motion at every `stride`-th frame of the episode `every`.

    Both are measured with the joints: frame t's come from frame t + `state_shift` (a known
    video/joint-state offset, in frames at the dataset's rate whatever the stride), clamped.
    """

    count = len(every["timestamp"])
    index = (torch.arange(0, count, stride) + priors.state_shift).clamp(0, count - 1)
    # Text columns (e.g. JSON records) come as lists: only tensors are measurements
    rows = {name: column[index] for name, column in every.items() if torch.is_tensor(column)}
    motion = None if priors.base_odometry is None else _base_motion(priors, rows)
    return _columns(rows, feature), motion


def _recorded_intrinsics(
    values: Tensor[[int, int]], source: IntrinsicsColumn, size: tuple[int, int]
) -> Tensor[[*Mat3x3]]:
    """A camera's K at the video resolution `size` (height, width) from its recorded column
    (rows: flattened K or fx, fy, cx, cy; the episode's first finite, non-zero row is used).

    `stretch`: each axis resized, `f' = f · s`, `c' = (c + ½) · s − ½`, `s = video / given`;
    `height`: one scale `s = H / H_given` on both focals, the principal point at the image
    centre `((W − 1)/2, (H − 1)/2)` (same vertical FOV). Pixel centres at integer coordinates.
    """

    finite = values.isfinite().all(1) & (values.abs().sum(1) > 0)
    if not finite.any():
        raise ValueError(f"{source.column}: no finite, non-zero intrinsics in this episode")
    row = values[int(finite.nonzero()[0, 0])]
    if source.layout == "matrix":
        fx, fy, cx, cy = (float(row[i]) for i in (0, 4, 2, 5))
    else:
        fx, fy, cx, cy = (float(v) for v in row[:4])
    recorded = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float64)
    height, width = size
    given_width, given_height = source.size or (width, height)
    if source.resize == "stretch":
        return camera.resize_intrinsics(recorded, width / given_width, height / given_height)
    scale = height / given_height
    intrinsics = camera.resize_intrinsics(recorded, scale, scale)
    intrinsics[0, 2], intrinsics[1, 2] = (width - 1) / 2, (height - 1) / 2
    return intrinsics


def recorded_pose(text: str | Tensor[[int]], source: PoseColumn) -> Tensor[[*Mat4x4]]:
    """A camera's recorded `world_from_camera` (OpenCV axes) from one row of its column. A JSON
    string is walked along the source's path; a numeric row is x, y, z, qw, qx, qy, qz."""

    if isinstance(text, str):
        item = json.loads(text)
        for step in source.path:
            key, _, value = step.partition("=")
            item = next(i for i in item if str(i[key]) == value) if value else item[key]
    else:
        values = torch.as_tensor(text, dtype=torch.float64).flatten()
        item = {"position": values[:3].tolist(), "orientation": values[3:7].tolist()}
    q = torch.tensor(item["orientation"], dtype=torch.float64)
    rotation = se3.quaternion_to_matrix(q / q.norm())
    if source.axes == "forward_left_up":
        rotation = rotation @ FORWARD_LEFT_UP_FROM_OPENCV.to(rotation)
    position = torch.tensor(item["position"], dtype=torch.float64)
    return se3.from_rotation_translation(rotation, position)


def _description(root: Path, robot: JsonObject, priors: DatasetPriors) -> Path:
    """The robot description: the corrected one of the priors, else the delivered one."""

    if priors.description:
        return PRIORS / priors.description
    if "urdf" not in robot:
        raise ValueError(f"{root}: no meta/robot.json description: the priors must name one")
    return root / "meta" / robot["urdf"]


def _state_feature(robot: JsonObject) -> str:
    """The joint-state feature `robot.json` names (`observation.state` without one)."""

    return robot.get("state", {}).get("feature", "observation.state")


def _columns(rows: Mapping[str, Tensor], feature: str | tuple[str, ...]) -> Tensor[[int, int]]:
    """A vector feature's columns as float64 [T, n]; several features are concatenated.

    `rows` is a slice of the dataset: its columns have any rank (bare `Tensor`).
    """

    names = (feature,) if isinstance(feature, str) else feature
    return torch.cat([rows[f].double().reshape(len(rows[f]), -1) for f in names], dim=1)


def _base_motion(priors: DatasetPriors, rows: Mapping[str, Tensor]) -> Tensor[[int, *Mat4x4]]:
    """The base's recorded motion (see `r4r.rago.odometry`) at the loaded timesteps, with the
    priors' fitted correction (`base_odometry.correction`) when there is one."""

    from r4r.rago import odometry

    spec = priors.base_odometry
    assert spec is not None
    motion = _recorded_motion(spec, rows)
    if spec.correction is None:
        return motion
    time = rows["timestamp"].double()
    step = float(time[-1] - time[0]) / max(len(time) - 1, 1)
    # The same OdometryGain seen through r4r.rago's absolute import (one class at run time)
    return odometry.corrected(motion, cast("checked_priors.OdometryGain", spec.correction), step)


def _recorded_motion(spec: BaseOdometry, rows: Mapping[str, Tensor]) -> Tensor[[int, *Mat4x4]]:
    """The base's motion as recorded (no correction)."""

    from r4r.rago import odometry

    values = _columns(rows, spec.feature)[:, list(spec.columns)]
    if spec.kind == "pose" and values.shape[1] == 7:
        return odometry.from_poses_3d(values[:, :3], values[:, 3:])
    if spec.kind == "pose":
        # x, y, θ; or x, y, qz, qw (a planar heading as a quaternion about z)
        heading = (
            2 * torch.atan2(values[:, 2], values[:, 3]) if values.shape[1] == 4 else values[:, 2]
        )
        return odometry.from_poses(values[:, 0], values[:, 1], heading)
    if spec.kind == "velocity":
        return odometry.from_velocity(values, rows["timestamp"].double())
    if spec.kind == "swerve":
        speeds = _columns(rows, spec.speed_feature or spec.feature)[:, list(spec.speed_columns)]
        wheels = torch.tensor(spec.wheels, dtype=torch.float64)
        twist = odometry.swerve_twist(values, speeds, wheels)
        return odometry.from_velocity(twist, rows["timestamp"].double())
    assert spec.radius is not None and spec.half_track is not None, "caster: radius, half_track"
    geometry = odometry.CasterGeometry(spec.radius, spec.half_track, spec.offset, spec.signs)
    steering = values[:, 2] if values.shape[1] > 2 else None
    return odometry.from_steps(odometry.caster_steps(values[:, :2], steering, geometry))


def _camera(
    key: str,
    info: JsonObject,
    link: str | None,
    prior: CameraPrior,
    frames: Tensor[[int, *RGB, int, int]],
    size: tuple[int, int],
) -> CameraStream[int, int, int]:
    moving = info.get("camera.mount") == "dynamic"
    if prior.mount is not None:
        moving = prior.mount == "moving"
    # Metadata links are unverified; "world" or a frame name is not a mount
    declared = link or info.get("camera.extrinsics_frame")
    mount = prior.link or (declared if moving and declared != "world" else None)
    return CameraStream(
        name=key.removeprefix(IMAGE_PREFIX),
        frames=frames,
        role=prior.role or info.get("camera.role", "other"),
        moving=moving,
        mount=mount,
        side=info.get("camera.side"),
        prior=prior,
        size=size,
    )


def _joints(robot: JsonObject, replaced: frozenset[str] = frozenset()) -> tuple[JointMap, ...]:
    """Arm joints and gripper joints of every arm in `robot.json`, in the description's units.

    Two layouts: `state.arms` (arm joints, then `gripper` entries with their own range) and
    `state.slices` (each joint with its `role`, `unit` and optional `sign`). A joint's
    `joint_ranges` entry (column range → joint range) applies to any joint; without one,
    degrees are converted to radians, radians and metres pass through. `normalised_percent`
    has no fixed mapping (it depends on each robot's calibration): such joints need a mapping
    in the dataset's priors. Joints in `replaced` (set by the priors) are skipped.
    """

    state = robot.get("state", {})
    signs = state.get("joint_signs", {})
    ranges = state.get("joint_ranges", {})
    joints = []
    for arm in state.get("arms", state.get("slices", [])):
        for joint in arm["joints"]:
            name = joint["urdf_joint"]
            if name in replaced:
                continue
            sign = float(joint.get("sign", signs.get(name, 1.0)))
            unit = joint.get("unit", state.get("joint_unit", "rad"))
            if name in ranges:
                source, target = tuple(ranges[name]["from"]), tuple(ranges[name]["to"])
            elif unit == "deg":
                source, target = (0.0, 180.0), (0.0, math.pi)
            elif unit in ("rad", "m"):
                source = target = None
            else:
                raise ValueError(f"{name}: state unit {unit!r} needs a mapping in the priors")
            joints.append(JointMap(name, joint["index"], sign, source, target))
        for gripper in arm.get("gripper", []):
            # No range: the state is already in the joint's units
            source, target = (tuple(gripper[k]) if gripper[k] else None for k in ("from", "to"))
            joints.append(JointMap(gripper["urdf_joint"], gripper["index"], 1.0, source, target))
    return tuple(joints)


def state_feature(root: Path) -> str | tuple[str, ...]:
    """The joint-state feature of a root: the priors' `state_feature`, else robot.json's."""

    opened = _root(root)
    return opened.priors.state_feature or _state_feature(opened.robot)


def episode_columns(
    root: Path, episode: int, features: Sequence[str], stride: int = 1
) -> dict[str, Tensor[[int, int]]]:
    """Every `stride`-th row of numeric `features` of `episode`, as float64 [T, n] (no shift)."""

    rows = _root(root).tables.rows(episode, list(features))
    out = {}
    for name in features:
        column = rows[name]
        assert torch.is_tensor(column), f"{name}: not numeric"
        out[name] = _columns({name: column}, name)[::stride]
    return out


def joint_action(root: Path) -> bool:
    """Whether the dataset's `action` is an absolute joint target in the state's columns: the
    priors say so (`action: joint_absolute`), or its feature info does (`vector.space: joint`,
    `vector.reference: absolute`) with the state's names and shape."""

    opened = _root(root)
    if opened.priors.action == "joint_absolute":
        return True
    features = opened.tables.features
    state = opened.priors.state_feature or _state_feature(opened.robot)
    action = features.get("action")
    if action is None or not isinstance(state, str) or state not in features:
        return False
    info = action.get("info", {})
    return (
        info.get("vector.space") == "joint"
        and info.get("vector.reference") == "absolute"
        and action.get("names") == features[state].get("names")
        and action.get("shape") == features[state].get("shape")
    )


def frame_rate(root: Path) -> float:
    """The dataset root's frame rate (`meta/info.json`, frames per second)."""

    return _root(root).tables.fps
