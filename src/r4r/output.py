"""r4r's per-frame output (docs/output-spec.md): column names, the pose layout and the tables.

Every value is per video frame, in the **episode frame** (the robot base frame at the episode's
first frame, `E = W_base(0)`), metres and radians:

    pose(X)_t = E⁻¹ · world_from_X_t  →  [x, y, z, qx, qy, qz, qw],  qw ≥ 0

`Layout` is what a dataset's episodes share (cameras, end effectors, joints): it names the
columns and carries each feature's `info`; `frame_table` writes one episode's rows.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import torch
from einops import rearrange

from r4r.effectors import aperture, ee_pose, fingertips, has_aperture, opening
from r4r.geometry import se3

if TYPE_CHECKING:
    from shape_extensions import IntTuple, IntVar
    from torch import Tensor

    from r4r.data.episode import JointMap
    from r4r.effectors import Effector
    from r4r.types import K1K2K3K4, Mat3x3, Mat4x4

POSE_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw")
INTRINSICS_NAMES = ("fx", "fy", "cx", "cy")
# Pinhole: OpenCV radial k1, k2, k3 (no tangential), the fourth entry 0; fisheye: OpenCV fisheye
# (Kannala–Brandt) k1..k4
DISTORTION_NAMES = {"pinhole": ("k1", "k2", "k3", "zero"), "fisheye": ("k1", "k2", "k3", "k4")}
# Schema metadata of a reconstruction shard: the dataset's `Layout` (JSON)
LAYOUT_KEY = b"r4r.layout"


def pose_vector[Bs: IntTuple](matrix: Tensor[[*Bs, 4, 4]]) -> Tensor[[*Bs, 7]]:
    """`[x, y, z, qx, qy, qz, qw]` of rigid transforms, the quaternion normalised with `qw ≥ 0`
    (`q` and `−q` are the same rotation: one sign makes the layout unique)."""

    q = se3.matrix_to_quaternion(matrix[..., :3, :3])
    q = q * torch.where(q[..., :1] < 0, -1.0, 1.0).to(q)
    q = q / q.norm(dim=-1, keepdim=True)
    return torch.cat((matrix[..., :3, 3], q[..., 1:], q[..., :1]), -1)


def pose_matrix[Bs: IntTuple](vector: Tensor[[*Bs, 7]]) -> Tensor[[*Bs, 4, 4]]:
    """The rigid transform of a `[x, y, z, qx, qy, qz, qw]` pose (inverse of `pose_vector`)."""

    q = torch.cat((vector[..., 6:], vector[..., 3:6]), -1)
    return se3.from_rotation_translation(se3.quaternion_to_matrix(q), vector[..., :3])


@dataclass(frozen=True, slots=True)
class CameraInfo:
    """A camera's column suffix and the facts its video feature's info states."""

    name: str
    lens: str  # pinhole | fisheye
    mount: str  # static | moving
    role: str


@dataclass(frozen=True, slots=True)
class Layout:
    """What every episode of a dataset shares: the columns and their feature info."""

    cameras: tuple[CameraInfo, ...]
    effectors: tuple[Effector, ...]
    # Description joints the state drives, in description order (`state_aligned`)
    joints: tuple[str, ...]
    mobile: bool
    action: bool  # whether `action_aligned` is written
    # Per `state_aligned` entry: its raw state column and the map to it (reversible)
    mapping: tuple[dict[str, object], ...] = field(default=())
    # The lag correction: `state_shift` (frames, whole) and the fitted `time_offset` (frames)
    lag: dict[str, float] = field(default_factory=dict)
    # Cameras the priors ignore (worn by a person, not placeable): their columns are written,
    # NaN in every frame, so every camera of the dataset is present (`camera.calibrated` false)
    ignored: tuple[CameraInfo, ...] = ()
    # False for a mobile base whose motion is not recorded: `observation.base.pose` is NaN in
    # every frame (the episode frame is then the base at each frame, cameras on the robot)
    base_pose: bool = True

    def features(self) -> dict[str, dict[str, object]]:
        """LeRobot feature entries (`dtype`, `shape`, `names`, `info`) of the r4r columns."""

        def entry(shape: list[int], names: object, info: dict[str, object] | None = None) -> dict:
            return {"dtype": "float32", "shape": shape, "names": names, "info": info or {}}

        out = {}
        for c in (*self.cameras, *self.ignored):
            prefix = f"observation.camera.{c.name}"
            out[f"{prefix}.pose"] = entry([7], list(POSE_NAMES), {"frame": "episode"})
            out[f"{prefix}.intrinsics"] = entry([4], list(INTRINSICS_NAMES))
            out[f"{prefix}.distortion"] = entry(
                [4], list(DISTORTION_NAMES[c.lens]), {"lens": c.lens}
            )
        for e in self.effectors:
            prefix = f"observation.ee.{e.arm}"
            info: dict[str, object] = {
                "ee.type": e.type,
                "ee.description_body": e.body_name,
                "ee.preset": e.preset,
                "ee.state_indices": [self.joints.index(j) for j in e.joints if j in self.joints],
                "ee.state_names": [j for j in e.joints if j in self.joints],
                "frame": "episode",
            }
            out[f"{prefix}.pose"] = entry([7], list(POSE_NAMES), info)
            if e.type == "parallel" and len(e.pads) == 2:
                out[f"{prefix}.gripper"] = entry([1], ["opening"], {"unit": "m"})
            if has_aperture(e):
                kind = "jaw opening" if f"{prefix}.gripper" in out else "thumb-index distance"
                out[f"{prefix}.aperture"] = entry([1], ["aperture"], {"unit": "m", "kind": kind})
            if e.fingertip_names:
                tips = len(e.fingertip_names)
                out[f"{prefix}.fingertips"] = entry(
                    [tips, 3], {"fingers": list(e.fingertip_names), "axes": ["x", "y", "z"]}
                )
            if e.actuation is not None:
                out[f"{prefix}.actuation"] = entry([1], ["actuation"])
        out["observation.base.pose"] = entry(
            [7], list(POSE_NAMES), {"frame": "episode", "available": self.base_pose}
        )
        joint_info: dict[str, object] = {"unit": "rad or m (description joints)"}
        out["observation.state_aligned"] = entry([len(self.joints)], list(self.joints), joint_info)
        if self.action:
            out["action_aligned"] = entry([len(self.joints)], list(self.joints), joint_info)
        return out

    def camera_info(self) -> dict[str, dict[str, object]]:
        """The `info` entries r4r sets on each camera's video feature."""

        return {
            f"observation.images.{c.name}": {
                "camera.lens": c.lens,
                "camera.extrinsics_frame": "episode",
                "camera.mount": c.mount,
                "camera.role": c.role,
                "camera.depth_key": None,
                "camera.calibrated": c not in self.ignored,
            }
            for c in (*self.cameras, *self.ignored)
        }

    def to_json(self) -> str:
        """The layout as shard metadata: features, camera info, the state mapping, robot."""

        return json.dumps(
            {
                "features": self.features(),
                "cameras": self.camera_info(),
                "state_mapping": list(self.mapping),
                "lag_frames": self.lag,
                "action_aligned": self.action,
                "robot": {"mobile": self.mobile, "base_pose": self.base_pose},
            }
        )


def state_mapping(
    joints: tuple[str, ...], maps: tuple[JointMap, ...], offsets: dict[str, float]
) -> tuple[dict, ...]:
    """Per aligned joint, the raw column and its map `q = t0 + (σ s − s0)(t1 − t0)/(s1 − s0)`
    (or `q = σ s`), plus r4r's fitted offset `δq` (`q' = q + δq`): enough to go back to the raw
    state (the lag is in `Layout.lag`)."""

    by_name = {m.name: m for m in maps}
    return tuple(
        {
            "joint": j,
            "column": by_name[j].index,
            "sign": by_name[j].sign,
            "from": by_name[j].source,
            "to": by_name[j].target,
            "offset": offsets.get(j, 0.0),
        }
        for j in joints
    )


@dataclass(frozen=True, slots=True)
class EpisodeOutput:
    """One episode's values at its `T` frames, in the world frame of the solution (`E` is
    applied by `frame_table`)."""

    episode: int
    frames: Tensor[[int]]  # frame index of each row
    timestamps: Tensor[[int]]
    world_from_camera: Tensor[[int, int, *Mat4x4]]  # [C, T]
    intrinsics: Tensor[[int, *Mat3x3]]  # [C]
    distortion: Tensor[[int, *K1K2K3K4]]  # [C]
    world_from_body: Tensor[[int, int, *Mat4x4]]  # [T, L]
    base: int | None  # the moving base body, if any
    state: Tensor[[int, int]]  # [T, J]: `state_aligned`
    action: Tensor[[int, int]] | None  # [T, J]: `action_aligned`
    actuation: dict[str, Tensor[[int]]] = field(default_factory=dict)
    # Cameras (indices) not calibrated in this episode: NaN pose, intrinsics and distortion
    uncalibrated: tuple[int, ...] = ()


def _vectors(values: Tensor[[int, int]]) -> pa.Array:
    """Rows of a [N, k] tensor as a fixed-size list column of float32."""

    flat = pa.array(rearrange(values.detach().cpu().float(), "n k -> (n k)").numpy())
    return pa.FixedSizeListArray.from_arrays(flat, values.shape[1])


def _scalars(values: Tensor[[int]]) -> pa.Array:
    """A [N] tensor as a float32 column (LeRobot stores shape-[1] features unwrapped)."""

    return pa.array(values.detach().cpu().float().numpy())


def frame_table(layout: Layout, out: EpisodeOutput) -> pa.Table:
    """One row per frame with every column of `layout` (module doc)."""

    world = out.world_from_body.double()
    steps = len(world)
    eye = torch.eye(4, dtype=torch.float64)
    episode_from_world = eye if out.base is None else se3.inverse(world[0, out.base])

    def pose(matrix: Tensor[[int, *Mat4x4]]) -> pa.Array:
        return _vectors(pose_vector(episode_from_world @ matrix.double()))

    columns: dict[str, pa.Array] = {
        "episode_index": pa.array(np.full(steps, out.episode, dtype=np.int64)),
        "frame_index": pa.array(out.frames.numpy().astype(np.int64)),
        "timestamp": pa.array(out.timestamps.numpy().astype(np.float32)),
    }
    for c, camera in enumerate(layout.cameras):
        k = out.intrinsics[c].double()
        fx_fy_cx_cy = torch.stack((k[0, 0], k[1, 1], k[0, 2], k[1, 2]))
        distortion = out.distortion[c].double()
        world_from_camera = out.world_from_camera[c].double()
        if c in out.uncalibrated:
            fx_fy_cx_cy, distortion = fx_fy_cx_cy.nan_to_num() * math.nan, distortion * math.nan
            world_from_camera = world_from_camera * math.nan
        prefix = f"observation.camera.{camera.name}"
        columns[f"{prefix}.pose"] = pose(world_from_camera)
        columns[f"{prefix}.intrinsics"] = _vectors(fx_fy_cx_cy.expand(steps, 4))
        columns[f"{prefix}.distortion"] = _vectors(distortion.expand(steps, 4))
    unknown = torch.full((steps, 4), math.nan, dtype=torch.float64)
    for camera in layout.ignored:
        prefix = f"observation.camera.{camera.name}"
        columns[f"{prefix}.pose"] = _vectors(torch.full((steps, 7), math.nan))
        columns[f"{prefix}.intrinsics"] = _vectors(unknown)
        columns[f"{prefix}.distortion"] = _vectors(unknown)
    for e in layout.effectors:
        prefix = f"observation.ee.{e.arm}"
        columns[f"{prefix}.pose"] = pose(ee_pose(e, world))
        width = opening(e, world)
        if width is not None:
            columns[f"{prefix}.gripper"] = _scalars(width)
        pinch = aperture(e, world)
        if pinch is not None:
            columns[f"{prefix}.aperture"] = _scalars(pinch)
        if e.fingertip_names:
            tips = fingertips(e, world)
            tips = (tips @ episode_from_world[:3, :3].T) + episode_from_world[:3, 3]
            inner = _vectors(rearrange(tips, "t f k -> (t f) k"))
            columns[f"{prefix}.fingertips"] = pa.FixedSizeListArray.from_arrays(
                inner, len(e.fingertip_names)
            )
        if e.actuation is not None:
            columns[f"{prefix}.actuation"] = _scalars(out.actuation[e.arm])
    base = eye.expand(steps, 4, 4) if out.base is None else world[:, out.base]
    columns["observation.base.pose"] = (
        pose(base) if layout.base_pose else _vectors(torch.full((steps, 7), math.nan))
    )
    columns["observation.state_aligned"] = _vectors(out.state)
    if layout.action:
        assert out.action is not None
        columns["action_aligned"] = _vectors(out.action)
    return pa.table(columns)


def aligned_joints(joint_names: tuple[str, ...], maps: tuple[JointMap, ...]) -> tuple[str, ...]:
    """The description joints a state drives, in description order."""

    driven = {m.name for m in maps}
    return tuple(j for j in joint_names if j in driven)


def select_joints[T: IntVar](
    qpos: Tensor[[T, int]],
    joint_names: tuple[str, ...],
    qpos_index: tuple[int, ...],
    joints: tuple[str, ...],
) -> Tensor[[T, int]]:
    """The columns of `qpos` of `joints` (description names)."""

    return qpos[:, [qpos_index[joint_names.index(j)] for j in joints]]
