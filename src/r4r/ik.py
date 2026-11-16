"""Joint states recovered by inverse kinematics, written as a side table (`r4r ik`).

A dataset that records each arm's end-effector pose but no joint state names, per arm, where
the pose is and what it is, in the priors' `ik` block:

```json
"ik": {"arm": {"feature": "observation.state", "position": [0, 1, 2], "scale": 1.0,
               "orientation": [3, 4, 5], "rotation": "rpy",
               "frame": {"body": "wx250s/base_link"}, "body": "wx250s/ee_gripper_link",
               "tool": {"orientation": [0.7071, 0, -0.7071, 0]},
               "joints": ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle",
                          "wrist_rotate"],
               "rest": [0, -0.3, 0.3, 0, 1.5, 0], "free_axis": null}}
```

- `position`: three state columns (x, y, z), or `{"value": v}` for a component the dataset
  does not record (a planar pusher's height); times `scale` (to metres).
- `orientation` and `rotation`: the columns and their kind: `rpy` (roll, pitch, yaw;
  `R = Rz(yaw) Ry(pitch) Rx(roll)`), `quat_xyzw`, `quat_wxyz`, `rotvec` (axis-angle), or
  `fixed` with no columns and the constant `value` (w, x, y, z).
- `frame`: the record's frame in the description: a pose on `body` (default: the world).
- `body` and `tool`: the recorded end-effector frame is `tool` (position, orientation w x y z)
  in `body`'s frame. Check it against forward kinematics of a dataset that has both.
- `joints`, `rest`: the chain IK moves and its rest pose (the redundancy is pulled there);
  `free_axis`: a tool-frame axis whose rotation the record leaves free; `limits: false` lets
  the joints past the description's limits where the record cannot be reached within them.

`r4r ik <dataset>` writes, per data file of the root, a parquet file into the priors'
`side_table` directory: `episode_index`, `frame_index`, `ik.<arm>` (float32, the joints in
`joints` order) and `ik.<arm>.error` (position mm, rotation deg). The priors then read them as
state: `"state_feature": ["observation.state", "ik.arm"]`, `joints` indexing past the
delivered columns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from r4r.geometry import se3
from r4r.rago.ik import Chain, IkConfig, branch, solve_ik
from r4r.rago.kinematics import forward_kinematics
from r4r.rago.robot import compile_model, load_robot

if TYPE_CHECKING:

    from torch import Tensor

    from r4r.data.priors import JsonObject
    from r4r.rago.robot import Robot
    from r4r.types import Mat4x4

type RotationKind = Literal["rpy", "quat_xyzw", "quat_wxyz", "rotvec", "fixed"]

# Episodes solved together in one batch
BATCH = 2048


@dataclass(frozen=True, slots=True)
class IkArm:
    """One arm's recorded end-effector pose and the chain that reaches it (module doc)."""

    feature: str
    position: tuple[int | float, int | float, int | float]  # column, or a constant (float)
    orientation: tuple[int, ...]
    rotation: RotationKind
    body: str
    joints: tuple[str, ...]
    rest: tuple[float, ...]
    scale: float = 1.0
    value: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    frame_body: str | None = None
    frame: tuple[tuple[float, ...], tuple[float, ...]] = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    tool: tuple[tuple[float, ...], tuple[float, ...]] = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    free_axis: tuple[float, float, float] | None = None
    # Hold the joints within the description's limits (off: the record wins where the
    # description's arm cannot reach it, e.g. a base placed slightly differently)
    limits: bool = True


def ik_arms(raw: JsonObject) -> dict[str, IkArm]:
    """The priors' `ik` block, one entry per arm."""

    allowed = {
        "feature", "position", "orientation", "rotation", "body", "joints", "rest", "scale",
        "value", "frame", "tool", "free_axis", "limits",
    }  # fmt: skip
    arms = {}
    for name, block in raw.items():
        unknown = set(block) - allowed
        if unknown:
            raise ValueError(f"ik.{name}: unknown keys {sorted(unknown)}")
        if len(block["rest"]) != len(block["joints"]):
            raise ValueError(f"ik.{name}: `rest` needs one value per joint")
        position = tuple(
            float(p["value"]) if isinstance(p, dict) else int(p) for p in block["position"]
        )
        frame, tool = block.get("frame", {}), block.get("tool", {})
        arms[name] = IkArm(
            feature=block["feature"],
            position=position,  # type: ignore[arg-type]  # three entries, checked by use
            orientation=tuple(block.get("orientation", ())),
            rotation=block["rotation"],
            body=block["body"],
            joints=tuple(block["joints"]),
            rest=tuple(float(v) for v in block["rest"]),
            scale=float(block.get("scale", 1.0)),
            value=tuple(block.get("value", (1.0, 0.0, 0.0, 0.0))),  # type: ignore[arg-type]
            frame_body=frame.get("body"),
            frame=_pose_fields(frame),
            tool=_pose_fields(tool),
            free_axis=tuple(block["free_axis"]) if block.get("free_axis") else None,  # type: ignore[arg-type]
            limits=bool(block.get("limits", True)),
        )
    return arms


def _pose_fields(raw: JsonObject) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return tuple(raw.get("position", (0.0, 0.0, 0.0))), tuple(
        raw.get("orientation", (1.0, 0.0, 0.0, 0.0))
    )


def _pose(fields: tuple[tuple[float, ...], tuple[float, ...]]) -> Tensor[[*Mat4x4]]:
    position, orientation = fields
    rotation = se3.quaternion_to_matrix(torch.tensor(orientation, dtype=torch.float64))
    return se3.from_rotation_translation(rotation, torch.tensor(position, dtype=torch.float64))


def recorded_rotation(values: Tensor[[int, int]], arm: IkArm) -> Tensor[[int, 3, 3]]:
    """Rotation matrices of the recorded orientation columns `values` `[T, k]`."""

    if arm.rotation == "fixed":
        value = torch.tensor(arm.value, dtype=torch.float64)
        return se3.quaternion_to_matrix(value).expand(len(values), 3, 3)
    if arm.rotation == "rpy":
        roll, pitch, yaw = values.unbind(1)
        return _about("z", yaw) @ _about("y", pitch) @ _about("x", roll)
    if arm.rotation == "rotvec":
        return se3.so3_exp(values)
    if arm.rotation == "quat_xyzw":
        values = values[:, [3, 0, 1, 2]]
    return se3.quaternion_to_matrix(values / values.norm(dim=1, keepdim=True))


def _about(axis: str, angle: Tensor[[int]]) -> Tensor[[int, 3, 3]]:
    c, s, one, zero = angle.cos(), angle.sin(), torch.ones_like(angle), torch.zeros_like(angle)
    rows = {
        "x": ((one, zero, zero), (zero, c, -s), (zero, s, c)),
        "y": ((c, zero, s), (zero, one, zero), (-s, zero, c)),
        "z": ((c, -s, zero), (s, c, zero), (zero, zero, one)),
    }[axis]
    return torch.stack([torch.stack(r, -1) for r in rows], -2)


def frame_from_record(state: Tensor[[int, int]], arm: IkArm) -> Tensor[[int, *Mat4x4]]:
    """The recorded end-effector poses in the record's frame, metres."""

    position = torch.stack(
        [
            state[:, p] if isinstance(p, int) else torch.full_like(state[:, 0], p)
            for p in arm.position
        ],
        1,
    )
    rotation = recorded_rotation(state[:, list(arm.orientation)], arm)
    return se3.from_rotation_translation(rotation, arm.scale * position)


@dataclass(frozen=True, slots=True)
class ArmModel:
    """An arm's chain on the description (cut to its branch) and where its record's frame
    sits in the world."""

    robot: Robot[int, int, int]
    chain: Chain[int]
    world_from_frame: Tensor[[*Mat4x4]]


def arm_model(description: Path, arm: IkArm) -> ArmModel:
    """The description compiled for `arm`'s IK: limits from MuJoCo, the record's frame at the
    rest pose (its body must not move with the chain)."""

    model = compile_model(description)
    full = load_robot(description)
    robot = branch(full, arm.body)
    joint = [full.joint_names.index(j) for j in arm.joints]  # MuJoCo's joint order
    limited = [arm.limits and bool(model.jnt_limited[j]) for j in joint]
    lower = [
        float(model.jnt_range[j][0]) if lim else -math.inf
        for j, lim in zip(joint, limited, strict=True)
    ]
    upper = [
        float(model.jnt_range[j][1]) if lim else math.inf
        for j, lim in zip(joint, limited, strict=True)
    ]
    chain = Chain(
        joints=arm.joints,
        qpos=tuple(full.joint_qpos[j] for j in joint),
        lower=torch.tensor(lower, dtype=torch.float64),
        upper=torch.tensor(upper, dtype=torch.float64),
        rest=torch.tensor(arm.rest, dtype=torch.float64),
        body=robot.body(arm.body),
        body_from_tool=_pose(arm.tool),
        free_axis=None
        if arm.free_axis is None
        else torch.tensor(arm.free_axis, dtype=torch.float64),
    )
    world_from_frame = _pose(arm.frame)
    if arm.frame_body is not None:
        rest = full.joint_reference.new_zeros(1, full.nq)
        rest[0, list(chain.qpos)] = chain.rest
        world_from_frame = forward_kinematics(full, rest)[0, full.body(arm.frame_body)] @ (
            world_from_frame
        )
    return ArmModel(robot, chain, world_from_frame)


def solve_episodes(
    model: ArmModel, arm: IkArm, states: list[Tensor[[int, int]]], config: IkConfig
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per episode (its state rows `[T, n]`): the joints `[T, J]` and errors `[T, 2]` (mm,
    deg), in the order of `states`. Episodes are batched by length (`BATCH` at a time): a
    batch runs as many frames as its longest episode."""

    order = sorted(range(len(states)), key=lambda i: len(states[i]))
    out: list[tuple[np.ndarray, np.ndarray]] = [None] * len(states)  # type: ignore[list-item]
    for start in range(0, len(order), BATCH):
        members = order[start : start + BATCH]
        batch = [states[i] for i in members]
        lengths = torch.tensor([len(s) for s in batch])
        frames = int(lengths.max())
        targets = torch.eye(4, dtype=torch.float64).repeat(len(batch), frames, 1, 1)
        for b, state in enumerate(batch):
            poses = model.world_from_frame @ frame_from_record(state, arm)
            targets[b, : len(state)] = poses
            targets[b, len(state) :] = poses[-1]
        solution = solve_ik(model.robot, model.chain, targets, lengths, config)
        for b, (index, length) in enumerate(zip(members, lengths.tolist(), strict=True)):
            joints = solution.joints[b, :length].numpy().astype(np.float32)
            error = torch.stack(
                (
                    1000 * solution.position_error[b, :length],
                    torch.rad2deg(solution.rotation_error[b, :length]),
                ),
                1,
            )
            out[index] = (joints, error.numpy().astype(np.float32))
    return out


def episode_states(path: Path, feature: str) -> tuple[np.ndarray, list[Tensor[[int, int]]]]:
    """A data file's episodes (in file order) and each one's `feature` rows, sorted by frame."""

    table = pq.read_table(path, columns=["episode_index", "frame_index", feature])
    table = table.sort_by([("episode_index", "ascending"), ("frame_index", "ascending")])
    episodes = table["episode_index"].to_numpy()
    values = np.asarray(table[feature].to_pylist(), dtype=np.float64).reshape(len(episodes), -1)
    starts = np.flatnonzero(np.r_[True, episodes[1:] != episodes[:-1]])
    ends = np.r_[starts[1:], len(episodes)]
    return episodes[starts], [
        torch.from_numpy(values[a:b]) for a, b in zip(starts, ends, strict=True)
    ]


def write_side_file(
    path: Path,
    episodes: np.ndarray,
    lengths: list[int],
    columns: dict[str, list[np.ndarray]],
    rows_per_group: int = 20_000,
) -> None:
    """One side-table file: `episode_index`, `frame_index` and the `columns` (per episode a
    `[T, n]` array), episodes in order, written atomically."""

    episode_index = np.repeat(episodes, lengths)
    frame_index = np.concatenate([np.arange(n) for n in lengths])
    arrays = {"episode_index": pa.array(episode_index), "frame_index": pa.array(frame_index)}
    for name, values in columns.items():
        flat = np.concatenate(values)
        arrays[name] = pa.FixedSizeListArray.from_arrays(
            pa.array(flat.reshape(-1), type=pa.float32()), flat.shape[1]
        )
    partial = path.with_suffix(".partial")
    pq.write_table(pa.table(arrays), partial, row_group_size=rows_per_group, compression="zstd")
    partial.rename(path)
