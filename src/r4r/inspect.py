"""What a dataset declares and how its robot moves: the first look before writing its priors.

`inspect_episode` reads one episode without decoding video and reports:
- cameras: metadata (role, side, static/dynamic, declared link) next to the priors;
- state columns that vary but drive no joint (a joint missing from the mapping);
- mapped joints whose values leave the description's limits (wrong sign, offset or unit);
- the body tree with how far each body moves over the episode (where a camera can sit).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from r4r.geometry import se3
from r4r.rago.kinematics import forward_kinematics
from r4r.rago.robot import compile_model, load_robot, qpos_from_state

if TYPE_CHECKING:
    from pathlib import Path

    from r4r.data.episode import Episode
    from r4r.data.priors import CameraPrior

# A state column "varies" beyond this standard deviation (its own units)
VARYING = 1e-3
# Joint values may exceed the description's limits by this share of the range
LIMIT_MARGIN = 0.05


@dataclass(frozen=True, slots=True)
class CameraReport:
    name: str
    size: tuple[int, int]
    role: str
    side: str | None
    moving: bool
    mount: str | None
    prior: str


@dataclass(frozen=True, slots=True)
class BodyReport:
    name: str
    depth: int
    translation: float  # metres: largest displacement of its origin from its first pose
    rotation: float  # degrees: largest rotation from its first pose
    cameras: tuple[str, ...]  # cameras declared on it


@dataclass(frozen=True, slots=True)
class Report:
    episode: str
    frames: int
    fps: float
    cameras: tuple[CameraReport, ...]
    unmapped: tuple[tuple[int, str, float], ...]  # (column, name, std) of varying columns
    out_of_range: tuple[tuple[str, float, float, float, float], ...]  # joint, min, max, low, high
    bodies: tuple[BodyReport, ...]

    def text(self) -> str:
        lines = [f"{self.episode}: {self.frames} frames at {self.fps:g} fps", "cameras:"]
        for c in self.cameras:
            kind = "moving" if c.moving else "static"
            lines.append(
                f"  {c.name:28s} {c.size[1]}x{c.size[0]}  {c.role:6s} {c.side or '-':6s} "
                f"{kind:6s} on {c.mount or '?':24s} prior: {c.prior}"
            )
        lines.append("state columns varying but unmapped:" + (" none" if not self.unmapped else ""))
        lines += [f"  [{i}] {name}  std {std:.3g}" for i, name, std in self.unmapped]
        lines.append("joints outside their limits:" + (" none" if not self.out_of_range else ""))
        lines += [
            f"  {n}: {lo:.3f}..{hi:.3f} vs limits {a:.3f}..{b:.3f}"
            for n, lo, hi, a, b in self.out_of_range
        ]
        lines.append("bodies (moves: translation cm / rotation deg):")
        for b in self.bodies:
            marks = f"  <- {', '.join(b.cameras)}" if b.cameras else ""
            moves = f"{b.translation * 100:6.1f} / {b.rotation:5.1f}"
            lines.append(f"  {moves}  {'  ' * b.depth}{b.name}{marks}")
        return "\n".join(lines)


def inspect_episode(episode: Episode[int, int], root: Path) -> Report:
    """The report of one episode (loaded with any `decode`, frames are not used)."""

    from r4r.data.lerobot import dataset_features, robot_record

    features = dataset_features(root)
    cameras = tuple(
        CameraReport(
            name=c.name,
            size=tuple(features[f"observation.images.{c.name}"]["shape"][:2]),
            role=c.role,
            side=c.side,
            moving=c.moving,
            mount=c.mount,
            prior=_describe(c.prior) or "none",
        )
        for c in episode.cameras
    )

    robot = load_robot(episode.robot).to("cpu", torch.float64)
    qpos = qpos_from_state(robot, episode.state, episode.joints)

    # State columns that vary but drive nothing
    mapped = {j.index for j in episode.joints}
    state_feature = episode.priors.state_feature or robot_record(root)["state"]["feature"]
    names = []
    for f in (state_feature,) if isinstance(state_feature, str) else state_feature:
        n = features[f].get("names")
        n = next(iter(n.values())) if isinstance(n, dict) else n
        width = features[f]["shape"][-1]
        names += n if isinstance(n, list) else [f"{f}[{i}]" for i in range(width)]
    count = episode.state.shape[1]
    state_names = (
        [str(n) for n in names]
        if isinstance(names, list) and len(names) == count
        else [str(i) for i in range(count)]
    )
    std = episode.state.std(0) if len(episode.state) > 1 else torch.zeros(episode.state.shape[1])
    unmapped = tuple(
        (i, state_names[i], float(std[i]))
        for i in range(episode.state.shape[1])
        if i not in mapped and float(std[i]) > VARYING
    )

    # Mapped joints beyond the description's limits
    model = compile_model(episode.robot)
    out_of_range = []
    for joint in episode.joints:
        j = robot.joint_names.index(joint.name)
        if not model.jnt_limited[j]:
            continue
        low, high = (float(v) for v in model.jnt_range[j])
        values = qpos[:, robot.joint_qpos[j]]
        margin = LIMIT_MARGIN * (high - low)
        lo, hi = float(values.min()), float(values.max())
        if lo < low - margin or hi > high + margin:
            out_of_range.append((joint.name, lo, hi, low, high))

    # How far each body moves, relative to its first pose
    world_from_body = forward_kinematics(robot, qpos)
    delta = se3.inverse(world_from_body[:1]) @ world_from_body
    translation = (world_from_body[..., :3, 3] - world_from_body[:1, :, :3, 3]).norm(dim=-1)
    rotation = se3.so3_log(delta[..., :3, :3]).norm(dim=-1)
    declared = {}
    for c in episode.cameras:
        if c.mount is not None:
            declared.setdefault(c.mount, []).append(c.name)
    bodies = []
    for b, name in enumerate(robot.body_names):
        depth, parent = 0, b
        while parent > 0:
            parent, depth = robot.body_parent[parent], depth + 1
        bodies.append(
            BodyReport(
                name,
                depth,
                float(translation[:, b].max()),
                math.degrees(float(rotation[:, b].max())),
                tuple(declared.get(name, [])),
            )
        )
    return Report(
        episode=episode.name,
        frames=len(episode.state),
        fps=episode.fps,
        cameras=cameras,
        unmapped=unmapped,
        out_of_range=tuple(out_of_range),
        bodies=tuple(_depth_first(robot.body_parent, bodies)),
    )


def _describe(prior: CameraPrior) -> str:
    """The fields that differ from the defaults, `name=value, ...` (empty: none)."""

    default = type(prior)()
    fields = getattr(prior, "__slots__", ())
    return ", ".join(
        f"{f}={getattr(prior, f)}" for f in fields if getattr(prior, f) != getattr(default, f)
    )


def _depth_first(parents: tuple[int, ...], bodies: list[BodyReport]) -> list[BodyReport]:
    children: dict[int, list[int]] = {}
    for b, p in enumerate(parents):
        if b:
            children.setdefault(p, []).append(b)
    order, stack = [], [0]
    while stack:
        b = stack.pop()
        order.append(bodies[b])
        stack.extend(reversed(children.get(b, [])))
    return order
