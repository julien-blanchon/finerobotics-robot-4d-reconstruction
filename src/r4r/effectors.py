"""End effectors: each arm's `ee` frame, gripper opening and fingertips, by forward kinematics.

The `ee` frame (docs/output-spec.md) is rigid on a description body `b`:

    world_from_ee_t = W_b(q_t) · body_from_ee

The axes mean the same for grippers and hands: z the approach direction (a hand: wrist →
middle-finger knuckle), y the closing direction (a hand: towards the thumb), x = y × z. A
left and a right end effector both get proper rotations (never mirrored). Origins differ:

- grippers and tools: the tool centre point, between the fingertips;
- hands: the palm centre.

A parallel gripper's opening is the distance of its two pad points along the frame's y, minus
the distance at which the jaws touch: `w_t = max((p_a − p_b) · y_t − closed, 0)` (m), the pads
ordered so that `p_a` lies on the +y side when open. A description whose jaws pass through each
other (HSR fingers past contact, the motor still closing on its spring) reads 0. Fingertips are
points on the last body of each finger: `p_t = W_f(q_t) · o_f`. The aperture is the opening of
a parallel gripper, the thumb–index fingertip distance `‖p_thumb − p_index‖` of a hand.

Presets hold the geometry of common grippers and hands. Their bodies are regular expressions:
`body` over the description, pads and fingertips over the bodies below it. Without an `ee`
block in the priors, an arm is found where a preset matches (pads and fingertips included);
of nested matches the deepest is kept; a matched pad or fingertip must end its branch of the
tree (a jaw or a finger tip, at most carrying one more link such as a linkage plate). Arms are
named `left`/`right` from the end effector's body name, else from its nearest ancestor that
names a side. Geometry from the descriptions' own frames and meshes (collision or visual
vertices at the jaw tips and finger pads).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from r4r.data.priors import EffectorPrior, RobotPoint
from r4r.geometry import se3

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.data.priors import ActuationColumn, DatasetPriors, EffectorType
    from r4r.rago.robot import Robot
    from r4r.types import XYZ, Mat4x4


def _orientation(
    x: tuple[float, float, float], y: tuple[float, float, float], z: tuple[float, float, float]
) -> tuple[float, float, float, float]:
    """The quaternion (w, x, y, z) of a frame whose axes are `x, y, z` in the body's frame."""

    rotation = torch.tensor([x, y, z], dtype=torch.float64).T
    return tuple(se3.matrix_to_quaternion(rotation).tolist())


IDENTITY = (1.0, 0.0, 0.0, 0.0)


def _tips(side: str, sign: float) -> dict[str, RobotPoint]:
    """Inspire hand fingertips: the farthest collision vertex of each finger's last body."""

    tips = {
        "thumb": ("thumb_distal", (0.0215, 0.0122, -0.0082)),
        "index": ("index_intermediate", (-0.0009, 0.0452, -0.006)),
        "middle": ("middle_intermediate", (-0.0011, 0.0486, -0.0059)),
        "ring": ("ring_intermediate", (-0.0004, 0.0452, -0.006)),
        "little": ("pinky_intermediate", (0.0001, 0.0373, -0.0067)),
    }
    return {
        name: RobotPoint(f"^{side}_{body}$", (x, sign * y, z))
        for name, (body, (x, y, z)) in tips.items()
    }


def _tip_links(side: str) -> dict[str, RobotPoint]:
    """Fourier hand fingertips: each finger's `<side>_<finger>_tip_link` frame."""

    fingers = {
        "thumb": "thumb",
        "index": "index",
        "middle": "middle",
        "ring": "ring",
        "little": "pinky",
    }
    return {name: RobotPoint(f"^{side}_{finger}_tip_link$") for name, finger in fingers.items()}


PRESETS: dict[str, EffectorPrior] = {
    # Franka Hand: TCP 0.1034 m along the hand's z (its `panda_hand_tcp`, the fingertips);
    # each finger slides along the hand's y, touching at q = 0
    "franka_hand": EffectorPrior(
        type="parallel",
        body="(panda|fr3)_hand$",
        position=(0.0, 0.0, 0.1034),
        orientation=IDENTITY,
        pads=(RobotPoint("leftfinger$"), RobotPoint("rightfinger$")),
        closed=0.0,
    ),
    # Robotiq 2F-85: TCP of the description (`robotiq_tcp`, 0.1493 m); jaws close along the
    # base's x; pads on the inner fingers (open 0.079 m, touching at the joint's limit)
    "robotiq_2f85": EffectorPrior(
        type="parallel",
        body="robotiq_85_base_link$",
        position=(0.0, 0.0, 0.1493),
        orientation=_orientation((0, -1, 0), (1, 0, 0), (0, 0, 1)),
        pads=(
            RobotPoint("left_inner_finger$", (-0.0078, 0.0, 0.035)),
            RobotPoint("right_inner_finger$", (-0.0078, 0.0, 0.035)),
        ),
        closed=0.0,
    ),
    # AgileX Piper: fingertips 0.1358 m along the gripper base's z, fingers slide along its y
    # (`link8` on the +y side)
    "piper": EffectorPrior(
        type="parallel",
        body="gripper_base$",
        position=(0.0, 0.0, 0.1358),
        orientation=IDENTITY,
        pads=(RobotPoint("link8$"), RobotPoint("link7$")),
        closed=0.0,
    ),
    # ARX X5: the gripper points along link6's x, fingers slide along its y (0.0498 m apart
    # at q = 0, the jaws touching); fingertips at x = 0.1576
    "arx_x5": EffectorPrior(
        type="parallel",
        body="link6$",
        position=(0.1576, 0.0, 0.0),
        orientation=_orientation((0, 0, -1), (0, 1, 0), (1, 0, 0)),
        pads=(RobotPoint("link7$"), RobotPoint("link8$")),
        closed=0.0498,
    ),
    # Rainbow RB-Y1: the gripper points along the end body's −z, fingers slide along its x;
    # fingertips 0.1328 m out, inner faces at x = −0.003 in each finger, touching at q = 0
    "rby1": EffectorPrior(
        type="parallel",
        body="ee_(left|right)$",
        position=(0.0, 0.0, -0.1328),
        orientation=_orientation((0, 1, 0), (1, 0, 0), (0, 0, -1)),
        pads=(
            RobotPoint("finger_.1$", (-0.003, 0.0, 0.0)),
            RobotPoint("finger_.2$", (-0.003, 0.0, 0.0)),
        ),
        closed=0.0,
    ),
    # AgiBot OmniPicker (G1): its description's centre link (0.14308 m along the base's z);
    # jaws on the base's y. The state drives the linkage's first joints (`*_joint1`), which carry
    # each jaw plate (`*_link4`) rigidly (the description leaves the plate joints at 0): pads on
    # the plates' inner faces near the tips, 0.121 m apart at the 0.785 rad limit; at q = 0 (jaw
    # tips touching) the rigid plates put them 8.4 mm past each other, hence `closed` < 0;
    # opening 0 to 0.129 m
    "omnipicker": EffectorPrior(
        type="parallel",
        body="gripper_[lr]_base_link$",
        position=(0.0, 0.0, 0.14308),
        orientation=IDENTITY,
        pads=(
            RobotPoint("outer_link4$", (0.0327, 0.0265, -0.00054)),
            RobotPoint("inner_link4$", (0.0327, -0.0265, 0.00054)),
        ),
        closed=-0.0084,
    ),
    # AgiBot G2: the same OmniPicker, its jaws along the base's x (r4r's corrected description
    # `agibot__g2`; the delivered one draws them along y, across the wrist camera's side)
    "omnipicker_g2": EffectorPrior(
        type="parallel",
        body="gripper_[lr]_base_link$",
        position=(0.0, 0.0, 0.14308),
        orientation=_orientation((0, -1, 0), (1, 0, 0), (0, 0, 1)),
        pads=(
            RobotPoint("outer_link4$", (0.0327, 0.0265, -0.00054)),
            RobotPoint("inner_link4$", (0.0327, -0.0265, 0.00054)),
        ),
        closed=-0.0084,
    ),
    # Toyota HSR: fingers rotate about the palm's x and close along its y; fingertip frames
    # (`hand_*_finger_tip_frame`, on the pads' inner edge) 4.4 mm apart at q = 0 (closed),
    # 0.137 m at the 1.24 rad limit; TCP between the closed fingertips, 0.093 m along z
    "hsr": EffectorPrior(
        type="parallel",
        body="hand_palm_link$",
        position=(0.0, 0.0, 0.093),
        orientation=IDENTITY,
        pads=(RobotPoint("hand_r_finger_tip_frame$"), RobotPoint("hand_l_finger_tip_frame$")),
        closed=0.0044,
    ),
    # I2RT YAM: the gripper points along its −z (fingertips 0.1468 m out); the tips slide along
    # its y, their inner faces touching at q = 0 and 0.094 m apart at the −0.04695 m limit
    "yam": EffectorPrior(
        type="parallel",
        body="^(left_|right_)?gripper$",
        position=(0.0, 0.0, -0.1468),
        orientation=_orientation((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
        pads=(
            RobotPoint("tip_right$", (-0.02399, 0.04451, -0.08032)),
            RobotPoint("tip_left$", (0.02399, -0.04451, -0.08032)),
        ),
        closed=0.0,
    ),
    # Unitree Z1: a fixed jaw (`gripperStator`, pointing along its x, fingertip at x = 0.1489)
    # and a jaw rotating about its y (`gripperMover`), closing along its z; jaws touch at
    # z = −0.009 at q = 0
    "z1": EffectorPrior(
        type="parallel",
        body="gripperStator$",
        position=(0.1489, 0.0, -0.009),
        orientation=_orientation((0, 1, 0), (0, 0, 1), (1, 0, 0)),
        pads=(
            RobotPoint("gripperMover$", (0.091, 0.0, -0.009)),
            RobotPoint("gripperStator$", (0.14, 0.0, -0.009)),
        ),
        closed=0.0,
    ),
    # RealSource RS-02 (`[lr]tool_*`): parallel linkage pads closing along the tool base's x,
    # the gripper along its z (pad tips at z = 0.229); pad origins 2.2 mm apart at q = 0 (the
    # pads touching), 0.106 m at q = 1
    "rs02_tool": EffectorPrior(
        type="parallel",
        body="[lr]tool_base_link$",
        position=(0.0, 0.0, 0.229),
        orientation=_orientation((0, -1, 0), (1, 0, 0), (0, 0, 1)),
        pads=(RobotPoint("tool_Right_Pad_Link$"), RobotPoint("tool_Left_Pad_Link$")),
        closed=0.0022,
    ),
    # Galaxea R1 Pro: fingers slide along the gripper link's y, the gripper along its −z
    # (fingertips 0.108 m out); each finger's frame sits on the other side of the centre from
    # its pad, so the pads are the inner faces near the tips (touching at q = 0)
    "galaxea_r1_pro": EffectorPrior(
        type="parallel",
        body="(left|right)_gripper_link$",
        position=(0.0, 0.0, -0.108),
        orientation=_orientation((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
        pads=(
            RobotPoint("gripper_finger_link1$", (-0.018, 0.0245, -0.1122)),
            RobotPoint("gripper_finger_link2$", (0.018, -0.0245, -0.1122)),
        ),
        closed=0.0,
    ),
    # Galaxea R1 Lite: fingers slide along the gripper link's y, the gripper along its x
    # (fingertips 0.0784 m out); finger origins 0.027 m apart when the fingers touch (q = 0)
    "galaxea_r1_lite": EffectorPrior(
        type="parallel",
        body="(left|right)_gripper_link$",
        position=(0.0784, 0.0, 0.0),
        orientation=_orientation((0, 0, -1), (0, 1, 0), (1, 0, 0)),
        pads=(RobotPoint("gripper_finger_link1$"), RobotPoint("gripper_finger_link2$")),
        closed=0.027,
    ),
    # Fourier hands (GR1): fingers along the base's −z, index (and thumb) side +x, both hands;
    # palm centre 0.05 m from the wrist; fingertips at the `*_tip_link` frames (≤ 5 mm inside
    # the finger ends)
    "fourier_hand_right": EffectorPrior(
        type="dexterous",
        body="^R_hand_base_link$",
        position=(0.0, 0.0, -0.05),
        orientation=_orientation((0, 1, 0), (1, 0, 0), (0, 0, -1)),
        fingertips=_tip_links("R"),
    ),
    "fourier_hand_left": EffectorPrior(
        type="dexterous",
        body="^L_hand_base_link$",
        position=(0.0, 0.0, -0.05),
        orientation=_orientation((0, 1, 0), (1, 0, 0), (0, 0, -1)),
        fingertips=_tip_links("L"),
    ),
    # Inspire hands (Unitree G1): the knuckles 0.137 m along the base's −y (z), the thumb on
    # the index side, −z for the right hand, +z for the left (y); palm centre 9 cm from the
    # wrist
    "inspire_right": EffectorPrior(
        type="dexterous",
        body="^R_hand_base_link$",
        position=(0.0, -0.09, 0.0),
        orientation=_orientation((-1, 0, 0), (0, 0, -1), (0, -1, 0)),
        fingertips=_tips("R", 1.0),
    ),
    "inspire_left": EffectorPrior(
        type="dexterous",
        body="^L_hand_base_link$",
        position=(0.0, -0.09, 0.0),
        orientation=_orientation((1, 0, 0), (0, 0, 1), (0, -1, 0)),
        fingertips=_tips("L", -1.0),
    ),
}

# Presets that share body names with another and apply only under this parent body
PARENTS = {"galaxea_r1_pro": "arm_link7$", "galaxea_r1_lite": "arm_link6$"}
PARENTS |= {"omnipicker": "arm_[lr]_link7$", "omnipicker_g2": "arm_[lr]_end_link$"}

# A body name on the left or right side (`left_…`, `…_l_…`, `L_…`, `ee_right`)
SIDES = (("left", re.compile(r"(^|_)(left|l)(_|$)", re.IGNORECASE)),)
SIDES += (("right", re.compile(r"(^|_)(right|r)(_|$)", re.IGNORECASE)),)


@dataclass(frozen=True, slots=True)
class Effector:
    """One arm's end effector resolved on a compiled description."""

    arm: str
    type: EffectorType
    preset: str | None
    body: int
    body_name: str
    body_from_ee: Tensor[[*Mat4x4]]
    # Parallel grippers: the jaws' pad points (body, offset) and their distance when touching
    pads: tuple[tuple[int, tuple[float, float, float]], ...]
    closed: float
    fingertip_names: tuple[str, ...]
    fingertip_bodies: tuple[int, ...]
    fingertip_offsets: Tensor[[int, *XYZ]]
    # The description's moving joints below `body` (the end effector's own), in order
    joints: tuple[str, ...]
    actuation: ActuationColumn | None


def effectors(robot: Robot[int, int, int], priors: DatasetPriors) -> tuple[Effector, ...]:
    """The end effectors of `priors.ee`, else those the presets find (module doc)."""

    if priors.ee:
        return tuple(_resolved(robot, arm, _with_preset(e)) for arm, e in priors.ee.items())
    found = [
        (robot.body(body), name)
        for name, preset in PRESETS.items()
        for body in robot.body_names
        if re.search(str(preset.body), body)
        and _points_found(robot, body, preset)
        and _parent_matches(robot, body, PARENTS.get(name))
    ]
    deepest = [
        (b, n) for b, n in found if not any(o != b and _under(robot, o, b) for o, _ in found)
    ]
    deepest.sort()
    arms = _arm_names([_lineage(robot, b) for b, _ in deepest])
    return tuple(
        _resolved(robot, arm, replace(PRESETS[n], preset=n, body=robot.body_names[b]))
        for arm, (b, n) in zip(arms, deepest, strict=True)
    )


def _with_preset(prior: EffectorPrior) -> EffectorPrior:
    """`prior` with its unset fields from its preset."""

    if prior.preset is None:
        return prior
    if prior.preset not in PRESETS:
        raise ValueError(f"ee preset {prior.preset!r}; known: {sorted(PRESETS)}")
    base = PRESETS[prior.preset]
    given = {f: getattr(prior, f) for f in EffectorPrior.__dataclass_fields__}
    return replace(base, **{f: v for f, v in given.items() if v is not None})


def _arm_names(lineages: list[list[str]]) -> list[str]:
    """`arm` for one end effector; `left`/`right` from each body's name, else from its nearest
    ancestor naming a side (`lineages`: a body's name, then its ancestors'); else `arm<i>`."""

    if len(lineages) == 1:
        return ["arm"]
    sides = [
        next((s for name in names for s, pattern in SIDES if pattern.search(name)), None)
        for names in lineages
    ]
    if None not in sides and len(set(sides)) == len(sides):
        return [str(s) for s in sides]
    return [f"arm{i}" for i in range(len(lineages))]


def _lineage(robot: Robot[int, int, int], body: int) -> list[str]:
    """The names of `body` and its ancestors, nearest first (world excluded)."""

    names = []
    while body > 0:
        names.append(robot.body_names[body])
        body = robot.body_parent[body]
    return names


def _parent_matches(robot: Robot[int, int, int], body: str, pattern: str | None) -> bool:
    parent = robot.body_parent[robot.body(body)]
    return pattern is None or re.search(pattern, robot.body_names[parent]) is not None


def _under(robot: Robot[int, int, int], body: int, ancestor: int) -> bool:
    while body > 0 and body != ancestor:
        body = robot.body_parent[body]
    return body == ancestor


def _below(robot: Robot[int, int, int], body: int) -> list[int]:
    return [b for b in range(len(robot.body_names)) if _under(robot, b, body)]


def _find(robot: Robot[int, int, int], body: int, pattern: str) -> int | None:
    """The first body below `body` (itself included) whose name matches `pattern`."""

    return next((b for b in _below(robot, body) if re.search(pattern, robot.body_names[b])), None)


def _points_found(robot: Robot[int, int, int], body: str, prior: EffectorPrior) -> bool:
    """Whether every pad and fingertip of `prior` is found below `body`, each on a body that
    ends its branch (a jaw or a finger tip, at most carrying one more link such as a linkage
    plate or a moving jaw; not an arm link like `link7`)."""

    def ends_branch(found: int) -> bool:
        children = [b for b, parent in enumerate(robot.body_parent) if parent == found and b]
        return all(c not in robot.body_parent for c in children)

    points = [*(prior.pads or ()), *(prior.fingertips or {}).values()]
    found = [_find(robot, robot.body(body), p.body) for p in points]
    return all(f is not None and ends_branch(f) for f in found)


def _resolved(robot: Robot[int, int, int], arm: str, prior: EffectorPrior) -> Effector:
    """An `EffectorPrior` (preset applied) on `robot`: its body named exactly or matched once."""

    where = f"ee.{arm}"
    if prior.body is None or prior.type is None:
        raise ValueError(f"{where}: needs a `body` and a `type` (or a `preset`)")
    if prior.body in robot.body_names:
        body = robot.body(prior.body)
    else:
        matches = [i for i, n in enumerate(robot.body_names) if re.search(prior.body, n)]
        if len(matches) != 1:
            names = [robot.body_names[i] for i in matches]
            raise ValueError(f"{where}: body {prior.body!r} matches {names}: name one")
        body = matches[0]

    def located(point: RobotPoint, what: str) -> int:
        found = _find(robot, body, point.body)
        if found is None:
            raise ValueError(
                f"{where}: {what} {point.body!r} is not below {robot.body_names[body]}"
            )
        return found

    rotation = se3.quaternion_to_matrix(
        torch.tensor(prior.orientation or IDENTITY, dtype=torch.float64)
    )
    position = torch.tensor(prior.position or (0.0, 0.0, 0.0), dtype=torch.float64)
    pads = tuple((located(p, "pad"), p.offset) for p in prior.pads or ())
    tips = prior.fingertips or {}
    below = set(_below(robot, body)) - {body}  # its own joints move the body itself
    return Effector(
        arm=arm,
        type=prior.type,
        preset=prior.preset,
        body=body,
        body_name=robot.body_names[body],
        body_from_ee=se3.from_rotation_translation(rotation, position),
        pads=pads,
        closed=prior.closed or 0.0,
        fingertip_names=tuple(tips),
        fingertip_bodies=tuple(located(p, f"fingertip {n}") for n, p in tips.items()),
        fingertip_offsets=torch.tensor(
            [p.offset for p in tips.values()], dtype=torch.float64
        ).reshape(len(tips), 3),
        joints=tuple(
            name for name, b in zip(robot.joint_names, robot.joint_body, strict=True) if b in below
        ),
        actuation=prior.actuation,
    )


def ee_pose[T: IntVar](
    effector: Effector, world: Tensor[[T, int, *Mat4x4]]
) -> Tensor[[T, *Mat4x4]]:
    """`world_from_ee_t = W_b(q_t) · body_from_ee` from `world_from_body [T, L]`."""

    return world[:, effector.body] @ effector.body_from_ee.to(world)


def opening[T: IntVar](effector: Effector, world: Tensor[[T, int, *Mat4x4]]) -> Tensor[[T]] | None:
    """A parallel gripper's opening `max((p_a − p_b) · y_t − closed, 0)` (m), `p_a` on the +y
    side when open (jaws passing through each other read 0); None without pads."""

    if effector.type != "parallel" or len(effector.pads) != 2:
        return None
    (a, offset_a), (b, offset_b) = effector.pads
    point_a = se3.transform_points(world[:, a], world.new_tensor(offset_a).expand(len(world), 1, 3))
    point_b = se3.transform_points(world[:, b], world.new_tensor(offset_b).expand(len(world), 1, 3))
    y = ee_pose(effector, world)[:, :3, 1]
    return (((point_a - point_b)[:, 0] * y).sum(-1) - effector.closed).clamp(min=0.0)


def fingertips[T: IntVar](
    effector: Effector, world: Tensor[[T, int, *Mat4x4]]
) -> Tensor[[T, int, *XYZ]]:
    """Fingertip positions `p_t = W_f(q_t) · o_f` [T, F, 3] in the world frame."""

    bodies = world[:, list(effector.fingertip_bodies)]  # [T, F, 4, 4]
    offsets = effector.fingertip_offsets.to(world)
    return (bodies[..., :3, :3] @ offsets[None, :, :, None])[..., 0] + bodies[..., :3, 3]


def has_aperture(effector: Effector) -> bool:
    """Whether the end effector has an aperture: a parallel gripper with pads, or fingertips
    named `thumb` and `index`."""

    pinch = {"thumb", "index"} <= set(effector.fingertip_names)
    return (effector.type == "parallel" and len(effector.pads) == 2) or pinch


def aperture[T: IntVar](effector: Effector, world: Tensor[[T, int, *Mat4x4]]) -> Tensor[[T]] | None:
    """The opening of a parallel gripper (`opening`), else the thumb–index fingertip distance
    `‖p_thumb − p_index‖` (m); None when the end effector has neither."""

    width = opening(effector, world)
    if width is not None:
        return width
    if not has_aperture(effector):
        return None
    tips = fingertips(effector, world)
    names = effector.fingertip_names
    return (tips[:, names.index("thumb")] - tips[:, names.index("index")]).norm(dim=-1)
