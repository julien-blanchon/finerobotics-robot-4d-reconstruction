"""Load a robot description into torch arrays: kinematic tree, cameras, free elements, priors.

MuJoCo compiles the MJCF (or URDF): defaults, includes, angle conventions and mesh
assets are resolved exactly as MuJoCo and mink see them. We copy the compiled arrays
into a `Robot` that forward kinematics (`kinematics.py`) evaluates in torch.

Unknown parts of a setup are declared in the MJCF itself, in MuJoCo's user-data
element, so the file stays loadable by any MuJoCo tool:

    <custom>
      <numeric name="r4r:camera:wrist"      data="0.02 0.10"/>   <!-- camera pose is free -->
      <numeric name="r4r:body:right_base"   data="0.10 0.05"/>   <!-- body offset is free -->
    </custom>

`data` is a Gaussian prior around the element's pose in the file: translation std (m),
rotation std (rad); `inf` leaves a component unconstrained.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import mujoco
import torch

from r4r.geometry import se3

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.data.episode import JointMap
    from r4r.data.priors import CameraPrior, DatasetPriors

PRIOR_PREFIX = "r4r:"
HINGE, SLIDE = int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)

# MuJoCo cameras look down −z with +y up; ours follow OpenCV (+z forward, +y down)
MUJOCO_FROM_OPENCV_CAMERA = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))

# Keep every URDF link as its own body (the wrist mount is often a fixed link), and skip
# visual meshes: MuJoCo cannot read COLLADA and kinematics never needs them. Kinematics
# ignores dynamics too: bound the tiny or zeroed masses and inertias some descriptions ship.
URDF_COMPILER = (
    '<mujoco><compiler fusestatic="false" discardvisual="true" strippath="false"'
    ' boundmass="1e-6" boundinertia="1e-12"/></mujoco>'
)

# Mesh formats MuJoCo decodes; geoms on other files (.dae, .glb collisions) are dropped
MESH_FORMATS = {".stl", ".obj", ".msh"}


SCOPES = ("dataset", "episode", "frame")
type Scope = Literal["dataset", "episode", "frame"]


@dataclass(frozen=True, slots=True)
class Freedom:
    """How far fitting may move an element (standard deviations, m and rad) and how often it
    may change: once for the dataset, per episode, or per frame."""

    translation_std: float
    rotation_std: float
    scope: Scope = "dataset"


@dataclass(frozen=True, slots=True)
class Robot[L: IntVar, J: IntVar, C: IntVar]:
    """A compiled kinematic tree in MuJoCo order (every parent precedes its children).

    Loaded in float64; `.to(device, dtype)` picks the working precision.

    Body 0 is MuJoCo's world body. `parent_from_body` is each body's offset at q = 0, before
    its joints. Joints act in order within a body; `joint_anchor`/`joint_axis` are in the body
    frame. Cameras are OpenCV `body_from_camera`.
    """

    body_names: tuple[str, ...]
    body_parent: tuple[int, ...]
    parent_from_body: Tensor[[L, 4, 4]]
    joint_names: tuple[str, ...]
    joint_body: tuple[int, ...]
    joint_kind: tuple[int, ...]
    joint_qpos: tuple[int, ...]
    joint_axis: Tensor[[J, 3]]
    joint_anchor: Tensor[[J, 3]]
    joint_reference: Tensor[[J]]
    # Mimic-style couplings: qpos[target] = c0 + c1 · qpos[source]
    coupling: tuple[tuple[int, int, float, float], ...]
    camera_names: tuple[str, ...]
    camera_body: tuple[int, ...]
    body_from_camera: Tensor[[C, 4, 4]]
    # Free elements: name → (translation std m, rotation std rad)
    free_cameras: dict[str, Freedom]
    free_bodies: dict[str, Freedom]
    nq: int

    def body(self, name: str) -> int:
        return self.body_names.index(name)

    def camera(self, name: str) -> int:
        return self.camera_names.index(name)

    def to(self, device: torch.device | str, dtype: torch.dtype = torch.float32) -> Self:
        return replace(
            self,
            parent_from_body=self.parent_from_body.to(device, dtype),
            joint_axis=self.joint_axis.to(device, dtype),
            joint_anchor=self.joint_anchor.to(device, dtype),
            joint_reference=self.joint_reference.to(device, dtype),
            body_from_camera=self.body_from_camera.to(device, dtype),
        )


@dataclass(frozen=True, slots=True)
class CameraMount:
    """A camera the description may lack: rigid on `body` (None = world), free with a prior.

    Added at `body_from_camera` (OpenCV axes) when known (a calibrated dataset), else at its
    body's origin looking along the body's +z (the usual wrist convention), to be searched.
    The prior's standard deviations bound how far fitting may move it.
    """

    name: str
    body: str | None
    translation_std: float
    rotation_std: float
    body_from_camera: tuple[tuple[float, ...], ...] | None = None  # 4×4, OpenCV camera axes
    scope: Scope = "dataset"

    @classmethod
    def from_prior(cls, name: str, prior: CameraPrior, link: str | None, moving: bool) -> Self:
        """The mount a dataset's priors describe; unknown parts get wide searchable priors."""

        known = prior.position is not None and prior.orientation is not None
        pose = None
        if known:
            rotation = se3.quaternion_to_matrix(
                torch.tensor(prior.orientation, dtype=torch.float64)
            )
            position = torch.tensor(prior.position, dtype=torch.float64)
            pose = tuple(map(tuple, se3.from_rotation_translation(rotation, position).tolist()))
        if not moving:
            unknown = (math.inf, math.inf)
        else:
            unknown = (SEARCH_TRANSLATION_STD, SEARCH_ROTATION_STD)
        return cls(
            name,
            (prior.link or link) if moving else None,
            prior.position_range if known else unknown[0],
            math.radians(prior.orientation_range_deg) if known else unknown[1],
            pose,
            prior.scope,
        )


@dataclass(frozen=True, slots=True)
class BodyMount:
    """A body of the description whose placement (offset from its parent) is not exactly
    known: e.g. a second arm's base, or a mobile robot's base per frame."""

    name: str
    translation_std: float
    rotation_std: float
    scope: Scope = "dataset"


def body_mounts(priors: DatasetPriors) -> tuple[BodyMount, ...]:
    """The free bodies a dataset's priors declare."""

    return tuple(
        BodyMount(name, b.position_range, math.radians(b.orientation_range_deg), b.scope)
        for name, b in priors.bodies.items()
    )


# Priors of a moving camera's mount still to be searched (m, rad)
SEARCH_TRANSLATION_STD, SEARCH_ROTATION_STD = 0.05, 0.5

# MuJoCo camera orientation whose optical axis (OpenCV +z) is the body's +z: 180° about x
LOOK_ALONG_BODY_Z = (0.0, 1.0, 0.0, 0.0)


def compile_model(
    path: Path,
    meshdir: Path | None = None,
    cameras: tuple[CameraMount, ...] = (),
    bodies: tuple[BodyMount, ...] = (),
) -> mujoco.MjModel:
    """Compile an MJCF or URDF file with MuJoCo, adding any `cameras` it does not define.

    `meshdir` overrides where meshes are read.
    """

    return _spec(path, meshdir, cameras, bodies).compile()


def describe(
    path: Path, cameras: tuple[CameraMount, ...], bodies: tuple[BodyMount, ...] = ()
) -> str:
    """MJCF of the description with `cameras` placed and their fitting ranges (`r4r:` numerics).

    Meshes are referenced from the original description's directory (absolute path).
    """

    meshdir = path.parent.resolve()
    spec = _spec(path, meshdir, cameras, bodies)
    spec.compile()  # validates the result
    for mesh in spec.meshes:
        mesh.file = str(meshdir / mesh.file)
    spec.meshdir = ""
    return spec.to_xml()


def _spec(
    path: Path,
    meshdir: Path | None,
    cameras: tuple[CameraMount, ...],
    bodies: tuple[BodyMount, ...] = (),
) -> mujoco.MjSpec:
    xml = path.read_text()
    if path.suffix.lower() == ".urdf" and "<mujoco>" not in xml:
        xml = re.sub(r"(<robot[^>]*>)", rf"\1{URDF_COMPILER}", xml, count=1)

    spec = mujoco.MjSpec.from_string(xml)
    spec.meshdir = str(meshdir or path.parent)
    _drop_undecodable_meshes(spec)
    _repair_inertias(spec)
    existing = {camera.name for camera in spec.cameras}
    numerics = {numeric.name for numeric in spec.numerics}
    for mount in cameras:
        if mount.name not in existing:
            parent = spec.worldbody if mount.body is None else spec.body(mount.body)
            if mount.body_from_camera is None:
                parent.add_camera(name=mount.name, quat=LOOK_ALONG_BODY_Z)
            else:
                pose = torch.tensor(mount.body_from_camera, dtype=torch.float64)
                pose = pose @ MUJOCO_FROM_OPENCV_CAMERA.double()  # its own inverse
                quat = se3.matrix_to_quaternion(pose[:3, :3]).tolist()
                parent.add_camera(name=mount.name, pos=pose[:3, 3].tolist(), quat=quat)
        prior = f"{PRIOR_PREFIX}camera:{mount.name}"
        if prior not in numerics:
            data = [mount.translation_std, mount.rotation_std, SCOPES.index(mount.scope)]
            spec.add_numeric(name=prior, data=data)
    for body in bodies:
        prior = f"{PRIOR_PREFIX}body:{body.name}"
        if prior not in numerics:
            data = [body.translation_std, body.rotation_std, SCOPES.index(body.scope)]
            spec.add_numeric(name=prior, data=data)
    return spec


def _drop_undecodable_meshes(spec: mujoco.MjSpec) -> None:
    """Remove meshes MuJoCo has no decoder for, and the geoms showing them."""

    dropped = {m.name for m in spec.meshes if Path(m.file).suffix.lower() not in MESH_FORMATS}
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.meshname in dropped:
            spec.delete(geom)
    for mesh in spec.meshes:
        if mesh.name in dropped:
            spec.delete(mesh)


def _repair_inertias(spec: mujoco.MjSpec) -> None:
    """Zero the inertia tensors MuJoCo rejects; `boundinertia` then makes them acceptable.

    Descriptions ship tensors that are not positive definite (every entry 1e-8) or whose
    principal moments break the triangle inequality. Kinematics never reads them.
    """

    for body in spec.bodies:
        if not body.explicitinertial:
            continue
        xx, yy, zz, xy, xz, yz = body.fullinertia
        moments = body.inertia.tolist()
        if not math.isnan(xx):
            tensor = torch.tensor([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=torch.float64)
            moments = torch.linalg.eigvalsh(tensor).tolist()
        a, b, c = sorted(moments)
        if not (a > 0 and a + b >= c):  # also catches NaN
            body.fullinertia = [math.nan, 0, 0, 0, 0, 0]
            body.inertia = [0, 0, 0]


def load_robot(
    path: Path,
    meshdir: Path | None = None,
    cameras: tuple[CameraMount, ...] = (),
    bodies: tuple[BodyMount, ...] = (),
) -> Robot:
    """Compile a robot description and extract the arrays forward kinematics needs."""

    model = compile_model(path, meshdir, cameras, bodies)
    unsupported = {int(k) for k in model.jnt_type} - {HINGE, SLIDE}
    assert not unsupported, f"only hinge/slide joints are supported, got types {unsupported}"

    def pose(pos: object, quat: object) -> Tensor:
        rotation = se3.quaternion_to_matrix(torch.as_tensor(quat, dtype=torch.float64))
        return se3.from_rotation_translation(rotation, torch.as_tensor(pos, dtype=torch.float64))

    body_names = tuple(model.body(i).name for i in range(model.nbody))
    camera_names = tuple(model.camera(i).name for i in range(model.ncam))
    priors = _priors(model)
    unknown = {name for kind, name in priors if name not in {*body_names, *camera_names}}
    assert not unknown, f"priors reference unknown elements: {sorted(unknown)}"

    return Robot(
        body_names=body_names,
        body_parent=tuple(int(p) for p in model.body_parentid),
        parent_from_body=pose(model.body_pos, model.body_quat),
        joint_names=tuple(model.joint(i).name for i in range(model.njnt)),
        joint_body=tuple(int(b) for b in model.jnt_bodyid),
        joint_kind=tuple(int(k) for k in model.jnt_type),
        joint_qpos=tuple(int(a) for a in model.jnt_qposadr),
        joint_axis=torch.as_tensor(model.jnt_axis, dtype=torch.float64),
        joint_anchor=torch.as_tensor(model.jnt_pos, dtype=torch.float64),
        joint_reference=torch.as_tensor(model.qpos0[model.jnt_qposadr], dtype=torch.float64),
        coupling=_couplings(model),
        camera_names=camera_names,
        camera_body=tuple(int(b) for b in model.cam_bodyid),
        body_from_camera=(
            pose(model.cam_pos, model.cam_quat) @ MUJOCO_FROM_OPENCV_CAMERA.double()
        ).reshape(model.ncam, 4, 4),
        free_cameras={name: f for (kind, name), f in priors.items() if kind == "camera"},
        free_bodies={name: f for (kind, name), f in priors.items() if kind == "body"},
        nq=int(model.nq),
    )


def _priors(model: mujoco.MjModel) -> dict[tuple[str, str], Freedom]:
    """`r4r:<kind>:<name>` numerics (translation std, rotation std[, scope index]) → freedoms."""

    priors = {}
    for i in range(model.nnumeric):
        full = model.numeric(i).name
        if not full.startswith(PRIOR_PREFIX):
            continue
        kind, _, name = full.removeprefix(PRIOR_PREFIX).partition(":")
        assert kind in {"camera", "body"}, f"unknown r4r element kind in {full!r}"
        data = [float(v) for v in model.numeric(i).data]
        scope = SCOPES[int(data[2])] if len(data) > 2 else "dataset"
        priors[kind, name] = Freedom(data[0], data[1], scope)
    return priors


def _couplings(model: mujoco.MjModel) -> tuple[tuple[int, int, float, float], ...]:
    """Linear joint equalities (URDF `<mimic>`) as (target qpos, source qpos, offset, gain)."""

    joint_equality = int(mujoco.mjtEq.mjEQ_JOINT)
    couplings = []
    for i in range(model.neq):
        if int(model.eq_type[i]) != joint_equality:
            continue
        target, source = model.eq_obj1id[i], model.eq_obj2id[i]
        offset, gain, *higher = model.eq_data[i, :5]
        assert not any(higher), "only linear joint couplings are supported"
        couplings.append(
            (
                int(model.jnt_qposadr[target]),
                int(model.jnt_qposadr[source]),
                float(offset),
                float(gain),
            )
        )
    return tuple(couplings)


def qpos_from_state[T: IntVar, S: IntVar](
    robot: Robot, state: Tensor[[T, S]], joints: tuple[JointMap, ...]
) -> Tensor[[T, int]]:
    """MuJoCo `qpos` from a dataset state trajectory; unmapped joints stay at 0.

    Mimic joints are filled later by forward kinematics (`apply_couplings`).
    """

    qpos = state.new_zeros(len(state), robot.nq)
    for joint in joints:
        value = joint.sign * state[:, joint.index]
        if joint.source is not None and joint.target is not None:
            (s0, s1), (t0, t1) = joint.source, joint.target
            value = t0 + (value - s0) * (t1 - t0) / (s1 - s0)
        qpos[:, robot.joint_qpos[robot.joint_names.index(joint.name)]] = value
    return qpos
