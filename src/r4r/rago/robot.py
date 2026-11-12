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

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Self

import mujoco
import torch

from r4r.geometry import se3

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

PRIOR_PREFIX = "r4r:"
HINGE, SLIDE = int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)

# MuJoCo cameras look down −z with +y up; ours follow OpenCV (+z forward, +y down)
MUJOCO_FROM_OPENCV_CAMERA = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))

# Keep every URDF link as its own body (the wrist mount is often a fixed link), and skip
# visual meshes: MuJoCo cannot read COLLADA and kinematics never needs them.
URDF_COMPILER = (
    '<mujoco><compiler fusestatic="false" discardvisual="true" strippath="false"/></mujoco>'
)


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
    free_cameras: dict[str, tuple[float, float]]
    free_bodies: dict[str, tuple[float, float]]
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


def compile_model(path: Path, meshdir: Path | None = None) -> mujoco.MjModel:
    """Compile an MJCF or URDF file with MuJoCo; `meshdir` overrides where meshes are read."""

    xml = path.read_text()
    if path.suffix.lower() == ".urdf" and "<mujoco>" not in xml:
        xml = re.sub(r"(<robot[^>]*>)", rf"\1{URDF_COMPILER}", xml, count=1)

    spec = mujoco.MjSpec.from_string(xml)
    spec.meshdir = str(meshdir or path.parent)
    return spec.compile()


def load_robot(path: Path, meshdir: Path | None = None) -> Robot:
    """Compile a robot description and extract the arrays forward kinematics needs."""

    model = compile_model(path, meshdir)
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
        free_cameras={name: std for (kind, name), std in priors.items() if kind == "camera"},
        free_bodies={name: std for (kind, name), std in priors.items() if kind == "body"},
        nq=int(model.nq),
    )


def _priors(model: mujoco.MjModel) -> dict[tuple[str, str], tuple[float, float]]:
    """`r4r:<kind>:<name>` numerics → {(kind, name): (translation std, rotation std)}."""

    priors = {}
    for i in range(model.nnumeric):
        full = model.numeric(i).name
        if not full.startswith(PRIOR_PREFIX):
            continue
        kind, _, name = full.removeprefix(PRIOR_PREFIX).partition(":")
        assert kind in {"camera", "body"}, f"unknown r4r element kind in {full!r}"
        translation_std, rotation_std = (float(v) for v in model.numeric(i).data)
        priors[kind, name] = (translation_std, rotation_std)
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
