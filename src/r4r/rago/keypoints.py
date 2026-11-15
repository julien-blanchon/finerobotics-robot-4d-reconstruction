"""Robot keypoints: pixels of known points on the robot, a direct handle on a camera's pose.

A robot point is a fixed offset `p` in a body's frame (a link origin, the gripper tip, the
flange). At timestep t forward kinematics puts it at `X_t = W_b(q_t) p`. Where a camera sees
it at pixel `u_t`, the camera's pose is tied to the robot with no triangulation:

    r_t = (π_K(P_t⁻¹ X_t) − u_t) / u_cam,   P_t = W_c(q_t) · mount   (world_from_camera)

`π_K` is the lens projection and `u_cam` the camera's pixel unit, as for the bundle's tracks
(`bundle.residuals`). The point is known in metres in the robot's frame. That pins cameras
that scene tracks cannot pin: a static camera that only sees the robot, a head camera that
never moves relative to the base, a wrist camera that sees the other arm.

The pixels are picks: clicked in a few frames (`r4r.data.priors.read_picks`). Pixels found
automatically were tried and dropped (SynGrasp, ground-truth cameras):
- KLT following a pick: it follows a surface patch, not the named point (a joint origin lies
  inside its housing). Read as the named point, followed pixels were 2-4 px off (median, up to
  15 px at 256²) and spoiled the PnP start; read as an unknown point rigid with the link (a
  track on a known body), they changed nothing (median error the same).
- Seeds where a start projects the points, followed by KLT: a seed's pixel is where the start
  puts it, and its track did not pull the start back: 6 of 7 starts 2-20 cm off ended further
  away. Points on a link's outline are not material points.

Pixels transferred from a calibrated session by dense matching do work (`transfer.py`, `r4r
picks`), at about 3× the error of clicks.

Start: PnP-RANSAC (`initialize.locate_camera`) on the picks, expressed in the frame of the body
the camera rides on (`carried`): the world for a static camera; its link for a moving one,
where the mount is constant over time.

Ref: Lepetit et al., EPnP, IJCV 2009 (start).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from r4r.geometry import camera, se3

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.data.priors import Pick, RobotPoint
    from r4r.types import UV, XYZ, Mat3x3, Mat4x4

    from .rig import Rig
    from .robot import Robot

# A camera needs this many keypoints for a PnP start
MIN_KEYPOINTS = 6


@dataclass(frozen=True, slots=True)
class Keypoints[M: IntVar]:
    """`M` observations of known robot points: camera `camera[m]` (index in the rig's names)
    at timestep `time[m]` sees the point `point[m]` (m, in the frame of body `body[m]`) at
    `pixel[m]`, with `weight[m]`."""

    camera: Tensor[[M]]
    time: Tensor[[M]]
    body: Tensor[[M]]
    point: Tensor[[M, *XYZ]]
    pixel: Tensor[[M, *UV]]
    weight: Tensor[[M]]

    def where(self, keep: Tensor[[M]]) -> Keypoints[int]:
        return Keypoints(
            self.camera[keep],
            self.time[keep],
            self.body[keep],
            self.point[keep],
            self.pixel[keep],
            self.weight[keep],
        )

    def shifted(self, steps: int) -> Keypoints[M]:
        """The same observations `steps` timesteps later (episodes pooled one after another)."""

        return Keypoints(
            self.camera, self.time + steps, self.body, self.point, self.pixel, self.weight
        )

    def to(self, like: Tensor) -> Keypoints[M]:  # `like`: read for device and dtype
        device = like.device
        return Keypoints(
            self.camera.to(device),
            self.time.to(device),
            self.body.to(device),
            self.point.to(like),
            self.pixel.to(like),
            self.weight.to(like),
        )


def join(parts: Sequence[Keypoints[int] | None]) -> Keypoints[int] | None:
    """All observations of `parts` (None when there are none)."""

    kept = [p for p in parts if p is not None and len(p.time)]
    if not kept:
        return None
    return Keypoints(
        torch.cat([p.camera for p in kept]),
        torch.cat([p.time for p in kept]),
        torch.cat([p.body for p in kept]),
        torch.cat([p.point for p in kept]),
        torch.cat([p.pixel for p in kept]),
        torch.cat([p.weight for p in kept]),
    )


def picked(
    picks: Sequence[Pick],
    points: Mapping[str, RobotPoint],
    robot: Robot[int, int, int],
    names: tuple[str, ...],
    stride: int,
) -> Keypoints[int] | None:
    """One episode's picks (frames at the video's own rate, decoded every `stride`-th) of
    cameras `names`. `points`: named points; a body's name is its origin."""

    camera, time, body, offset, pixel = [], [], [], [], []
    for pick in picks:
        if pick.camera not in names:
            continue
        if pick.frame % stride:
            raise ValueError(
                f"pick in frame {pick.frame} of {pick.camera} is not a decoded frame (stride "
                f"{stride}): pick frames that are multiples of the stride"
            )
        for name, xy in pick.pixels.items():
            point = points.get(name)
            if point is None and name not in robot.body_names:
                raise ValueError(f"picked point {name!r} is neither a robot point nor a body")
            camera.append(names.index(pick.camera))
            time.append(pick.frame // stride)
            body.append(robot.body(point.body if point else name))
            offset.append(point.offset if point else (0.0, 0.0, 0.0))
            pixel.append(xy)
    if not camera:
        return None
    return Keypoints(
        torch.tensor(camera),
        torch.tensor(time),
        torch.tensor(body),
        torch.tensor(offset, dtype=torch.float64),
        torch.tensor(pixel, dtype=torch.float64),
        torch.ones(len(camera), dtype=torch.float64),
    )


def world_points[M: IntVar](
    world_from_body: Tensor[[int, int, *Mat4x4]], keypoints: Keypoints[M]
) -> Tensor[[M, *XYZ]]:
    """`X = W_b(q_t) p` of each observation (metres, world frame)."""

    pose = world_from_body[keypoints.time, keypoints.body]
    return se3.transform_points(pose, keypoints.point[:, None].to(pose))[:, 0]


def carried[M: IntVar](
    world_from_body: Tensor[[int, int, *Mat4x4]], keypoints: Keypoints[M], carrier: int
) -> Tensor[[M, *XYZ]]:
    """Each observation's point in the frame of body `carrier` at its time,
    `W_carrier(q_t)⁻¹ W_b(q_t) p`: where a camera rigid on `carrier` sees a fixed point set, so
    PnP on these gives `carrier_from_camera` (the world body: a static camera's pose)."""

    local = se3.inverse(world_from_body[keypoints.time, carrier])
    local = local @ world_from_body[keypoints.time, keypoints.body]
    return se3.transform_points(local, keypoints.point[:, None].to(local))[:, 0]


def residuals[K: IntVar](
    rig: Rig[int, int, int],
    keypoints: Keypoints[K],
    view_intrinsics: Tensor[[K, *Mat3x3]],
    world_from_body: Tensor[[int, int, *Mat4x4]],
    min_depth: float,
) -> tuple[Tensor[[K, *UV]], Tensor[[K]]]:
    """Per observation: `r` of the module doc (projected − observed, per pixel unit) and
    whether the point is more than `min_depth` (m) in front of the camera. `view_intrinsics`:
    each observation's K."""

    world = world_points(world_from_body, keypoints)
    world_from_camera = rig.camera_poses(keypoints.camera, keypoints.time, world_from_body)
    local = se3.transform_points(se3.inverse(world_from_camera), world[:, None])[:, 0]
    lens = rig.lens()
    distortion = fisheye = None
    if lens is not None:
        distortion, fisheye = lens.distortion[keypoints.camera], lens.fisheye[keypoints.camera]
    projected = camera.project(view_intrinsics, local[:, None], distortion, fisheye)[:, 0]
    unit = rig.pixel_unit[keypoints.camera]
    valid = local[:, 2] > min_depth
    difference = torch.where(valid[:, None], (projected - keypoints.pixel) / unit[:, None], 0.0)
    return difference, valid
