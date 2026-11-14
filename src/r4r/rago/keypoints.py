"""Robot keypoints: pixels of known points on the robot, a direct handle on a camera's pose.

A robot point is a fixed offset `p` in a body's frame (a link origin, the gripper tip, the
flange). At timestep t forward kinematics puts it at `X_t = W_b(q_t) p`. Where a camera sees
it at pixel `u_t`, the camera's pose is tied to the robot with no triangulation:

    r_t = (π_K(P_t⁻¹ X_t) − u_t) / u_cam,   P_t = W_c(q_t) · mount   (world_from_camera)

`π_K` is the lens projection and `u_cam` the camera's pixel unit, as for the bundle's tracks
(`bundle.residuals`). The point is known in metres in the robot's frame. That pins cameras
that scene tracks cannot pin: a static camera that only sees the robot, a head camera that
never moves relative to the base, a wrist camera that sees the other arm.

Where the pixels come from:
- picks: pixels clicked in a few frames (`r4r.data.priors.read_picks`);
- seeds: where a camera's current pose projects the points, every `every` timesteps (a
  refinement within the basin of that pose, not a search).
Both are followed `span` timesteps forwards and backwards by KLT (forward–backward checked).
A followed pixel drifts off the point, by 1–10 px over 10–40 frames on a moving link (SynGrasp,
256²), so it weighs `followed_weight` of a pick.

Start: PnP-RANSAC (`initialize.locate_camera`) on the pooled 3D–2D pairs, the points expressed
in the frame of the body the camera rides on (`carried`): the world for a static camera; its
link for a moving one, where the mount is constant over time. The bundle then adds the
residuals above to its tracks'.

Ref: Lucas & Kanade 1981 (KLT); Lepetit et al., EPnP, IJCV 2009 (start).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from einops import rearrange

from r4r.geometry import camera, se3

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.data.priors import Pick, RobotPoint
    from r4r.types import RGB, UV, XYZ, Mat3x3, Mat4x4

    from .lens import Lens
    from .rig import Rig
    from .robot import Robot

# A camera needs this many keypoint observations for a PnP start
MIN_KEYPOINTS = 12
# Seeds this close to the image border (pixels) are not followed
BORDER = 4


@dataclass(frozen=True, slots=True, kw_only=True)
class KeypointConfig:
    # Follow each pick or seed this many timesteps forwards and backwards
    span: int = 10
    # Seeds: every this many timesteps
    every: int = 5
    # Weight of a followed (or seeded) pixel, a pick weighing 1
    followed_weight: float = 0.3
    # KLT window (pixels), pyramid levels, forward–backward agreement (pixels)
    window: int = 15
    levels: int = 3
    max_error: float = 0.5


@dataclass(frozen=True, slots=True)
class Keypoints[M: IntVar]:
    """`M` observations of known robot points: camera `camera[m]` (index in the rig's names)
    at timestep `time[m]` sees the point `point[m]` (m, in the frame of body `body[m]`) at
    `pixel[m]`, with `weight[m]` (a pick 1, a followed pixel less)."""

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
        *(torch.cat([getattr(p, f) for p in kept]) for f in Keypoints.__dataclass_fields__)
    )


def world_points[M: IntVar](
    world_from_body: Tensor[[int, int, *Mat4x4]], keypoints: Keypoints[M]
) -> Tensor[[M, *XYZ]]:
    """`X = W_b(q_t) p` of each observation (metres, world frame)."""

    pose = world_from_body[keypoints.time, keypoints.body]
    return se3.transform_points(pose, keypoints.point[:, None])[:, 0]


def picked[T: IntVar](
    picks: Sequence[Pick],
    points: Mapping[str, RobotPoint],
    robot: Robot[int, int, int],
    names: tuple[str, ...],
    videos: Mapping[str, Tensor[[T, *RGB, int, int]]],
    stride: int,
    config: KeypointConfig,
) -> Keypoints[int] | None:
    """One episode's picks (frames at the video's own rate, decoded every `stride`-th), each
    followed through its camera's video. `points`: named points; a body's name is its origin."""

    parts = []
    for pick in picks:
        if pick.camera not in names:
            continue
        if pick.frame % stride:
            raise ValueError(
                f"pick in frame {pick.frame} of {pick.camera} is not a decoded frame (stride "
                f"{stride}): pick frames that are multiples of the stride"
            )
        body, offset = [], []
        for name in pick.pixels:
            point = points.get(name)
            if point is None and name not in robot.body_names:
                raise ValueError(f"picked point {name!r} is neither a robot point nor a body")
            body.append(robot.body(point.body if point else name))
            offset.append(point.offset if point else (0.0, 0.0, 0.0))
        video = videos[pick.camera]
        time = torch.full((len(body),), pick.frame // stride)
        pixel = torch.tensor(list(pick.pixels.values()), dtype=torch.float64)
        parts.append(
            _observed(names.index(pick.camera), time, body, offset, pixel, video, config, 1.0)
        )
    return join(parts)


def seeded[T: IntVar](
    world_from_body: Tensor[[T, int, *Mat4x4]],
    world_from_camera: Tensor[[T, *Mat4x4]],
    intrinsics: Tensor[[*Mat3x3]],
    lens: Lens[1] | None,
    points: Sequence[tuple[int, tuple[float, float, float]]],
    video: Tensor[[T, *RGB, int, int]],
    camera_index: int,
    config: KeypointConfig,
) -> Keypoints[int] | None:
    """`points` (body, offset) where the camera's pose projects them every `every` timesteps,
    inside the image and in front of it, followed through the video."""

    steps = len(world_from_body)
    times = torch.arange(0, steps, config.every)
    body = torch.tensor([b for b, _ in points]).repeat(len(times))
    offset = torch.tensor([o for _, o in points], dtype=world_from_body.dtype)
    time = times.repeat_interleave(len(points))
    world = se3.transform_points(
        world_from_body[time, body].cpu(), offset.repeat(len(times), 1)[:, None]
    )
    local = se3.transform_points(se3.inverse(world_from_camera[time].cpu()), world)[:, 0]
    k = intrinsics.cpu().expand(len(time), 3, 3)
    distortion = fisheye = None
    if lens is not None:
        distortion = lens.distortion.cpu().expand(len(time), -1)
        fisheye = lens.fisheye.cpu().expand(len(time))
    pixel = camera.project(k, local[:, None], distortion, fisheye)[:, 0]
    height, width = video.shape[-2:]
    keep = (local[:, 2] > 0.05) & (pixel >= BORDER).all(-1)
    keep &= (pixel[:, 0] < width - 1 - BORDER) & (pixel[:, 1] < height - 1 - BORDER)
    if not bool(keep.any()):
        return None
    offsets = [points[i % len(points)][1] for i in keep.nonzero()[:, 0].tolist()]
    return _observed(
        camera_index,
        time[keep],
        body[keep].tolist(),
        offsets,
        pixel[keep],
        video,
        config,
        config.followed_weight,
    )


def _observed[T: IntVar](
    camera_index: int,
    time: Tensor[[int]],
    body: list[int],
    offset: list[tuple[float, float, float]],
    pixel: Tensor[[int, *UV]],
    video: Tensor[[T, *RGB, int, int]],
    config: KeypointConfig,
    weight: float,
) -> Keypoints[int]:
    """Observations at the seeds (weight `weight`) and along their KLT follow-up."""

    seed, followed_time, followed_pixel = follow(video, time, pixel, config)
    body_t = torch.tensor(body)
    offset_t = torch.tensor(offset, dtype=torch.float64)
    weights = torch.cat(
        (
            torch.full((len(time),), weight, dtype=torch.float64),
            torch.full((len(seed),), config.followed_weight, dtype=torch.float64),
        )
    )
    return Keypoints(
        torch.full((len(weights),), camera_index),
        torch.cat((time, followed_time)),
        torch.cat((body_t, body_t[seed])),
        torch.cat((offset_t, offset_t[seed])),
        torch.cat((pixel.double(), followed_pixel)),
        weights,
    )


def follow[T: IntVar, N: IntVar](
    video: Tensor[[T, *RGB, int, int]],
    time: Tensor[[N]],
    pixel: Tensor[[N, *UV]],
    config: KeypointConfig,
) -> tuple[Tensor[[int]], Tensor[[int]], Tensor[[int, *UV]]]:
    """KLT from each seed (`time`, `pixel`) up to `span` timesteps forwards and backwards,
    stopped where the forward–backward check fails: per followed observation, its seed, its
    timestep and its pixel."""

    gray: dict[int, np.ndarray] = {}

    def image(t: int) -> np.ndarray:
        if t not in gray:
            frame = rearrange(video[t], "c h w -> h w c").cpu().numpy()
            gray[t] = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return gray[t]

    window = (config.window, config.window)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

    def lucas_kanade(
        a: np.ndarray, b: np.ndarray, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        moved, ok, _ = cv2.calcOpticalFlowPyrLK(
            a, b, points, points.copy(), winSize=window, maxLevel=config.levels, criteria=criteria
        )
        return moved, ok[:, 0] == 1

    seeds, times, pixels = [], [], []
    for start in torch.unique(time).tolist():
        mine = (time == start).nonzero()[:, 0].numpy()
        for direction in (1, -1):
            active: np.ndarray = pixel[mine].float().numpy().copy()
            ids = mine.copy()
            for step in range(1, config.span + 1):
                t = start + direction * step
                if not 0 <= t < len(video) or not len(ids):
                    break
                before, after = image(t - direction), image(t)
                ahead, ok = lucas_kanade(before, after, active)
                back, ok_b = lucas_kanade(after, before, ahead)
                keep = ok & ok_b
                keep &= np.linalg.norm(back - active, axis=1) < config.max_error
                active, ids = ahead[keep], ids[keep]
                seeds.append(ids)
                times.append(np.full(len(ids), t))
                pixels.append(active.astype(np.float64))
    if not seeds:
        return (
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, 2, dtype=torch.float64),
        )
    return (
        torch.from_numpy(np.concatenate(seeds)),
        torch.from_numpy(np.concatenate(times)),
        torch.from_numpy(np.concatenate(pixels)),
    )


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
