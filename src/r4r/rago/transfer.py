"""Robot keypoints transferred from calibrated frames by dense matching: picks without clicks.

In a reference frame whose camera is known (a calibrated station), the robot's visible surface
is known: the ray through pixel `u` first meets the description's geometry at a point `p` rigid
with body `b` (MuJoCo ray cast at the frame's joints). RoMa v2 matches the reference with a
target frame of an uncalibrated camera of the same robot; the warp at `u` is where the target
sees `p`:

    u' = w_ab(u),  kept when  c_ab(u) > certainty  and  ‖w_ba(u') − u‖ < cycle

`c_ab` is RoMa's overlap probability; the forward-backward check drops points hidden or
ambiguous in the target. `(b, p) ↔ u'` is a robot keypoint (`keypoints.py`) at the target's
timestep. Surface points, not named ones: a joint origin lies inside its housing, and the
warp carries the surface in front of it, off the origin by about `r · sin Δθ` for a change of
viewing direction `Δθ` (SynGrasp, 256²: named points gave a median camera error of 23 cm,
surface points 5.6 cm; docs/dataset-onboarding.md). The geometry only says which surface point a
reference pixel shows; the solve never sees it.

Each target frame is matched with the reference frame of nearest joint configuration
(`‖q − q_ref‖`, radians and metres) that shows enough of the robot: it then looks most alike
(gripper opening, wrist roll). The reference is cropped around its points (with a margin) so
the robot fills the matched image. The pixels are written as picks (`r4r.data.priors.Pick`) for
`r4r rig --picks`: a PnP-RANSAC start, then the bundle's Cauchy loss handles outliers left.

Ref: Edstedt et al., RoMa v2, arXiv:2511.15706.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from r4r.data.priors import Pick, RobotPoint
from r4r.geometry import se3

from .kinematics import forward_kinematics

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.matching import Matches, RoMaV2
    from r4r.types import RGB, UV, XYZ, Mat3x3, Mat4x4

    from .robot import Robot

# Pairs matched at once (each encodes two frames)
BATCH = 8
# Reference frames tried per target frame, nearest first, for one with MIN_POINTS points
CANDIDATES = 8
MIN_POINTS = 8
# Rays are cast this far at most (m)
MAX_DISTANCE = 100.0


@dataclass(frozen=True, slots=True, kw_only=True)
class TransferConfig:
    # Minimum RoMa overlap probability at the reference pixel
    certainty: float = 0.1
    # Maximum forward-backward distance (reference pixels)
    cycle: float = 3.0
    # Reference crop: the box of its points grown by this fraction of its size on each side
    # (inf: the whole frame)
    margin: float = 0.5
    # Target frames per episode, spread over its joint configurations
    frames: int = 16
    # Surface points per reference frame, drawn from a grid of `grid` rays per image row
    points: int = 96
    grid: int = 64


@dataclass(frozen=True, slots=True)
class Reference[R: IntVar, NQ: IntVar]:
    """`R` frames of calibrated static cameras (one image size): images, joint positions, and
    each frame's camera pose (base frame, OpenCV axes) and intrinsics."""

    frames: Tensor[[R, *RGB, int, int]]  # uint8
    qpos: Tensor[[R, NQ]]
    world_from_camera: Tensor[[R, *Mat4x4]]
    intrinsics: Tensor[[R, *Mat3x3]]


@dataclass(frozen=True, slots=True)
class SurfacePoints[M: IntVar]:
    """`M` reference pixels and the robot surface point each shows: body index and offset (m,
    in the body's frame)."""

    pixels: Tensor[[M, *UV]]
    body: Tensor[[M]]
    offset: Tensor[[M, *XYZ]]


@dataclass(frozen=True, slots=True)
class Transferred[N: IntVar, P: IntVar]:
    """Up to `P` surface points in each of `N` target frames: their pixels there, the point
    (body, offset), and whether it passed the checks (padding never does)."""

    pixels: Tensor[[N, P, *UV]]
    body: Tensor[[N, P]]
    offset: Tensor[[N, P, *XYZ]]
    kept: Tensor[[N, P]]


def surface_points[NQ: IntVar](
    model: mujoco.MjModel,
    robot: Robot[int, int, int],
    qpos: Tensor[[NQ]],
    world_from_camera: Tensor[[*Mat4x4]],
    intrinsics: Tensor[[*Mat3x3]],
    size: tuple[int, int],
    bodies: Collection[int],
    config: TransferConfig,
    seed: int = 0,
) -> SurfacePoints[int]:
    """Up to `config.points` pixels (at random) of a grid whose rays first meet a geom of
    `bodies`, away from its outline (the four grid neighbours meet the same body), with the
    point met: `p = W_b(q)⁻¹ (o + t d)`, ray `o + t d` through the pixel (`d = R K⁻¹ (u, 1)`).

    `model` is the description compiled with its geometry (`robot.compile_model`), `robot` the
    same description's kinematics; scene geometry is not modelled, so occluders are not known.
    """

    data = mujoco.MjData(model)
    data.qpos[:] = qpos.double().cpu().numpy()
    mujoco.mj_kinematics(model, data)
    height, width = size
    step = width / config.grid
    columns, rows = np.arange(step / 2, width, step), np.arange(step / 2, height, step)
    u, v = np.meshgrid(columns, rows)
    pixels = np.stack((u, v, np.ones_like(u)), -1).reshape(-1, 3)
    pose = world_from_camera.double().cpu().numpy()
    direction = pixels @ np.linalg.inv(intrinsics.double().cpu().numpy()).T @ pose[:3, :3].T
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    origin = pose[:3, 3].copy()
    geom = np.full(len(direction), -1, dtype=np.int32)
    distance = np.full(len(direction), -1.0)
    mujoco.mj_multiRay(
        model,
        data,
        origin,
        direction.reshape(-1).copy(),
        None,
        1,
        -1,
        geom,
        distance,
        None,
        len(direction),
        MAX_DISTANCE,
    )
    body = np.where(geom >= 0, model.geom_bodyid[np.maximum(geom, 0)], -1).reshape(u.shape)
    inner = np.isin(body, list(bodies))
    centre = body[1:-1, 1:-1]
    inner[1:-1, 1:-1] &= (centre == body[:-2, 1:-1]) & (centre == body[2:, 1:-1])
    inner[1:-1, 1:-1] &= (centre == body[1:-1, :-2]) & (centre == body[1:-1, 2:])
    inner[[0, -1], :] = inner[:, [0, -1]] = False
    index = np.flatnonzero(inner)
    if len(index) > config.points:
        index = np.sort(np.random.default_rng(seed).choice(index, config.points, replace=False))
    met = torch.from_numpy(origin + direction[index] * distance[index, None])
    hit = torch.from_numpy(body.reshape(-1)[index]).long()
    world_from_body = forward_kinematics(robot, qpos[None].double().cpu())[0]
    offset = se3.transform_points(se3.inverse(world_from_body[hit]), met[:, None])[:, 0]
    return SurfacePoints(torch.from_numpy(pixels[index, :2]), hit, offset)


def spread_frames[T: IntVar, NQ: IntVar](qpos: Tensor[[T, NQ]], count: int) -> Tensor[[int]]:
    """`count` timesteps (sorted) far apart in joint space: farthest-point sampling from the
    middle timestep."""

    chosen = [len(qpos) // 2]
    distance = (qpos - qpos[chosen[0]]).norm(dim=-1)
    while len(chosen) < min(count, len(qpos)):
        chosen.append(int(distance.argmax()))
        distance = torch.minimum(distance, (qpos - qpos[chosen[-1]]).norm(dim=-1))
    return torch.tensor(sorted(chosen), device=qpos.device)


def nearest[R: IntVar, N: IntVar, NQ: IntVar](
    reference_qpos: Tensor[[R, NQ]], qpos: Tensor[[N, NQ]], count: int
) -> Tensor[[N, int]]:
    """For each joint configuration, the `count` nearest reference frames, nearest first."""

    distance = torch.cdist(qpos, reference_qpos.to(qpos))
    return distance.topk(min(count, len(reference_qpos)), dim=-1, largest=False).indices


def read_warp[B: IntVar, P: IntVar](
    matches: Matches[B], pixels_a: Tensor[[B, P, *UV]]
) -> tuple[Tensor[[B, P, *UV]], Tensor[[B, P]], Tensor[[B, P]]]:
    """At pixels of image a: the matched pixel in image b, the certainty, and how far b's
    warp brings that pixel back from where it started (pixels of a). Bilinear in the warp.

    Pixels ↔ grid_sample coordinates: `x = 2 (u + ½) / W − 1`.
    """

    def normalised(pixels: Tensor[[B, P, *UV]], size: tuple[int, int]) -> Tensor[[B, P, *UV]]:
        return 2 * (pixels + 0.5) / pixels.new_tensor((size[1], size[0])) - 1

    def pixel_of(x: Tensor[[B, P, *UV]], size: tuple[int, int]) -> Tensor[[B, P, *UV]]:
        return (x + 1) / 2 * x.new_tensor((size[1], size[0])) - 0.5

    def sampled(field: Tensor[[B, int, int, int]], at: Tensor[[B, P, *UV]]) -> Tensor[[B, P, int]]:
        grid = rearrange(at, "b p uv -> b 1 p uv").to(field.dtype)
        values = F.grid_sample(
            rearrange(field, "b h w c -> b c h w"), grid, mode="bilinear", align_corners=False
        )
        return rearrange(values, "b c 1 p -> b p c")

    at_a = normalised(pixels_a, matches.size_a)
    at_b = sampled(matches.warp_ab, at_a)
    certainty = sampled(rearrange(matches.certainty_ab, "b h w -> b h w 1"), at_a)[..., 0]
    back = sampled(matches.warp_ba, at_b)
    cycle = (pixel_of(back, matches.size_a) - pixels_a).norm(dim=-1)
    return pixel_of(at_b, matches.size_b).to(pixels_a), certainty.to(pixels_a), cycle.to(pixels_a)


def crop_box(
    pixels: Tensor[[int, *UV]], size: tuple[int, int], aspect: float, margin: float
) -> tuple[int, int, int, int]:
    """Box (x0, y0, x1, y1, pixel edges) around `pixels` grown by `margin` × its size on each
    side, widened to `aspect` (width / height) and kept inside an image of `size` (H, W)."""

    height, width = size
    if math.isinf(margin) or not len(pixels):
        return 0, 0, width, height
    low, high = pixels.amin(0), pixels.amax(0)
    centre = (low + high) / 2
    half = (high - low) / 2 * (1 + 2 * margin) + 16
    half_w, half_h = float(half[0]), float(half[1])
    half_w, half_h = max(half_w, half_h * aspect), max(half_h, half_w / aspect)
    half_w, half_h = min(half_w, width / 2), min(half_h, height / 2)
    x0 = min(max(float(centre[0]) - half_w, 0.0), width - 2 * half_w)
    y0 = min(max(float(centre[1]) - half_h, 0.0), height - 2 * half_h)
    return round(x0), round(y0), round(x0 + 2 * half_w), round(y0 + 2 * half_h)


def match_points[N: IntVar, P: IntVar](
    model: RoMaV2,
    references: Tensor[[N, *RGB, int, int]],
    pixels: Tensor[[N, P, *UV]],
    seen: Tensor[[N, P]],
    targets: Tensor[[N, *RGB, int, int]],
    margin: float,
) -> tuple[Tensor[[N, P, *UV]], Tensor[[N, P]], Tensor[[N, P]]]:
    """Reference frame `i` (uint8, cropped to its `seen` pixels, resized to the target's size)
    matched with target frame `i`: each pixel's match in the target, its certainty and its
    forward-backward distance (reference pixels)."""

    device = next(model.parameters()).device
    height, width = targets.shape[-2:]
    size = (references.shape[-2], references.shape[-1])
    found, certainty, cycle = [], [], []
    for start in range(0, len(targets), BATCH):
        crops, cropped, zoom = [], [], []
        for i in range(start, min(start + BATCH, len(targets))):
            x0, y0, x1, y1 = crop_box(pixels[i][seen[i]], size, width / height, margin)
            image = references[i, :, y0:y1, x0:x1].to(device).float() / 255
            crops.append(
                F.interpolate(image[None], (height, width), mode="bilinear", antialias=True)[0]
            )
            # Reference pixels → crop pixels: `u' = (u − x0 + ½) · s − ½`, `s` per axis
            scale = pixels.new_tensor((width / (x1 - x0), height / (y1 - y0)))
            cropped.append((pixels[i] - pixels.new_tensor((x0, y0)) + 0.5) * scale - 0.5)
            zoom.append(float(scale.mean()))
        target = targets[start : start + len(crops)].to(device).float() / 255
        encoding = model.encode(torch.cat((torch.stack(crops), target)))
        a = torch.arange(len(crops), device=device)
        matches = model.match(encoding[a], encoding[a + len(crops)])
        matched = read_warp(matches, torch.stack(cropped).to(device))
        found.append(matched[0].cpu())
        certainty.append(matched[1].cpu())
        cycle.append(matched[2].cpu() / torch.tensor(zoom, dtype=matched[2].dtype)[:, None])
    return (
        torch.cat(found).to(pixels),
        torch.cat(certainty).to(pixels),
        torch.cat(cycle).to(pixels),
    )


def transfer[N: IntVar](
    matcher: RoMaV2,
    model: mujoco.MjModel,
    robot: Robot[int, int, int],
    reference: Reference[int, int],
    targets: Tensor[[N, *RGB, int, int]],
    target_qpos: Tensor[[N, int]],
    bodies: Collection[int],
    config: TransferConfig,
) -> Transferred[N, int]:
    """Surface points of `bodies` carried into each target frame (uint8) from its nearest
    reference frame showing at least `MIN_POINTS` of them (`surface_points`)."""

    size = (reference.frames.shape[-2], reference.frames.shape[-1])
    candidates = nearest(reference.qpos, target_qpos.to(reference.qpos), CANDIDATES)
    chosen, found = [], []
    for row in candidates.tolist():
        for r in row:
            points = surface_points(
                model,
                robot,
                reference.qpos[r],
                reference.world_from_camera[r],
                reference.intrinsics[r],
                size,
                bodies,
                config,
                seed=r,
            )
            if len(points.body) >= MIN_POINTS:
                break
        chosen.append(r)
        found.append(points)
    count = config.points
    like = reference.world_from_camera
    pixels = like.new_zeros(len(found), count, 2)
    body = torch.zeros(len(found), count, dtype=torch.long)
    offset = like.new_zeros(len(found), count, 3)
    seen = torch.zeros(len(found), count, dtype=torch.bool)
    for i, points in enumerate(found):
        m = len(points.body)
        pixels[i, :m], body[i, :m], offset[i, :m] = (
            points.pixels.to(like),
            points.body,
            points.offset,
        )
        seen[i, :m] = True
    moved, certainty, cycle = match_points(
        matcher, reference.frames[torch.tensor(chosen)], pixels, seen, targets, config.margin
    )
    kept = seen & (certainty > config.certainty) & (cycle < config.cycle)
    return Transferred(moved, body, offset, kept)


def as_picks[N: IntVar, P: IntVar](
    transferred: Transferred[N, P],
    body_names: Sequence[str],
    episode: int,
    camera_name: str,
    frames: Sequence[int],
) -> tuple[dict[str, RobotPoint], list[Pick]]:
    """The kept points as named robot points (`e<episode>f<frame>p<j>`) and one pick per
    target frame (the video's own frame index) that kept any."""

    points, picks = {}, []
    for i, frame in enumerate(frames):
        pixels = {}
        for j in transferred.kept[i].nonzero()[:, 0].tolist():
            name = f"e{episode}f{frame}p{j}"
            offset = transferred.offset[i, j].tolist()
            points[name] = RobotPoint(
                body_names[int(transferred.body[i, j])], (offset[0], offset[1], offset[2])
            )
            xy = transferred.pixels[i, j].tolist()
            pixels[name] = (xy[0], xy[1])
        if pixels:
            picks.append(Pick(episode, camera_name, frame, pixels))
    return points, picks
