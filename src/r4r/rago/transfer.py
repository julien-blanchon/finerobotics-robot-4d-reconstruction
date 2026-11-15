"""Robot keypoints transferred from calibrated frames by dense matching: picks without clicks.

In a reference frame whose camera is known (a calibrated station), every robot point's pixel
is known, `u = π_K(P⁻¹ W_b(q) p)`. RoMa v2 matches the reference with a target frame of an
uncalibrated camera of the same robot; the warp at `u` is the point's pixel in the target:

    u' = w_ab(u),  kept when  c_ab(u) > certainty  and  ‖w_ba(u') − u‖ < cycle

`c_ab` is RoMa's overlap probability; the forward-backward check drops points hidden or
ambiguous in the target. Each target frame is matched with the reference frame of nearest
joint configuration (`‖q − q_ref‖`, radians and metres): the robot then looks most alike
(gripper opening, wrist roll). The reference is cropped around its projected points (with a
margin) so the robot fills the matched image.

The pixels are picks (`r4r.data.priors.Pick`), as if clicked, for `r4r rig --picks`: a PnP-RANSAC
start, then the bundle's Cauchy loss handles the outliers left.

Limits (measured in docs/dataset-onboarding.md): a named point is rarely a surface point (a
joint origin lies inside its housing); the warp carries the surface in front of it, which
projects onto the point only from the reference's viewing direction, so transferred pixels
drift by about `r · sin Δθ` (housing radius `r`, change of viewing direction `Δθ`). A point
hidden in the reference by another link transfers that link's surface.

Ref: Edstedt et al., RoMa v2, arXiv:2511.15706.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from einops import rearrange

from r4r.data.priors import Pick
from r4r.geometry import camera, se3

from .kinematics import forward_kinematics

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.data.priors import RobotPoint
    from r4r.matching import Matches, RoMaV2
    from r4r.types import RGB, UV, Mat3x3, Mat4x4

    from .robot import Robot

# Pairs matched at once (each encodes two frames)
BATCH = 8
# A projected point counts as seen this far inside the image (pixels) and in front (m)
BORDER = 2.0
MIN_DEPTH = 0.05
# A reference frame needs this many seen points to be used
MIN_SEEN = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class TransferConfig:
    # Minimum RoMa overlap probability at the reference pixel
    certainty: float = 0.1
    # Maximum forward-backward distance (reference pixels)
    cycle: float = 3.0
    # Reference crop: the box of its seen points grown by this fraction of its size on each
    # side (inf: the whole frame)
    margin: float = 0.5
    # Target frames per episode, spread over its joint configurations
    frames: int = 8


@dataclass(frozen=True, slots=True)
class Reference[R: IntVar, NQ: IntVar, P: IntVar]:
    """`R` frames of a calibrated camera: images, joint positions and the pixels of `P`
    robot points, with whether each is seen."""

    frames: Tensor[[R, *RGB, int, int]]  # uint8
    qpos: Tensor[[R, NQ]]
    pixels: Tensor[[R, P, *UV]]
    seen: Tensor[[R, P]]


@dataclass(frozen=True, slots=True)
class Transferred[N: IntVar, P: IntVar]:
    """`P` points in `N` target frames: pixels, RoMa certainty, forward-backward distance
    (reference pixels), whether the point passed both checks, and the reference frame used."""

    pixels: Tensor[[N, P, *UV]]
    certainty: Tensor[[N, P]]
    cycle: Tensor[[N, P]]
    kept: Tensor[[N, P]]
    reference: Tensor[[N]]


def project_points[T: IntVar, NQ: IntVar](
    robot: Robot[int, int, int],
    qpos: Tensor[[T, NQ]],
    points: Sequence[RobotPoint],
    world_from_camera: Tensor[[*Mat4x4]],
    intrinsics: Tensor[[*Mat3x3]],
    size: tuple[int, int],
) -> tuple[Tensor[[T, int, *UV]], Tensor[[T, int]]]:
    """Pixels of `points` at every timestep in a static camera of image `size` (H, W), and
    whether each is seen (in front, inside the image): `u = π_K(P⁻¹ W_b(q_t) p)`."""

    world = forward_kinematics(robot, qpos)
    body = torch.tensor([robot.body(p.body) for p in points], device=qpos.device)
    offset = qpos.new_tensor([p.offset for p in points])
    camera_from_body = se3.inverse(world_from_camera) @ world[:, body]
    local = se3.transform_points(camera_from_body, offset[None, :, None])[..., 0, :]
    t = len(qpos)
    pixels = camera.project(intrinsics.expand(t, len(points), 3, 3), local[..., None, :])[..., 0, :]
    height, width = size
    inside = (pixels >= -0.5 + BORDER).all(-1)
    inside &= (pixels[..., 0] <= width - 0.5 - BORDER) & (pixels[..., 1] <= height - 0.5 - BORDER)
    return pixels, inside & (local[..., 2] > MIN_DEPTH)


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
    reference: Reference[R, NQ, int], qpos: Tensor[[N, NQ]]
) -> Tensor[[N]]:
    """For each joint configuration, the reference frame nearest to it among those that see
    at least `MIN_SEEN` points."""

    distance = torch.cdist(qpos, reference.qpos.to(qpos))
    usable = reference.seen.sum(-1) >= MIN_SEEN
    return distance.masked_fill(~usable.to(qpos.device), torch.inf).argmin(-1)


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
    low, high = pixels.amin(0), pixels.amax(0)
    centre = (low + high) / 2
    half = (high - low) / 2 * (1 + 2 * margin) + 16
    half_w, half_h = float(half[0]), float(half[1])
    half_w, half_h = max(half_w, half_h * aspect), max(half_h, half_w / aspect)
    half_w, half_h = min(half_w, width / 2), min(half_h, height / 2)
    x0 = min(max(float(centre[0]) - half_w, 0.0), width - 2 * half_w)
    y0 = min(max(float(centre[1]) - half_h, 0.0), height - 2 * half_h)
    return round(x0), round(y0), round(x0 + 2 * half_w), round(y0 + 2 * half_h)


def transfer[N: IntVar, P: IntVar](
    model: RoMaV2,
    reference: Reference[int, int, P],
    targets: Tensor[[N, *RGB, int, int]],
    target_qpos: Tensor[[N, int]],
    config: TransferConfig,
) -> Transferred[N, P]:
    """The reference's points in each target frame (uint8), each matched with its nearest
    reference frame (cropped to its seen points, resized to the target's size)."""

    device = next(model.parameters()).device
    chosen = nearest(reference, target_qpos.to(reference.qpos)).cpu()
    height, width = targets.shape[-2:]
    found, certainty, cycle = [], [], []
    for start in range(0, len(targets), BATCH):
        crops, pixels, zoom = [], [], []
        for r in chosen[start : start + BATCH].tolist():
            x0, y0, x1, y1 = crop_box(
                reference.pixels[r][reference.seen[r]],
                (reference.frames.shape[-2], reference.frames.shape[-1]),
                width / height,
                config.margin,
            )
            image = reference.frames[r, :, y0:y1, x0:x1].to(device).float() / 255
            crops.append(
                F.interpolate(image[None], (height, width), mode="bilinear", antialias=True)[0]
            )
            # Reference pixels → crop pixels: `u' = (u − x0 + ½) · s − ½`, `s` per axis
            scale = reference.pixels.new_tensor((width / (x1 - x0), height / (y1 - y0)))
            offset = reference.pixels.new_tensor((x0, y0))
            pixels.append((reference.pixels[r] - offset + 0.5) * scale - 0.5)
            zoom.append(float(scale.mean()))
        target = targets[start : start + BATCH].to(device).float() / 255
        encoding = model.encode(torch.cat((torch.stack(crops), target)))
        n = len(crops)
        index = torch.arange(n, device=device)
        matched = read_warp(
            model.match(encoding[index], encoding[index + n]), torch.stack(pixels).to(device)
        )
        found.append(matched[0].cpu())
        certainty.append(matched[1].cpu())
        # Forward-backward distance back in reference pixels
        cycle.append(matched[2].cpu() / torch.tensor(zoom, dtype=matched[2].dtype)[:, None])
    like = reference.pixels
    pixels_t = torch.cat(found).to(like)
    certainty_t, cycle_t = torch.cat(certainty).to(like), torch.cat(cycle).to(like)
    kept = reference.seen[chosen].to(like.device)
    kept &= (certainty_t > config.certainty) & (cycle_t < config.cycle)
    return Transferred(pixels_t, certainty_t, cycle_t, kept, chosen)


def as_picks[N: IntVar, P: IntVar](
    transferred: Transferred[N, P],
    names: Sequence[str],
    episode: int,
    camera_name: str,
    frames: Sequence[int],
) -> list[Pick]:
    """One pick per target frame (the video's own frame index) with its kept points."""

    picks = []
    for i, frame in enumerate(frames):
        pixels = {
            name: (float(xy[0]), float(xy[1]))
            for name, xy, kept in zip(
                names, transferred.pixels[i].tolist(), transferred.kept[i].tolist(), strict=True
            )
            if kept
        }
        if pixels:
            picks.append(Pick(episode, camera_name, frame, pixels))
    return picks
