"""Robot keypoints transferred from calibrated frames by dense matching: picks without clicks.

In a reference frame whose camera is known (a calibrated station), every robot point's pixel
is known, `u = π_K(P⁻¹ W_b(q) p)`. RoMa v2 matches the reference with a target frame of an
uncalibrated camera of the same robot; the warp at `u` is the point's pixel in the target:

    u' = w_ab(u),  kept when  c_ab(u) > certainty  and  ‖w_ba(u') − u‖ < cycle

`c_ab` is RoMa's overlap probability; the forward-backward check drops points hidden or
ambiguous in the target. Each target frame is matched with the reference frames of nearest
joint configuration (`‖q − q_ref‖`, radians and metres): the robot then looks most alike
(gripper opening, wrist roll). With several, a point's pixel is the median over those where it
passes, kept when they are a majority. The reference is cropped around its projected points (with a
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
    # Reference frames matched with each target frame (nearest joint configurations); a point
    # is kept where it passes the checks for a majority of them
    references: int = 1


@dataclass(frozen=True, slots=True)
class Reference[R: IntVar, NQ: IntVar, P: IntVar]:
    """`R` frames of a calibrated camera: images, joint positions and the pixels of `P`
    robot points, with whether each is seen."""

    frames: Tensor[[R, *RGB, int, int]]  # uint8
    qpos: Tensor[[R, NQ]]
    pixels: Tensor[[R, P, *UV]]
    seen: Tensor[[R, P]]


@dataclass(frozen=True, slots=True)
class Matched[N: IntVar, NRef: IntVar, P: IntVar]:
    """`P` points carried from `NRef` reference frames (`reference`, indices) into each of `N`
    target frames: pixels, RoMa certainty, forward-backward distance (reference pixels) and
    whether the reference sees the point."""

    pixels: Tensor[[N, NRef, P, *UV]]
    certainty: Tensor[[N, NRef, P]]
    cycle: Tensor[[N, NRef, P]]
    seen: Tensor[[N, NRef, P]]
    reference: Tensor[[N, NRef]]


@dataclass(frozen=True, slots=True)
class Transferred[N: IntVar, P: IntVar]:
    """`P` points in `N` target frames: pixel (median over the references that passed the
    checks), how many passed, and whether that is a majority of the references."""

    pixels: Tensor[[N, P, *UV]]
    votes: Tensor[[N, P]]
    kept: Tensor[[N, P]]


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
    reference: Reference[R, NQ, int], qpos: Tensor[[N, NQ]], count: int
) -> Tensor[[N, int]]:
    """For each joint configuration, the `count` reference frames nearest to it (nearest
    first) among those that see at least `MIN_SEEN` points."""

    distance = torch.cdist(qpos, reference.qpos.to(qpos))
    usable = reference.seen.sum(-1) >= MIN_SEEN
    distance = distance.masked_fill(~usable.to(qpos.device), torch.inf)
    return distance.topk(min(count, len(reference.qpos)), dim=-1, largest=False).indices


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


def match_points[N: IntVar, NRef: IntVar, P: IntVar](
    model: RoMaV2,
    reference: Reference[int, int, P],
    chosen: Tensor[[N, NRef]],
    targets: Tensor[[N, *RGB, int, int]],
    margin: float,
) -> Matched[N, NRef, P]:
    """Each target frame (uint8) matched with its `chosen` reference frames, each cropped to
    its seen points (`crop_box`) and resized to the target's size."""

    device = next(model.parameters()).device
    count = chosen.shape[1]
    pairs = rearrange(chosen.cpu(), "n r -> (n r)").tolist()
    height, width = targets.shape[-2:]
    found, certainty, cycle = [], [], []
    for start in range(0, len(pairs), BATCH):
        crops, pixels, zoom = [], [], []
        for r in pairs[start : start + BATCH]:
            x0, y0, x1, y1 = crop_box(
                reference.pixels[r][reference.seen[r]],
                (reference.frames.shape[-2], reference.frames.shape[-1]),
                width / height,
                margin,
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
        index = torch.arange(start, start + len(crops)) // count
        target = targets[index].to(device).float() / 255
        encoding = model.encode(torch.cat((torch.stack(crops), target)))
        n = len(crops)
        a = torch.arange(n, device=device)
        matched = read_warp(
            model.match(encoding[a], encoding[a + n]), torch.stack(pixels).to(device)
        )
        found.append(matched[0].cpu())
        certainty.append(matched[1].cpu())
        # Forward-backward distance back in reference pixels
        cycle.append(matched[2].cpu() / torch.tensor(zoom, dtype=matched[2].dtype)[:, None])
    like = reference.pixels

    def per_target(parts: list[Tensor]) -> Tensor:  # [N · NRef, P, ...] → [N, NRef, P, ...]
        return rearrange(torch.cat(parts).to(like), "(n r) ... -> n r ...", r=count)

    return Matched(
        per_target(found),
        per_target(certainty),
        per_target(cycle),
        reference.seen[chosen.to(reference.seen.device)],
        chosen,
    )


def pooled[N: IntVar, NRef: IntVar, P: IntVar](
    matched: Matched[N, NRef, P], certainty: float, cycle: float
) -> Transferred[N, P]:
    """A point passes for one reference when it is seen there, `c > certainty` and the
    forward-backward distance is below `cycle`; its pixel is the median over the references
    where it passes, kept when they are a majority (`> NRef / 2`)."""

    passed = matched.seen & (matched.certainty > certainty) & (matched.cycle < cycle)
    masked = matched.pixels.masked_fill(~passed[..., None], torch.nan)
    pixels = masked.nanmedian(dim=1).values
    votes = passed.sum(1)
    kept = 2 * votes > matched.reference.shape[1]
    return Transferred(torch.nan_to_num(pixels), votes, kept)


def transfer[N: IntVar, P: IntVar](
    model: RoMaV2,
    reference: Reference[int, int, P],
    targets: Tensor[[N, *RGB, int, int]],
    target_qpos: Tensor[[N, int]],
    config: TransferConfig,
) -> Transferred[N, P]:
    """The reference's points in each target frame (uint8), from its `config.references`
    reference frames of nearest joint configuration."""

    chosen = nearest(reference, target_qpos.to(reference.qpos), config.references)
    matched = match_points(model, reference, chosen, targets, config.margin)
    return pooled(matched, config.certainty, config.cycle)


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
