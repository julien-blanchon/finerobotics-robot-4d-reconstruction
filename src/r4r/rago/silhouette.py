"""A static camera of one setup fitted to the robot's silhouette, on the CPU.

Some static cameras are re-placed between setups (source repos, stations, sessions) where
scene matches, neighbour seeds and transferred points cannot pin them. On a static camera the
robot is the main thing that moves, so its mask comes free from background subtraction, and
the description's surface, posed by the recorded joints, must cover it. The only place robot
meshes enter a solve; everything else stays mesh-free.

1. Frames: per sampled episode, `frames` evenly spaced frames; their per-pixel median is the
   episode's background. Its `rest` frame (joints nearest their median) is the robot the
   background shows.
2. Masks: a frame's CIELAB distance to the background above `threshold`, opened and closed.
   Foreground is where the robot is now and where it rests in the background (its ghost),
   plus held objects, cables, shadows and people.
3. Model: points sampled on the description's surfaces (area-weighted), posed by forward
   kinematics at each frame (`X = W_b(q_t) · x`) and at rest. A point that moved more than
   `moved` from its resting place must be on the mask; one that did not is background.
4. Loss over a setup's frames, `ρ(r) = r² / (1 + r²)` with `r = d / σ` (Geman-McClure: bounded,
   so masks' clutter and occluded parts cost at most 1 each):
   - each moved point to the mask: `d` its pixel's distance to the mask (distance transform);
   - each mask pixel (a sample) to the nearest projected point, now or at rest.
   Finger points count `finger_weight` (opening and held objects are uncertain).
5. Fit: Adam on the camera's rotation, centre (and log focal when free) from every start
   (seeds × perturbations) at once, σ annealed through `sigmas`; the lowest loss wins.
6. Score, on frames of other episodes than the fit's: the share of moved points inside the
   mask (`precision`) and the IoU of the drawn robot (now ∪ rest) with the mask. Low scores
   flag a setup whose camera is still wrong.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from einops import einsum, rearrange, repeat

from r4r.geometry import se3

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import mujoco
    from torch import Tensor

    from r4r.types import RGB, UV, XYZ, Mat4x4

    from .robot import BodyMount, CameraMount, Robot


# A frame is a candidate when this share of the robot's points moved from rest, and its mask
# covers less than this share of the image (more is a lighting change)
MOVED_SHARE = 0.2
MAX_AREA = 0.4
# A point explains mask pixels when its weight (moved share × finger weight) exceeds this
EXPLAINS = 0.05
# Joint values beyond this (rad or m) are corrupt state rows (robomind2 agilex: −292.66)
CORRUPT = 10.0


@dataclass(frozen=True, slots=True, kw_only=True)
class SilhouetteConfig:
    # Frames decoded per episode (background median, candidates) and kept per episode
    frames: int = 40
    kept: int = 3
    # Mask: CIELAB distance to the background, opening and closing kernels (pixels)
    threshold: float = 20.0
    open_px: int = 3
    close_px: int = 7
    # Surface points per body (area-weighted samples) and mask pixels sampled per frame
    points: int = 150
    samples: int = 300
    # A point moved this far from its resting place (m) counts fully as foreground
    moved: float = 0.04
    finger_weight: float = 0.3
    # Weight of the mask-to-model term against the model-to-mask one
    coverage_weight: float = 0.5
    # Robust scale schedule (pixels) and Adam steps per scale; learning rates (rad, m, log f)
    sigmas: tuple[float, ...] = (40.0, 20.0, 10.0, 5.0)
    steps: int = 60
    # Every start runs the first `coarse` scales; the `keep` best run the rest
    coarse: int = 2
    keep: int = 3
    # Surface points per body that explain mask pixels (a subset: the nearest-point search)
    explainers: int = 20
    lr_rotation: float = 0.01
    lr_position: float = 0.01
    lr_focal: float = 0.01
    # Starts per seed: perturbations of the seed's orientation (degrees) and position (m)
    turn_deg: float = 10.0
    shift: float = 0.1
    # Score: a moved point within this many pixels of the mask is inside it; a setup passes
    # with at least these held-out precision and IoU
    inside_px: float = 3.0
    pass_precision: float = 0.7
    pass_iou: float = 0.3


@dataclass(frozen=True, slots=True)
class Views:
    """A setup's frames prepared for the fit: per frame `F` the posed surface points now and at
    rest [F, N, 3], each point's weight (moved, finger), the mask's distance transform [F, H, W]
    (pixels to the nearest mask pixel), sampled mask pixels [F, M, 2] and the masks."""

    now: Tensor[[int, int, *XYZ]]
    rest: Tensor[[int, int, *XYZ]]
    weight: Tensor[[int, int]]
    distance: Tensor[[int, int, int]]
    samples: Tensor[[int, int, *UV]]
    masks: np.ndarray


@dataclass(frozen=True, slots=True)
class SilhouetteFit:
    """The fitted `base_from_camera`, focal (pixels) and loss, and the start it came from."""

    pose: Tensor[[*Mat4x4]]
    focal: float
    loss: float
    start: int


def surface_points(
    model: mujoco.MjModel, count: int, fingers: str = r"link7$|link8$|finger"
) -> tuple[Tensor[[int]], Tensor[[int, *XYZ]], Tensor[[int]]]:
    """`count` points per body on its drawn surfaces (`viz.overlay.surfaces`), area-weighted:
    each point's body, its position in the body frame and whether the body is a finger."""

    import re

    from r4r.viz.overlay import surfaces

    by_body: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for body, vertices, faces in surfaces(model):
        by_body.setdefault(body, []).append((vertices, faces))
    rng = np.random.default_rng(0)
    bodies, points, finger = [], [], []
    for body, parts in sorted(by_body.items()):
        triangles = np.concatenate([v[f] for v, f in parts])  # [faces, 3 corners, 3]
        area = np.linalg.norm(
            np.cross(*(triangles[:, 1:] - triangles[:, :1]).transpose(1, 0, 2)), axis=-1
        )
        if area.sum() <= 0:
            continue
        chosen = rng.choice(len(triangles), count, p=area / area.sum())
        a, b = rng.random((2, count, 1))
        flip = a + b > 1
        a, b = np.where(flip, 1 - a, a), np.where(flip, 1 - b, b)
        corner = triangles[chosen]
        points.append(
            corner[:, 0] + a * (corner[:, 1] - corner[:, 0]) + b * (corner[:, 2] - corner[:, 0])
        )
        bodies.append(np.full(count, body))
        finger.append(np.full(count, bool(re.search(fingers, model.body(body).name))))
    return (
        torch.from_numpy(np.concatenate(bodies)),
        torch.from_numpy(np.concatenate(points)),
        torch.from_numpy(np.concatenate(finger)),
    )


def background(frames: Tensor[[int, *RGB, int, int]]) -> np.ndarray:
    """The per-pixel median of uint8 frames [T, 3, H, W] -> [H, W, 3] uint8 (RGB)."""

    return np.median(rearrange(frames, "t c h w -> t h w c").numpy(), 0).astype(np.uint8)


def change_mask(frame: np.ndarray, back: np.ndarray, config: SilhouetteConfig) -> np.ndarray:
    """Pixels that differ from the background: CIELAB distance above `threshold`, opened and
    closed; [H, W] uint8 (255 foreground)."""

    # Auto exposure and white balance change between frames: the background scaled per
    # channel to the frame's median brightness first
    gain = np.median(frame.reshape(-1, 3), 0) / np.maximum(np.median(back.reshape(-1, 3), 0), 1)
    scaled = np.clip(back * gain, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_back = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB).astype(np.float32)
    mask = (np.linalg.norm(lab - lab_back, axis=-1) > config.threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((config.open_px,) * 2, np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((config.close_px,) * 2, np.uint8))


def rest_frame(qpos: Tensor[[int, int]]) -> int:
    """The frame whose joints are nearest their per-joint median: the robot the background
    (per-pixel median of the frames) mostly shows."""

    median = qpos.median(0).values
    return int((qpos - median).abs().sum(-1).argmin())


def posed(
    world_from_body: Tensor[[int, int, *Mat4x4]], body: Tensor[[int]], local: Tensor[[int, *XYZ]]
) -> Tensor[[int, int, *XYZ]]:
    """Surface points `X = W_b(q_t) · x` per frame [T, N, 3]."""

    pose = world_from_body[:, body]  # [T, N, 4, 4]
    return einsum(pose[..., :3, :3], local, "t n i j, n j -> t n i") + pose[..., :3, 3]


def views(
    now: Tensor[[int, int, *XYZ]],
    rest: Tensor[[int, int, *XYZ]],
    finger: Tensor[[int]],
    masks: Sequence[np.ndarray],
    config: SilhouetteConfig,
) -> Views:
    """`Views` from posed points now and at rest and each frame's mask."""

    moved = torch.linalg.norm(now - rest, dim=-1) / config.moved
    weight = moved.clamp(max=1.0) * torch.where(finger, config.finger_weight, 1.0)
    distance, samples = [], []
    rng = np.random.default_rng(0)
    for mask in masks:
        distance.append(cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5))
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            samples.append(np.full((config.samples, 2), np.nan))
            continue
        pick = rng.choice(len(xs), config.samples, replace=len(xs) < config.samples)
        samples.append(np.stack([xs[pick], ys[pick]], -1).astype(np.float64))
    # float32: the fit's precision is pixels
    return Views(
        now.float(),
        rest.float(),
        weight.float(),
        torch.from_numpy(np.stack(distance)).float(),
        torch.from_numpy(np.stack(samples)).float(),
        np.stack(masks),
    )


def project(
    rotation: Tensor[[int, 3, 3]],
    centre: Tensor[[int, *XYZ]],
    focal: Tensor[[int]],
    size: tuple[int, int],
    points: Tensor[[int, int, *XYZ]],
) -> tuple[Tensor[[int, int, int, *UV]], Tensor[[int, int, int]]]:
    """Pixels of points [F, N, 3] in each camera `S` (`world_from_camera` rotation, centre,
    focal; principal point at the image centre): [S, F, N, 2] and whether in front."""

    height, width = size
    local = einsum(rotation, points[None] - centre[:, None, None], "s j i, s f n j -> s f n i")
    depth = local[..., 2]
    xy = local[..., :2] / depth.clamp(min=1e-3)[..., None]
    centre_px = torch.tensor([(width - 1) / 2, (height - 1) / 2], dtype=points.dtype)
    return focal[:, None, None, None] * xy + centre_px, depth > 0.05


def _rho(distance: Tensor, sigma: float) -> Tensor:
    r = (distance / sigma) ** 2
    return r / (1 + r)


def loss(
    data: Views,
    rotation: Tensor[[int, 3, 3]],
    centre: Tensor[[int, *XYZ]],
    focal: Tensor[[int]],
    sigma: float,
    config: SilhouetteConfig,
) -> Tensor[[int]]:
    """Per camera `S`: weighted mean of moved points' robust distance to the mask, plus
    `coverage_weight` × mean of mask samples' robust distance to the robot (now or at rest)."""

    _frames, height, width = data.distance.shape
    step = max(config.points // config.explainers, 1)
    uv, front = project(rotation, centre, focal, (height, width), data.now)
    grid = torch.stack([uv[..., 0] / (width - 1), uv[..., 1] / (height - 1)], -1) * 2 - 1
    count = len(rotation)
    # Every camera's points in one lookup per frame (no copy of the distance maps)
    sampled = torch.nn.functional.grid_sample(
        data.distance[:, None],
        rearrange(grid, "s f n uv -> f (s n) 1 uv"),
        padding_mode="border",
        align_corners=True,
    )
    distance = rearrange(sampled, "f 1 (s n) 1 -> s f n", s=count)
    inside = (grid.abs() <= 1).all(-1) & front
    weight = data.weight[None] * inside
    # Points outside the image say nothing (the mask ends at the border)
    precision = (_rho(distance, sigma) * weight).sum((1, 2)) / weight.sum((1, 2)).clamp(min=1)

    # Only points that moved explain a change (a static body near the camera would explain
    # anything): where they are now and where the background shows them
    rest_uv, rest_front = project(rotation, centre, focal, (height, width), data.rest[:, ::step])
    explain = torch.cat([uv[:, :, ::step], rest_uv], 2)
    moving = (data.weight[:, ::step] > EXPLAINS)[None]
    valid = torch.cat([front[:, :, ::step] & moving, rest_front & moving], 2)
    samples = data.samples[None].expand(count, -1, -1, -1)
    with torch.no_grad():
        gap = torch.cdist(samples.nan_to_num(-1e4), explain)  # [S, F, M, 2N]
        gap = gap.masked_fill(~valid[:, :, None], math.inf)
        nearest = gap.argmin(-1)
    chosen = torch.gather(explain, 2, repeat(nearest, "s f m -> s f m uv", uv=2))
    near = torch.linalg.norm(samples - chosen, dim=-1)
    known = ~samples[..., 0].isnan() & valid.any(-1)[..., None]
    coverage = (_rho(near.nan_to_num(0.0), sigma) * known).sum((1, 2)) / known.sum((1, 2)).clamp(
        min=1
    )
    return precision + config.coverage_weight * coverage


def starts(
    seeds: Sequence[tuple[Tensor[[*Mat4x4]], float]], config: SilhouetteConfig
) -> tuple[Tensor[[int, 3, 3]], Tensor[[int, *XYZ]], Tensor[[int]]]:
    """Each seed (pose, focal) and its perturbations: turned by ±`turn_deg` about the camera's
    x and y axes, moved by ±`shift` along its optical axis."""

    rotations, centres, focals = [], [], []
    turn = math.radians(config.turn_deg)
    offsets = [(0.0, 0.0, 0.0)] + [
        (sx * turn, sy * turn, 0.0) for sx in (-1, 0, 1) for sy in (-1, 0, 1) if sx or sy
    ]
    for pose, focal in seeds:
        for ox, oy, _ in offsets:
            for depth in (0.0, -config.shift, config.shift) if ox == oy == 0 else (0.0,):
                turned = pose[:3, :3] @ se3.so3_exp(torch.tensor([ox, oy, 0.0], dtype=pose.dtype))
                rotations.append(turned)
                centres.append(pose[:3, 3] + depth * pose[:3, 2])
                focals.append(focal)
    return torch.stack(rotations), torch.stack(centres), torch.tensor(focals, dtype=torch.float64)


def fit(
    data: Views,
    seeds: Sequence[tuple[Tensor[[*Mat4x4]], float]],
    free_focal: bool,
    config: SilhouetteConfig,
) -> SilhouetteFit:
    """The camera from every start at once through the coarse scales, then the `keep` best
    through the fine ones (module doc, step 5)."""

    rotation, centre, focal = (x.to(data.now) for x in starts(seeds, config))
    coarse, fine = config.sigmas[: config.coarse], config.sigmas[config.coarse :]
    rotation, centre, focal, losses = _descend(
        data, rotation, centre, focal, coarse, free_focal, config
    )
    best = torch.argsort(losses)[: config.keep]
    rotation, centre, focal, losses = _descend(
        data, rotation[best], centre[best], focal[best], fine, free_focal, config
    )
    winner = int(losses.argmin())
    pose = se3.from_rotation_translation(rotation[winner], centre[winner]).double()
    return SilhouetteFit(pose, float(focal[winner]), float(losses[winner]), int(best[winner]))


def _descend(
    data: Views,
    rotation0: Tensor[[int, 3, 3]],
    centre0: Tensor[[int, *XYZ]],
    focal0: Tensor[[int]],
    sigmas: Sequence[float],
    free_focal: bool,
    config: SilhouetteConfig,
) -> tuple[Tensor[[int, 3, 3]], Tensor[[int, *XYZ]], Tensor[[int]], Tensor[[int]]]:
    """Adam on every camera's pose (and log focal) through `sigmas`: the cameras and their
    losses at the last scale.

    The pose moves in the camera frame about a pivot `p` on the optical axis at the robot's
    median depth: `X_c = E (X_c0 − p) + p + t`, `E = exp(ω)`, i.e. `R = R0 Eᵀ`,
    `c = c0 + R0 p − R (p + t)`. Turning about the robot rather than the camera centre keeps
    rotation and translation nearly independent (a camera turned about its centre moves the
    whole image, which a translation then undoes: a slow valley for Adam).
    """

    with torch.no_grad():
        local = einsum(rotation0, data.now[None] - centre0[:, None, None], "s j i, s f n j -> s f n i")
        depth = local[..., 2].flatten(1).median(1).values.clamp(min=0.1)
    pivot = torch.zeros_like(centre0)
    pivot[:, 2] = depth
    turn = torch.zeros_like(centre0, requires_grad=True)
    move = torch.zeros_like(centre0, requires_grad=True)
    zoom = torch.zeros_like(focal0, requires_grad=True)
    groups = [
        {"params": [turn], "lr": config.lr_rotation},
        {"params": [move], "lr": config.lr_position},
    ]
    if free_focal:
        groups.append({"params": [zoom], "lr": config.lr_focal})
    optimiser = torch.optim.Adam(groups)

    def camera() -> tuple[Tensor, Tensor, Tensor]:
        rotation = rotation0 @ se3.so3_exp(turn).mT
        centre = (
            centre0
            + einsum(rotation0, pivot, "s i j, s j -> s i")
            - einsum(rotation, pivot + move, "s i j, s j -> s i")
        )
        return rotation, centre, focal0 * torch.exp(zoom)

    for sigma in sigmas:
        for _ in range(config.steps):
            optimiser.zero_grad()
            loss(data, *camera(), sigma, config).sum().backward()
            optimiser.step()
    with torch.no_grad():
        rotation, centre, focal = camera()
        return rotation, centre, focal, loss(data, rotation, centre, focal, sigmas[-1], config)


@dataclass(frozen=True, slots=True)
class SilhouetteScore:
    """On held-out frames: the share of moved surface points inside the mask, and the IoU of
    the drawn robot (now ∪ rest) with the mask (medians over frames)."""

    precision: float
    iou: float
    frames: int


def score(
    data: Views,
    pose: Tensor[[*Mat4x4]],
    focal: float,
    drawn: Sequence[np.ndarray],
    config: SilhouetteConfig,
) -> SilhouetteScore:
    """`SilhouetteScore` of a camera on `data` (frames not used in its fit); `drawn` are the
    robot's rendered silhouettes (now ∪ rest) with that camera, one per frame."""

    frames, height, width = data.distance.shape
    camera = pose.to(data.now)
    focal_t = torch.tensor([focal]).to(data.now)
    uv, front = project(
        camera[None, :3, :3], camera[None, :3, 3], focal_t, (height, width), data.now
    )
    u, v = uv[0, ..., 0].round().long(), uv[0, ..., 1].round().long()
    inside = front[0] & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    near = torch.zeros_like(inside)
    index = torch.arange(frames)[:, None].expand_as(u)
    near[inside] = data.distance[index[inside], v[inside], u[inside]] <= config.inside_px
    moved = (data.weight > 0.5) & inside
    # Pooled over frames: many frames show few moved points
    precision = (near & moved).sum() / moved.sum().clamp(min=1)
    ious = []
    for mask, robot in zip(data.masks, drawn, strict=True):
        union = np.logical_or(mask > 0, robot > 0).sum()
        ious.append(np.logical_and(mask > 0, robot > 0).sum() / max(union, 1))
    return SilhouetteScore(float(precision), float(np.median(ious)), frames)


def render(
    world_from_body: np.ndarray,
    world_from_camera: np.ndarray,
    focal: float,
    size: tuple[int, int],
    geoms: Sequence[tuple[int, np.ndarray, np.ndarray]],
) -> np.ndarray:
    """The robot's silhouette [H, W] uint8 (255 on the robot): every surface triangle
    (`viz.overlay.surfaces`) filled at its projection."""

    height, width = size
    out = np.zeros((height, width), np.uint8)
    camera_from_world = np.linalg.inv(world_from_camera)
    intrinsics = np.array([[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]])
    for body, vertices, faces in geoms:
        pose = camera_from_world @ world_from_body[body]
        local = vertices @ pose[:3, :3].T + pose[:3, 3]
        ok = (local[:, 2] > 0.02)[faces].all(1)
        pixel = (local @ intrinsics.T)[:, :2] / np.maximum(local[:, 2:3], 1e-6)
        polygons = np.round(pixel[faces[ok]] * 4).astype(np.int64)
        near = (np.abs(polygons) < 8 * max(width, height) * 4).all((1, 2))
        if near.any():
            cv2.fillPoly(out, list(polygons[near].astype(np.int32)), 255, shift=2)
    return out


@dataclass(frozen=True, slots=True)
class EpisodeFrames:
    """One episode's frames kept for a fit or a score (module doc, steps 1-3): the posed surface
    points now and at rest [F, N, 3], which points are fingers, the masks, the bodies' poses
    now [F, L, 4, 4] and at rest [L, 4, 4] (for drawing) and the frame indices."""

    now: Tensor[[int, int, *XYZ]]
    rest: Tensor[[int, int, *XYZ]]
    finger: Tensor[[int]]
    masks: list[np.ndarray]
    world: Tensor[[int, int, *Mat4x4]]
    world_rest: Tensor[[int, *Mat4x4]]
    frames: list[int]


def episode_frames(
    images: Tensor[[int, *RGB, int, int]],
    world: Tensor[[int, int, *Mat4x4]],
    qpos: Tensor[[int, int]],
    frames: Sequence[int],
    model: tuple[Tensor[[int]], Tensor[[int, *XYZ]], Tensor[[int]]],
    config: SilhouetteConfig,
) -> EpisodeFrames:
    """From an episode's `frames` (decoded `images`, bodies' poses `world`, joints `qpos`): the
    background, the rest frame and the `kept` frames whose masks are largest (below 40 % of the
    image: a lighting change, not the robot) among those where the robot moved from rest, at
    least 3 frames apart. `model`: `surface_points`."""

    body, local, finger = model
    back = background(images)
    rest = rest_frame(qpos)
    now = posed(world, body, local)
    at_rest = now[rest : rest + 1].expand_as(now)
    moved = (torch.linalg.norm(now - at_rest, dim=-1) / config.moved).clamp(max=1).mean(1)
    pictures = rearrange(images, "t c h w -> t h w c").numpy()
    masks = [change_mask(p, back, config) for p in pictures]
    area = torch.tensor([float((m > 0).mean()) for m in masks])
    order = torch.argsort(area * (moved > MOVED_SHARE) * (area < MAX_AREA), descending=True)
    kept: list[int] = []
    for i in order.tolist():
        if all(abs(i - j) >= 3 for j in kept):
            kept.append(i)
        if len(kept) == config.kept:
            break
    return EpisodeFrames(
        now[kept],
        at_rest[kept],
        finger,
        [masks[i] for i in kept],
        world[kept],
        world[rest],
        [frames[i] for i in kept],
    )


def stacked(episodes: Sequence[EpisodeFrames], config: SilhouetteConfig) -> Views:
    """`Views` of several episodes' kept frames together."""

    return views(
        torch.cat([e.now for e in episodes]),
        torch.cat([e.rest for e in episodes]),
        episodes[0].finger,
        [m for e in episodes for m in e.masks],
        config,
    )


def scored(
    episodes: Sequence[EpisodeFrames],
    pose: Tensor[[*Mat4x4]],
    focal: float,
    geoms: Sequence[tuple[int, np.ndarray, np.ndarray]],
    config: SilhouetteConfig,
) -> SilhouetteScore:
    """`score` of a camera on episodes' kept frames, the robot drawn now ∪ at rest."""

    drawn = []
    camera = pose.double().numpy()
    for episode in episodes:
        rest = episode.world_rest.double().numpy()
        for k, mask in enumerate(episode.masks):
            size = mask.shape[:2]
            now = render(episode.world[k].double().numpy(), camera, focal, size, geoms)
            both = now | render(rest, camera, focal, size, geoms)
            # Decimated meshes leave pinholes
            drawn.append(cv2.morphologyEx(both, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)))
    return score(stacked(episodes, config), pose, focal, drawn, config)


@dataclass(frozen=True, slots=True)
class Loaded:
    """An episode's kept frames for one static camera, the camera's prior placement
    (`base_from_camera` or None, focal or None, the focal's relative range) and the description's
    drawn surfaces (for `scored`)."""

    frames: EpisodeFrames
    prior: tuple[Tensor[[*Mat4x4]] | None, float | None, float | None]
    geoms: list[tuple[int, np.ndarray, np.ndarray]]


def load(root: Path, number: int, camera: str, config: SilhouetteConfig) -> Loaded | None:
    """Episode `number` of the dataset `root` prepared for `camera` (module doc, steps 1-3):
    `frames` frames decoded alone, the description posed by the priors' corrected kinematics.
    None when its state holds corrupt values (|q| > 10)."""

    from r4r.data.lerobot import keyframes, load_episode

    from .kinematics import corrected_qpos, forward_kinematics
    from .robot import CameraMount, body_mounts, qpos_from_state

    episode = load_episode(root, number, decode=())
    mounts = tuple(
        CameraMount.from_prior(c.name, c.prior, c.mount, c.moving) for c in episode.cameras
    )
    bodies = body_mounts(episode.priors)
    robot, model, geoms = _model(episode.robot, mounts, bodies, config.points)
    qpos = qpos_from_state(robot, episode.state.to(torch.float64), episode.joints)
    fitted = episode.priors.kinematics
    if fitted is not None:
        qpos = corrected_qpos(robot, qpos, fitted.offsets, fitted.time_offset)
    if bool((qpos.abs() > CORRUPT).any()):
        return None
    frames = np.linspace(0, len(qpos) - 1, config.frames).round().astype(int).tolist()
    images = keyframes(root, number, frames, [camera])[camera]
    world = forward_kinematics(robot, qpos[frames])
    kept = episode_frames(images, world, qpos[frames], frames, model, config)
    prior = next(c.prior for c in episode.cameras if c.name == camera)
    pose = None
    if prior.position is not None and prior.orientation is not None:
        rotation = se3.quaternion_to_matrix(torch.tensor(prior.orientation, dtype=torch.float64))
        pose = se3.from_rotation_translation(
            rotation, torch.tensor(prior.position, dtype=torch.float64)
        )
    return Loaded(kept, (pose, prior.focal, prior.focal_range), geoms)


@functools.cache
def _model(
    path: Path, mounts: tuple[CameraMount, ...], bodies: tuple[BodyMount, ...], points: int
) -> tuple[Robot[int, int, int], tuple[Tensor[[int]], Tensor[[int, *XYZ]], Tensor[[int]]], list]:
    """A description's kinematics, surface points and drawn surfaces, once per process."""

    from r4r.viz.overlay import surfaces

    from .robot import compile_model, load_robot

    robot = load_robot(path, cameras=mounts, bodies=bodies).to("cpu", torch.float64)
    model = compile_model(path, None, mounts, bodies, visual=True)
    return robot, surface_points(model, points), surfaces(model)
