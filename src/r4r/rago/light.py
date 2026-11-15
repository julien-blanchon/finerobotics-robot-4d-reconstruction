"""Episode-scope unknowns from a few short bursts of frames, on the CPU: the light path of
`r4r reconstruct`.

Once a dataset (or station) is calibrated, an episode may still have a few 6-DoF unknowns of
its own: a re-placed robot base, a re-placed static camera, a recorded pose that does not fit.
They are fitted from `bursts` short runs of frames only (decoded alone,
`data.lerobot.keyframes`), not the whole video:

1. KLT tracks within each burst, per camera (`klt.track_video`): sub-pixel, on the static
   scene and on the moving robot alike.
2. Links between cameras at each burst's middle frame: SIFT (RootSIFT, on small videos
   upscaled), Lowe's ratio test and mutual matching, then a fundamental-matrix RANSAC on
   undistorted coordinates (no pose needed); they join tracks across cameras (`merge`).
3. Levenberg–Marquardt on the free rows only (`solver.levenberg_marquardt`), every other
   mount, body, focal and lens held, a moving base on its (corrected) record. As in the bundle,
   each track is attached to the body that explains it best (the world, or a moving link: the
   robot is the calibration target that pins a static camera, `bundle.attach`).

The same tracks with nothing free give the **fit check**: the median reprojection error of the
held values, `e = median_m ‖π(P_m⁻¹ X̂) − p_m‖` (pixels, tracks attached to their best body).

Ref: Lowe, Distinctive Image Features from Scale-Invariant Keypoints, IJCV 2004; Arandjelović
     & Zisserman, Three things everyone should know to improve object retrieval, CVPR 2012
     (RootSIFT); Hartley & Zisserman, Multiple View Geometry, ch. 11 (RANSAC on F).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from einops import rearrange

from r4r.geometry import camera, se3

from .bundle import WORLD, BundleConfig, Observations, attach, moving_bodies, residuals
from .klt import KLTConfig, track_video
from .merge import merge_tracks
from .rig import Rig
from .robot import Freedom
from .solver import LMConfig, levenberg_marquardt
from .tracks import Tracks

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.types import RGB, UV, Mat3x3, Mat4x4

    from .lens import Lens
    from .robot import Robot


@dataclass(frozen=True, slots=True, kw_only=True)
class LightConfig:
    # Bursts of `burst` timesteps, `every` apart, spread over the episode
    bursts: int = 6
    burst: int = 16
    every: int = 4
    klt: KLTConfig = field(default_factory=lambda: KLTConfig(features=400))
    # SIFT on images upscaled to at least `side` pixels wide (small videos give few features),
    # at most `features` per image, contrast threshold `contrast`; Lowe's ratio; RANSAC
    # threshold (pixels at the camera's focal); pairs with fewer verified matches are left out
    side: int = 480
    features: int = 4000
    contrast: float = 0.005
    ratio: float = 0.85
    ransac: float = 1.5
    min_matches: int = 12
    # Moments per burst matched between cameras
    links: int = 2
    # A static camera's track moving less than this (pixels) is on the still scene; the light
    # solve keeps at most `tracks` tracks
    still: float = 2.0
    tracks: int = 1500
    # The fit check flags an episode whose median epipolar distance exceeds this (pixels;
    # DROID's 320×180 exteriors: 4-19 px where the record fits, 36 px on episode 1000)
    flag: float = 25.0
    bundle: BundleConfig = field(
        default_factory=lambda: BundleConfig(cauchy=0.01, lm=LMConfig(rounds=2, iterations=15))
    )


@dataclass(frozen=True, slots=True)
class FitCheck:
    """A median error (pixels; NaN when nothing matched) and how many matches or observations
    it is over."""

    median_px: float
    observations: int

    def flagged(self, config: LightConfig) -> bool:
        return self.observations >= config.min_matches and self.median_px > config.flag


def burst_steps(steps: int, config: LightConfig) -> list[list[int]]:
    """`config.bursts` runs of `config.burst` timesteps `config.every` apart, spread evenly over
    an episode of `steps` timesteps (fewer in a short episode)."""

    span = (config.burst - 1) * config.every
    if steps <= span + 1:
        return [list(range(0, steps, config.every))]
    count = max(1, min(config.bursts, steps // (span + 1)))
    starts = [round(i * (steps - 1 - span) / max(count - 1, 1)) for i in range(count)]
    return [list(range(s, s + span + 1, config.every)) for s in starts]


@dataclass(frozen=True, slots=True)
class Links[K: IntVar]:
    """`K` verified matches between two cameras at one moment: each end's view and pixel."""

    view_a: Tensor[[K]]
    pixel_a: Tensor[[K, *UV]]
    view_b: Tensor[[K]]
    pixel_b: Tensor[[K, *UV]]


def burst_links(
    frames: Sequence[Tensor[[int, *RGB, int, int]]],
    bursts: Sequence[int],
    intrinsics: Tensor[[int, *Mat3x3]],
    lens: Lens[int] | None,
    static: Sequence[bool],
    config: LightConfig,
) -> Links[int]:
    """Verified SIFT matches between every pair of cameras at `config.links` moments of each
    burst (`frames`: per camera, the same `K` moments, bursts of `bursts[i]` frames in order); view
    `c · K + k` is camera c at moment k (module doc, step 2). Two `static` cameras keep one
    relative pose: their matches of every moment are verified by a single RANSAC, which
    rejects far more wrong matches than one RANSAC per moment."""

    cameras, count = len(frames), len(frames[0])
    first = np.cumsum([0, *bursts[:-1]]).tolist()
    # OpenCV's stubs lack SIFT_create (in the main module since 4.4)
    sift = cv2.SIFT_create(  # type: ignore[pyrefly:missing-attribute]  # see above
        nfeatures=config.features, contrastThreshold=config.contrast
    )
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    found: dict[tuple[int, int], list[tuple[int, np.ndarray, np.ndarray]]] = {}
    moments = [
        start + (length - 1) * (i + 1) // (config.links + 1)
        for start, length in zip(first, bursts, strict=True)
        for i in range(config.links)
    ]
    for k in moments:
        features = [_features(sift, frames[c][k], config.side) for c in range(cameras)]
        rays = [_undistorted(intrinsics, lens, c, features[c][0]) for c in range(cameras)]
        for a in range(cameras):
            for b in range(a + 1, cameras):
                pairs = _candidates(matcher, features[a], features[b], config)
                if pairs is None:
                    continue
                ia, ib = pairs
                xa, xb = rays[a][ia], rays[b][ib]
                finite = np.isfinite(xa).all(1) & np.isfinite(xb).all(1)
                pixels = np.concatenate((features[a][0][ia], features[b][0][ib]), 1)
                rays_ab = np.concatenate((xa, xb), 1)
                found.setdefault((a, b), []).append((k, pixels[finite], rays_ab[finite]))
    view_a, pixel_a, view_b, pixel_b = [], [], [], []
    for (a, b), moments in found.items():
        focal = float(min(intrinsics[a, 0, 0], intrinsics[b, 0, 0]))
        groups = [moments] if static[a] and static[b] else [[m] for m in moments]
        for group in groups:
            rays_ab = np.concatenate([m[2] for m in group])
            keep = _consistent(rays_ab[:, :2], rays_ab[:, 2:], focal, config)
            if keep is None:
                continue
            moment = np.concatenate([np.full(len(m[1]), m[0]) for m in group])[keep]
            pixels = np.concatenate([m[1] for m in group])[keep]
            view_a.append(torch.from_numpy(a * count + moment))
            view_b.append(torch.from_numpy(b * count + moment))
            pixel_a.append(torch.from_numpy(pixels[:, :2]))
            pixel_b.append(torch.from_numpy(pixels[:, 2:]))
    dtype = intrinsics.dtype
    if not view_a:
        none = torch.zeros(0, dtype=torch.long)
        return Links(none, torch.zeros(0, 2, dtype=dtype), none, torch.zeros(0, 2, dtype=dtype))
    return Links(
        torch.cat(view_a),
        torch.cat(pixel_a).to(dtype),
        torch.cat(view_b),
        torch.cat(pixel_b).to(dtype),
    )


def burst_tracks[C: IntVar](
    frames: Sequence[Tensor[[int, *RGB, int, int]]],
    bursts: Sequence[int],
    intrinsics: Tensor[[C, *Mat3x3]],
    lens: Lens[C] | None,
    links: Links[int],
    config: LightConfig,
) -> Tracks[int, int, C]:
    """KLT tracks within each burst and camera, joined across cameras by `links` (module doc,
    steps 1-2). View `c · K + k` is camera c at moment k."""

    cameras, count = len(frames), len(frames[0])
    first = np.cumsum([0, *bursts[:-1]]).tolist()
    track, view, pixel, next_id = [], [], [], 0
    for c, video in enumerate(frames):
        for start, length in zip(first, bursts, strict=True):
            found = track_video(video[start : start + length], config.klt)
            track.append(found.track + next_id)
            view.append(c * count + start + found.frame)
            pixel.append(found.pixel)
            next_id += int(found.track.max()) + 1 if len(found.track) else 0
    dtype = intrinsics.dtype
    ones = torch.ones(sum(len(t) for t in track), dtype=dtype)
    tracks = Tracks(
        view_camera=torch.arange(cameras).repeat_interleave(count),
        view_time=torch.arange(count).repeat(cameras),
        track=torch.cat(track),
        view=torch.cat(view),
        pixel=torch.cat(pixel).to(dtype),
        weight=ones,
        moving=ones > 0,
        intrinsics=intrinsics,
        lens=lens,
    )
    if not len(links.view_a):
        return tracks
    return merge_tracks(tracks, links.view_a, links.pixel_a, links.view_b, links.pixel_b)


def informative[C: IntVar](
    tracks: Tracks[int, int, C], static: Tensor[[C]], alone: bool, config: LightConfig
) -> Tracks[int, int, C]:
    """The tracks that can inform the free unknowns: seen by several cameras, or by one static
    camera and moving in it by more than `config.still` pixels (on the robot); also those of
    a single moving camera when `alone` (a free moving mount, pinned by its own video). At most
    `config.tracks` of them, evenly subsampled. A static camera's still tracks (all rays from
    one centre) and a held moving camera's own scene tracks constrain nothing here."""

    _, track = torch.unique(tracks.track, return_inverse=True)
    count = int(track.max()) + 1 if len(track) else 0
    camera = tracks.view_camera[tracks.view]
    seen = torch.zeros(count, len(static), dtype=torch.bool)
    seen[track, camera] = True
    low = torch.full((count, 2), math.inf, dtype=tracks.pixel.dtype)
    high = torch.full((count, 2), -math.inf, dtype=tracks.pixel.dtype)
    low = low.scatter_reduce(0, track[:, None].expand(-1, 2), tracks.pixel, "amin")
    high = high.scatter_reduce(0, track[:, None].expand(-1, 2), tracks.pixel, "amax")
    cameras = seen.sum(1)
    in_static = (seen & static[None]).any(1)
    moves = (high - low).norm(dim=1) > config.still
    keep = (cameras > 1) | (in_static & moves) | (alone & ~in_static)
    chosen = keep.nonzero()[:, 0]
    if len(chosen) > config.tracks:
        chosen = chosen[torch.linspace(0, len(chosen) - 1, config.tracks).long()]
    wanted = torch.zeros(count, dtype=torch.bool)
    wanted[chosen] = True
    mask = wanted[track]
    return replace(
        tracks,
        track=track[mask],
        view=tracks.view[mask],
        pixel=tracks.pixel[mask],
        weight=tracks.weight[mask],
        moving=tracks.moving[mask],
    )


def check(
    links: Links[int],
    world_from_view: Tensor[[int, *Mat4x4]],
    view_camera: Tensor[[int]],
    intrinsics: Tensor[[int, *Mat3x3]],
    lens: Lens[int] | None,
    static: Tensor[[int]],
) -> FitCheck:
    """The fit check (module doc): the epipolar distance of each verified match under the held
    poses `world_from_view` (per view), in the image where it is smaller,
    `d = min(|x_bᵀ E x_a| / ‖(E x_a)₁,₂‖ · f_b, |x_aᵀ Eᵀ x_b| / ‖(Eᵀ x_b)₁,₂‖ · f_a)`, `E = [t]ₓ R`
    of the relative pose `b_from_a`, `x` normalised coordinates (a near camera magnifies the
    far one's pixel noise: the smaller side measures the geometry); its median over the
    matches between two `static` cameras when there are some (verified over all moments at
    once), else over all."""

    camera_a, camera_b = view_camera[links.view_a], view_camera[links.view_b]
    both = static[camera_a] & static[camera_b]
    if bool(both.any()):
        links = Links(
            links.view_a[both], links.pixel_a[both], links.view_b[both], links.pixel_b[both]
        )
        camera_a, camera_b = camera_a[both], camera_b[both]
    if not len(links.view_a):
        return FitCheck(math.nan, 0)
    xa = _normalised(intrinsics, lens, camera_a, links.pixel_a)
    xb = _normalised(intrinsics, lens, camera_b, links.pixel_b)
    relative = se3.inverse(world_from_view[links.view_b]) @ world_from_view[links.view_a]
    essential = se3.skew(relative[:, :3, 3]) @ relative[:, :3, :3]
    line_b = (essential @ xa[..., None])[..., 0]
    line_a = (essential.transpose(1, 2) @ xb[..., None])[..., 0]
    residual = (xb * line_b).sum(-1).abs()
    pixels = torch.minimum(
        residual / line_b[:, :2].norm(dim=-1).clamp_min(1e-12) * intrinsics[camera_b, 0, 0],
        residual / line_a[:, :2].norm(dim=-1).clamp_min(1e-12) * intrinsics[camera_a, 0, 0],
    )
    pixels = pixels[torch.isfinite(pixels)]
    if not len(pixels):
        return FitCheck(math.nan, 0)
    return FitCheck(float(pixels.median()), len(pixels))


def _normalised[K: IntVar](
    intrinsics: Tensor[[int, *Mat3x3]],
    lens: Lens[int] | None,
    camera_index: Tensor[[K]],
    pixel: Tensor[[K, *UV]],
) -> Tensor[[K, 3]]:
    """Homogeneous normalised coordinates `(x/z, y/z, 1)` of pixels (NaN behind the camera)."""

    ray = camera.rays(
        intrinsics[camera_index],
        pixel[:, None].to(intrinsics),
        None if lens is None else lens.distortion[camera_index],
        None if lens is None else lens.fisheye[camera_index],
    )[:, 0]
    z = ray[:, 2:]
    return torch.where(z > 0.05, ray / z.clamp_min(0.05), math.nan)


def _features(
    sift: cv2.SIFT, image: Tensor[[*RGB, int, int]], side: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """SIFT keypoints `[n, 2]` (pixel centres at integer coordinates, in `image`) and RootSIFT
    descriptors (`√(d / ‖d‖₁)`), on the image upscaled by `u = max(1, side / W)`
    (`x = (x_u + ½) / u − ½`)."""

    gray = cv2.cvtColor(rearrange(image, "c h w -> h w c").numpy(), cv2.COLOR_RGB2GRAY)
    up = max(1.0, side / gray.shape[1])
    if up > 1.0:
        gray = cv2.resize(gray, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    points, descriptors = sift.detectAndCompute(gray, None)
    xy = (np.array([p.pt for p in points], dtype=np.float64).reshape(-1, 2) + 0.5) / up - 0.5
    if descriptors is not None:
        descriptors = descriptors / (descriptors.sum(1, keepdims=True) + 1e-7)
        descriptors = np.sqrt(descriptors).astype(np.float32)
    return xy, descriptors


def _undistorted(
    intrinsics: Tensor[[int, *Mat3x3]], lens: Lens[int] | None, c: int, xy: np.ndarray
) -> np.ndarray:
    """Each keypoint's normalised pinhole coordinates `(x/z, y/z)` (NaN behind the camera)."""

    ray = camera.rays(
        intrinsics[c],
        torch.from_numpy(xy).to(intrinsics),
        None if lens is None else lens.distortion[c],
        None if lens is None else lens.fisheye[c],
    )
    z = ray[:, 2:]
    return torch.where(z > 0.05, ray[:, :2] / z.clamp_min(0.05), math.nan).numpy()


def _candidates(
    matcher: cv2.BFMatcher,
    a: tuple[np.ndarray, np.ndarray | None],
    b: tuple[np.ndarray, np.ndarray | None],
    config: LightConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Indices of `a`'s and `b`'s keypoints that match each other (ratio test, mutual)."""

    if a[1] is None or b[1] is None or len(a[0]) < 8 or len(b[0]) < 8:
        return None
    forward = matcher.knnMatch(a[1], b[1], k=2)
    backward = {m.queryIdx: m.trainIdx for m in matcher.match(b[1], a[1])}
    pairs = [
        (m.queryIdx, m.trainIdx)
        for m, *rest in forward
        if rest
        and m.distance < config.ratio * rest[0].distance
        and backward.get(m.trainIdx) == m.queryIdx
    ]
    if not pairs:
        return None
    first, second = (np.array(i) for i in zip(*pairs, strict=True))
    return first, second


def _consistent(
    xa: np.ndarray, xb: np.ndarray, focal: float, config: LightConfig
) -> np.ndarray | None:
    """Which matches (normalised coordinates) one fundamental matrix explains (RANSAC,
    threshold `ransac / focal`); None with fewer than `min_matches`."""

    if len(xa) < config.min_matches:
        return None
    _, inliers = cv2.findFundamentalMat(xa, xb, cv2.FM_RANSAC, config.ransac / focal, 0.999)
    if inliers is None or int(inliers.sum()) < config.min_matches:
        return None
    return inliers[:, 0].astype(bool)


def held_robot[L: IntVar, J: IntVar, C: IntVar](
    robot: Robot[L, J, C], free: Collection[str]
) -> Robot[L, J, C]:
    """`robot` with only the cameras and bodies in `free` free; a frame-scope body is held at its
    recorded motion (zero std: it follows `bodies_at`), others at their mounts and offsets."""

    bodies = {n: f for n, f in robot.free_bodies.items() if n in free}
    bodies |= {
        n: Freedom(0.0, 0.0, "frame") for n, f in robot.free_bodies.items() if f.scope == "frame"
    }
    cameras = {n: f for n, f in robot.free_cameras.items() if n in free}
    return replace(robot, free_cameras=cameras, free_bodies=bodies)


def motion_at[T: IntVar](
    robot: Robot[int, int, int], base: str | None, motion: Tensor[[T, *Mat4x4]] | None
) -> Tensor[[int, T, *Mat4x4]] | None:
    """Starting corrections of `robot`'s free bodies (body order) at each timestep: the recorded
    `motion` for the moving `base`, identity for the others; None without a record."""

    if base is None or motion is None:
        return None
    free = sorted(robot.body(n) for n in robot.free_bodies)
    out = torch.eye(4, dtype=motion.dtype).repeat(len(free), len(motion), 1, 1)
    out[free.index(robot.body(base))] = motion
    return out


def light_solve[C: IntVar, T: IntVar](
    robot: Robot[int, int, int],
    qpos: Tensor[[T, int]],
    tracks: Tracks[int, int, C],
    names: tuple[str, ...],
    start_at: Tensor[[C, T, *Mat4x4]],
    bodies_at: Tensor[[int, T, *Mat4x4]] | None,
    config: LightConfig,
    focal_std: Tensor[[C]] | None = None,
) -> tuple[Rig[T, int, C], FitCheck]:
    """Levenberg–Marquardt on `robot`'s free cameras and bodies (`held_robot`) over the burst
    tracks, and on the focals with a nonzero `focal_std` (log-normal prior; default: all
    held); returns the solved rig (its rows carry the solution, `extended`) and its error."""

    rig = _rig(robot, qpos, tracks, names, start_at, bodies_at, focal_std)
    obs = _observations(tracks, qpos)
    assign = _assigner(rig, tracks, obs, config)
    state = {"body": assign()[obs.track]}

    def residual_fn() -> tuple[Tensor[[int, *UV]], Tensor[[int]]]:
        difference, valid, _ = residuals(rig, tracks, obs, state["body"].clamp_min(WORLD))
        return difference, valid

    def weight_fn(valid: Tensor[[int]]) -> Tensor[[int]]:
        return torch.where(state["body"] >= 0, obs.weight, 0.0) * valid

    def reassign() -> None:
        state["body"] = assign()[obs.track]

    time = tracks.view_time[obs.view]
    steps = len(tracks.view_time)
    cauchy = config.bundle.cauchy
    lm = config.bundle.lm
    levenberg_marquardt(rig, residual_fn, weight_fn, time, obs.track, steps, cauchy, lm, reassign)
    return rig, FitCheck(*_error(rig, tracks, obs, config))


def extended[C: IntVar, T: IntVar](
    solved: Rig[int, int, C],
    robot: Robot[int, int, int],
    qpos: Tensor[[T, int]],
    start_at: Tensor[[C, T, *Mat4x4]],
    bodies_at: Tensor[[int, T, *Mat4x4]] | None,
) -> tuple[Tensor[[C, T, *Mat4x4]], Tensor[[T, int, *Mat4x4]]]:
    """Every camera's `world_from_camera` and every body's `world_from_body` at every timestep
    of the episode (`qpos`), with the burst solution's dataset- and episode-scope rows
    (frame-scope bodies follow `bodies_at`)."""

    names = solved.names
    rig = Rig(
        robot,
        qpos,
        names,
        start_at[:, 0],
        solved.initial_intrinsics,
        0.0,
        start_at=start_at,
        bodies_at=bodies_at,
    )
    with torch.no_grad():
        for mine, theirs in ((rig.cameras, solved.cameras), (rig.bodies, solved.bodies)):
            for element in range(mine.row.shape[0]):
                if bool((mine.row[element] != mine.row[element, 0]).any()):
                    continue  # frame scope: held at its start
                mine.delta[mine.row[element, 0]] = theirs.delta[theirs.row[element, 0]]
        steps = len(qpos)
        camera = torch.arange(len(names)).repeat_interleave(steps)
        time = torch.arange(steps).repeat(len(names))
        world = rig.world_from_body()
        poses = rig.camera_poses(camera, time, world)
        # A re-placed robot root (a free body on the world): poses in the robot's frame, i.e.
        # the world moved back by the root's correction `C = W'_r W_r⁻¹` (`C⁻¹ · pose`)
        for name, freedom in robot.free_bodies.items():
            root = robot.body(name)
            if robot.body_parent[root] == 0 and freedom.scope != "frame":
                change = world[0, root] @ se3.inverse(rig.kinematics[0, root])
                poses = se3.inverse(change) @ poses
                world = se3.inverse(change) @ world
    return rearrange(poses, "(c t) i j -> c t i j", c=len(names)), world


def _rig[C: IntVar, T: IntVar](
    robot: Robot[int, int, int],
    qpos: Tensor[[T, int]],
    tracks: Tracks[int, int, C],
    names: tuple[str, ...],
    start_at: Tensor[[C, T, *Mat4x4]],
    bodies_at: Tensor[[int, T, *Mat4x4]] | None,
    focal_std: Tensor[[C]] | None = None,
) -> Rig[T, int, C]:
    """A rig over the burst timesteps: lenses held, focals held unless `focal_std`, cameras at
    `start_at`."""

    lens = tracks.lens
    if lens is not None:
        lens = replace(lens, std=torch.zeros_like(lens.std))
    return Rig(
        robot,
        qpos,
        names,
        start_at[:, 0],
        tracks.intrinsics.to(qpos),
        0.0 if focal_std is None else focal_std.to(qpos),
        start_at=start_at,
        bodies_at=bodies_at,
        lens=lens,
    )


def _observations(tracks: Tracks[int, int, int], like: Tensor) -> Observations[int]:
    """The tracks' observations with dense track ids (as `bundle_adjust` builds them)."""

    _, track = torch.unique(tracks.track, return_inverse=True)
    count = int(track.max()) + 1 if len(track) else 0
    return Observations(tracks.view, tracks.pixel.to(like), tracks.weight.to(like), track, count)


def _assigner(
    rig: Rig[int, int, int],
    tracks: Tracks[int, int, int],
    obs: Observations[int],
    config: LightConfig,
) -> Callable[[], Tensor[[int]]]:
    """Each track's body under the rig's current values (`bundle.attach`)."""

    moving = torch.ones(obs.count, dtype=torch.bool)
    candidates = moving_bodies(rig)
    bundle = config.bundle

    def assign() -> Tensor[[int]]:
        with torch.no_grad():
            return attach(rig, tracks, obs, moving, candidates, bundle.cauchy, bundle.outlier_cost)

    return assign


def _error(
    rig: Rig[int, int, int],
    tracks: Tracks[int, int, int],
    obs: Observations[int],
    config: LightConfig,
) -> tuple[float, int]:
    """Median reprojection error (pixels) of the tracks a body explains, and their count."""

    body = _assigner(rig, tracks, obs, config)()[obs.track]
    with torch.no_grad():
        difference, valid, _ = residuals(rig, tracks, obs, body.clamp_min(WORLD))
    unit = rig.pixel_unit[tracks.view_camera[obs.view]]
    pixels = (difference.norm(dim=-1) * unit)[valid & (body >= 0)]
    if not len(pixels):
        return math.nan, 0
    return float(pixels.median()), len(pixels)
