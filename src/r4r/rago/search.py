"""Global search for a moving camera's mount and focal length, without any initial guess.

The kinematic bundle converges from ~25 cm / 10° away, not from anywhere. Instead of trusting
one starting point, short bundles start from the 24 axis-aligned orientations of the camera on
its link (at the description's mount position) × a few focal-length hypotheses `f = α · f₀`
(α ∈ `focal_factors`, f₀ the prior or guess, pixels), on a subsample
of the camera's own tracks, all in one batched bundle; the start with the best score is kept.
Scores count dropped observations as outliers, so they compare across starts. Only the static
world is used (no link attachment): on the camera's own videos, links invite spurious fits.

Ref: the 24 rotations of the cube group (octahedral group O).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch

from .bundle import BundleConfig, bundle_adjust

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.types import Mat3x3, Mat4x4

    from .robot import Robot
    from .tracks import Tracks


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchConfig:
    focal_factors: tuple[float, ...] = (0.2, 0.35, 0.5, 0.75, 1.0, 1.5)
    # Subsample: this share of the tracks, observations every `every`-th timestep, and at
    # most `max_views` timesteps (long episodes: memory grows with views × starts)
    track_share: float = 1 / 3
    every: int = 3
    max_views: int = 400
    bundle: BundleConfig = field(
        default_factory=lambda: BundleConfig(
            iterations=150,
            learning_rate=1e-2,
            focal_std=0.5,
            reassign_every=50,
            attach_links=False,
            # Many independent starts in one bundle: Adam steps each parameter on its own; LM
            # would couple their step acceptance and need a Jacobian product per start
            optimizer="adam",
        )
    )
    seed: int = 0


@dataclass(frozen=True, slots=True)
class SearchResult[C: IntVar]:
    # Body the best start rides on; mounts of all `C` cameras (the searched one on `body`)
    body: int
    body_from_camera: Tensor[[C, *Mat4x4]]
    intrinsics: Tensor[[C, *Mat3x3]]
    # Scores of every start (body-major, then orientation, then focal factor): lower is better,
    # inf for a start that diverged (non-finite parameters)
    scores: Tensor[[int]]


def cube_rotations() -> Tensor[[24, *Mat3x3]]:
    """The 24 rotations mapping the axes onto signed axes: the signed permutation matrices
    with `det R = +1`."""

    rotations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            matrix = torch.zeros(3, 3, dtype=torch.float64)
            matrix[range(3), permutation] = torch.tensor(signs, dtype=torch.float64)
            if torch.linalg.det(matrix) > 0:
                rotations.append(matrix)
    return torch.stack(rotations)


def search_mount[T: IntVar, NQ: IntVar, C: IntVar](
    robot: Robot[int, int, int],
    qpos: Tensor[[T, NQ]],
    tracks: Tracks[int, int, C],
    names: tuple[str, ...],
    camera: str,
    config: SearchConfig,
    bodies: tuple[int, ...] = (),
) -> SearchResult[C]:
    """Best start for `camera`'s mount and its focal length, from its own tracks only.

    Starts ride on each of `bodies` (default: the camera's own body), at the description's
    mount position on the camera's own body and at the origin of the others. All starts are
    solved at once: each is a copy of the camera (its own body, mount and focal length)
    seeing its own copy of the tracks, so one bundle optimises them independently (Adam's
    steps are per parameter) and scores them per copy.
    """

    c = names.index(camera)
    generator = torch.Generator().manual_seed(config.seed)
    mine = tracks.view_camera[tracks.view] == c
    ids = torch.unique(tracks.track[mine])
    share = max(1, int(len(ids) * config.track_share))
    chosen = ids[torch.randperm(len(ids), generator=generator)[:share].to(ids.device)]
    keep = mine & torch.isin(tracks.track, chosen)
    every = max(config.every, -(-len(torch.unique(tracks.view_time)) // config.max_views))
    keep &= tracks.view_time[tracks.view] % every == 0
    view, track = tracks.view[keep], tracks.track[keep]

    index = robot.camera(camera)
    bodies = bodies or (robot.camera_body[index],)
    rotations = cube_rotations().to(qpos)
    factors = qpos.new_tensor(config.focal_factors)
    per_body = len(rotations) * len(factors)
    copies = len(bodies) * per_body
    # Views of the camera, renumbered, then repeated once per copy
    views, view = torch.unique(view, return_inverse=True)
    _, track = torch.unique(track, return_inverse=True)
    n_views, n_tracks = len(views), int(track.max()) + 1
    offset = torch.arange(copies, device=view.device)[:, None]
    intrinsics = tracks.intrinsics[c].to(qpos).expand(copies, 3, 3).clone()
    intrinsics[:, :2, :2] *= factors.repeat(len(rotations) * len(bodies))[:, None, None]
    batch = replace(
        tracks,
        view_camera=torch.arange(copies, device=view.device).repeat_interleave(n_views),
        view_time=tracks.view_time[views].repeat(copies),
        track=(track[None] + offset * n_tracks).flatten(),
        view=(view[None] + offset * n_views).flatten(),
        pixel=tracks.pixel[keep].repeat(copies, 1),
        weight=tracks.weight[keep].repeat(copies),
        moving=tracks.moving[keep].repeat(copies),
        intrinsics=intrinsics,
        lens=None if tracks.lens is None else tracks.lens.select(view.new_full((copies,), c)),
        view_scaling=None
        if tracks.view_scaling is None
        else tracks.view_scaling[views].repeat(copies, 1, 1),
    )
    copy_names = tuple(f"{camera}#{i}" for i in range(copies))
    # Prior centres: the description's mount, moved to the origin of other bodies
    starts = robot.body_from_camera[index].to(qpos).repeat(len(bodies), 1, 1)
    for b, body in enumerate(bodies):
        if body != robot.camera_body[index]:
            starts[b, :3, 3] = 0.0
    virtual = replace(
        robot,
        camera_names=copy_names,
        camera_body=tuple(b for b in bodies for _ in range(per_body)),
        body_from_camera=starts.repeat_interleave(per_body, 0),
        free_cameras=dict.fromkeys(copy_names, robot.free_cameras[camera]),
    )
    initial = virtual.body_from_camera.to(qpos).clone()
    initial[:, :3, :3] = rotations.repeat_interleave(len(factors), 0).repeat(len(bodies), 1, 1)
    # One pixel unit for all starts: per-start units would favour long focal guesses
    unit = tracks.intrinsics[c, 0, 0].to(qpos).expand(copies)
    result = bundle_adjust(virtual, qpos, batch, copy_names, initial, config.bundle, unit)
    # A start whose parameters diverged scores as the outlier cap (no valid observation), like
    # a poor fit: never pick it
    finite = torch.isfinite(result.body_from_camera).all(-1).all(-1)
    finite &= torch.isfinite(result.intrinsics).all(-1).all(-1)
    if result.lens is not None:
        finite &= torch.isfinite(result.lens.distortion).all(-1)
    if not bool(finite.any()):
        raise ValueError(f"search_mount({camera!r}): every start diverged (non-finite parameters)")
    scores = torch.where(finite, result.camera_score, torch.inf)
    best = int(scores.argmin())
    body_from_camera = robot.body_from_camera[[robot.camera(n) for n in names]].to(qpos).clone()
    body_from_camera[c] = result.body_from_camera[best]
    best_intrinsics = tracks.intrinsics.to(qpos).clone()
    best_intrinsics[c] = result.intrinsics[best]
    return SearchResult(bodies[best // per_body], body_from_camera, best_intrinsics, scores)
