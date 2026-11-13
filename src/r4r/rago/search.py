"""Global search for a moving camera's mount and focal length, without any initial guess.

The kinematic bundle converges from ~25 cm / 10° away, not from anywhere. Instead of trusting
one starting point, short bundles start from the 24 axis-aligned orientations of the camera on
its link (at the description's mount position) × a few focal-length hypotheses, on a subsample
of the tracks; the start with the best score is kept. Scores count dropped observations as
outliers, so they compare across starts. Only the static world is used (no link attachment):
on the camera's own videos, links invite spurious fits.

Ref: the 24 rotations of the cube group (octahedral group O).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch

from .bundle import BundleConfig, bundle_adjust

if TYPE_CHECKING:
    from torch import Tensor

    from .robot import Robot
    from .tracks import Tracks


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchConfig:
    focal_factors: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5)
    # Subsample: this share of the tracks, observations every `every`-th timestep
    track_share: float = 1 / 3
    every: int = 3
    bundle: BundleConfig = field(
        default_factory=lambda: BundleConfig(
            iterations=150, learning_rate=1e-2, focal_std=0.5, reassign_every=50, attach_links=False
        )
    )
    seed: int = 0


@dataclass(frozen=True, slots=True)
class SearchResult:
    body_from_camera: Tensor[[int, 4, 4]]
    intrinsics: Tensor[[int, 3, 3]]
    # Scores of every start (orientation-major, then focal factor): lower is better
    scores: Tensor[[int]]


def cube_rotations() -> Tensor[[24, 3, 3]]:
    """The 24 rotations mapping the axes onto signed axes."""

    rotations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            matrix = torch.zeros(3, 3, dtype=torch.float64)
            matrix[range(3), permutation] = torch.tensor(signs, dtype=torch.float64)
            if torch.linalg.det(matrix) > 0:
                rotations.append(matrix)
    return torch.stack(rotations)


def search_mount(
    robot: Robot,
    qpos: Tensor[[int, int]],
    tracks: Tracks,
    names: tuple[str, ...],
    camera: str,
    config: SearchConfig,
) -> SearchResult:
    """Best start for `camera`'s mount (others at their description) and its focal length."""

    c = names.index(camera)
    generator = torch.Generator().manual_seed(config.seed)
    ids = torch.unique(tracks.track)
    share = max(1, int(len(ids) * config.track_share))
    chosen = ids[torch.randperm(len(ids), generator=generator)[:share].to(ids.device)]
    keep = torch.isin(tracks.track, chosen)
    keep &= tracks.view_time[tracks.view] % config.every == 0
    subsample = replace(
        tracks,
        track=tracks.track[keep],
        view=tracks.view[keep],
        pixel=tracks.pixel[keep],
        weight=tracks.weight[keep],
        moving=tracks.moving[keep],
    )

    description = robot.body_from_camera[[robot.camera(n) for n in names]].to(qpos)
    best, scores = None, []
    for rotation in cube_rotations().to(qpos):
        initial = description.clone()
        initial[c, :3, :3] = rotation
        for factor in config.focal_factors:
            intrinsics = tracks.intrinsics.to(qpos).clone()
            intrinsics[c, :2, :2] *= factor
            run = replace(subsample, intrinsics=intrinsics)
            result = bundle_adjust(robot, qpos, run, names, initial, config.bundle)
            scores.append(result.score)
            if best is None or result.score < best.score:
                best = result
    assert best is not None
    return SearchResult(best.body_from_camera, best.intrinsics, torch.tensor(scores))
