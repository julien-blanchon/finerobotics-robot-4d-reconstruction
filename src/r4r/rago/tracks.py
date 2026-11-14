"""Static point tracks from ARROW, encoded on many small subsets of the episode's images.

ARROW is trained on sets of 2–24 images, so instead of one huge encoding we encode subsets of
`subset_size` images: all cameras at a few timesteps, half of the subsets in a local time
window (overlap for moving cameras), half spread over the episode (diverse robot motion).
Inside a subset, every grid pixel of every view is followed into the other views. A track is
static if ARROW sees its point static towards every target, moving if it moves towards any
(a point on the robot, a grasped object, a person); an observation is kept where the point is
visible and inside the image (and, for a static track, displaced by less than `static_rho`).
Moving tracks are calibration targets once attached to the link they move with (bundle).

Only ARROW's 2D answers (where the point is) enter calibration: its depth and field of view
are unreliable on robot footage (research log). Each subset also yields ARROW's relative poses
and per-view intrinsics fits, used only to initialise.

Ref: ARROW arXiv:2610.01314 (queries; displacement encoding App. B.1, Eq. 9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from einops import rearrange, repeat

from r4r.arrow import queries
from r4r.arrow.images import model_images
from r4r.geometry import alignment, camera

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.arrow.images import Resize
    from r4r.arrow.models.arrow import ARROW

    from .lens import Lens


@dataclass(frozen=True, slots=True, kw_only=True)
class TrackConfig:
    subsets: int = 48
    subset_size: int = 16
    # Local subsets span at most this many timesteps; the rest are spread over the episode
    window: int = 24
    # Source pixels per view, spread over the native image: (columns, rows)
    grid: tuple[int, int] = (32, 18)
    # ARROW's encoded displacement ‖ρ‖ = log(1 + ‖d‖/ε): static below, moving above (on DROID
    # the static background is exactly 0 and a moving arm sits at 3–4)
    static_rho: float = 0.5
    moving_rho: float = 1.0
    min_visibility: float = 0.5
    keep_moving: bool = True
    seed: int = 0


@dataclass(frozen=True, slots=True)
class Tracks[M: IntVar, V: IntVar, C: IntVar]:
    """`M` observations of static tracks over `V` views of `C` cameras.

    Observation `m` sees track `track[m]` in view `view[m]` at native `pixel[m]`; `moving[m]`
    when its track's point moves (it is then rigid with some link, or an outlier).

    `view_scaling [V]`, when a camera's recorded intrinsics change between episodes (randomising
    simulators): view v sees through `view_scaling[v] · intrinsics[view_camera[v]]`, i.e. its
    episode's recorded K times the inverse of the reference episode's (identity where they are
    equal). A focal correction of the camera then scales every episode's focal alike.
    """

    view_camera: Tensor[[V]]
    view_time: Tensor[[V]]
    track: Tensor[[M]]
    view: Tensor[[M]]
    pixel: Tensor[[M, 2]]
    weight: Tensor[[M]]
    moving: Tensor[[M]]
    intrinsics: Tensor[[C, 3, 3]]  # median of ARROW's per-view fits (initialisation only)
    lens: Lens | None = None  # distortion per camera; None: ideal pinholes
    view_scaling: Tensor[[V, 3, 3]] | None = None  # None: every view sees its camera's K

    def view_intrinsics(self, intrinsics: Tensor[[C, 3, 3]]) -> Tensor[[V, 3, 3]]:
        """Each view's intrinsics from its camera's `intrinsics` (with `view_scaling`)."""

        k = intrinsics[self.view_camera]
        return k if self.view_scaling is None else self.view_scaling.to(k) @ k


@dataclass(frozen=True, slots=True)
class SubsetPoses:
    """ARROW's poses of a subset's views relative to its first view (ARROW units)."""

    views: Tensor[[int]]
    reference_from_view: Tensor[[int, 4, 4]]


def sample_subsets(
    view_camera: Tensor[[int]], view_time: Tensor[[int]], config: TrackConfig
) -> list[Tensor[[int]]]:
    """View indices of each subset: every camera at `subset_size // cameras` timesteps."""

    cameras = int(view_camera.max()) + 1
    times = int(view_time.max()) + 1
    per_subset = max(2, config.subset_size // cameras)
    generator = torch.Generator().manual_seed(config.seed)
    lookup = torch.full((cameras, times), -1, dtype=torch.long)
    lookup[view_camera.cpu(), view_time.cpu()] = torch.arange(len(view_time))

    subsets = []
    for i in range(config.subsets):
        local = i % 2 == 0
        span = min(config.window, times) if local else times
        start = int(torch.randint(0, times - span + 1, (1,), generator=generator))
        chosen = start + torch.randperm(span, generator=generator)[:per_subset].sort().values
        views = lookup[:, chosen].flatten()
        subsets.append(views[views >= 0].to(view_camera.device))
    return subsets


def tracks_from_arrow(
    model: ARROW,
    frames: Tensor[[int, 3, int, int]],
    view_camera: Tensor[[int]],
    view_time: Tensor[[int]],
    resize: Resize,
    config: TrackConfig,
) -> tuple[Tracks, list[SubsetPoses]]:
    """Encode every subset, collect static tracks and ARROW's subset poses."""

    device = view_camera.device
    width, height = resize.native
    columns, rows = config.grid
    xs = torch.linspace(0.04, 0.96, columns, device=device) * (width - 1)
    ys = torch.linspace(0.04, 0.96, rows, device=device) * (height - 1)
    native = rearrange(
        torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1), "r c two -> (r c) two"
    )
    model_xy = camera.resize_pixels(native, *resize.scale)
    n = len(native)

    observations: list[tuple[Tensor, ...]] = []
    poses, focal_fits = [], {c: [] for c in range(int(view_camera.max()) + 1)}
    next_track = 0
    for views in sample_subsets(view_camera, view_time, config):
        s = len(views)
        encoding = model.encode(model_images(frames[views.cpu()].to(device), resize)[None])

        # Own-camera pointmaps: ARROW's relative poses and intrinsics (initialisation only)
        own = model.decode(encoding, queries.per_frame(model_xy, s, None))
        shared = model.decode(encoding, queries.per_frame(model_xy, s, 0))
        points = rearrange(own.points[0], "(s n) c -> s n c", s=s)
        weight = rearrange((own.point_confidence * shared.point_confidence)[0], "(s n) -> s n", s=s)
        target = rearrange(shared.points[0], "(s n) c -> s n c", s=s)
        poses.append(SubsetPoses(views, alignment.procrustes(points, target, weight)))
        for i, v in enumerate(views.tolist()):
            fit = camera.fit_intrinsics(
                points[i], native, own.point_confidence[0, i * n : (i + 1) * n]
            )
            focal_fits[int(view_camera[v])].append(fit)

        # Every grid pixel of every view, followed into every other view of the subset
        source, target_view = (
            repeat(torch.arange(s, device=device), "a -> (a b)", b=s),
            repeat(torch.arange(s, device=device), "b -> (a b)", a=s),
        )
        other = source != target_view
        source, target_view = source[other], target_view[other]
        found = model.decode(encoding, queries.pairs(model_xy, source, target_view))
        uv = rearrange(found.uv[0], "(p n) c -> p n c", n=n)
        pixel = camera.resize_pixels(
            uv * uv.new_tensor(resize.model) - 0.5, 1 / resize.scale[0], 1 / resize.scale[1]
        )
        rho = rearrange(found.displacement[0].norm(dim=-1), "(p n) -> p n", n=n)
        visible = rearrange(found.visibility_logits[0], "(p n) -> p n", n=n).sigmoid()
        confidence = rearrange(found.point_confidence[0], "(p n) -> p n", n=n)

        # A track = (source view, grid pixel); dropped if it moves towards any target
        moving = (
            torch.zeros(s, n, device=device).index_reduce_(
                0, source, (rho > config.moving_rho).float(), "amax"
            )
            > 0
        )
        inside = (pixel >= 0).all(-1) & (pixel <= pixel.new_tensor([width - 1, height - 1])).all(-1)
        seen_well = (visible > config.min_visibility) & inside
        keep = seen_well & (rho < config.static_rho) & ~moving[source]
        if config.keep_moving:
            keep |= seen_well & moving[source]
        pair, pixel_index = keep.nonzero(as_tuple=True)
        track_id = next_track + source[pair] * n + pixel_index

        # Source observations of tracks seen at least once elsewhere
        seen = torch.unique(track_id)
        source_view, source_pixel = (seen - next_track) // n, (seen - next_track) % n
        observations.append(
            (
                seen,
                views[source_view],
                native[source_pixel],
                torch.ones_like(seen, dtype=native.dtype),
                moving.flatten()[seen - next_track],
            )
        )
        observations.append(
            (
                track_id,
                views[target_view[pair]],
                pixel[pair, pixel_index],
                confidence[pair, pixel_index].log(),
                moving[source[pair], pixel_index],
            )
        )
        next_track += s * n
        del encoding

    track, view, pixels, weights, is_moving = (
        torch.cat(parts) for parts in zip(*observations, strict=True)
    )
    intrinsics = torch.stack(
        [torch.stack(fits).median(0).values for _, fits in sorted(focal_fits.items())]
    )
    tracks = Tracks(
        view_camera, view_time, track, view, pixels, weights.clamp_min(0.1), is_moving, intrinsics
    )
    return tracks, poses
