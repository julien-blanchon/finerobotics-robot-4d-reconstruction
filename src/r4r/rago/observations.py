"""What calibration consumes: views, static correspondences, intrinsics, initial relative poses.

Everything visual is in ARROW's units (an unknown global scale) and camera frames; pixels are
native image pixels. Built from ARROW predictions in `from_arrow`, or synthesised in tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch

from r4r.arrow import queries
from r4r.geometry import alignment, camera

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.arrow.images import Resize
    from r4r.arrow.models.arrow import ARROW, Encoding


@dataclass(frozen=True, slots=True)
class Observations[V: IntVar, K: IntVar, C: IntVar]:
    """`V` views (one image each), `K` correspondences between views, `C` cameras.

    Correspondence `k` is one static physical point seen in views `view_a[k]` and `view_b[k]`:
    at `pixel_a`/`pixel_b`, with camera-frame coordinates `point_a`/`point_b` (ARROW units).
    `reference_from_view` are ARROW's relative camera poses (ARROW units), used to initialise.
    """

    view_camera: Tensor[[V]]
    view_time: Tensor[[V]]
    view_a: Tensor[[K]]
    view_b: Tensor[[K]]
    pixel_a: Tensor[[K, 2]]
    pixel_b: Tensor[[K, 2]]
    point_a: Tensor[[K, 3]]
    point_b: Tensor[[K, 3]]
    weight: Tensor[[K]]
    intrinsics: Tensor[[C, 3, 3]]
    reference_from_view: Tensor[[V, 4, 4]]

    def to(self, device: torch.device | str, dtype: torch.dtype) -> Observations[V, K, C]:
        """Move to `device`; floating tensors to `dtype`."""

        def move(x: Tensor) -> Tensor:
            return x.to(device, dtype) if x.is_floating_point() else x.to(device)

        return Observations(*(move(getattr(self, f.name)) for f in fields(self)))


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationConfig:
    # Source pixels per view, spread over the native image: (columns, rows)
    grid: tuple[int, int] = (24, 14)
    # Pair each view with views this many timesteps away (both signs; 0 = other cameras)
    offsets: tuple[int, ...] = (0, 1, 2, 4, 8)
    # Static if ARROW's source→target displacement is below this (fraction of scene size)
    static_displacement: float = 0.01
    min_visibility: float = 0.5
    chunk: int = 100_000


# ARROW encodes displacement d as ρ(d) = d/‖d‖ · log(1 + ‖d‖/ε)  (ARROW App. B.1, Eq. 9)
DISPLACEMENT_EPS = 1e-4


def observations_from_arrow(
    model: ARROW,
    encoding: Encoding,
    view_camera: Tensor[[int]],
    view_time: Tensor[[int]],
    resize: Resize,
    config: ObservationConfig,
) -> Observations:
    """Query ARROW for everything calibration needs. Views are the encoding's frames, in order."""

    device = encoding.images.device
    views = len(view_time)
    width, height = resize.native
    scale_x, scale_y = resize.scale

    # Native source pixels, and where they fall in the model image
    columns, rows = config.grid
    xs = torch.linspace(0.05, 0.95, columns, device=device) * (width - 1)
    ys = torch.linspace(0.05, 0.95, rows, device=device) * (height - 1)
    native = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2)
    model_xy = camera.resize_pixels(native, scale_x, scale_y)
    n = len(native)

    # Own-camera pointmaps and the same points in the middle view's camera
    reference = views // 2
    own = model.decode(encoding, queries.per_frame(model_xy, views, None), config.chunk)
    shared = model.decode(encoding, queries.per_frame(model_xy, views, reference), config.chunk)
    points = own.points[0].reshape(views, n, 3)
    weight = (own.point_confidence * shared.point_confidence)[0].reshape(views, n)
    reference_from_view = alignment.procrustes(
        points, shared.points[0].reshape(views, n, 3), weight
    )

    intrinsics = torch.stack(
        [
            camera.fit_intrinsics(
                points[view_camera == c].reshape(-1, 3),
                native.repeat(int((view_camera == c).sum()), 1),
                own.point_confidence[0].reshape(views, n)[view_camera == c].reshape(-1),
            )
            for c in range(int(view_camera.max()) + 1)
        ]
    )

    # View pairs at the configured time offsets (same camera: nonzero offsets only)
    a, b = torch.meshgrid(
        torch.arange(views, device=device), torch.arange(views, device=device), indexing="ij"
    )
    offset = (view_time[b] - view_time[a]).abs()
    wanted = torch.isin(offset, torch.tensor(config.offsets, device=device))
    wanted &= (view_camera[a] != view_camera[b]) | (offset > 0)
    source, target = a[wanted], b[wanted]

    found = model.decode(encoding, queries.pairs(model_xy, source, target), config.chunk)
    uv = found.uv[0].reshape(-1, n, 2)
    pixel_b = camera.resize_pixels(uv * uv.new_tensor(resize.model) - 0.5, 1 / scale_x, 1 / scale_y)
    displacement = found.displacement[0].reshape(-1, n, 3).norm(dim=-1)
    static = displacement < math.log1p(config.static_displacement / DISPLACEMENT_EPS)
    visible = found.visibility_logits[0].reshape(-1, n).sigmoid() > config.min_visibility
    inside = (pixel_b >= 0).all(-1) & (pixel_b <= pixel_b.new_tensor([width - 1, height - 1])).all(
        -1
    )
    keep = static & visible & inside
    pair, pixel = keep.nonzero(as_tuple=True)

    return Observations(
        view_camera=view_camera,
        view_time=view_time,
        view_a=source[pair],
        view_b=target[pair],
        pixel_a=native[pixel],
        pixel_b=pixel_b[pair, pixel],
        point_a=points[source[pair], pixel],
        point_b=found.points[0].reshape(-1, n, 3)[pair, pixel],
        weight=weight[source[pair], pixel] * found.point_confidence[0].reshape(-1, n)[pair, pixel],
        intrinsics=intrinsics,
        reference_from_view=reference_from_view,
    )
