"""What calibration consumes: views, static correspondences, intrinsics, initial relative poses.

Everything visual is in ARROW's units (an unknown global scale) and camera frames; pixels are
native image pixels. Built from ARROW predictions in `from_arrow`, or synthesised in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor


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
