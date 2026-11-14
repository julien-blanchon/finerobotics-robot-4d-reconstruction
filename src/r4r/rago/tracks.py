"""Point tracks: observations of 3D points across the views of a rig's cameras.

A track is one point; an observation is where a view sees it. Tracks come from KLT along each
camera's video (`klt`), joined across cameras by RoMa links (`merge`). Static points are the
calibration target; moving ones (a point on the robot, a grasped object) are once the bundle
attaches them to the link they move with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.types import UV, Mat3x3

    from .lens import Lens


@dataclass(frozen=True, slots=True)
class Tracks[M: IntVar, V: IntVar, C: IntVar]:
    """`M` observations of tracks over `V` views of `C` cameras.

    Observation `m` sees track `track[m]` in view `view[m]` at native `pixel[m]`; `moving[m]`
    when its track's point moves (it is then rigid with some link, or an outlier).

    `view_scaling [V]`, when a camera's recorded intrinsics change between episodes (randomising
    simulators): view v sees through `K_v = S_v · K_c` with `S_v = K_e(v) · K_ref⁻¹`, its
    episode's recorded K over the reference episode's (identity where they are equal). A focal
    correction of the camera then scales every episode's focal alike.
    """

    view_camera: Tensor[[V]]
    view_time: Tensor[[V]]
    track: Tensor[[M]]
    view: Tensor[[M]]
    pixel: Tensor[[M, *UV]]
    weight: Tensor[[M]]
    moving: Tensor[[M]]
    intrinsics: Tensor[[C, *Mat3x3]]  # per camera: the prior or current estimate
    lens: Lens | None = None  # distortion per camera; None: ideal pinholes
    view_scaling: Tensor[[V, *Mat3x3]] | None = None  # None: every view sees its camera's K

    def view_intrinsics(self, intrinsics: Tensor[[C, *Mat3x3]]) -> Tensor[[V, *Mat3x3]]:
        """Each view's intrinsics `K_v = S_v · K_c(v)` from its camera's (with `view_scaling`)."""

        k = intrinsics[self.view_camera]
        return k if self.view_scaling is None else self.view_scaling.to(k) @ k
