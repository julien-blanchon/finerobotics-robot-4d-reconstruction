"""Hybrid: kinematic poses plus a small per-view correction with a Gaussian prior.

`world_from_view = world_from_body(q_t) · mount · exp(δ_view)`, penalised by
`‖δ_t‖²/σ_t² + ‖δ_r‖²/σ_r²`: σ → 0 recovers the hard constraint, a finite σ absorbs small FK
or timestamp errors per view.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from r4r.geometry import se3

if TYPE_CHECKING:
    from torch import Tensor

    from .observations import Observations
    from .rig import Rig


class Hybrid(nn.Module):
    def __init__(self, rig: Rig, views: int, translation_std: float, rotation_std: float) -> None:
        super().__init__()
        self.rig = rig
        self.view_delta = nn.Parameter(rig.log_scale.new_zeros(views, 6))
        self.std = (translation_std, rotation_std)

    def forward(self, observations: Observations) -> tuple[Tensor[[int, 4, 4]], Tensor[[]]]:
        """(world_from_view, penalty)."""

        kinematic = self.rig.camera_poses(observations.view_camera, observations.view_time)
        poses = kinematic @ se3.se3_exp(self.view_delta)
        deviation = (self.view_delta[:, :3] ** 2).sum() / self.std[0] ** 2
        deviation = deviation + (self.view_delta[:, 3:] ** 2).sum() / self.std[1] ** 2
        return poses, self.rig.prior() + deviation

    def calibrated_mounts(self, observations: Observations) -> Tensor[[int, 4, 4]]:
        return self.rig.mounts()
