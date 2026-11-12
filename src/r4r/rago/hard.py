"""Hard constraint: every view's pose is kinematics times its camera mount, exactly.

The only unknowns are the free mounts and the scale. Kinematic consistency holds by
construction; errors in FK or timestamps become bias in the mounts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from torch import Tensor

    from .observations import Observations
    from .rig import Rig


class Hard(nn.Module):
    def __init__(self, rig: Rig) -> None:
        super().__init__()
        self.rig = rig

    def forward(self, observations: Observations) -> tuple[Tensor[[int, 4, 4]], Tensor[[]]]:
        """(world_from_view, penalty)."""

        poses = self.rig.camera_poses(observations.view_camera, observations.view_time)
        return poses, self.rig.prior()

    def calibrated_mounts(self, observations: Observations) -> Tensor[[int, 4, 4]]:
        return self.rig.mounts()
