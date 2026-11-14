"""Lens distortion of a rig's cameras: model, coefficients and how far calibration may move them.

Each camera is a pinhole with radial distortion (OpenCV `k1, k2, k3`) or a Kannala–Brandt
fisheye (OpenCV fisheye `k1..k4`), see `r4r.geometry.camera`: both map a ray at `s` (`r/z`, or
the angle θ) to the normalised radius `ρ = s · (1 + k1 s² + k2 s⁴ + k3 s⁶ + k4 s⁸)`.

The coefficients are a dataset prior like the focal length: given (held, or refined within a
range), or unknown (refined from zero: an ideal pinhole, or the equidistant fisheye) with a
wide Gaussian prior, residual `(k − k_prior) / σ_k` per coefficient. A rig whose cameras are
all ideal pinholes carries no `Lens` at all (the plain pinhole path).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from r4r.geometry.camera import COEFFICIENTS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.data.priors import CameraPrior
    from r4r.types import K1K2K3K4

# Prior std of each coefficient of a fisheye whose distortion is unknown (real Kannala–Brandt
# coefficients are within ±0.1); its higher terms rarely matter, k1 most
UNKNOWN_FISHEYE_STD = 0.1
# Coefficients of a pinhole's radial model (OpenCV k1, k2, k3; the fourth stays 0)
PINHOLE_COEFFICIENTS = 3


@dataclass(frozen=True, slots=True)
class Lens[C: IntVar]:
    """Per camera: `fisheye [C]` (Kannala–Brandt, else radial pinhole), `distortion [C, 4]`
    and `std [C, 4]`, the prior std of each coefficient around its value (0: held)."""

    fisheye: Tensor[[C]]
    distortion: Tensor[[C, *K1K2K3K4]]
    std: Tensor[[C, *K1K2K3K4]]

    def to(self, like: Tensor) -> Lens[C]:  # `like`: any tensor, read for device and dtype
        """On `like`'s device, coefficients in its dtype."""

        return Lens(self.fisheye.to(like.device), self.distortion.to(like), self.std.to(like))

    def select[V: IntVar](self, camera: Tensor[[V]]) -> Lens[V]:
        """The lenses of cameras `camera` (repeated or reordered)."""

        return Lens(self.fisheye[camera], self.distortion[camera], self.std[camera])

    def with_distortion(self, camera: int, distortion: Tensor[[*K1K2K3K4]]) -> Lens[C]:
        """The same lenses with camera `camera`'s coefficients replaced (prior centre moved)."""

        values = self.distortion.clone()
        values[camera] = distortion
        return Lens(self.fisheye, values, self.std)


def lens_from_priors(priors: Sequence[CameraPrior], like: Tensor) -> Lens[int] | None:
    """The lenses of cameras with these priors; None when all are ideal, held pinholes.

    `distortion` gives the coefficients (missing: zero), held unless `distortion_range` says
    how far each may move. A fisheye without `distortion` is unknown: refined from the
    equidistant model within `UNKNOWN_FISHEYE_STD` unless `distortion_range` says otherwise.
    """

    fisheye, distortion, std = [], [], []
    for prior in priors:
        is_fisheye = prior.lens == "fisheye"
        size = COEFFICIENTS if is_fisheye else PINHOLE_COEFFICIENTS
        given = tuple(prior.distortion or ())
        assert len(given) <= size, f"{prior.lens} lens: at most {size} distortion coefficients"
        unknown = is_fisheye and prior.distortion is None
        spread = prior.distortion_range
        if spread is None:
            spread = UNKNOWN_FISHEYE_STD if unknown else 0.0
        fisheye.append(is_fisheye)
        distortion.append([*given, *[0.0] * (COEFFICIENTS - len(given))])
        std.append([spread] * size + [0.0] * (COEFFICIENTS - size))
    if not any(fisheye) and not any(any(d) or any(s) for d, s in zip(distortion, std, strict=True)):
        return None
    return Lens(
        torch.tensor(fisheye, device=like.device),
        like.new_tensor(distortion),
        like.new_tensor(std),
    )
