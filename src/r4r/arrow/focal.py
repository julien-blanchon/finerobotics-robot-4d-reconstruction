"""A camera's focal length from ARROW's own-camera pointmaps, pooled over many frames.

Frames are encoded in sets of `SET_SIZE` interleaved frames (each set spans the whole video);
every frame's pointmap on a pixel grid gives a weighted pinhole fit (ARROW App. C.3), and the
focal is the median over frames. With the gen2 adapter (`adapter.load_adapter`), a second pass
gives the first estimate back as a draft, which gen2 corrects rather than copies: measured on
simulated cameras with ground truth, this self-draft is the most accurate of the estimates
compared (`docs/dataset-onboarding.md`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from einops import rearrange

from r4r.geometry import camera

from .images import Resize, model_images
from .queries import per_frame, pixel_grid

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.types import RGB

    from .models.arrow import ARROW
    from .models.camera_encoder import CameraEncoder
    from .models.encoder import SlotHook

SET_SIZE = 16
GRID_STRIDE = 12  # native pixels between queried pixels


def frame_focals[N: IntVar, H: IntVar, W: IntVar](
    model: ARROW, frames: Tensor[[N, *RGB, H, W]], slot: SlotHook | None = None
) -> Tensor[[N]]:
    """fx in native pixels of each frame (uint8 frames of one camera): the weighted fit of
    `x = fx · X/Z + cx` to the frame's own-camera points (`camera.fit_intrinsics`)."""

    count, _, height, width = frames.shape
    device = next(model.parameters()).device
    resize = Resize.for_size(width, height)
    native = pixel_grid(height, width, GRID_STRIDE, device).double()
    model_xy = camera.resize_pixels(native, *resize.scale).float()
    sets = max(1, count // SET_SIZE)
    focal = torch.full((count,), torch.nan, dtype=torch.float64)
    for s in range(sets):
        views = torch.arange(s, count, sets)
        encoding = model.encode(model_images(frames[views].to(device), resize)[None], slot)
        own = model.decode(encoding, per_frame(model_xy, len(views), None), chunk=16_384)
        points = rearrange(own.points[0].double(), "(s n) c -> s n c", s=len(views))
        weight = rearrange(own.point_confidence[0].double(), "(s n) -> s n", s=len(views))
        for i, v in enumerate(views.tolist()):
            focal[v] = camera.fit_intrinsics(points[i], native, weight[i])[0, 0].cpu()
    return focal


def self_drafted_focal[N: IntVar, H: IntVar, W: IntVar](
    model: ARROW, encoder: CameraEncoder, frames: Tensor[[N, *RGB, H, W]]
) -> tuple[float, float]:
    """(no-draft focal, focal with the no-draft focal as draft): medians over frames."""

    first = float(frame_focals(model, frames).nanmedian())
    size_wh = (int(frames.shape[-1]), int(frames.shape[-2]))
    second = frame_focals(model, frames, encoder.focal_slot(first, size_wh))
    return first, float(second.nanmedian())
