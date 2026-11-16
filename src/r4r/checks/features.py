"""DINOv2 patch features of a few frames: the one feature helper of the image checks.

A small self-supervised ViT (DINOv2 ViT-S/14 with registers, 21 M parameters, via `timm`)
gives per-patch descriptors whose cosine similarity matches parts across views and
episodes, and a global (class-token) descriptor per frame. Frames are resized so that the
patch grid has `rows × cols` patches of 14 px, keeping the aspect; descriptors are
L2-normalised.

Pixels and patches: the delivered frame `H × W` is resized to `14·R × 14·Cl`, so patch `(r,
c)` covers frame pixels centred on `((c + ½)·W/Cl − ½, (r + ½)·H/R − ½)` (pixel centres at
integer coordinates). `sample` reads descriptors at frame pixels by bilinear interpolation
of the patch grid.

Ref: Oquab et al., DINOv2, arXiv:2304.07193; Darcet et al., registers, arXiv:2309.16588.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from einops import rearrange

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor, nn

    from r4r.types import RGB, UV

MODEL = "vit_small_patch14_reg4_dinov2.lvd142m"
PATCH = 14
WIDTH = 384
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class Features[S: IntVar, R: IntVar, Cl: IntVar]:
    """Unit descriptors per patch `[S, R, Cl, D]` and per frame `[S, D]` (class token), for
    frames of `size` (height, width) in pixels."""

    patches: Tensor[[S, R, Cl, 384]]
    global_: Tensor[[S, 384]]
    size: tuple[int, int]


@functools.cache
def load_model(device: str = "cpu", name: str = MODEL) -> nn.Module:
    """The pretrained ViT in eval mode (weights from the Hugging Face cache, `HF_HOME`)."""

    import timm  # in the env's `parity` group; the checks are optional

    model = timm.create_model(name, pretrained=True, num_classes=0, dynamic_img_size=True)
    return model.eval().to(device)


def grid(height: int, width: int, long_side: int) -> tuple[int, int]:
    """Patch rows and columns for a frame: its longer side resized to about `long_side` px."""

    scale = long_side / max(height, width)
    return max(1, round(height * scale / PATCH)), max(1, round(width * scale / PATCH))


@torch.inference_mode()
def extract[S: IntVar, H: IntVar, W: IntVar](
    images: Tensor[[S, *RGB, H, W]], long_side: int = 448, device: str = "cpu", batch: int = 16
) -> Features[S, int, int]:
    """Features of uint8 RGB frames (one size), the longer side at about `long_side` px."""

    model = load_model(device)
    height, width = int(images.shape[-2]), int(images.shape[-1])
    rows, cols = grid(height, width, long_side)
    mean = torch.tensor(IMAGENET_MEAN, device=device)[:, None, None]
    std = torch.tensor(IMAGENET_STD, device=device)[:, None, None]
    patches, global_ = [], []
    autocast = torch.autocast("cuda", torch.bfloat16, enabled=device.startswith("cuda"))
    for start in range(0, len(images), batch):
        x = images[start : start + batch].to(device).float() / 255
        x = F.interpolate(x, (rows * PATCH, cols * PATCH), mode="bilinear", antialias=True)
        with autocast:
            tokens = model.forward_features((x - mean) / std).float()
        skip = model.num_prefix_tokens  # class token, then registers
        grid_tokens = rearrange(tokens[:, skip:], "s (r c) d -> s r c d", r=rows, c=cols)
        patches.append(F.normalize(grid_tokens, dim=-1))
        global_.append(F.normalize(tokens[:, 0], dim=-1))
    return Features(torch.cat(patches), torch.cat(global_), (height, width))


def sample[S: IntVar, N: IntVar](
    features: Features[S, int, int], pixels: Tensor[[S, N, *UV]]
) -> Tensor[[S, N, 384]]:
    """Unit descriptors at frame pixels (bilinear between patch centres; module doc)."""

    height, width = features.size
    # Frame pixel -> patch-grid coordinate -> grid_sample's [-1, 1] (align_corners=False)
    gx = (pixels[..., 0] + 0.5) / width * 2 - 1
    gy = (pixels[..., 1] + 0.5) / height * 2 - 1
    where = torch.stack([gx, gy], -1)[:, :, None].to(features.patches)
    maps = rearrange(features.patches, "s r c d -> s d r c")
    out = F.grid_sample(maps, where, mode="bilinear", padding_mode="border", align_corners=False)
    return F.normalize(rearrange(out[..., 0], "s d n -> s n d"), dim=-1)


def upsampled[S: IntVar](
    features: Features[S, int, int], factor: int
) -> Tensor[[S, int, int, 384]]:
    """Patch descriptors bilinearly upsampled by `factor` per axis and renormalised: a finer
    grid for locating parts below the patch size."""

    maps = rearrange(features.patches, "s r c d -> s d r c")
    fine = F.interpolate(maps, scale_factor=factor, mode="bilinear", align_corners=False)
    return F.normalize(rearrange(fine, "s d r c -> s r c d"), dim=-1)
