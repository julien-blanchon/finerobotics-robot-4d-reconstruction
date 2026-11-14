"""RoMa v2 dense matcher: encode frames once, then match any pairs of them in batches.

Weight-compatible with the released `romav2.0.1.pt` (same parameter names, see
`load_checkpoint` for the mapping) and numerically matched to the reference "precise" setting:
frames are squashed to 800² (coarse match + first refinement) and 1280² (second refinement),
and both directions are predicted. Each frame's DINOv3 descriptors and fine features are
computed once in `encode`, however many pairs use it. The 1280² refinement only runs for
frames larger than 800² (`fine`): below, it refines past the frames' own resolution, at about
half the cost of a match (set R4R_ROMA_FINE=1 to force it).

Ref: Edstedt et al., RoMa v2, arXiv:2511.15706.
Adapted from https://github.com/Parskatt/RoMaV2 (MIT).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .dinov3 import OUTPUT_BLOCKS, DINOv3
from .fine_features import FineFeatures
from .matcher import Matcher
from .refiner import ConvRefiner

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

CHECKPOINT_URL = "https://github.com/Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt"
COARSE_SIZE = 800
FINE_SIZE = 1280
ANCHOR_SIZE = 512  # displacement embeddings are scaled by image size / 512
STRIDES = (4, 2, 1)  # refiner order, coarse to fine
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Compiled regions specialise on batch size 1 and on 32- vs 64-bit indexing: up to ~12 graphs
# per region, above dynamo's default limit of 8 (fullgraph=True makes that a hard error)
RECOMPILE_LIMIT = 32


def default_checkpoint() -> Path:
    """Where torch.hub caches the release: `$TORCH_HOME/hub/checkpoints/romav2.0.1.pt`."""

    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    torch_home = Path(os.environ.get("TORCH_HOME", cache / "torch"))
    return torch_home / "hub" / "checkpoints" / Path(CHECKPOINT_URL).name


@dataclass(frozen=True, slots=True)
class Encoding[S: IntVar]:
    """Per-frame features: DINOv3 descriptors at 800² and VGG features at both resolutions.

    `coarse_fine` and `fine` hold strides 1, 2, 4 (channels-last, bf16). About 0.5 GB a frame.
    """

    descriptors: Tensor[[S, 50, 50, 2048]]
    coarse_fine: tuple[
        Tensor[[S, 800, 800, 64]], Tensor[[S, 400, 400, 128]], Tensor[[S, 200, 200, 256]]
    ]
    # None: no 1280² refinement (frames no larger than the coarse size)
    fine: (
        tuple[Tensor[[S, 1280, 1280, 64]], Tensor[[S, 640, 640, 128]], Tensor[[S, 320, 320, 256]]]
        | None
    )
    size: tuple[int, int]  # input (H, W), for pixel coordinates

    def __getitem__(self, index: Tensor[[int]] | slice) -> Encoding:
        """Select frames, e.g. `encoding[pairs[:, 0]]`."""

        return Encoding(
            self.descriptors[index],
            (self.coarse_fine[0][index], self.coarse_fine[1][index], self.coarse_fine[2][index]),
            None
            if self.fine is None
            else (self.fine[0][index], self.fine[1][index], self.fine[2][index]),
            self.size,
        )


@dataclass(frozen=True, slots=True)
class Matches[B: IntVar]:
    """Dense matches in both directions on the 1280² working grid.

    Warps use grid_sample's normalised coordinates (`align_corners=False`):
    `pixel = (x + 1) / 2 · W − ½` in the other image. `certainty` is the predicted overlap
    probability; `precision` the 2×2 inverse covariance of the warp in working-grid pixels⁻².
    """

    warp_ab: Tensor[[B, 1280, 1280, 2]]
    warp_ba: Tensor[[B, 1280, 1280, 2]]
    certainty_ab: Tensor[[B, 1280, 1280]]
    certainty_ba: Tensor[[B, 1280, 1280]]
    precision_ab: Tensor[[B, 1280, 1280, 2, 2]]
    precision_ba: Tensor[[B, 1280, 1280, 2, 2]]
    size_a: tuple[int, int]
    size_b: tuple[int, int]


def resize[S: IntVar, H: IntVar, W: IntVar](
    x: Tensor[[S, H, W, int]], size: int
) -> Tensor[[S, int, int, int]]:
    channels_first = F.interpolate(
        rearrange(x, "s h w c -> s c h w"), size=(size, size), mode="bilinear", align_corners=False
    )
    return rearrange(channels_first, "s c h w -> s h w c")


class RoMaV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.descriptor = DINOv3()
        self.matcher = Matcher()
        self.refiners: nn.ModuleDict = nn.ModuleDict(
            {
                "4": ConvRefiner(256, 192, 79, radius=3),
                "2": ConvRefiner(128, 48, 23, radius=1),
                "1": ConvRefiner(64, 12, 8, radius=None),
            }
        )
        self.refiner_features = FineFeatures()

    @classmethod
    def from_checkpoint(cls, path: Path | None = None, device: str = "cuda") -> Self:
        with torch.device("meta"):
            model = cls()
        state = load_checkpoint(path or default_checkpoint())
        model.load_state_dict(
            {name: tensor.to(device) for name, tensor in state.items()}, strict=True, assign=True
        )
        return model.eval()

    @torch.inference_mode()
    @torch._dynamo.config.patch(recompile_limit=RECOMPILE_LIMIT)
    def encode[S: IntVar, H: IntVar, W: IntVar](self, images: Tensor[[S, 3, H, W]]) -> Encoding[S]:
        """RGB in [0, 1], any size; squashed (bicubic, antialiased) to the working sizes."""

        mean = images.new_tensor(IMAGENET_MEAN)[:, None, None]
        std = images.new_tensor(IMAGENET_STD)[:, None, None]

        def normalised(size: int) -> Tensor:
            resized = F.interpolate(images, size=(size, size), mode="bicubic", antialias=True)
            return (resized - mean) / std

        refine = max(images.shape[2:]) > COARSE_SIZE or os.environ.get("R4R_ROMA_FINE") == "1"
        with torch.autocast(images.device.type, dtype=torch.bfloat16):
            coarse = normalised(COARSE_SIZE)
            descriptors = self.descriptor(coarse)
            coarse_fine = self.refiner_features(coarse)
            fine_features = self.refiner_features(normalised(FINE_SIZE)) if refine else None
        return Encoding(descriptors, coarse_fine, fine_features, (images.shape[2], images.shape[3]))

    @torch.inference_mode()
    @torch._dynamo.config.patch(recompile_limit=RECOMPILE_LIMIT)
    def match[B: IntVar](self, a: Encoding[B], b: Encoding[B]) -> Matches[B]:
        """Match frame `a[i]` with `b[i]` for every `i`."""

        out = self.matcher(a.descriptors, b.descriptors)
        warp: Tensor[[int, int, int, 2]] = out[..., :2]
        # Precision slots start at zero; the refiners accumulate into them
        confidence: Tensor[[int, int, int, 4]] = F.pad(out[..., 2:], (0, 3))
        stages = [(COARSE_SIZE, a.coarse_fine, b.coarse_fine)]
        if a.fine is not None and b.fine is not None:
            stages.append((FINE_SIZE, a.fine, b.fine))
        for size, features_a, features_b in stages:
            if size == FINE_SIZE:
                # The coarse stage's precision is in its own pixels: restart it, keep the overlap
                confidence = F.pad(confidence[..., :1], (0, 3))
            for stride, refiner, feature_a, feature_b in zip(
                STRIDES,
                self.refiners.values(),  # insertion order: strides 4, 2, 1
                reversed(features_a),
                reversed(features_b),
                strict=True,
            ):
                warp = resize(warp, size // stride)
                confidence = resize(confidence, size // stride)
                # Frames a then b on one batch axis (see ConvRefiner.forward)
                features = torch.cat((feature_a, feature_b))
                warp, confidence = refiner(features, warp, confidence, size / ANCHOR_SIZE)

        n = len(a.descriptors)
        precision = rearrange(confidence[..., [1, 2, 2, 3]], "s h w (i j) -> s h w i j", i=2)
        certainty = confidence[..., 0].sigmoid()
        return Matches(
            warp[:n],
            warp[n:],
            certainty[:n],
            certainty[n:],
            precision[:n],
            precision[n:],
            a.size,
            b.size,
        )


def load_checkpoint(path: Path) -> dict[str, Tensor]:
    """The released state dict, mapped to this module's names (on CPU, memory-mapped).

    The only checkpoint format is a pickled `.pt`: it loads with `weights_only=True` (tensors
    only, no code). Mapping: `f.*` → `descriptor.*`; DINOv3 blocks past the last output block
    and the unused `mask_token` are dropped; the masked QKV bias is folded
    (`qkv.bias · qkv.bias_mask`, which is all zeros in this release).
    """

    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    mapped = {}
    for name, tensor in state.items():
        if not name.startswith("f."):
            mapped[name] = tensor
            continue
        parts = name.split(".")
        if name == "f.mask_token" or name.endswith(".bias_mask"):
            continue
        if parts[1] == "blocks" and int(parts[2]) > OUTPUT_BLOCKS[-1]:
            continue
        if name.endswith("attn.qkv.bias"):
            tensor = tensor * state[name + "_mask"]
        mapped["descriptor." + name.removeprefix("f.")] = tensor
    return mapped
