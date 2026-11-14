"""ARROW: encode a set of images once, then answer any number of point queries.

Weight-compatible with the released `ifradlin/arrow` checkpoint (same parameter names) and
numerically matched to the reference implementation (bf16 autocast for the encoder and the
decoder trunk on CUDA, fp32 context projection and heads).

Ref: Fradlin et al., ARROW, arXiv:2610.01314.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self, cast

import torch
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn

from .decoder import Decoder
from .encoder import PATCH_SIZE, Encoder, SlotHook
from .query_embed import ID_WIDTH, QueryEmbedding

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar
    from torch import Tensor

    from r4r.types import RGB, UV, XYZ

    from ..queries import Queries

PRETRAINED = "ifradlin/arrow"
DECODER_WIDTH: Final = 1280
DECODER_DEPTH: Final = 8
DECODER_HEADS: Final = 16
RGB_CROP: Final = 9
OUTPUTS = ("points", "point_confidence", "uv", "visibility_logits", "normals", "displacement")


@dataclass(frozen=True, slots=True)
class Encoding[B: IntVar, S: IntVar, H: IntVar, W: IntVar, N: IntVar]:
    """Everything the decoder needs: input images, ID tokens and cached per-block keys/values
    (`[blocks, B, heads, N context tokens, head width]`)."""

    images: Tensor[[B, S, *RGB, H, W]]
    id_tokens: Tensor[[B, S, 3072]]
    keys: Tensor[[8, B, 16, N, 80]]
    values: Tensor[[8, B, 16, N, 80]]


@dataclass(frozen=True, slots=True)
class Predictions[B: IntVar, Q: IntVar]:
    """Per query, at the target frame's time; 3D quantities in the reference camera.

    Points are in ARROW's normalised scene scale (not metric). `uv` is normalised target-image
    position (`pixel = uv · [W, H] − ½`); visibility is `sigmoid(visibility_logits)`.
    """

    points: Tensor[[B, Q, *XYZ]]
    point_confidence: Tensor[[B, Q]]
    uv: Tensor[[B, Q, *UV]]
    visibility_logits: Tensor[[B, Q]]
    normals: Tensor[[B, Q, *XYZ]]
    displacement: Tensor[[B, Q, *XYZ]]

    @staticmethod
    def concat[B2: IntVar](parts: list[Predictions[B2, int]]) -> Predictions[B2, int]:
        """Join query chunks, in order."""

        return Predictions(*(torch.cat([getattr(p, name) for p in parts], 1) for name in OUTPUTS))


def head[D: IntVar, Out: IntVar](
    width: Int[D], out: Int[Out]
) -> nn.Sequential[nn.LayerNorm, nn.Linear[D, D // 2], nn.GELU, nn.Linear[D // 2, Out]]:
    return nn.Sequential(
        nn.LayerNorm(width), nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, out)
    )


class Heads[D: IntVar](nn.Module):
    """One small MLP per output, with the output activations: confidence `1 + exp(·)`,
    unit normals."""

    def __init__(self, width: Int[D]) -> None:
        super().__init__()
        self.points = head(width, 3)
        self.point_confidence = head(width, 1)
        self.uv = head(width, 2)
        self.visibility_logits = head(width, 1)
        self.normals = head(width, 3)
        self.displacement = head(width, 3)

    def forward[B: IntVar, Q: IntVar](self, tokens: Tensor[[B, Q, D]]) -> Predictions[B, Q]:
        return Predictions(
            points=self.points(tokens),
            point_confidence=1 + self.point_confidence(tokens)[..., 0].exp(),
            uv=self.uv(tokens),
            visibility_logits=self.visibility_logits(tokens)[..., 0],
            normals=F.normalize(self.normals(tokens), dim=-1, eps=1e-6),
            displacement=self.displacement(tokens),
        )


class ARROW(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.query_embed = QueryEmbedding(DECODER_WIDTH, RGB_CROP)
        self.context_proj = nn.Linear(ID_WIDTH, DECODER_WIDTH)
        self.decoder = Decoder(DECODER_WIDTH, DECODER_DEPTH, DECODER_HEADS)
        self.heads = Heads(DECODER_WIDTH)

    @classmethod
    def from_pretrained(cls, source: str | Path = PRETRAINED, device: str = "cuda") -> Self:
        """Load the released FP32 weights from a local directory or the Hugging Face Hub."""

        path = Path(source)
        file = (
            path / "model.safetensors"
            if path.is_dir()
            else hf_hub_download(str(source), "model.safetensors")
        )
        with torch.device("meta"):
            model = cls()
        model.load_state_dict(load_file(file, device=device), strict=True, assign=True)
        return model.eval()

    @torch.inference_mode()
    def encode[B: IntVar, S: IntVar, H: IntVar, W: IntVar](
        self, images: Tensor[[B, S, *RGB, H, W]], slot: SlotHook | None = None
    ) -> Encoding[B, S, H, W, int]:
        """Encode RGB in [0, 1]; any size (replicate-padded to a multiple of 14 internally,
        hence a context length `S · ⌈H/14⌉ · ⌈W/14⌉` left `int`).

        `slot` optionally rewrites the encoder's block-13 special tokens (pose conditioning).
        """

        b, _, _, h, w = images.shape
        pad_h, pad_w = -h % PATCH_SIZE, -w % PATCH_SIZE
        padded = images
        if pad_h or pad_w:
            flat = F.pad(
                rearrange(images, "b f c h w -> (b f) c h w"),
                (0, pad_w, 0, pad_h),
                mode="replicate",
            )
            padded = rearrange(flat, "(b f) c h w -> b f c h w", b=b)

        with self.autocast(images.device):
            context, id_tokens = self.encoder(padded, slot)
        context = self.context_proj(rearrange(context, "b f p c -> b (f p) c"))
        # SDPA runs in bf16 under autocast anyway: caching in bf16 halves memory, same result
        dtype = torch.bfloat16 if images.device.type == "cuda" else context.dtype
        with self.autocast(images.device):
            keys, values = self.decoder.keys_values(context, dtype)
        return Encoding(images, id_tokens, keys, values)

    @torch.inference_mode()
    def decode[B: IntVar, Q: IntVar](
        self,
        encoding: Encoding[B, int, int, int, int],
        queries: Queries[B, Q],
        chunk: int = 100_000,
    ) -> Predictions[B, Q]:
        """Answer queries in chunks of `chunk`.

        Queries are independent, so chunking changes results only at bf16 kernel noise (~1e-3),
        the same noise the reference shows across chunk sizes.
        """

        parts: list[Predictions[B, int]] = []
        for start in range(0, max(queries.count, 1), chunk):
            part = queries.slice(start, min(start + chunk, queries.count))
            with self.autocast(encoding.images.device):
                tokens = self.query_embed(encoding.images, encoding.id_tokens, part)
                tokens = self.decoder(tokens, encoding.keys, encoding.values)
            parts.append(self.heads(tokens))  # heads run in fp32, outside autocast
        # The chunks' lengths sum to Q
        return cast("Predictions[B, Q]", parts[0] if len(parts) == 1 else Predictions.concat(parts))

    @staticmethod
    def autocast(device: torch.device) -> torch.autocast:
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
