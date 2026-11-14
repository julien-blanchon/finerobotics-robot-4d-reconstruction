"""Load the gen2 adapter into ARROW: LoRA merged into the encoder, and its camera encoder.

The adapter (`experiments/arrow_pose/train_lora.py`, "gen2") holds, for each adapted linear
layer `<name>`, `<name>.down [R, In]` and `<name>.up [Out, R]`; the layer becomes
`W + (alpha / R) · up @ down`, so the merged model runs at no extra cost. Keys `cam_enc.*` are
the trained camera encoder that takes focal drafts.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors.torch import load_file

from .models.camera_encoder import CameraEncoder

if TYPE_CHECKING:
    from .models.arrow import ARROW

ALPHA = 16.0  # training's LoRA alpha (scale = alpha / rank)


@torch.no_grad()
def load_adapter(model: ARROW, path: Path) -> CameraEncoder:
    """Merge the adapter at `path` (safetensors) into `model` in place; its camera encoder."""

    state = load_file(str(path))
    layers = dict(model.named_modules())
    for key in [k for k in state if k.endswith(".down")]:
        name = key.removesuffix(".down")
        down, up = state[key], state[f"{name}.up"]
        weight = layers[name].weight
        delta = (ALPHA / down.shape[0]) * (up.double() @ down.double())
        weight += delta.to(weight)
    encoder = CameraEncoder()
    prefix = "cam_enc."
    encoder.load_state_dict(
        {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
    )
    return encoder.to(next(model.parameters()).device).eval()
