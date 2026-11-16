"""Robot masks from a text-prompted segmentation model (SAM 3), for `r4r silhouette --masks`.

Background subtraction fails where moved objects fill the change mask (a bag lifted by both
arms) or where lighting and writing change more than the robot does. SAM 3 segments "robot
arm" in each frame without a camera prior. The model and `transformers` (≥ 5 for `Sam3Model`)
are not r4r dependencies: this module runs in a separate environment on a GPU.

    python -m r4r.rago.segment DATASET CAMERA SELECTION.json OUTDIR [--prompt "robot arm"]

`SELECTION.json` (`r4r silhouette … --select`): setup → episode → frame indices. Writes
`OUTDIR/<setup>.npz`: per episode `e<episode>` the masks bit-packed along the width
`[F, H, ⌈W/8⌉]`, `e<episode>_frames` and `e<episode>_width`. One file per setup (scratch
inodes).

Ref: SAM 3 (Meta, 2025), `facebook/sam3` on the Hugging Face hub
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

MODEL = "facebook/sam3"
# Instances above this score are the robot; pixels above this probability belong to them
SCORE, PIXEL = 0.4, 0.5
BATCH = 8


def read(path: Path) -> dict[int, dict[int, np.ndarray]]:
    """A setup's masks: episode → frame index → [H, W] uint8 (255 on the robot)."""

    stored = np.load(path)
    out: dict[int, dict[int, np.ndarray]] = {}
    for key in stored.files:
        if not key[1:].isdigit():
            continue
        width = int(stored[f"{key}_width"])
        bits = np.unpackbits(stored[key], axis=-1)[..., :width]
        frames = stored[f"{key}_frames"].tolist()
        out[int(key[1:])] = {f: b * 255 for f, b in zip(frames, bits, strict=True)}
    return out


def segmenter(prompt: str, device: str = "cuda") -> Callable[[np.ndarray], np.ndarray]:
    """A function from RGB frames [N, H, W, 3] uint8 to robot masks [N, H, W] bool: the union
    of SAM 3's instances for `prompt` (bf16)."""

    import torch
    from transformers import Sam3Model, Sam3Processor

    model = Sam3Model.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    processor = Sam3Processor.from_pretrained(MODEL)

    def masks(images: np.ndarray) -> np.ndarray:
        out = np.zeros(images.shape[:3], bool)
        for start in range(0, len(images), BATCH):
            chunk = list(images[start : start + BATCH])
            inputs = processor(images=chunk, text=[prompt] * len(chunk), return_tensors="pt")
            inputs = {
                k: v.to(device, torch.bfloat16) if torch.is_floating_point(v) else v.to(device)
                for k, v in inputs.items()
            }
            with torch.no_grad():
                found = processor.post_process_instance_segmentation(
                    model(**inputs),
                    threshold=SCORE,
                    mask_threshold=PIXEL,
                    target_sizes=inputs["original_sizes"].tolist(),
                )
            for i, instances in enumerate(found):
                if len(instances["masks"]):
                    out[start + i] = instances["masks"].bool().any(0).cpu().numpy()
        return out

    return masks


def main() -> None:
    from r4r.cli import dataset_root
    from r4r.data.lerobot import keyframes

    args = sys.argv[1:]
    prompt = args[args.index("--prompt") + 1] if "--prompt" in args else "robot arm"
    dataset, camera, selection, outdir = args[:4]
    root, folder = dataset_root(dataset), Path(outdir)
    folder.mkdir(parents=True, exist_ok=True)
    segment = segmenter(prompt)
    for setup, episodes in json.loads(Path(selection).read_text()).items():
        path = folder / f"{setup}.npz"
        if path.exists():
            continue
        start, arrays = time.perf_counter(), {}
        for episode, frames in episodes.items():
            images = keyframes(root, int(episode), frames, [camera])[camera]
            masks = segment(images.permute(0, 2, 3, 1).numpy())
            arrays[f"e{episode}"] = np.packbits(masks, axis=-1)
            arrays[f"e{episode}_frames"] = np.array(frames)
            arrays[f"e{episode}_width"] = np.array(masks.shape[-1])
        np.savez_compressed(path, **arrays)
        print(f"{setup}: {len(episodes)} episodes in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
