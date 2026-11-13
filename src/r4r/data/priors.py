"""Per-dataset priors: what a human or agent established about a dataset, overriding its metadata.

A dataset's own metadata (joint mapping, camera links, labels) is often incomplete or wrong.
Priors live in this repository as `priors/<family>/<dataset>.json`, one per dataset root, and
take precedence over `meta/robot.json` and the camera feature info:

```json
{
  "base": "fixed",
  "joints": [{"index": 16, "urdf_joint": "idx11_head_joint1", "sign": 1.0}],
  "removed_joints": ["joint8"],
  "cameras": {
    "wrist_left": {"mount": "moving", "link": "panda_hand", "position": [0.05, 0.0, 0.04],
                   "position_range": 0.05, "optical_axis": [0, 0, 1],
                   "optical_axis_range_deg": 30, "lens": "pinhole"}
  },
  "notes": "free text: what was checked and how"
}
```

`joints` add or replace (by URDF joint name) entries of the state mapping; `removed_joints`
drops wrong ones. Camera positions and optical axes are in the frame of `link` (moving) or
of the robot base (static); ranges bound what fitting may change.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .episode import JointMap

PRIORS = Path(__file__).resolve().parents[3] / "priors"


@dataclass(frozen=True, slots=True)
class CameraPrior:
    mount: Literal["static", "moving"] | None = None
    link: str | None = None
    position: tuple[float, float, float] | None = None
    position_range: float = math.inf  # metres
    optical_axis: tuple[float, float, float] | None = None
    optical_axis_range_deg: float = 180.0
    lens: Literal["pinhole", "fisheye"] = "pinhole"


@dataclass(frozen=True, slots=True)
class DatasetPriors:
    base: Literal["fixed", "mobile"] = "fixed"
    joints: tuple[JointMap, ...] = ()
    removed_joints: tuple[str, ...] = ()
    cameras: dict[str, CameraPrior] = field(default_factory=dict)
    notes: str = ""

    def joint_map(self, metadata: tuple[JointMap, ...]) -> tuple[JointMap, ...]:
        """The dataset's joint mapping with these priors applied."""

        replaced = {j.name for j in self.joints} | set(self.removed_joints)
        return tuple(j for j in metadata if j.name not in replaced) + self.joints


def priors_path(root: Path) -> Path:
    """`priors/<family>/<dataset>.json` for a delivery root `<family>/datasets/<dataset>`."""

    return PRIORS / root.parent.parent.name / f"{root.name}.json"


def load_priors(root: Path) -> DatasetPriors:
    """The priors of a dataset root (empty when none were written)."""

    path = priors_path(root)
    if not path.exists():
        return DatasetPriors()
    raw = json.loads(path.read_text())
    joints = tuple(
        JointMap(
            j["urdf_joint"],
            j["index"],
            float(j.get("sign", 1.0)),
            tuple(j["from"]) if j.get("from") else None,
            tuple(j["to"]) if j.get("to") else None,
        )
        for j in raw.get("joints", [])
    )
    cameras = {
        name: CameraPrior(
            mount=c.get("mount"),
            link=c.get("link"),
            position=tuple(c["position"]) if c.get("position") else None,
            position_range=float(c.get("position_range", math.inf)),
            optical_axis=tuple(c["optical_axis"]) if c.get("optical_axis") else None,
            optical_axis_range_deg=float(c.get("optical_axis_range_deg", 180.0)),
            lens=c.get("lens", "pinhole"),
        )
        for name, c in raw.get("cameras", {}).items()
    }
    return DatasetPriors(
        base=raw.get("base", "fixed"),
        joints=joints,
        removed_joints=tuple(raw.get("removed_joints", [])),
        cameras=cameras,
        notes=raw.get("notes", ""),
    )
