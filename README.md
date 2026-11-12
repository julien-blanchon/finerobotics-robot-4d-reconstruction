# r4r: robot 4D reconstruction

Metric, robot-frame 4D reconstruction of robot episodes (synchronized RGB + joint states + robot
model): depth, normals, 3D tracks, camera intrinsics, poses and mounts.

- **ARROW** ([arXiv:2610.01314](https://arxiv.org/abs/2610.01314)), re-implemented: feed-forward
  reconstruction and tracking from the images.
- **RAGO**, robot-aware geometric optimisation in the spirit of Calib3R
  ([arXiv:2509.08813](https://arxiv.org/abs/2509.08813)): fits the unknown camera mounts, arm
  placements and scale so that ARROW's geometry agrees with forward kinematics.

Status: early development. See `STYLE.md` for conventions and `docs/` (to come) for usage.

## Setup (JUPITER)

```bash
source jupiter/env.sh   # environments and caches on /e/scratch
uv sync
uv run pytest
```
