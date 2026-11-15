"""Axis glossary and named fixed-size axes for shape annotations.

Dimension type parameters mean the same thing in every file:

    Bs  leading batch dims (IntTuple)   B  batch            S  images in an encoding
    N   points / pixels                  Q  queries          P  patch tokens per frame
    D   model width                      NHead  heads        HD  head width
    H, W  image height / width           C  cameras          T  robot timesteps
    J   joints                           L  kinematic links  K  correspondences
    M   observations (track samples)     V  views (camera × timestep)
    E   episodes                         G  groups of episodes
    F   frames in a query pattern        NB  free bodies
    NQ  qpos width (MuJoCo generalised coordinates)
    NState  recorded robot-state width   NWheel  wheels of a drive
    NPair   image or motion pairs
    In, Out, Hidden  layer input / output / hidden width
    R, Cl   patch-grid rows / columns
    Ho, Wo  output height / width        Side  side of a square resize
    Radius  correlation window radius    Depth  decoder blocks   Crop  query RGB crop side
    Start, Stop  slice bounds

A fixed-size axis is named by an alias that spells its components, so the name shows the
size: `Tensor[[N, *XYZ]]` is `Tensor[[N, 3]]`. Pyrefly accepts these only as unpacked
`IntTuple` aliases, and only in a shape with no other unpacked part: inside `[*Bs, ...]`
write the literal (`Tensor[[*Bs, 3]]`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shape_extensions import IntTuple
    from torch import Tensor

# Points, translations, directions, axis-angle vectors (metres or radians)
type XYZ = IntTuple[3]
# Homogeneous points
type XYZW = IntTuple[4]
# Pixel coordinates (x right, y down; pixels)
type UV = IntTuple[2]
# Planar positions and planar twists: (x, y) and (vx, vy, ω)
type XY = IntTuple[2]
type XYTheta = IntTuple[3]
# Unit quaternion, scalar first
type WXYZ = IntTuple[4]
# se(3) tangent vector, translational part first: (v, ω)
type VxVyVzWxWyWz = IntTuple[6]
# Rotation or intrinsics matrix; rigid transform
type Mat3x3 = IntTuple[3, 3]
type Mat4x4 = IntTuple[4, 4]
# Pinhole intrinsics as a vector
type FxFyCxCy = IntTuple[4]
# Radial lens coefficients (Brown–Conrady k1..k3 with k4 = 0, or Kannala–Brandt k1..k4)
type K1K2K3K4 = IntTuple[4]
# Colour channels
type RGB = IntTuple[3]

# A 0-d tensor (a loss, a prior, a reduced statistic)
type Scalar = Tensor[[]]
