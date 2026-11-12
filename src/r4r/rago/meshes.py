"""The robot's own surface: points sampled on its collision geometry, in body frames.

Forward kinematics places them in the world at every timestep, so the robot is a calibration
target of known shape and metric size, visible in most exterior views. Samples are area-
weighted over mesh faces and box faces (the geometry types robot descriptions use for
collision); other primitives are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch

from r4r.geometry import se3

from .robot import CameraMount, compile_model

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

MESH, BOX = int(mujoco.mjtGeom.mjGEOM_MESH), int(mujoco.mjtGeom.mjGEOM_BOX)

# The 12 triangles of a unit cube's surface ([-1, 1]³), as corner indices
CUBE_CORNERS = torch.tensor(
    [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)], dtype=torch.float64
)
CUBE_FACES = torch.tensor(
    [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
     [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class Surface[P: IntVar]:
    """`P` surface points: the body each belongs to and its position in that body's frame."""

    body: Tensor[[P]]
    points: Tensor[[P, 3]]

    def to(self, device: torch.device | str, dtype: torch.dtype) -> Surface[P]:
        return Surface(self.body.to(device), self.points.to(device, dtype))


def load_surface(
    path: Path,
    count: int,
    meshdir: Path | None = None,
    cameras: tuple[CameraMount, ...] = (),
    seed: int = 0,
) -> Surface:
    """`count` area-weighted samples on the collision geometry of a robot description.

    `cameras` must match `load_robot`'s so that body indices agree.
    """

    model = compile_model(path, meshdir, cameras)
    triangles, bodies = [], []
    for geom in range(model.ngeom):
        kind = int(model.geom_type[geom])
        if kind == MESH:
            mesh = int(model.geom_dataid[geom])
            vertices = torch.as_tensor(
                model.mesh_vert[
                    model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]
                ],
                dtype=torch.float64,
            )
            faces = torch.as_tensor(
                model.mesh_face[
                    model.mesh_faceadr[mesh] : model.mesh_faceadr[mesh] + model.mesh_facenum[mesh]
                ]
            ).long()
            local = vertices[faces]  # (F, 3 corners, 3)
        elif kind == BOX:
            local = (CUBE_CORNERS * torch.as_tensor(model.geom_size[geom], dtype=torch.float64))[
                CUBE_FACES
            ]
        else:
            continue
        rotation = se3.quaternion_to_matrix(
            torch.as_tensor(model.geom_quat[geom], dtype=torch.float64)
        )
        body_from_geom = se3.from_rotation_translation(
            rotation, torch.as_tensor(model.geom_pos[geom], dtype=torch.float64)
        )
        triangles.append(
            se3.transform_points(body_from_geom, local.reshape(-1, 3)).reshape(-1, 3, 3)
        )
        bodies.append(torch.full((len(local),), int(model.geom_bodyid[geom])))
    assert triangles, f"{path}: no mesh or box collision geometry to sample"

    triangle, body = torch.cat(triangles), torch.cat(bodies)
    edge_a, edge_b = triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
    area = torch.linalg.cross(edge_a, edge_b).norm(dim=-1) / 2

    # Area-weighted triangles, uniform barycentric points within each
    generator = torch.Generator().manual_seed(seed)
    chosen = torch.multinomial(area, count, replacement=True, generator=generator)
    u, v = torch.rand(2, count, 1, generator=generator, dtype=torch.float64)
    flip = (u + v) > 1
    u, v = torch.where(flip, 1 - u, u), torch.where(flip, 1 - v, v)
    points = triangle[chosen, 0] + u * edge_a[chosen] + v * edge_b[chosen]
    return Surface(body[chosen], points)
