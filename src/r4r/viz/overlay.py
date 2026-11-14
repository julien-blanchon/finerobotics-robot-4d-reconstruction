"""Visual check of a solved rig: the robot (its description's meshes, posed by the corrected
forward kinematics) drawn over every camera's frames with the solved cameras.

If cameras, intrinsics, kinematics and timing are right, the drawn robot sits on the real one
in every camera and every frame; residuals alone can miss a wrong camera (a static camera
with few links to the others fits its own tracks from any pose). A check for people and
agents, never part of the solve. Input: a solution saved by `r4r rig --save`.
"""

from __future__ import annotations

import colorsys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import mujoco
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

ALPHA = 0.45
MAX_FACES = 4000  # per geom (decimated beyond)
AXIS_LENGTH = 0.06  # metres, end-effector frames
LABEL_HEIGHT = 28

type Surface = tuple[int, np.ndarray, np.ndarray]  # body, vertices in its frame [V, 3], faces


def surfaces(model: mujoco.MjModel) -> list[Surface]:
    """Per geom to draw: its body, its vertices in the body frame and its triangles. Per body,
    its visual geoms if it has any, else its other (collision) geoms: visual meshes can be
    missing for some links (unreadable files dropped) and present for others."""

    visual = {
        g
        for g in range(model.ngeom)
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0
    }
    with_visual = {int(model.geom_bodyid[g]) for g in visual}
    chosen = [
        g for g in range(model.ngeom) if (g in visual) == (int(model.geom_bodyid[g]) in with_visual)
    ]
    out = []
    for g in chosen:
        mesh = _mesh(model, g)
        if mesh is None:
            continue
        vertices, faces = mesh
        if len(faces) > MAX_FACES:
            faces = faces[np.random.default_rng(0).choice(len(faces), MAX_FACES, replace=False)]
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[g])
        placed = vertices @ rotation.reshape(3, 3).T + model.geom_pos[g]
        out.append((int(model.geom_bodyid[g]), placed, faces))
    return out


def _mesh(model: mujoco.MjModel, geom: int) -> tuple[np.ndarray, np.ndarray] | None:
    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[geom])
        va, vn = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        fa, fn = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        return model.mesh_vert[va : va + vn].astype(np.float64), model.mesh_face[fa : fa + fn]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return _box(model.geom_size[geom])
    if kind in (
        mujoco.mjtGeom.mjGEOM_SPHERE,
        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
    ):
        return _round(kind, model.geom_size[geom])
    return None


def _box(size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * size
    faces = np.array(
        [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
         [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    )  # fmt: skip
    return corners, faces


def _round(kind: int, size: np.ndarray, n: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Rings along z: a sphere, ellipsoid, capsule (hemispheres apart) or cylinder."""

    radius = size[0]
    straight = kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER)
    half = size[1] if straight else 0.0
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rings = np.linspace(-1, 1, 7)
    vertices = []
    for z in rings:
        if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r, h = radius, z * half
        else:
            r, h = radius * np.sqrt(max(0.0, 1 - z * z)), z * radius + np.sign(z) * half
        vertices += [[r * np.cos(a), r * np.sin(a), h] for a in angles]
    points = np.array(vertices)
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        points = points / radius * size
    faces = [
        face
        for i in range(len(rings) - 1)
        for j in range(n)
        for face in (
            [i * n + j, i * n + (j + 1) % n, i * n + j + n],
            [i * n + (j + 1) % n, i * n + (j + 1) % n + n, i * n + j + n],
        )
    ]
    return points, np.array(faces)


def _colour(body: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((body * 0.618) % 1.0, 0.75, 1.0)
    return int(255 * r), int(255 * g), int(255 * b)


def draw(
    image: np.ndarray,
    world_from_body: np.ndarray,
    world_from_camera: np.ndarray,
    intrinsics: np.ndarray,
    geoms: list[Surface],
    frames: list[int],
) -> np.ndarray:
    """`image` (RGB) with the robot's surfaces (one colour per body, translucent) and the
    frames of the bodies `frames` (RGB axes for x, y, z)."""

    camera_from_world = np.linalg.inv(world_from_camera)
    layer = image.copy()
    width = image.shape[1]
    for body, vertices, faces in geoms:
        pose = camera_from_world @ world_from_body[body]
        local = vertices @ pose[:3, :3].T + pose[:3, 3]
        pixel = (local @ intrinsics.T)[:, :2] / np.maximum(local[:, 2:3], 1e-6)
        ok = (local[:, 2] > 0.02)[faces].all(1)
        polygons = np.round(pixel[faces[ok]] * 4).astype(np.int32)  # 2 bits of sub-pixel
        near = (np.abs(polygons[..., 0]) < 16 * width).all(1)
        if near.any():
            cv2.fillPoly(layer, list(polygons[near]), _colour(body), cv2.LINE_AA, shift=2)
    out = cv2.addWeighted(layer, ALPHA, image, 1 - ALPHA, 0)
    for body in frames:
        pose = camera_from_world @ world_from_body[body]
        if pose[2, 3] < 0.05:
            continue
        points = np.vstack([pose[:3, 3], pose[:3, 3] + AXIS_LENGTH * pose[:3, :3].T])
        pixel = (points @ intrinsics.T)[:, :2] / points[:, 2:3]
        origin = tuple(int(v) for v in pixel[0])
        for axis, colour in enumerate([(255, 40, 40), (40, 255, 40), (60, 120, 255)]):
            end = tuple(int(v) for v in pixel[1 + axis])
            cv2.line(out, origin, end, colour, 2, cv2.LINE_AA)
    return out


def leaves(model: mujoco.MjModel) -> list[int]:
    """Bodies without children (end effectors, fingers, tool frames)."""

    parents = {int(p) for p in model.body_parentid[1:]}
    return [b for b in range(1, model.nbody) if b not in parents]


def overlay_frames(
    solution: dict,
    videos: dict[str, np.ndarray],
    model: mujoco.MjModel,
    episode_index: int,
    times: list[int],
    height: int,
    label: str,
) -> Iterator[np.ndarray]:
    """One composite frame per timestep in `times` (of the episode `episode_index` of the
    solution): every camera side by side, the robot drawn over each, a label on top."""

    start = sum(solution["lengths"][:episode_index])
    steps = solution["lengths"][episode_index]
    world = solution["world_from_body"][start : start + steps].double().numpy()
    mounts = solution["mounts_at"][:, start : start + steps].double().numpy()
    intrinsics = solution["intrinsics"].double().numpy()
    geoms, frames = surfaces(model), leaves(model)
    names = solution["names"]
    widths = [round(v.shape[2] * height / v.shape[1]) // 2 * 2 for v in map(videos.get, names)]
    for t in times:
        tiles = []
        for c, name in enumerate(names):
            image = np.ascontiguousarray(videos[name][t])
            camera = world[t, solution["camera_body"][c]] @ mounts[c, t]
            tile = draw(image, world[t], camera, intrinsics[c], geoms, frames)
            tiles.append(cv2.resize(tile, (widths[c], height), interpolation=cv2.INTER_AREA))
        frame = np.zeros((height + LABEL_HEIGHT, sum(widths), 3), dtype=np.uint8)
        frame[LABEL_HEIGHT:] = np.hstack(tiles)
        text = f"{label}  t {t}/{steps}"
        cv2.putText(frame, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        yield frame


def write_video(frames: Iterator[np.ndarray], path: Path, fps: float) -> int:
    """H.264 MP4 (plays in browsers); returns the number of frames."""

    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with av.open(str(path), mode="w") as container:
        stream = None
        for frame in frames:
            if stream is None:
                stream = container.add_stream("libx264", rate=max(round(fps), 1))
                stream.height, stream.width = frame.shape[0], frame.shape[1] // 2 * 2
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "20"}
            even = frame[: stream.height // 2 * 2, : stream.width]
            container.mux(stream.encode(av.VideoFrame.from_ndarray(even, format="rgb24")))
            count += 1
        if stream is not None:
            container.mux(stream.encode())
    return count
