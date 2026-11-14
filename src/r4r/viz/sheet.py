"""A contact sheet to match a dataset's videos with its robot description by eye.

Rows are moments of the episode. Each row shows every camera's frame, then the robot's
skeleton (body origins joined to their parents, from forward kinematics) in three
orthographic views of the base frame: top (x→right, y→up), side (x→right, z→up) and front
(y→left, z→up). Bodies carrying a declared camera, branch points and leaves are labelled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

from r4r.rago.kinematics import forward_kinematics
from r4r.rago.robot import qpos_from_state

if TYPE_CHECKING:
    from torch import Tensor

    from r4r.data.episode import Episode
    from r4r.rago.robot import Robot
    from r4r.types import RGB, XYZ

TILE = 240  # pixels per camera tile side
VIEW = 360  # pixels per skeleton view side
VIEWS = (("top", 0, 1), ("side", 0, 2), ("front", 1, 2))  # (name, horizontal axis, vertical axis)
WHITE, GREY, YELLOW = (255, 255, 255), (130, 130, 130), (255, 210, 0)


def contact_sheet(
    episode: Episode[int, int], robot: Robot[int, int, int], moments: int = 4
) -> np.ndarray:
    """RGB image: one row per moment, camera tiles then the three skeleton views."""

    qpos = qpos_from_state(robot, episode.state, episode.joints)
    world_from_body = forward_kinematics(robot, qpos).double().cpu()
    origins = world_from_body[..., :3, 3]  # [T, L, 3]
    steps = len(qpos)
    times = torch.linspace(0, steps - 1, moments).round().long().tolist()

    # One scale for all views and moments: the robot's extent over the episode
    low = origins.flatten(0, 1).amin(0) - 0.1
    high = origins.flatten(0, 1).amax(0) + 0.1
    extent = float((high - low).max())
    labelled = _labelled_bodies(robot, episode)

    rows = []
    for t in times:
        tiles = [
            _camera_tile(c.frames[t] if len(c.frames) else None, c.name) for c in episode.cameras
        ]
        for name, horizontal, vertical in VIEWS:
            tile = np.zeros((VIEW, VIEW, 3), np.uint8)

            def to_pixel(
                point: Tensor[[*XYZ]],
                h: int = horizontal,
                v: int = vertical,
                mirror: bool = name == "front",
            ) -> tuple[int, int]:
                sign = -1.0 if (mirror and h == 1) else 1.0
                centre = (low + high) / 2
                x = sign * float(point[h] - centre[h]) / extent + 0.5
                y = 0.5 - float(point[v] - centre[v]) / extent
                return int(x * (VIEW - 1)), int(y * (VIEW - 1))

            for body, parent in enumerate(robot.body_parent):
                if body > 0 and parent > 0:
                    cv2.line(
                        tile,
                        to_pixel(origins[t, parent]),
                        to_pixel(origins[t, body]),
                        GREY,
                        1,
                        cv2.LINE_AA,
                    )
            for body, label in labelled.items():
                point = to_pixel(origins[t, body])
                cv2.circle(tile, point, 2, YELLOW if label.endswith("*") else WHITE, -1)
                cv2.putText(
                    tile,
                    label,
                    (point[0] + 3, point[1] - 3),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.7,
                    WHITE,
                    1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                tile, f"{name} t={t}", (4, 12), cv2.FONT_HERSHEY_PLAIN, 0.8, YELLOW, 1, cv2.LINE_AA
            )
            tiles.append(tile)
        height = max(t.shape[0] for t in tiles)
        tiles = [np.pad(t, ((0, height - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
        rows.append(np.concatenate(tiles, 1))
    return np.concatenate(rows, 0)


def _camera_tile(frame: Tensor[[*RGB, int, int]] | None, name: str) -> np.ndarray:
    tile = np.zeros((TILE, TILE, 3), np.uint8)
    if frame is not None:
        image = frame.permute(1, 2, 0).cpu().numpy()
        h, w = image.shape[:2]
        scale = TILE / max(h, w)
        resized = cv2.resize(
            image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
        )
        tile[: resized.shape[0], : resized.shape[1]] = resized
    cv2.putText(tile, name, (4, TILE - 6), cv2.FONT_HERSHEY_PLAIN, 0.8, YELLOW, 1, cv2.LINE_AA)
    return tile


def _labelled_bodies(robot: Robot[int, int, int], episode: Episode[int, int]) -> dict[int, str]:
    """Bodies worth a label, by priority: declared camera links (marked *), leaves, branches."""

    children: dict[int, int] = {}
    for body, parent in enumerate(robot.body_parent):
        if body:
            children[parent] = children.get(parent, 0) + 1
    labels = {
        robot.body(c.mount): f"{c.mount}*" for c in episode.cameras if c.mount in robot.body_names
    }
    for count in (0, 2):  # leaves, then branch points (2 or more children)
        for body, name in enumerate(robot.body_names):
            if body and (children.get(body, 0) == 0 if count == 0 else children.get(body, 0) >= 2):
                labels.setdefault(body, name)
    return labels
