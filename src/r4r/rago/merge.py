"""Join tracks across cameras: links between two views' pixels make their tracks one.

Each camera has its own tracks (KLT along its video). A link — a correspondence between a
pixel in one view and a pixel in another (RoMa v2 between two cameras' frames) — joins the
tracks whose observations lie within `radius` of its two ends; joined tracks become one
(connected components). An end with no observation nearby becomes a new observation, so a
link always connects something: two unknown ends make a new two-view track.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    from .tracks import Tracks

RADIUS = 1.0  # pixels


def merge_tracks[K: IntVar](
    tracks: Tracks,
    view_a: Tensor[[K]],
    pixel_a: Tensor[[K, 2]],
    view_b: Tensor[[K]],
    pixel_b: Tensor[[K, 2]],
    radius: float = RADIUS,
) -> Tracks:
    """`tracks` with every link's two ends in one track."""

    track, view, pixel = tracks.track, tracks.view, tracks.pixel.to(pixel_a)
    track_a, added_a = _attach(view, pixel, track, view_a, pixel_a, radius)
    next_id = int(track.max()) + 1 if len(track) else 0
    # Unattached ends: new observations, each its own new track for now
    new_a = torch.where(added_a, next_id + torch.arange(len(view_a), device=track.device), track_a)
    next_id += len(view_a)
    track_b, added_b = _attach(view, pixel, track, view_b, pixel_b, radius)
    new_b = torch.where(added_b, next_id + torch.arange(len(view_b), device=track.device), track_b)
    next_id += len(view_b)

    label = _components(next_id, new_a, new_b)
    keep_a, keep_b = added_a, added_b
    merged_track = torch.cat((track, new_a[keep_a], new_b[keep_b]))
    _, dense = torch.unique(label[merged_track], return_inverse=True)
    count = int(keep_a.sum() + keep_b.sum())
    return replace(
        tracks,
        track=dense,
        view=torch.cat((view, view_a[keep_a], view_b[keep_b])),
        pixel=torch.cat(
            (tracks.pixel, pixel_a[keep_a].to(tracks.pixel), pixel_b[keep_b].to(tracks.pixel))
        ),
        weight=torch.cat((tracks.weight, tracks.weight.new_ones(count))),
        moving=torch.cat((tracks.moving, tracks.moving.new_ones(count))),
    )


def _attach[K: IntVar](
    view: Tensor[[int]],
    pixel: Tensor[[int, 2]],
    track: Tensor[[int]],
    query_view: Tensor[[K]],
    query_pixel: Tensor[[K, 2]],
    radius: float,
) -> tuple[Tensor[[K]], Tensor[[K]]]:
    """Track of the nearest observation in the same view within `radius` (−1 and True when
    none: the end is to be added)."""

    found = torch.full_like(query_view, -1)
    for v in torch.unique(query_view).tolist():
        mine = (view == v).nonzero()[:, 0]
        asked = (query_view == v).nonzero()[:, 0]
        if not len(mine):
            continue
        distance, nearest = torch.cdist(query_pixel[asked], pixel[mine]).min(1)
        found[asked] = torch.where(distance <= radius, track[mine[nearest]], -1)
    return found, found < 0


def _components[K: IntVar](count: int, a: Tensor[[K]], b: Tensor[[K]]) -> Tensor[[int]]:
    """Connected-component label (smallest id) of `count` nodes joined by edges a–b."""

    label = torch.arange(count, device=a.device)
    while True:
        low = torch.minimum(label[a], label[b])
        updated = label.scatter_reduce(0, a, low, "amin").scatter_reduce(0, b, low, "amin")
        updated = updated[updated]  # pointer jumping
        if torch.equal(updated, label):
            return label
        label = updated
