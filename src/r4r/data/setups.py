"""Setups (stations) of a whole dataset from two thumbnails per episode, on the CPU.

A camera that is not on the robot keeps its view while it stays in place; re-placing it (a new
session, another table) changes the whole background. Per episode, a small grayscale thumbnail
of its first frame and one of a later frame (`LATER` of the way in) give:

- the scene: gradient magnitudes `g = ‖∇I‖` of the first thumbnail (insensitive to gain);
- a still mask: pixels whose thumbnail barely changes between the two moments. The robot, the
  operator and moved objects change; the masked comparison is about the background only (the
  robot's projected region, wherever the camera is, without knowing where it is).

Two episodes `i`, `j` differ by `d_ij = 1 − q₇₅(ncc_k(g_i, g_j))`: normalised
cross-correlations over the pixels still in both, per block `k` of a grid, of which the upper
quartile is kept (a re-placed camera changes every block, an object moved between episodes a
few). A break before episode `i` scores

    s_i = median_{a ∈ [i−w, i), b ∈ [i, i+w)} d_ab − max(median within the left window, median
          within the right one)

and is kept where `s_i > threshold` and `s_i` is the window's maximum: one odd episode
(mismatched streams, an occluded view) changes few of the `w²` cross pairs and leaves the median.
Each run between breaks is one setup.
"""

from __future__ import annotations

import bisect
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

# Thumbnail size (width, height); the later moment, as a fraction of the episode; blocks
# compared separately (rows, columns)
SIZE = (96, 72)
BLOCKS = (3, 4)
LATER = 0.6
# A pixel is still when its standardised thumbnail (zero mean, unit std: auto-exposure and
# lighting drift cancel) changes by less than this
STILL = 0.35


@dataclass(frozen=True, slots=True, kw_only=True)
class SetupConfig:
    window: int = 4
    threshold: float = 0.15
    # Fewer still pixels shared than this fraction: the whole thumbnails are compared
    min_overlap: float = 0.1
    # An episode this much farther from its neighbours than its run's typical pair is odd
    odd: float = 0.4


@dataclass(frozen=True, slots=True)
class Thumbnails:
    """Per episode: the first frame's gradient magnitude `[E, h, w]` and its still mask."""

    episodes: np.ndarray
    gradient: np.ndarray
    still: np.ndarray


def thumbnails(root: Path, episodes: Sequence[int], camera: str) -> Thumbnails:
    """Two decoded frames of `camera` per episode (its first and one `LATER` in), as gradient
    thumbnails and still masks (module doc). Episodes sharing a video file share one opening."""

    from .lerobot import _root, seek_frames, video_key

    tables = _root(root).tables
    key = video_key(tables.video_keys, camera)
    by_file: dict[Path, list[tuple[int, float, float]]] = {}
    for e in episodes:
        files = tables.files(e)
        path = files.videos[key][0]
        start, later = files.times(key, [0.0, LATER * (tables.length(e) - 1) / tables.fps])
        by_file.setdefault(path, []).append((e, start, later))
    gradient, still = {}, {}
    for path, rows in by_file.items():
        times = sorted({t for _, a, b in rows for t in (a, b)})
        frames = seek_frames(path, times, 0.5 / tables.fps, gap=5.0)
        at = {t: i for i, t in enumerate(times)}
        for e, a, b in rows:
            first, last = (_gray(frames[at[t]]) for t in (a, b))
            gradient[e] = _gradient(first)
            change = cv2.GaussianBlur(_standard(first) - _standard(last), (5, 5), 0)
            still[e] = np.abs(change) < STILL
    order = np.asarray(list(episodes))
    return Thumbnails(
        order,
        np.stack([gradient[e] for e in order]),
        np.stack([still[e] for e in order]),
    )


def _gray(frame: object) -> np.ndarray:
    """A uint8 RGB frame `[3, H, W]` as a float grayscale thumbnail of `SIZE`."""

    image = np.ascontiguousarray(np.asarray(frame).transpose(1, 2, 0))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


def _standard(gray: np.ndarray) -> np.ndarray:
    return (gray - gray.mean()) / max(float(gray.std()), 1e-6)


def _gradient(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return np.hypot(gx, gy)


def distance(a: int, b: int, found: Thumbnails, config: SetupConfig) -> float:
    """`d_ab = 1 − q_75(ncc_k(g_a, g_b))` over the blocks `k` of a `BLOCKS` grid, each over its
    pixels still in both episodes (all its pixels when too few are). Re-placing the camera
    changes every block; an object moved between episodes changes a few, which the upper
    quantile ignores."""

    rows, columns = BLOCKS
    height, width = found.gradient.shape[1:]
    similar = []
    for i in range(rows):
        for j in range(columns):
            window = (
                slice(i * height // rows, (i + 1) * height // rows),
                slice(j * width // columns, (j + 1) * width // columns),
            )
            mask = found.still[a][window] & found.still[b][window]
            if mask.mean() < config.min_overlap:
                mask = np.ones_like(mask)
            x, y = found.gradient[a][window][mask], found.gradient[b][window][mask]
            x, y = x - x.mean(), y - y.mean()
            scale = float(np.sqrt((x * x).sum() * (y * y).sum()))
            if scale > 0:
                similar.append(float((x * y).sum()) / scale)
    return 1.0 - float(np.quantile(similar, 0.75)) if similar else 0.0


def breaks(found: Thumbnails, config: SetupConfig) -> tuple[list[int], list[int]]:
    """Indices `i` (into `found.episodes`) where a new setup starts, and those of odd episodes
    (module doc): an episode farther than `config.odd` from its run's typical pair,
    `median_j d_ij − median_{a,b} d_ab` over its neighbours `j` within `window` in its run, is a
    setup of its own (a one-off view between two episodes of a run)."""

    count, w = len(found.episodes), config.window
    cache: dict[tuple[int, int], float] = {}

    def d(a: int, b: int) -> float:
        key = (min(a, b), max(a, b))
        if key not in cache:
            cache[key] = distance(*key, found, config)
        return cache[key]

    def within(first: int, last: int) -> float:
        pairs = [d(a, b) for a in range(first, last) for b in range(a + 1, last)]
        return float(np.median(pairs)) if pairs else 0.0

    score = np.zeros(count)
    for i in range(1, count):
        left, right = range(max(0, i - w), i), range(i, min(count, i + w))
        cross = float(np.median([d(a, b) for a in left for b in right]))
        score[i] = cross - max(within(left.start, left.stop), within(right.start, right.stop))
    cuts = [
        i
        for i in range(1, count)
        if score[i] > config.threshold and score[i] == score[max(1, i - w) : i + w].max()
    ]
    odd = []
    for a, b in zip([0, *cuts], [*cuts, count], strict=True):
        typical = within(a, b) if b - a <= 2 * w else within(a, a + 2 * w)
        for i in range(a, b):
            near = [d(i, j) for j in range(max(a, i - w), min(b, i + w + 1)) if j != i]
            if len(near) >= 2 and float(np.median(near)) - typical > config.odd:
                odd.append(i)
    return cuts, odd


def setups(found: Thumbnails, config: SetupConfig) -> list[tuple[int, int]]:
    """Runs of episodes with one setup: (first, last) episode numbers, inclusive; an odd
    episode is a run of its own and splits its run."""

    cuts, odd = breaks(found, config)
    edges = sorted({0, *cuts, *odd, *(i + 1 for i in odd), len(found.episodes)})
    return [
        (int(found.episodes[a]), int(found.episodes[b - 1])) for a, b in itertools.pairwise(edges)
    ]


def labelled(
    runs: Sequence[tuple[int, int]],
    known: dict[int, str],
    episodes: Iterable[int] | None = None,
    prefix: str = "",
) -> tuple[dict[int, str], list[int]]:
    """A station per episode of `runs`: each label of `known` (labelled episodes) goes to the
    run holding most of its episodes, the other runs are `<prefix>auto-<first episode>`; and
    the labelled episodes whose run says otherwise. With `episodes` (the thumbnailed ones, e.g.
    one priors group's) a run spans only those of them between its first and last episode."""

    def members(first: int, last: int) -> Iterable[int]:
        if chosen is None:
            return range(first, last + 1)
        return chosen[bisect.bisect_left(chosen, first) : bisect.bisect_right(chosen, last)]

    chosen = None if episodes is None else sorted(episodes)
    count: dict[str, list[int]] = {}
    for i, (first, last) in enumerate(runs):
        for e in members(first, last):
            if e in known:
                count.setdefault(known[e], [0] * len(runs))[i] += 1
    owner = {max(range(len(runs)), key=c.__getitem__): label for label, c in count.items()}
    stations, relabelled = {}, []
    for i, (first, last) in enumerate(runs):
        name = owner.get(i, f"{prefix}auto-{first}")
        for e in members(first, last):
            stations[e] = name
            if e in known and known[e] != name:
                relabelled.append(e)
    return stations, relabelled


def write_stations(
    path: Path,
    stations: dict[int, str],
    relabelled: Sequence[int],
    camera: str | None = None,
) -> None:
    """Set the `station` of the episodes table at `path` (created if absent), or with `camera`
    that camera's own setup key `station.<camera>`: rows of known episodes get their station,
    other episodes one row with only it; the recorded poses of `relabelled` episodes (a
    station's values given to an episode of another setup) are cleared (only `camera`'s, with
    `camera`)."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist() if path.exists() else []
    pose = ("x", "y", "z", "qw", "qx", "qy", "qz")
    column = "station" if camera is None else f"station.{camera}"
    seen = set()
    for row in rows:
        episode = int(row["episode"])
        seen.add(episode)
        if episode in stations:
            row[column] = stations[episode]
        if episode in relabelled and (camera is None or row.get("camera") == camera):
            row.update(dict.fromkeys(pose))
    rows += [
        {"episode": e, "camera": None, column: s} for e, s in stations.items() if e not in seen
    ]
    columns = list(dict.fromkeys(k for row in rows for k in row))
    table = pa.Table.from_pylist([{k: row.get(k) for k in columns} for row in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.sort_by("episode"), path)
