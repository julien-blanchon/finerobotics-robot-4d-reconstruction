"""Read a LeRobot v3 root's tables with pyarrow: only the rows and columns an episode needs.

LeRobot's own loader converts every `meta/episodes` file (with its per-episode statistics) and
filters every data file into an Arrow cache for each new set of episodes: minutes and
gigabytes per call on a dataset of millions of episodes. Here an episode costs its row of
`meta/episodes` (the index columns of its file, read once per file) and the row groups of its
data file that hold it (found from the row groups' `episode_index` statistics). Both are kept
for the next episode, so consecutive episodes of a process share the reads.

A side table (`SideTable`, the priors' `side_table`) adds features the delivery dropped, kept
outside the root: a directory of parquet files with `episode_index`, `frame_index` and the
features' columns, one row per frame and a row group per episode (e.g. joint states recovered
from a simulator's upstream records). `LeRobotTables.rows` reads a side feature like any other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from einops import rearrange

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from .priors import JsonObject

# A column of an episode's rows: a tensor with one row per frame, or text (e.g. JSON records)
type Column = Tensor | list[str]

# Columns of `meta/episodes` that locate an episode's rows and video segments
INDEX_COLUMNS = ("episode_index", "length", "data/chunk_index", "data/file_index")
VIDEO_COLUMNS = ("chunk_index", "file_index", "from_timestamp")


@dataclass(frozen=True, slots=True)
class EpisodeFiles:
    """Where an episode lives: its data file and, per video key, its video file and the
    episode's start in it (seconds)."""

    data: Path
    videos: dict[str, tuple[Path, float]]


class LeRobotTables:
    """A LeRobot v3 root's `meta/info.json`, `meta/episodes` and `data` tables, read lazily."""

    def __init__(self, root: Path, side: Path | None = None) -> None:
        self.root = root
        self.info: JsonObject = json.loads((root / "meta" / "info.json").read_text())
        self.side = None if side is None else SideTable(side)
        self.features: JsonObject = self.info["features"] | (
            {} if self.side is None else self.side.features
        )
        self.fps = float(self.info["fps"])
        self.video_keys = tuple(k for k, f in self.features.items() if f["dtype"] == "video")
        self.total_episodes = int(self.info["total_episodes"])
        # The `meta/episodes` files (in episode order) and the (first, last) episode of those
        # whose footer was read
        self._episode_files: list[Path] | None = None
        self._spans: dict[int, tuple[int, int]] = {}
        # The index columns of the last `meta/episodes` file read
        self._episode_rows: tuple[Path, dict[str, np.ndarray]] | None = None
        # The last data file opened, with each row group's (first, last) episode
        self._data: tuple[Path, pq.ParquetFile, np.ndarray] | None = None

    def files(self, episode: int) -> EpisodeFiles:
        """The data and video files of `episode`, from its row of `meta/episodes`."""

        row = self._episode_row(episode)
        data = self.info["data_path"].format(
            chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])
        )
        videos = {}
        for key in self.video_keys:
            path = self.info["video_path"].format(
                video_key=key,
                chunk_index=int(row[f"videos/{key}/chunk_index"]),
                file_index=int(row[f"videos/{key}/file_index"]),
            )
            videos[key] = (self.root / path, float(row[f"videos/{key}/from_timestamp"]))
        return EpisodeFiles(self.root / data, videos)

    def rows(self, episode: int, columns: Sequence[str]) -> dict[str, Column]:
        """`columns` of `episode`'s frames, in file order: numbers as tensors `[T, ...]` of the
        stored dtype (vectors and matrices flattened to `[T, n]`), text as lists."""

        side = [] if self.side is None else [c for c in columns if c in self.side.features]
        path = self.files(episode).data
        if self._data is None or self._data[0] != path:
            handle = pq.ParquetFile(path)
            self._data = (path, handle, _row_group_episodes(handle))
        _, handle, spans = self._data
        groups = np.flatnonzero((spans[:, 0] <= episode) & (spans[:, 1] >= episode)).tolist()
        wanted = list(dict.fromkeys(["episode_index", *(c for c in columns if c not in side)]))
        table = handle.read_row_groups(groups, columns=wanted)
        table = table.filter(pa.array(table["episode_index"].to_numpy() == episode))
        out = {name: _column(table[name]) for name in wanted[1:]}
        if side:
            assert self.side is not None
            extra = self.side.rows(episode, side)
            if len(extra[side[0]]) != table.num_rows:
                raise ValueError(
                    f"{self.side.directory}: episode {episode} has {len(extra[side[0]])} rows, "
                    f"the dataset {table.num_rows}"
                )
            out |= extra
        return {name: out[name] for name in columns}

    def length(self, episode: int) -> int:
        """How many frames `episode` has (its row of `meta/episodes`)."""

        return int(self._episode_row(episode)["length"])

    def simulated(self) -> bool | None:
        """Whether the dataset is simulated (`source/is_simulation` of its first episode in
        `meta/episodes`; None when the column is absent)."""

        path = min((self.root / "meta" / "episodes").glob("*/*.parquet"))
        if "source/is_simulation" not in pq.read_schema(path).names:
            return None
        value = pq.read_table(path, columns=["source/is_simulation"])[0][0].as_py()
        return None if value is None else bool(value)

    def episode_statistic(self, stat: str) -> tuple[np.ndarray, np.ndarray]:
        """Every episode's `stats/<stat>` of `meta/episodes` (e.g. `observation.state/q99`):
        the episode indices `[E]` and the values `[E, n]` (float64), in episode order."""

        episodes, values = [], []
        for path in sorted((self.root / "meta" / "episodes").glob("*/*.parquet")):
            table = pq.read_table(path, columns=["episode_index", f"stats/{stat}"])
            episodes.append(table["episode_index"].to_numpy())
            column = table[f"stats/{stat}"].to_pylist()
            values.append(np.asarray(column, dtype=np.float64).reshape(len(column), -1))
        order = np.concatenate(episodes)
        rows = np.concatenate(values)
        sort = np.argsort(order, kind="stable")
        return order[sort], rows[sort]

    def _episode_row(self, episode: int) -> dict[str, float]:
        path = self._episode_file(episode)
        if self._episode_rows is None or self._episode_rows[0] != path:
            names = [
                *INDEX_COLUMNS,
                *(f"videos/{k}/{c}" for k in self.video_keys for c in VIDEO_COLUMNS),
            ]
            table = pq.read_table(path, columns=names)
            self._episode_rows = (path, {n: table[n].to_numpy() for n in names})
        rows = self._episode_rows[1]
        i = int(np.searchsorted(rows["episode_index"], episode))
        if i == len(rows["episode_index"]) or rows["episode_index"][i] != episode:
            raise IndexError(f"{self.root}: no episode {episode} in {path}")
        return {name: values[i].item() for name, values in rows.items()}

    def _episode_file(self, episode: int) -> Path:
        """The `meta/episodes` file holding `episode`: the files hold consecutive runs of
        episodes in path order, so a binary search reads a few footers, not all."""

        if self._episode_rows is not None:  # most often the file of the previous episode
            rows = self._episode_rows[1]["episode_index"]
            if rows[0] <= episode <= rows[-1]:
                return self._episode_rows[0]
        if self._episode_files is None:
            self._episode_files = sorted((self.root / "meta" / "episodes").glob("*/*.parquet"))
        low, high = 0, len(self._episode_files) - 1
        while low <= high:
            middle = (low + high) // 2
            if middle not in self._spans:
                self._spans[middle] = _episode_span(self._episode_files[middle])
            first, last = self._spans[middle]
            if episode < first:
                high = middle - 1
            elif episode > last:
                low = middle + 1
            else:
                return self._episode_files[middle]
        raise IndexError(f"{self.root}: no episode {episode} in meta/episodes")


class SideTable:
    """Features kept apart from a root (see the module doc): parquet files of a directory,
    each holding a run of episodes; an episode's rows are found from the row groups'
    `episode_index` statistics, sorted by `frame_index`."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.paths = sorted(directory.glob("*.parquet"))
        self.paths = [p for p in self.paths if "frame_index" in pq.read_schema(p).names]
        if not self.paths:
            raise FileNotFoundError(f"{directory}: no side table files")
        schema = pq.read_schema(self.paths[0])
        self.features: JsonObject = {
            name: {
                "dtype": str(_leaf(schema.field(name).type)),
                "shape": [_width(schema.field(name).type)],
            }
            for name in schema.names
            if name not in ("episode_index", "frame_index")
        }
        # Each file's row groups' (first, last) episode, read when first needed
        self._spans: dict[Path, np.ndarray] = {}
        self._open: tuple[Path, pq.ParquetFile] | None = None

    def rows(self, episode: int, columns: Sequence[str]) -> dict[str, Column]:
        """`columns` of `episode`'s frames in frame order (as `LeRobotTables.rows`)."""

        for path in self.paths:
            if path not in self._spans:
                handle = pq.ParquetFile(path)
                self._spans[path] = _row_group_episodes(handle)
                self._open = (path, handle)
            spans = self._spans[path]
            groups = np.flatnonzero((spans[:, 0] <= episode) & (spans[:, 1] >= episode)).tolist()
            if not groups:
                continue
            if self._open is None or self._open[0] != path:
                self._open = (path, pq.ParquetFile(path))
            wanted = ["episode_index", "frame_index", *columns]
            table = self._open[1].read_row_groups(groups, columns=wanted)
            table = table.filter(pa.array(table["episode_index"].to_numpy() == episode))
            if not table.num_rows:  # row groups spanning the episode without holding it
                continue
            table = table.sort_by("frame_index")
            return {name: _column(table[name]) for name in columns}
        raise IndexError(f"{self.directory}: no episode {episode}")


def _leaf(kind: pa.DataType) -> pa.DataType:
    while pa.types.is_list(kind) or pa.types.is_fixed_size_list(kind):
        kind = kind.value_type
    return kind


def _width(kind: pa.DataType) -> int:
    return kind.list_size if pa.types.is_fixed_size_list(kind) else 1


def _episode_span(path: Path) -> tuple[int, int]:
    """The first and last `episode_index` of a `meta/episodes` file (footer statistics)."""

    spans = _row_group_episodes(pq.ParquetFile(path))
    return int(spans[:, 0].min()), int(spans[:, 1].max())


def _row_group_episodes(handle: pq.ParquetFile) -> np.ndarray:
    """Each row group's (first, last) `episode_index` `[groups, 2]`, from the footer's
    statistics (the column itself where a row group has none)."""

    meta = handle.metadata
    # Leaf column index (nested columns before it count one per leaf)
    column = next(
        i for i in range(meta.num_columns) if meta.schema.column(i).path == "episode_index"
    )
    spans = []
    for g in range(meta.num_row_groups):
        stats = meta.row_group(g).column(column).statistics
        if stats is not None and stats.has_min_max:
            spans.append((stats.min, stats.max))
            continue
        values = handle.read_row_group(g, columns=["episode_index"])["episode_index"].to_numpy()
        spans.append((values.min(), values.max()))
    return np.array(spans, dtype=np.int64).reshape(-1, 2)


def _column(values: pa.ChunkedArray) -> Column:
    """One column as a tensor `[T]` or `[T, n]` (lists flattened), or a list of strings."""

    kind = values.type
    if pa.types.is_string(kind) or pa.types.is_large_string(kind):
        return values.to_pylist()
    count = len(values)
    flat = values.combine_chunks() if values.num_chunks != 1 else values.chunk(0)
    nested = False
    while pa.types.is_list(flat.type) or pa.types.is_fixed_size_list(flat.type):
        flat = flat.flatten()
        nested = True
    tensor = torch.from_numpy(np.array(flat.to_numpy(zero_copy_only=False)))
    return rearrange(tensor, "(t n) -> t n", t=count) if nested else tensor
