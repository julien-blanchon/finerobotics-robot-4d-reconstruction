"""Merge `r4r reconstruct` shards into a LeRobot v3 dataset (docs/output-spec.md, packaging).

- **twin** (default): `<dest>/<dataset>--r4r`, a geometry-twin-style root next to the main one:
  the index columns (`episode_index, frame_index, index, timestamp, task_index`) and r4r's
  columns, `meta/episodes` linking each episode back with `source/episode_index`. Videos are
  not copied: `videos` is a link to the main root's, and each episode keeps its video
  segments, so the twin opens with LeRobot's loader, images included.
- **full**: `<dest>/<dataset>`, the main dataset's rows with r4r's columns added (videos linked
  likewise): everything in one dataset.

Only reconstructed episodes are written (renumbered from 0, in order). Files stay few: one data
file per `rows_per_file` frames (a row group per episode), one `meta/episodes` file, and
`meta/r4r.json` (quality and provenance; the per-episode part in
`meta/r4r.episodes.parquet` past `JSON_EPISODES` episodes).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from r4r.output import LAYOUT_KEY
from r4r.reconstruction import EPISODES_KEY

if TYPE_CHECKING:
    from collections.abc import Iterator

    from r4r.data.priors import JsonObject

INDEX_COLUMNS = ("episode_index", "frame_index", "index", "timestamp", "task_index")
CHUNKS_SIZE = 1000
# Past this many episodes the per-episode records go to a parquet file next to `r4r.json`
JSON_EPISODES = 20_000
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
# Rows sampled for a feature's statistics (evenly spread)
STATS_ROWS = 1_000_000


@dataclass(frozen=True, slots=True)
class MergeConfig:
    full: bool = False
    rows_per_file: int = 2_000_000


def merge(root: Path, reconstruction: Path, dest: Path, config: MergeConfig) -> Path:
    """Write the merged dataset of `root` (a LeRobot v3 root) from the shards under
    `reconstruction` (the `r4r reconstruct` output: `<family>/<dataset>/` or that directory
    itself) into `dest`; returns the new root (module doc)."""

    shards = _shards(root, reconstruction)
    layout, records = _summaries(shards)
    done = sorted(int(r["episode"]) for r in records.values() if "error" not in r)
    if not done:
        raise ValueError(f"{reconstruction}: no reconstructed episode")
    info = json.loads((root / "meta" / "info.json").read_text())
    target = dest / (root.name if config.full else f"{root.name}--r4r")
    if target.exists():
        raise FileExistsError(f"{target}: exists")
    partial = target.with_name(f".{target.name}.{os.getpid()}.partial")
    (partial / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    main_episodes = _main_episodes(root, done, config.full)
    episodes, sample = _write_data(root, info, shards, done, main_episodes, partial, config)
    pq.write_table(episodes, partial / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    features = _features(info, layout, config.full)
    _write_info(
        partial, info, features, layout, len(done), int(sum(episodes["length"].to_pylist()))
    )
    _write_stats(partial, root, features, sample)
    shutil.copyfile(root / "meta" / "tasks.parquet", partial / "meta" / "tasks.parquet")
    for name in ("robot.json", "provenance.json"):
        if (root / "meta" / name).exists():
            shutil.copyfile(root / "meta" / name, partial / "meta" / name)
    if (root / "meta" / "robot").exists():
        (partial / "meta" / "robot").symlink_to((root / "meta" / "robot").resolve())
    if (root / "videos").exists():
        (partial / "videos").symlink_to((root / "videos").resolve())
    _write_side_file(partial, root, layout, records, done, _checks(shards[0].parent))
    partial.rename(target)
    return target


def _shards(root: Path, reconstruction: Path) -> list[Path]:
    nested = reconstruction / root.parent.parent.name / root.name
    found = sorted((nested if nested.is_dir() else reconstruction).glob("shard-*.parquet"))
    if not found:
        raise FileNotFoundError(f"{reconstruction}: no shard-*.parquet for {root.name}")
    return found


def _checks(directory: Path) -> dict[int, JsonObject]:
    """Per source episode, the `r4r check` results in `directory/checks.parquet` (none when
    the checks were not run): the mean sync ρ and its flag, and per camera its sync ρ and
    estimated pixel error (`r4r.checks.episode`)."""

    from r4r.checks.episode import CheckConfig, episode_sync

    path = directory / "checks.parquet"
    if not path.exists():
        return {}
    rows: dict[int, list[dict]] = {}
    for row in pq.read_table(path).to_pylist():
        rows.setdefault(int(row["episode"]), []).append(row)

    def value(x: float) -> float | None:
        return None if not np.isfinite(x) else round(float(x), 3)

    out: dict[int, JsonObject] = {}
    for episode, found in rows.items():
        rho, flagged = episode_sync(found, CheckConfig())
        cameras: dict[str, dict[str, float | None]] = {}
        for r in found:
            if r["method"] == "sync":
                cameras.setdefault(r["camera"], {})["sync_rho"] = value(1.0 - r["score"])
            elif r["method"] == "camera":
                cameras.setdefault(r["camera"], {})["error_px"] = value(r["estimate_px"])
        out[episode] = {"sync_rho": value(rho), "sync_flagged": flagged, "cameras": cameras}
    return out


def _summaries(shards: list[Path]) -> tuple[JsonObject, dict[int, JsonObject]]:
    """The dataset's layout and each episode's record (the last shard holding it wins)."""

    layout, records = None, {}
    for path in shards:
        metadata = pq.read_schema(path).metadata or {}
        if LAYOUT_KEY in metadata:
            layout = json.loads(metadata[LAYOUT_KEY])
        for record in json.loads(metadata.get(EPISODES_KEY, b"[]")):
            records[int(record["episode"])] = record
    if layout is None:
        raise ValueError("no shard carries a layout (every episode failed?)")
    return layout, records


def _main_episodes(root: Path, episodes: list[int], full: bool) -> pa.Table:
    """The main dataset's `meta/episodes` rows of `episodes` (index and video columns; every
    column for a full copy), sorted by episode."""

    wanted = np.asarray(episodes, dtype=np.int64)
    parts = []
    for path in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
        names = pq.read_schema(path).names
        keep = names if full else [n for n in names if _twin_column(n)]
        span = pq.read_table(path, columns=["episode_index"])["episode_index"].to_numpy()
        inside = np.isin(span, wanted)
        if not inside.any():
            continue
        parts.append(pq.read_table(path, columns=keep).filter(pa.array(inside)))
    table = pa.concat_tables(parts, promote_options="permissive").sort_by("episode_index")
    if table.num_rows != len(episodes):
        raise ValueError(f"{root}: {len(episodes) - table.num_rows} episodes not in meta/episodes")
    return table


def _twin_column(name: str) -> bool:
    return name in {"episode_index", "length", "tasks", "data/chunk_index", "data/file_index"} or (
        name.startswith("videos/")
    )


def _main_rows(root: Path, info: dict, row: dict, columns: list[str] | None) -> pa.Table:
    """One episode's rows of the main dataset's data file (`columns`; None: all)."""

    path = root / info["data_path"].format(
        chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]
    )
    handle = _open(path)
    groups = _groups(handle, row["episode_index"])
    names = None if columns is None else ["episode_index", *columns]
    table = handle.read_row_groups(groups, columns=names)
    table = table.filter(pa.array(table["episode_index"].to_numpy() == row["episode_index"]))
    return table if columns is None else table.select(columns)


_OPEN: dict[Path, tuple[pq.ParquetFile, np.ndarray]] = {}


def _open(path: Path) -> pq.ParquetFile:
    if path not in _OPEN:
        _OPEN.clear()  # one data file at a time: episodes are read in order
        handle = pq.ParquetFile(path)
        _OPEN[path] = (handle, _spans(handle))
    return _OPEN[path][0]


def _groups(handle: pq.ParquetFile, episode: int) -> list[int]:
    spans = next(s for h, s in _OPEN.values() if h is handle)
    return np.flatnonzero((spans[:, 0] <= episode) & (spans[:, 1] >= episode)).tolist()


def _spans(handle: pq.ParquetFile) -> np.ndarray:
    from r4r.data.tables import _row_group_episodes

    return _row_group_episodes(handle)


def _r4r_rows(shards: list[Path]) -> Iterator[tuple[int, pa.Table]]:
    """Each episode's r4r rows, shard by shard, episodes in order."""

    for path in shards:
        table = pq.read_table(path)
        if not table.num_rows:
            continue
        table = table.sort_by([("episode_index", "ascending"), ("frame_index", "ascending")])
        episode = table["episode_index"].to_numpy()
        cuts = np.flatnonzero(np.diff(episode)) + 1
        for start, stop in zip([0, *cuts.tolist()], [*cuts.tolist(), len(episode)], strict=True):
            yield int(episode[start]), table.slice(start, stop - start)


def _write_data(
    root: Path,
    info: dict,
    shards: list[Path],
    done: list[int],
    main: pa.Table,
    target: Path,
    config: MergeConfig,
) -> tuple[pa.Table, pa.Table]:
    """Data files of the merged dataset and its `meta/episodes` table; also a sample of the
    rows for the statistics."""

    main_rows = {r["episode_index"]: r for r in main.to_pylist()}
    wanted = set(done)
    writer, rows_in_file = None, 0
    chunk = file = 0
    total = 0
    meta: list[dict[str, object]] = []
    samples: list[pa.Table] = []
    every = max(1, sum(main["length"].to_pylist()) // STATS_ROWS)
    columns = None if config.full else ["frame_index", "timestamp", "task_index"]
    new = 0
    for source, rows in _r4r_rows(shards):
        if source not in wanted:
            continue
        if meta and source <= int(str(meta[-1]["source/episode_index"])):
            raise ValueError(f"episode {source}: shards are not in episode order")
        row = main_rows[source]
        base = _main_rows(root, info, row, columns)
        length = base.num_rows
        frames = rows["frame_index"].to_numpy()
        if len(frames) != length or not np.array_equal(frames, np.arange(length)):
            raise ValueError(
                f"episode {source}: frames {len(frames)} of {length} (reconstruct with --stride 1)"
            )
        ours = rows.drop_columns(["episode_index", "frame_index", "timestamp"])
        index = pa.array(np.arange(total, total + length, dtype=np.int64))
        base = _set(base, "episode_index", pa.array(np.full(length, new, dtype=np.int64)))
        base = _set(base, "index", index)
        table = base
        for name in ours.column_names:
            table = _set(table, name, ours[name])
        if not config.full:
            table = table.select([*INDEX_COLUMNS, *ours.column_names])
        if writer is None or rows_in_file >= config.rows_per_file:
            if writer is not None:
                writer.close()
                file += 1
                if file == CHUNKS_SIZE:
                    chunk, file = chunk + 1, 0
            path = target / info["data_path"].format(chunk_index=chunk, file_index=file)
            path.parent.mkdir(parents=True, exist_ok=True)
            writer, rows_in_file = pq.ParquetWriter(path, table.schema), 0
        writer.write_table(table, row_group_size=max(length, 1))
        if new % every == 0:
            samples.append(table.select(ours.column_names))
        entry = {k: v for k, v in row.items() if not k.startswith("stats/")} if config.full else row
        meta.append(
            entry
            | {
                "episode_index": new,
                "length": length,
                "dataset_from_index": total,
                "dataset_to_index": total + length,
                "data/chunk_index": chunk,
                "data/file_index": file,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                "source/episode_index": source,
            }
        )
        rows_in_file += length
        total += length
        new += 1
    assert writer is not None
    writer.close()
    return pa.Table.from_pylist(meta), pa.concat_tables(samples)


def _set(table: pa.Table, name: str, values: pa.Array | pa.ChunkedArray) -> pa.Table:
    if name in table.column_names:
        return table.set_column(table.column_names.index(name), name, values)
    return table.append_column(name, values)


def _features(info: dict, layout: dict, full: bool) -> dict[str, dict]:
    """The merged dataset's features: the index and video features (every main feature for a
    full copy), the video features' info with r4r's camera facts, then r4r's columns."""

    main = info["features"]
    keep = (
        main
        if full
        else {k: v for k, v in main.items() if k in INDEX_COLUMNS or v["dtype"] == "video"}
    )
    features = {k: dict(v) for k, v in keep.items()}
    from r4r.data.lerobot import camera_name

    extras = {camera_name(k): v for k, v in layout["cameras"].items()}
    for key, feature in features.items():
        if feature.get("dtype") == "video" and camera_name(key) in extras:
            feature["info"] = {**feature.get("info", {}), **extras[camera_name(key)]}
    return features | layout["features"]


def _write_info(
    target: Path, info: dict, features: dict, layout: dict, episodes: int, frames: int
) -> None:
    out = {k: v for k, v in info.items() if k not in {"features", "tables"}}
    out |= {
        "total_episodes": episodes,
        "total_frames": frames,
        "splits": {"train": f"0:{episodes}"},
        "chunks_size": CHUNKS_SIZE,
        "features": features,
        "robot": {"mobile": layout["robot"]["mobile"]}
        | ({"base_pose": False} if layout["robot"].get("base_pose") is False else {}),
    }
    (target / "meta" / "info.json").write_text(json.dumps(out, indent=2) + "\n")


def _write_stats(target: Path, root: Path, features: dict, sample: pa.Table) -> None:
    """`meta/stats.json`: r4r's columns from the sampled rows; the main dataset's entries for
    the others (videos, a full copy's columns)."""

    main = root / "meta" / "stats.json"
    stats = json.loads(main.read_text()) if main.exists() else {}
    stats = {k: v for k, v in stats.items() if k in features}
    for name in sample.column_names:
        values = _numbers(sample[name])
        stats[name] = {
            "min": values.min(0).tolist(),
            "max": values.max(0).tolist(),
            "mean": values.mean(0).tolist(),
            "std": values.std(0).tolist(),
            "count": [len(values)],
            **{
                f"q{round(q * 100):02d}": np.quantile(values, q, axis=0).tolist() for q in QUANTILES
            },
        }
    (target / "meta" / "stats.json").write_text(json.dumps(stats) + "\n")


def _numbers(column: pa.ChunkedArray) -> np.ndarray:
    """A fixed-size list column as float64 [N, ...]."""

    flat = column.combine_chunks()
    shape = [len(flat)] if pa.types.is_fixed_size_list(flat.type) else [len(flat), 1]
    while pa.types.is_fixed_size_list(flat.type):
        shape.append(flat.type.list_size)
        flat = flat.flatten()
    return np.asarray(flat.to_numpy(zero_copy_only=False), dtype=np.float64).reshape(shape)


def _write_side_file(
    target: Path,
    root: Path,
    layout: dict,
    records: dict[int, JsonObject],
    done: list[int],
    checks: dict[int, JsonObject],
) -> None:
    """`meta/r4r.json`: per dataset, provenance (r4r commit, priors hash, date), the state
    mapping and the action's treatment; per episode, quality (route, median px, track share,
    fit check)."""

    from r4r.data.priors import episode_table_path, priors_path

    def quality(new: int, source: int) -> dict[str, object]:
        r = records[source]
        px = next(
            (r[k] for k in ("median_px", "light_px", "check_after_px", "check_px") if k in r), None
        )
        return {
            "episode_index": new,
            "source_episode_index": source,
            "route": r.get("route"),
            "median_px": px,
            "track_share": r.get("inliers"),
            "flagged": bool(r.get("flagged", False)),
            "cameras": r.get("cameras"),
            "checks": checks.get(source),
        }

    episodes = [quality(n, s) for n, s in enumerate(done)]
    hashed = hashlib.sha256()
    for path in (priors_path(root), episode_table_path(root)):
        if path.exists():
            hashed.update(path.read_bytes())
    side = {
        "r4r_commit": _commit(),
        "priors_sha256": hashed.hexdigest(),
        "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "source": str(root),
        "failed_episodes": sorted(e for e, r in records.items() if "error" in r),
        "state_mapping": layout["state_mapping"],
        "lag_frames": layout.get("lag_frames", {}),
        "action": "action_aligned: the action through the state's joint map (no lag correction)"
        if layout["action_aligned"]
        else "untouched: not an absolute joint target in the state's columns",
    }
    if len(episodes) > JSON_EPISODES:
        pq.write_table(pa.Table.from_pylist(episodes), target / "meta" / "r4r.episodes.parquet")
        side["episodes"] = "r4r.episodes.parquet"
    else:
        side["episodes"] = episodes
    (target / "meta" / "r4r.json").write_text(json.dumps(side, indent=1) + "\n")


def _commit() -> str | None:
    """The r4r checkout's commit (with `+dirty` when it has changes), if known."""

    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit + ("+dirty" if dirty else "")
