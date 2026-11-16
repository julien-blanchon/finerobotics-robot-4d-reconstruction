"""Which path each episode takes in `r4r reconstruct`, and what a whole dataset will cost.

An episode's unknowns come from its priors (with its station's values) and what is recorded for
it. Values of dataset or station scope are held once calibrated (written by `r4r rig --write`,
or recorded); only episode-scope ones may stay free:

- **fast**: every value held or recorded: `W_b(q_t) · mount`, no video read.
- **light**: episode-scope unknowns with a start (a re-placed robot base, a re-placed static
  camera, an episode the fit check flags): a few bursts of frames, sparse tracks, LM on those
  unknowns only (`rago.light`, CPU).
- **full**: its setup was never calibrated (`setup`: a mount, pose, focal or placement of
  dataset or station scope, an uncorrected odometry), or an unknown of its own has no start
  (`missing`): the whole rig on the episode (GPU).

Held values:
- a moving camera's mount when the priors give it (dataset scope) or the episode records it;
- a static camera's pose when recorded for the episode, or given for the dataset with ranges
  (a pose without ranges is only a start);
- a focal when given or recorded; distortion and kinematic refinements at their priors' values;
- a dataset-scope body at its written correction (or with zero ranges);
- a frame-scope base at its recorded motion when the record is a pose or carries a fitted
  `base_odometry.correction`.

`dataset_plan` counts the routes over a dataset from its priors and episodes table alone (no
episode is loaded) and costs them with the per-episode timings measured on this cluster
(`COST`): a setup that is not calibrated costs one `r4r rig` on a sample, after which its
episodes take the route their own unknowns allow. A setup is an episode's common `station`, or
for a camera with its own setup key (`station.<camera>`) that camera's station: a value missing
for such a camera belongs to its own setup.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence
    from pathlib import Path

    from r4r.data.priors import DatasetPriors

type Route = Literal["fast", "light", "full"]

# Why a lone re-placed camera's setup is to be placed (picks), not rigged
LONE = "lone camera re-placed: place it per station"
# A detected setup with fewer episodes is a one-off view (odd episode), not calibrated
MIN_SETUP = 10


@dataclass(frozen=True, slots=True)
class Unknowns:
    """What an episode leaves free: `free` cameras and bodies (episode scope, with a start) for
    the light solve; `setup`: values of its setup not calibrated yet; `missing`: unknowns of
    its own with no start; `recorded`: episode-scope cameras held at their record, freed (from
    it) when the fit check flags the episode."""

    free: tuple[str, ...] = ()
    setup: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    recorded: tuple[str, ...] = ()

    @property
    def route(self) -> Route:
        if self.setup or self.missing:
            return "full"
        return "light" if self.free else "fast"

    @property
    def calibrated_route(self) -> Route:
        """The route once its setup is calibrated (one `r4r rig --write`)."""

        if self.missing:
            return "full"
        return "light" if self.free else "fast"


def episode_unknowns(
    priors: DatasetPriors,
    cameras: Sequence[str],
    moving: Collection[str],
    poses: Collection[str],
    intrinsics: Collection[str],
) -> Unknowns:
    """The unknowns of an episode with `cameras` (names), of which `moving` ride on a link,
    `poses` have a recorded pose (static: in the base frame; moving: on its link) and
    `intrinsics` recorded intrinsics, under `priors` (station applied). See the module doc."""

    states = [(n, _camera_state(priors, n, n in moving, n in poses)) for n in cameras]
    states += [(n, _body_state(priors, n)) for n in priors.bodies]
    focal = [
        f"{n}: focal unknown"
        for n in cameras
        if n in priors.cameras
        and n not in intrinsics
        and priors.cameras[n].focal is None
        and priors.cameras[n].intrinsics is None
    ]
    free = [n for n, state in states if state == "free"]
    setup = focal + [f"{n}: {s.removeprefix('setup: ')}" for n, s in states if s[:6] == "setup:"]
    missing = [f"{n}: {s.removeprefix('missing: ')}" for n, s in states if s[:8] == "missing:"]
    if len(cameras) < 2 and (free or missing):
        # Nothing to match it with, and a static camera's own tracks do not pin it (field
        # notes): placed once per setup instead (`r4r stations`, then picks)
        setup += [f"{n}: {LONE}" for n in cameras]
        free, missing = [], []
    recorded = [n for n, state in states if state == "recorded"]
    return Unknowns(tuple(free), tuple(setup), tuple(missing), tuple(recorded))


def _body_state(priors: DatasetPriors, name: str) -> str:
    """`held`, `free` (episode scope), `missing: …` (a moving base with nothing recorded) or
    `setup: …` (why the setup's value is not calibrated)."""

    body = priors.bodies[name]
    if body.scope == "episode":
        return "free"
    if body.scope == "frame":
        odometry = priors.base_odometry
        if odometry is None:
            return "missing: frame scope with no recorded motion"
        if odometry.kind != "pose" and odometry.correction is None:
            return "setup: odometry not calibrated (no base_odometry.correction)"
        return "held"
    fitted = body.position is not None and body.orientation is not None
    if not fitted and (body.position_range or body.orientation_range_deg):
        return "setup: placement not calibrated"
    return "held"


def _camera_state(priors: DatasetPriors, name: str, moving: bool, posed: bool) -> str:
    """`held`, `recorded` (held at its record for the episode), `free` (episode scope, from a
    start), `missing: …` (episode scope with no start) or `setup: …` (why the setup's value is
    not calibrated)."""

    prior = priors.cameras.get(name)
    if prior is None:
        return "setup: no camera priors"
    if prior.mount is not None:  # a station's own mount (streams swapped between setups)
        moving = prior.mount == "moving"
    if posed or prior.pose is not None:
        return "recorded" if prior.scope == "episode" else "held"
    known = prior.position is not None and prior.orientation is not None
    if prior.scope == "episode":
        return "free" if known else "missing: episode scope with no start"
    if not known:
        return "setup: mount not calibrated" if moving else "setup: pose not calibrated"
    if not moving and (math.isinf(prior.position_range) or prior.orientation_range_deg >= 180):
        return "setup: pose only a start (no ranges)"
    return "held"


@dataclass(frozen=True, slots=True, kw_only=True)
class Cost:
    """Per-episode timings measured with `r4r reconstruct` on a JUPITER login node (4 CPU
    threads, 2026-11-15) and the share of real episodes the fit check flags."""

    # Fast path: the state and forward kinematics (rby1, syngrasp, sim-sawyer: 13-40 ms per
    # episode), seconds per episode and per frame
    fast_episode: float = 0.03
    fast_frame: float = 2e-5
    # Bursts (decode, KLT, SIFT links), seconds per camera (DROID's check: 1.1 s, 3 cameras);
    # the light LM and its extension to every frame (M1: 5 s per episode in all, DROID 9 s)
    keyframes_camera: float = 0.35
    light_solve: float = 4.0
    # Real episodes the fit check flags (DROID: 12 of 34), solved light
    flagged: float = 0.35
    # Full path: GPU seconds per second of video (8-10 GPU s per 25 s episode; KLT on CPU
    # alongside, CPU seconds per camera and second of video)
    full_gpu: float = 0.36
    full_cpu: float = 0.035
    # One `r4r rig` on a sample of a setup (GPU seconds: 10-18 min measured)
    calibration_gpu: float = 900.0
    # A station whose only new value is one static camera, fitted from a neighbour station
    # (`r4r station`): CPU seconds (bursts, PnP, LM), GPU seconds (RoMa links and transferred
    # points; 20-60 s measured). Share of such stations that pass the fit check; the rest
    # still need a rig (openneodata arx5, leave-one-out over 8 full-rig stations with RoMa
    # links: 2 passed, 1 of them correctly, 2026-11-15)
    station_cpu: float = 60.0
    station_gpu: float = 40.0
    station_pass: float = 0.125


COST = Cost()


@dataclass(frozen=True, slots=True)
class DatasetPlan:
    """Routes of a dataset's episodes and their estimated cost."""

    name: str
    hours: float
    episodes: int
    routes: Counter[str] = field(default_factory=Counter)
    checked: int = 0  # fast episodes that get the fit check (real data)
    setups: int = 0  # setups (stations) to calibrate first: one `r4r rig` each (GPU)
    neighbours: int = 0  # setups whose one static camera is fitted from a neighbour station
    placements: int = 0  # setups of a lone camera to place (picks, no GPU)
    odd: int = 0  # episodes of one-off setups (fewer than MIN_SETUP episodes), not calibrated
    reasons: Counter[str] = field(default_factory=Counter)
    cpu_hours: float = 0.0
    gpu_hours: float = 0.0


def dataset_plan(
    root: Path, episodes: Sequence[int] | None = None, cost: Cost = COST
) -> DatasetPlan:
    """Count the routes of `episodes` (default: all) of a dataset root from its priors and its
    episodes table (stations, recorded poses and intrinsics), and cost them (module doc)."""

    from r4r.data.lerobot import camera_names, episode_group, simulated
    from r4r.data.priors import episode_table_path, load_priors
    from r4r.data.tables import LeRobotTables

    tables = LeRobotTables(root)
    priors = load_priors(root)
    names, moving = camera_names(root)
    selection = range(tables.total_episodes) if episodes is None else episodes
    groups = _record_groups(episode_table_path(root), selection, lambda e: episode_group(root, e))
    frames = tables.info.get("total_frames", 0) / max(tables.total_episodes, 1)
    seconds = frames / tables.fps
    real = simulated(root) is False
    routes, reasons = Counter(), Counter()
    # A setup is a common station (camera None) or a camera's own station (camera, station)
    sizes: Counter[Setup] = Counter()
    for (_, station, own, _, _), count in groups.items():
        sizes[None, station] += count
        for setup in own:
            sizes[setup] += count
    # Setups to calibrate by kind: a whole rig, from a neighbour station, a lone camera placed
    kinds: dict[str, set[Setup]] = {"rig": set(), "neighbour": set(), "lone": set()}
    checked = odd = 0
    cpu = gpu = 0.0
    for (group, station, own, poses, intrinsics), count in groups.items():
        keys = dict(own)
        episode_priors = priors.for_episode(group, station, keys)
        unknowns = episode_unknowns(episode_priors, names, moving, poses, intrinsics)
        by_setup = setup_reasons(unknowns.setup, station, keys)
        if any(s[1] is not None and sizes[s] < MIN_SETUP for s in by_setup):
            odd += count
        else:
            for setup, why in by_setup.items():
                kinds[_setup_kind(why, moving, bool(priors.stations))].add(setup)
        if unknowns.setup:
            reasons.update(dict.fromkeys(unknowns.setup, count))
        reasons.update(dict.fromkeys(unknowns.missing, count))
        route = unknowns.calibrated_route
        routes[route] += count
        cpu += count * (cost.fast_episode + cost.fast_frame * frames)
        if route == "full":
            gpu += count * cost.full_gpu * seconds
            cpu += count * cost.full_cpu * seconds * len(names)
            continue
        check = real and bool(unknowns.recorded)
        checked += count if check and route == "fast" else 0
        if route == "light" or check:
            cpu += count * cost.keyframes_camera * len(names)
        if route == "light":
            cpu += count * cost.light_solve
        elif check:
            cpu += count * cost.flagged * cost.light_solve
    rigs, neighbours = len(kinds["rig"]), len(kinds["neighbour"])
    failing = neighbours * (1 - cost.station_pass)
    gpu += (rigs + failing) * cost.calibration_gpu + neighbours * cost.station_gpu
    cpu += neighbours * cost.station_cpu
    hours = tables.info.get("total_frames", 0) / tables.fps / 3600 * len(selection)
    return DatasetPlan(
        name=f"{root.parent.parent.name}/{root.name}",
        hours=hours / max(tables.total_episodes, 1),
        episodes=len(selection),
        routes=routes,
        checked=checked,
        setups=rigs,
        neighbours=neighbours,
        placements=len(kinds["lone"]),
        odd=odd,
        reasons=reasons,
        cpu_hours=cpu / 3600,
        gpu_hours=gpu / 3600,
    )


type Setup = tuple[str | None, str | None]
type RecordKey = tuple[
    str | None, str | None, frozenset[tuple[str, str]], frozenset[str], frozenset[str]
]


def _record_groups(
    path: Path, episodes: Sequence[int] | range, group_of: Callable[[int], str | None]
) -> Counter[RecordKey]:
    """How many of `episodes` share each (priors group, station, per-camera stations (camera,
    station), cameras with a recorded pose, cameras with recorded intrinsics) of the episodes
    table (episodes without a row: none recorded); `group_of` gives an episode's priors
    group."""

    import numpy as np

    wanted = np.asarray(episodes, dtype=np.int64)
    groups: Counter[RecordKey] = Counter()
    if not path.exists():
        for e in wanted.tolist():
            groups[group_of(e), None, frozenset(), frozenset(), frozenset()] += 1
        return groups
    import pyarrow.parquet as pq

    from r4r.data.priors import STATION_PREFIX

    table = pq.read_table(path)
    columns = set(table.column_names)
    episode = table["episode"].to_numpy()
    inside = np.isin(episode, wanted)
    camera = table["camera"].to_pylist() if "camera" in columns else [None] * len(episode)
    station = table["station"].to_pylist() if "station" in columns else [None] * len(episode)
    own = {
        c.removeprefix(STATION_PREFIX): table[c].to_pylist()
        for c in table.column_names
        if c.startswith(STATION_PREFIX)
    }
    posed = _not_null(table, ("x", "qw"))
    focal = _not_null(table, ("fx",))
    record: dict[int, tuple[str | None, dict[str, str], set[str], set[str]]] = {}
    for i in np.flatnonzero(inside).tolist():
        entry = record.setdefault(int(episode[i]), (None, {}, set(), set()))
        if station[i] is not None and entry[0] is None:
            entry = record[int(episode[i])] = (str(station[i]), *entry[1:])
        for name, values in own.items():
            if values[i] is not None:
                entry[1].setdefault(name, str(values[i]))
        if camera[i] is not None and posed[i]:
            entry[2].add(camera[i])
        if camera[i] is not None and focal[i]:
            entry[3].add(camera[i])
    for e, (station_name, setups, poses, intrinsics) in record.items():
        key = (frozenset(setups.items()), frozenset(poses), frozenset(intrinsics))
        groups[group_of(e), station_name, *key] += 1
    for e in wanted.tolist():
        if e not in record:
            groups[group_of(e), None, frozenset(), frozenset(), frozenset()] += 1
    return +groups


def setup_reasons(
    reasons: Sequence[str], station: str | None, camera_stations: Mapping[str, str]
) -> dict[Setup, list[str]]:
    """`reasons` (`name: why`) by the setup they belong to: a camera with its own station
    (`camera_stations`) to (camera, its station), the rest to the common (None, `station`)."""

    out: dict[Setup, list[str]] = {}
    for reason in reasons:
        name = reason.split(":")[0]
        own = camera_stations.get(name)
        out.setdefault((None, station) if own is None else (name, own), []).append(reason)
    return out


def _setup_kind(reasons: Sequence[str], moving: Collection[str], stations: bool) -> str:
    """How a setup missing `reasons` is calibrated: `lone` (a lone camera placed), `neighbour`
    (one static camera fitted from a neighbour station, when the priors have `stations`) or
    `rig`."""

    if all(r.endswith(LONE) for r in reasons):
        return "lone"
    return "neighbour" if stations and _from_neighbour(reasons, moving) else "rig"


def _from_neighbour(reasons: Sequence[str], moving: Collection[str]) -> bool:
    """Whether a setup's missing values are one static camera's pose and focal, plus focals a
    neighbour station gives (`rago.station` fits the rest)."""

    static = {r.split(":")[0] for r in reasons if "placement" not in r} - set(moving)
    allowed = all(
        r.endswith(
            (
                "focal unknown",
                "pose not calibrated",
                "pose only a start (no ranges)",
                "placement not calibrated",
            )
        )
        for r in reasons
    )
    return allowed and len(static) == 1


def _not_null(table: object, columns: tuple[str, ...]) -> list[bool]:
    """Per row: whether every one of `columns` is present and not null."""

    import pyarrow as pa

    assert isinstance(table, pa.Table)
    out = [True] * table.num_rows
    for name in columns:
        if name not in table.column_names:
            return [False] * table.num_rows
        valid = table[name].is_valid().to_pylist()
        out = [a and bool(b) for a, b in zip(out, valid, strict=True)]
    return out


def describe(plans: Sequence[DatasetPlan]) -> str:
    """The plans as a Markdown table, with totals and each dataset's main reasons."""

    lines = [
        "| dataset | hours | episodes | fast | light | full | checked | rig setups | "
        "neighbour setups | place setups | odd | CPU h | GPU h |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {p.name} | {p.hours:.0f} | {p.episodes} | {p.routes['fast']} | "
        f"{p.routes['light']} | {p.routes['full']} | {p.checked} | {p.setups} | "
        f"{p.neighbours} | {p.placements} | {p.odd} | {p.cpu_hours:.1f} | {p.gpu_hours:.2f} |"
        for p in plans
    )
    cpu, gpu = sum(p.cpu_hours for p in plans), sum(p.gpu_hours for p in plans)
    hours, count = sum(p.hours for p in plans), sum(p.episodes for p in plans)
    lines.append(f"| total | {hours:.0f} | {count} | | | | | | | | | {cpu:.1f} | {gpu:.2f} |")
    for p in plans:
        for reason, count in p.reasons.most_common(3):
            lines.append(f"- {p.name}: {count} episodes: {reason}")
    return "\n".join(lines)
