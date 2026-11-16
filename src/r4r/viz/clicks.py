"""Held-out validation by clicked points: the real jaw or fingertip against the drawn one.

A reconstruction (`r4r reconstruct … --episodes …`, one shard) gives every camera's pose,
intrinsics and lens and every end effector's frame and opening in the episode frame. Per
episode one moment per view is picked where an end effector is in the image: in a scene view
(static, head) the frame where a gripper reaches farthest from where it started, in a wrist view
the frame where another arm is nearest the image centre. The raw frame is saved with a pixel
grid (nothing drawn) for clicking the real point; the click is scored against the drawn points:

- jaws (parallel gripper, opening recorded): `TCP ± y · w/2`, the nearer of the two;
- jaw line (the opening constant over the episode, i.e. not recorded): the segment between the
  jaws fully open, `TCP ± y · open/2` (a click lands on one pad, up to half the stroke from the
  modelled one);
- fingertips (hands): the nearest fingertip.

Each target also stores the drawn points moved along the end effector's approach axis (z) by
`SHIFTS`: `score` reports the error with one shift for the whole dataset (the median of each
click's best one) next to the plain error. A plain error that drops at a constant shift is the
TCP depth (the drawn point short of or past the real fingertips), not the camera.

Projection uses OpenCV (`cv2.projectPoints`, `cv2.fisheye.projectPoints`) on the columns as
`docs/output-spec.md` defines them, independently of r4r's own camera code.

    python -m r4r.viz.clicks sample DATASET N SEED [--exclude 1,2,3]
    python -m r4r.viz.clicks make SHARD DATASET CLICKS.json SHEETS_DIR [--open 0.08] [--views 1]
        [--no-sheets]
    python -m r4r.viz.clicks redraw SHARD CLICKS.json OUT.json [--open 0.08]
    python -m r4r.viz.clicks score CLICKS.json

`make` writes the targets (with the drawn points) into CLICKS.json and the sheets
`SHEETS_DIR/sheet_<i>.png` (two targets each: the whole frame and a zoom, both gridded in the
frame's pixels). The clicks go into the same JSON (`"click": [u, v]`, `null` when the point is
not visible); `score` prints each error and the median, p90 and max in pixels and in % of the
image width. `redraw` re-scores the same clicks on a new reconstruction of their episodes
(after a description or priors change): each target's drawn points are recomputed at its
episode, camera, arm and frame.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

MARGIN = 0.06  # a scene target's TCP lies this fraction of the image size inside its border
OTHER_ARM = 0.25  # a wrist view's target: an end effector at least this far in front (m)
SHIFTS = np.round(np.arange(-0.10, 0.1001, 0.01), 3)  # along the approach axis (m)
SHOW = 448  # height of the whole frame and side of the zoom on a sheet (display pixels)


def _matrix(pose: np.ndarray) -> np.ndarray:
    """4×4 from the spec's `x, y, z, qx, qy, qz, qw` rows [T, 7] -> [T, 4, 4]."""

    x, y, z, w = pose[:, 3], pose[:, 4], pose[:, 5], pose[:, 6]
    out = np.zeros((len(pose), 4, 4))
    out[:, 0, :3] = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], 1)
    out[:, 1, :3] = np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], 1)
    out[:, 2, :3] = np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1)
    out[:, :3, 3] = pose[:, :3]
    out[:, 3, 3] = 1.0
    return out


def _column(table: pa.Table, name: str) -> np.ndarray | None:
    if name not in table.column_names:
        return None
    return np.array(table.column(name).to_pylist(), dtype=np.float64)


def _episodes(shard: Path) -> Iterator[tuple[int, pa.Table, dict, list[str], dict]]:
    """Per episode: its frame indices, camera columns and end-effector points."""

    from r4r.data.lerobot import camera_name

    table = pq.read_table(shard)
    layout = json.loads(table.schema.metadata[b"r4r.layout"])
    cams = {camera_name(k): v for k, v in layout["cameras"].items()}
    arms = sorted({k.split(".")[2] for k in layout["features"] if k.startswith("observation.ee.")})
    index = np.array(table.column("episode_index").to_pylist())
    for episode in dict.fromkeys(index.tolist()):
        part = table.take(np.flatnonzero(index == episode))
        yield episode, part, cams, arms, layout


def _arm_points(
    part: pa.Table, arm: str, opened: float
) -> tuple[np.ndarray, str, np.ndarray | None] | None:
    """[T, P, 3] points of one end effector, the metric (`jaws`, `line`, `tips`, `tcp`) and its
    approach axis [T, 3] (None for fingertips)."""

    prefix = f"observation.ee.{arm}"
    pose = _column(part, f"{prefix}.pose")
    if pose is None:
        return None
    tips = _column(part, f"{prefix}.fingertips")
    if tips is not None:
        return tips, "tips", None
    frame = _matrix(pose)
    tcp, y = frame[:, :3, 3], frame[:, :3, 1]
    width = _column(part, f"{prefix}.gripper")
    if width is None:
        return tcp[:, None], "tcp", frame[:, :3, 2]
    width = width.reshape(-1)
    if np.nanstd(width) < 1e-4:  # not recorded: the jaw line, fully open
        width = np.full_like(width, opened)
        metric = "line"
    else:
        metric = "jaws"
    half = (width / 2)[:, None] * y
    return np.stack([tcp + half, tcp - half], 1), metric, frame[:, :3, 2]


def _project(
    part: pa.Table, camera: str, lens: str, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pixels [T, P, 2] and depths [T, P] of episode-frame points [T, P, 3] in `camera`."""

    prefix = f"observation.camera.{camera}"
    pose = _matrix(_column(part, f"{prefix}.pose"))
    k = _column(part, f"{prefix}.intrinsics")
    dist = _column(part, f"{prefix}.distortion")
    finite = np.isfinite(pose).all(axis=(1, 2))
    inv = np.full_like(pose, np.nan)
    inv[finite] = np.linalg.inv(pose[finite])  # uncalibrated cameras are NaN
    local = np.einsum("tij,tpj->tpi", inv[:, :3, :3], points) + inv[:, None, :3, 3]
    uv = np.full((*points.shape[:2], 2), np.nan)
    for t in range(len(points)):
        if not np.isfinite(pose[t]).all() or not np.isfinite(local[t]).all():
            continue
        matrix = np.array([[k[t, 0], 0, k[t, 2]], [0, k[t, 1], k[t, 3]], [0, 0, 1]])
        p = local[t][:, None].astype(np.float64)
        if lens == "fisheye":
            out, _ = cv2.fisheye.projectPoints(p, np.zeros(3), np.zeros(3), matrix, dist[t])
        else:
            coeffs = np.array([dist[t, 0], dist[t, 1], 0.0, 0.0, dist[t, 2]])
            out, _ = cv2.projectPoints(p[:, 0], np.zeros(3), np.zeros(3), matrix, coeffs)
        uv[t] = out[:, 0]
    return uv, local[..., 2]


def _grid(image: np.ndarray, x0: float, y0: float, scale: float, step: int) -> np.ndarray:
    """`image` (a crop at `x0, y0` of the frame, shown at `scale`) with lines every `step`
    frame pixels, labelled in frame pixels every second line."""

    h, w = image.shape[:2]
    out = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC)
    first = lambda origin: int(np.ceil(origin / step) * step)  # noqa: E731
    for x in range(first(x0), int(x0 + w), step):
        px = round((x - x0) * scale)
        major = x % (2 * step) == 0
        cv2.line(out, (px, 0), (px, out.shape[0]), (0, 255, 255) if major else (90, 90, 90), 1)
        if major:
            cv2.putText(out, str(x), (px + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
    for y in range(first(y0), int(y0 + h), step):
        py = round((y - y0) * scale)
        major = y % (2 * step) == 0
        cv2.line(out, (0, py), (out.shape[1], py), (0, 255, 255) if major else (90, 90, 90), 1)
        if major:
            cv2.putText(out, str(y), (2, py - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
    return out


def _tile(
    image: np.ndarray, centre: np.ndarray, rng: np.random.Generator, label: str
) -> np.ndarray:
    """The whole frame and a zoom near `centre` (moved at random, so the zoom's centre does
    not give the drawn point away), side by side."""

    h, w = image.shape[:2]
    whole = _grid(image, 0, 0, SHOW / h, 32 if w <= 960 else 64)
    side = int(np.clip(0.45 * min(w, h), 112, 320))
    c = centre + rng.uniform(-side / 8, side / 8, 2)
    x0 = int(np.clip(c[0] - side / 2, 0, max(w - side, 0)))
    y0 = int(np.clip(c[1] - side / 2, 0, max(h - side, 0)))
    crop = image[y0 : y0 + side, x0 : x0 + side]
    zoom = _grid(crop, x0, y0, SHOW / side, 8 if side <= 160 else 16)
    pad = np.zeros((SHOW, max(SHOW - zoom.shape[1], 0), 3), np.uint8)
    zoom = np.hstack([zoom[:SHOW, :SHOW], pad])[:SHOW]
    if zoom.shape[0] < SHOW:
        zoom = np.vstack([zoom, np.zeros((SHOW - zoom.shape[0], zoom.shape[1], 3), np.uint8)])
    row = np.hstack([whole, np.full((SHOW, 6, 3), 255, np.uint8), zoom])
    bar = np.zeros((22, row.shape[1], 3), np.uint8)
    cv2.putText(bar, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return np.vstack([bar, row])


def _targets(
    episode: int, part: pa.Table, cams: dict, arms: list[str], opened: float, turn: int, views: int
) -> list[dict]:
    """At most `views` scene targets (distinct cameras) and one wrist target of the episode
    (cameras and arms rotate with `turn`)."""

    frames = np.array(part.column("frame_index").to_pylist())
    points = {a: p for a in arms if (p := _arm_points(part, a, opened)) is not None}
    # Calibrated in this episode: the dataset's flag, and finite intrinsics (a camera written
    # as NaN in this episode by `--uncalibrated nan`)
    shown = {
        c
        for c, v in cams.items()
        if v["camera.calibrated"]
        and np.isfinite(_column(part, f"observation.camera.{c}.intrinsics")).any()
    }
    scene = [c for c, v in cams.items() if v["camera.role"] != "wrist" and c in shown]
    wrist = [c for c, v in cams.items() if v["camera.role"] == "wrist" and c in shown]
    out = []
    for kind, cameras, wanted in (("scene", scene, views), ("wrist", wrist, 1)):
        cameras = cameras[turn % len(cameras) :] + cameras[: turn % len(cameras)] if cameras else []
        taken = 0
        for camera in cameras:
            if taken == wanted:
                break
            best = None
            order = arms[turn % len(arms) :] + arms[: turn % len(arms)] if arms else []
            for arm in order:
                if arm not in points:
                    continue
                pts, metric, axis = points[arm]
                uv, depth = _project(part, camera, cams[camera]["camera.lens"], pts)
                mid, z = np.nanmean(uv, 1), np.nanmean(depth, 1)
                w, h = _size(part, camera)
                inside = (
                    (mid[:, 0] > MARGIN * w)
                    & (mid[:, 0] < (1 - MARGIN) * w)
                    & (mid[:, 1] > MARGIN * h)
                    & (mid[:, 1] < (1 - MARGIN) * h)
                )
                if kind == "scene":
                    inside &= z > 0.1
                    tcp = pts.mean(1)
                    score = np.linalg.norm(tcp - tcp[0], axis=1)
                else:
                    inside &= z > OTHER_ARM
                    score = -np.hypot(mid[:, 0] - w / 2, mid[:, 1] - h / 2)
                if not inside.any():
                    continue
                score = np.where(inside, score, -np.inf)
                t = int(np.argmax(score))
                # the first arm in turn order wins unless another reaches clearly farther
                if best is None or (kind == "wrist" and score[t] > best[0]):
                    best = (score[t], arm, t, metric, uv[t], w, h, pts, axis)
            if best is None:
                continue
            _, arm, t, metric, uv, w, h, pts, axis = best
            along = _along(part, camera, cams[camera]["camera.lens"], pts, axis, t)
            out.append(
                {
                    "key": f"{episode}_{camera}_{arm}_{int(frames[t])}",
                    "episode": int(episode),
                    "camera": camera,
                    "arm": arm,
                    "kind": kind,
                    "frame": int(frames[t]),
                    "row": t,
                    "metric": metric,
                    "width": w,
                    "height": h,
                    "drawn": np.round(uv, 2).tolist(),
                    "along": along,
                    "click": None,
                }
            )
            taken += 1
    return out


def _along(
    part: pa.Table, camera: str, lens: str, pts: np.ndarray, axis: np.ndarray | None, t: int
) -> list:
    """Pixels [S, P, 2] of row `t`'s points moved by each of `SHIFTS` along `axis` ([] for
    fingertips)."""

    if axis is None:
        return []
    moved = pts[t][None, None] + SHIFTS[:, None, None] * axis[t][None, None]
    moved = np.repeat(moved, len(pts), 0)  # rows: frames of the episode
    shifted = [_project(part, camera, lens, moved[:, j])[0][t] for j in range(len(SHIFTS))]
    return np.round(np.stack(shifted), 2).tolist()


def redraw(shard: Path, clicks: Path, out: Path, opened: float) -> None:
    """The targets of `clicks` (their clicks kept) with the drawn points of a new
    reconstruction `shard` at the same episode, camera, arm and frame: a description or priors
    change re-scored on the same clicks."""

    data = json.loads(clicks.read_text())
    parts = {episode: (part, cams) for episode, part, cams, _, _ in _episodes(shard)}
    kept = []
    for t in data["targets"]:
        if t["episode"] not in parts:
            print(f"{t['key']}: episode not in {shard}, dropped")
            continue
        part, cams = parts[t["episode"]]
        frames = np.array(part.column("frame_index").to_pylist())
        rows = np.flatnonzero(frames == t["frame"])
        found = _arm_points(part, t["arm"], opened)
        if not len(rows) or found is None:
            print(f"{t['key']}: frame or arm missing, dropped")
            continue
        pts, metric, axis = found
        row, lens = int(rows[0]), cams[t["camera"]]["camera.lens"]
        uv, _ = _project(part, t["camera"], lens, pts)
        t |= {
            "row": row,
            "metric": metric,
            "drawn": np.round(uv[row], 2).tolist(),
            "along": _along(part, t["camera"], lens, pts, axis, row),
        }
        kept.append(t)
    out.write_text(json.dumps(data | {"targets": kept}, indent=1))
    print(f"{len(kept)} targets redrawn -> {out}")


def _size(part: pa.Table, camera: str) -> tuple[int, int]:
    k = _column(part, f"observation.camera.{camera}.intrinsics")
    # cx, cy are pixel centres of the delivered video: (W − 1)/2, (H − 1)/2
    return round(2 * np.nanmedian(k[:, 2]) + 1), round(2 * np.nanmedian(k[:, 3]) + 1)


def make(
    shard: Path,
    dataset: str,
    clicks: Path,
    sheets: Path,
    opened: float,
    views: int,
    no_sheets: bool = False,
) -> None:
    from r4r.cli import dataset_root
    from r4r.data.lerobot import keyframes

    root = dataset_root(dataset)
    rng = np.random.default_rng(0)
    targets = []
    for turn, (episode, part, cams, arms, _) in enumerate(_episodes(shard)):
        found = _targets(episode, part, cams, arms, opened, turn, views)
        targets += found
        print(episode, [t["key"] for t in found] or "no target")
    sheets.mkdir(parents=True, exist_ok=True)
    tiles = []
    for t in [] if no_sheets else targets:
        image = keyframes(root, t["episode"], [t["frame"]], [t["camera"]])[t["camera"]][0]
        image = cv2.cvtColor(image.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
        t["width"], t["height"] = image.shape[1], image.shape[0]
        tiles.append(_tile(image, np.nanmean(np.array(t["drawn"]), 0), rng, t["key"]))
    for i in range(0, len(tiles), 2):
        pair = tiles[i : i + 2]
        width = max(p.shape[1] for p in pair)
        pair = [
            np.hstack([p, np.zeros((p.shape[0], width - p.shape[1], 3), np.uint8)]) for p in pair
        ]
        cv2.imwrite(str(sheets / f"sheet_{i // 2}.png"), np.vstack(pair))
    old = json.loads(clicks.read_text()) if clicks.exists() else {}
    known = {t["key"]: t.get("click") for t in old.get("targets", [])}
    sizes = {t["key"]: (t["width"], t["height"]) for t in old.get("targets", [])}
    for t in targets:
        t["click"] = known.get(t["key"])
        if no_sheets and t["key"] in sizes:
            t["width"], t["height"] = sizes[t["key"]]
    clicks.parent.mkdir(parents=True, exist_ok=True)
    clicks.write_text(json.dumps({"dataset": dataset, "targets": targets}, indent=1))
    print(f"{len(targets)} targets -> {clicks}, sheets in {sheets}")


def error(target: dict, shift: float | None = None) -> float:
    """Pixel distance of the click to the drawn point(s) (module doc), the points moved by
    `shift` (m, one of `SHIFTS`) along the approach axis when given."""

    c, drawn = np.array(target["click"], float), np.array(target["drawn"], float)
    if shift is not None and target.get("along"):
        drawn = np.array(target["along"][int(np.argmin(np.abs(SHIFTS - shift)))], float)
    if target["metric"] == "line":
        a, b = drawn[0], drawn[1]
        s = np.clip(np.dot(c - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0, 1)
        return float(np.linalg.norm(a + s * (b - a) - c))
    return float(np.min(np.linalg.norm(drawn - c, axis=1)))


def score(clicks: Path) -> None:
    data = json.loads(clicks.read_text())
    rows = [t for t in data["targets"] if t.get("click") is not None]
    groups: dict[str, list[float]] = {}
    percent: dict[str, list[float]] = {}
    for t in rows:
        e = error(t)
        d = np.array(t["click"], float) - np.nanmean(np.array(t["drawn"], float), 0)
        print(
            f"{t['key']:44s} {t['metric']:5s} {e:6.1f} px  ({100 * e / t['width']:4.1f}% of "
            f"{t['width']}x{t['height']})  drawn - click mean du {-d[0]:+.0f} dv {-d[1]:+.0f}"
        )
        for g in ("all", t["kind"], t["camera"]):
            groups.setdefault(g, []).append(e)
            percent.setdefault(g, []).append(100 * e / t["width"])
    for g, v in groups.items():
        v, p = np.array(v), np.array(percent[g])
        print(
            f"{g:20s} n {len(v):3d}  median {np.median(v):5.1f}  p90 {np.percentile(v, 90):5.1f}  "
            f"max {v.max():5.1f} px  |  median {np.median(p):4.1f}% width"
        )
    shifted = [t for t in rows if t.get("along")]
    if shifted:
        best = [float(SHIFTS[np.argmin([error(t, x) for x in SHIFTS])]) for t in shifted]
        shift = float(np.median(best))
        v = np.array([error(t, shift) for t in shifted])
        low, high = (100 * np.percentile(best, q) for q in (10, 90))
        print(
            f"TCP moved {100 * shift:+.0f} cm along the approach axis (median of the clicks' best "
            f"shifts, p10-p90 {low:+.0f}..{high:+.0f} cm): n {len(v)}  median {np.median(v):5.1f}  "
            f"p90 {np.percentile(v, 90):5.1f}  max {v.max():5.1f} px"
        )
    skipped = len(data["targets"]) - len(rows)
    print(f"{skipped} targets without a click")


def sample(dataset: str, n: int, seed: int, exclude: set[int]) -> list[int]:
    """`n` random episodes spread over data files and stations (round robin over the
    (station, file) cells, a random episode from each)."""

    from r4r.cli import dataset_root
    from r4r.data.lerobot import _root, episode_count
    from r4r.data.priors import load_episode_record

    root = dataset_root(dataset)
    rng = np.random.default_rng(seed)
    total = episode_count(root)
    tables = _root(root).tables
    pool = rng.permutation(total)[: max(40 * n, 400)]
    cells: dict[tuple, list[int]] = {}
    for e in pool.tolist():
        if e in exclude:
            continue
        record = load_episode_record(root, e)
        key = (None if record is None else record.station, str(tables.files(e).data))
        cells.setdefault(key, []).append(e)
    order = list(cells)
    rng.shuffle(order)
    picked: list[int] = []
    while len(picked) < n and any(cells.values()):
        for key in order:
            if cells[key] and len(picked) < n:
                picked.append(cells[key].pop(0))
    for e in picked:
        record = load_episode_record(root, e)
        station = None if record is None else record.station
        print(e, station, tables.files(e).data.name)
    return picked


if __name__ == "__main__":
    args = sys.argv[1:]

    def option(name: str, default: str) -> str:
        return args[args.index(name) + 1] if name in args else default

    if args[0] == "sample":
        excluded = {int(x) for x in option("--exclude", "").split(",") if x}
        print(" ".join(map(str, sample(args[1], int(args[2]), int(args[3]), excluded))))
    elif args[0] == "make":
        opened, views = float(option("--open", "0.08")), int(option("--views", "1"))
        make(
            Path(args[1]),
            args[2],
            Path(args[3]),
            Path(args[4]),
            opened,
            views,
            "--no-sheets" in args,
        )
    elif args[0] == "redraw":
        redraw(Path(args[1]), Path(args[2]), Path(args[3]), float(option("--open", "0.08")))
    elif args[0] == "score":
        score(Path(args[1]))
