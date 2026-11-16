"""Shared evaluation of per-episode camera checks against clicked fingertips and known errors.

Every check (motion, edges, rest template, DINO features, ...) writes the same result table,
one row per (dataset, episode, camera, method, perturbation), as parquet (`write_rows`):

- `dataset` (str): delivery-relative name as `r4r reconstruct` takes it (`droid/datasets/droid`);
- `episode` (int): source episode index; `camera` (str): video key suffix;
- `role` (str): `scene` (static or head camera) or `wrist`;
- `method` (str): the check (`edge`, `flow`, `rest`, `dino-feature`, ...);
- `perturbation` (str): `""` for the camera as reconstructed, else e.g. `t:5cm`, `r:2deg`,
  `f:+10%`;
- `score` (float): the check's raw score, higher = worse; NaN when it could not run;
- `estimate_px` (float): the check's own estimate of the camera's pixel error (NaN: none);
- `confidence` (float): evidence for the score in [0, 1] (NaN: not given);
- `shift_u`, `shift_v` (float): a 2D correction the check proposes, pixels added to the drawn
  points (NaN: none);
- `truth_px` (float): the known error. Synthetic perturbation: the projected displacement of
  the robot; exact simulated camera: 0; real data: NaN (joined from clicks here);
- `seconds` (float): CPU seconds the check spent on the episode (all its cameras).

The ground truth of real data is the held-out click error per (episode, camera): the median
over that camera's clicked targets of `r4r.viz.clicks.error` on a click set redrawn on the
same reconstruction (`python -m r4r.viz.clicks redraw`). A camera is bad when its error
exceeds `BAD_PX`.

    python -m r4r.checks.eval RESULTS.parquet [...] --clicks CLICKS.json [...] [--by kind]

prints per method: rows, bad rows, AUC of `score` for bad (Mann-Whitney), Spearman ρ with the
truth, the median |estimate − truth|, seconds per episode, and the click error before and after
the proposed 2D shift (correction gain); `--by kind|dataset|role|perturbation` splits the table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BAD_PX = 15.0  # a camera whose click error exceeds this is bad (docs/verification.md)

COLUMNS = {
    "dataset": pa.string(),
    "episode": pa.int64(),
    "camera": pa.string(),
    "role": pa.string(),
    "method": pa.string(),
    "perturbation": pa.string(),
    "score": pa.float64(),
    "estimate_px": pa.float64(),
    "confidence": pa.float64(),
    "shift_u": pa.float64(),
    "shift_v": pa.float64(),
    "truth_px": pa.float64(),
    "seconds": pa.float64(),
}

# Dataset kinds for the robustness split (family/name -> tags); unknown datasets: "other"
KINDS = {
    "abc-130k": "bimanual",
    "agibot-world": "humanoid,fisheye",
    "airoa-moma-5k": "mobile",
    "droid": "arm",
    "galaxea-open-world": "mobile,bimanual",
    "hiw-500": "humanoid,people",
    "molmoact2-yam": "bimanual",
    "openneodata/aloha": "bimanual",
    "openneodata/arx5": "arm",
    "openneodata/flexiv-rizon4": "arm",
    "openneodata/ur5e": "arm",
    "rdt-ft-data": "bimanual",
    "realsource-world": "humanoid",
    "robomind2": "bimanual",
    "robomind/agilex-real": "bimanual",
    "robomind/ur5e-real": "arm",
    "rw-rl": "mobile,bimanual",
    "xvla-soft-fold": "bimanual",
    "molmobot": "sim,mobile",
    "interndata-a1-sim": "sim,bimanual",
    "nvidia-kitchen": "sim,mobile",
}


def kind(dataset: str) -> str:
    family, name = dataset.split("/")[0], dataset.split("/")[-1]
    return KINDS.get(f"{family}/{name}", KINDS.get(family, "other"))


def write_rows(rows: list[dict], path: Path) -> None:
    """The rows (dicts with any subset of `COLUMNS`, missing ones NaN or empty) as parquet."""

    def value(row: dict, name: str, kind: pa.DataType) -> object:
        if name in row and row[name] is not None:
            return row[name]
        return "" if kind == pa.string() else (0 if kind == pa.int64() else float("nan"))

    arrays = {n: pa.array([value(r, n, k) for r in rows], k) for n, k in COLUMNS.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(arrays), path)


def click_truth(clicks: Path) -> dict[tuple[str, int, str], tuple[float, list[dict]]]:
    """(dataset, episode, camera) -> (median click error in pixels, its clicked targets)."""

    from r4r.viz.clicks import error

    data = json.loads(clicks.read_text())
    found: dict[tuple[str, int, str], list[dict]] = {}
    for t in data["targets"]:
        if t.get("click") is not None:
            found.setdefault((data["dataset"], int(t["episode"]), t["camera"]), []).append(t)
    return {k: (float(np.median([error(t) for t in v])), v) for k, v in found.items()}


def shifted_error(targets: list[dict], du: float, dv: float) -> float:
    """The median click error with the drawn points moved by (du, dv) pixels."""

    from r4r.viz.clicks import error

    shift = np.array([du, dv])
    moved = [t | {"drawn": (np.array(t["drawn"], float) + shift).tolist()} for t in targets]
    return float(np.median([error(t) for t in moved]))


def auc(score: np.ndarray, bad: np.ndarray) -> float:
    """P(score of a bad row > score of a good row), ties counted half (Mann-Whitney U)."""

    pos, neg = score[bad], score[~bad]
    if not len(pos) or not len(neg):
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    for v in np.unique(x):  # ties: their mean rank
        same = x == v
        ranks[same] = ranks[same].mean()
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(_ranks(a), _ranks(b))[0, 1])


def load(results: list[Path], clicks: list[Path]) -> list[dict]:
    """Every result row, real rows joined with their click truth (rows without one dropped)
    and with `click_after` (the click error after the row's shift, NaN without a shift)."""

    truth: dict[tuple[str, int, str], tuple[float, list[dict]]] = {}
    for path in clicks:
        truth |= click_truth(path)
    rows = []
    for path in results:
        rows += pq.read_table(path).to_pylist()
    out = []
    for row in rows:
        row["kind"] = kind(row["dataset"])
        row["click_after"] = float("nan")
        if np.isnan(row["truth_px"]):
            key = (row["dataset"], row["episode"], row["camera"])
            if key not in truth:
                continue
            row["truth_px"], targets = truth[key]
            if row["perturbation"] == "" and np.isfinite(row["shift_u"]):
                row["click_after"] = shifted_error(targets, row["shift_u"], row["shift_v"])
        out.append(row)
    return out


def summary(rows: list[dict]) -> dict[str, float]:
    """The figures of one group of rows (module doc)."""

    ok = [r for r in rows if np.isfinite(r["score"])]
    score = np.array([r["score"] for r in ok])
    truth = np.array([r["truth_px"] for r in ok])
    estimate = np.array([r["estimate_px"] for r in ok])
    bad = truth > BAD_PX
    seconds = {(r["dataset"], r["episode"], r["perturbation"]): r["seconds"] for r in ok}
    finite = np.isfinite(estimate)
    before = np.array([r["truth_px"] for r in ok if np.isfinite(r["click_after"])])
    after = np.array([r["click_after"] for r in ok if np.isfinite(r["click_after"])])
    return {
        "rows": len(rows),
        "scored": len(ok),
        "bad": int(bad.sum()),
        "auc": auc(score, bad),
        "spearman": spearman(score, truth),
        "est_err_px": float(np.median(np.abs(estimate[finite] - truth[finite])))
        if finite.any()
        else float("nan"),
        "s_per_episode": float(np.median(list(seconds.values()))) if seconds else float("nan"),
        "click_before": float(np.median(before)) if len(before) else float("nan"),
        "click_after": float(np.median(after)) if len(after) else float("nan"),
        "worse_5px": int((after > before + 5).sum()),
    }


def table(rows: list[dict], by: str | None) -> str:
    """One line per method (and per value of `by`)."""

    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        keys = [""] if by is None else str(r[by]).split(",")
        for k in keys:
            groups.setdefault((r["method"], k), []).append(r)
    head = (
        f"{'method':14s} {by or '':22s} {'rows':>5s} {'bad':>4s} {'AUC':>5s} {'rho':>5s} "
        f"{'|est-truth|':>11s} {'s/ep':>6s} {'click->after':>14s} {'worse':>5s}"
    )
    lines = [head]
    for (method, k), group in sorted(groups.items()):
        s = summary(group)
        lines.append(
            f"{method:14s} {k:22s} {s['scored']:5d} {s['bad']:4d} {s['auc']:5.2f} "
            f"{s['spearman']:5.2f} {s['est_err_px']:11.1f} {s['s_per_episode']:6.2f} "
            f"{s['click_before']:6.1f}->{s['click_after']:6.1f} {s['worse_5px']:5d}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    by, lists, into = None, {"results": [], "clicks": []}, "results"
    args = iter(sys.argv[1:])
    for arg in args:
        if arg == "--by":
            by = next(args)
        elif arg == "--clicks":
            into = "clicks"
        else:
            lists[into].append(Path(arg))
    print(table(load(lists["results"], lists["clicks"]), by))
