"""Levenberg–Marquardt for the kinematic bundle: few unknowns, many observations.

The objective is the bundle's (`bundle.bundle_adjust`): the mean robust reprojection cost
`ρ(s) = log(1 + s)`, `s = |r|² / δ²`, plus the Gaussian priors. Each step linearises the 2D
residuals `r` (points re-triangulated in closed form inside them, as in the Adam loop) and
solves the reweighted normal equations (IRLS: weight `w / (1 + s)`, the robust cost's slope):

    (H + λ diag H) Δ = −g,   H = Σ ω Jᵀ J + Jₚᵀ Jₚ / T,   g = Σ ω Jᵀ r + Jₚᵀ rₚ / T

accepting Δ only if the true robust objective decreases (λ shrinks), else retrying with a
larger λ. Tens of steps replace the Adam loop's ~1000.

The Jacobian of the residuals comes from forward-mode products, compressed: an observation
depends on a single row of each scoped element (the dataset row, or its episode's row: tracks
never span episodes), so all rows of one (element, degree of freedom) share one product, and
each observation's entry is credited to its own row's column. The cost is one product per
(element, dof) and free focal, whatever the number of episodes. Frame-scope rows (a mobile base
per frame) break this (a track spans frames): `supports` says no, and the caller uses Adam.

Ref: Levenberg 1944; Marquardt 1963; Triggs et al., Bundle Adjustment — A Modern Synthesis,
     1999 (robust IRLS, damping); Curtis, Powell & Reid 1974 (compressed Jacobians).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor

    from .rig import Corrections, Rig


@dataclass(frozen=True, slots=True, kw_only=True)
class LMConfig:
    # Rounds of: converge with the tracks' bodies fixed, then re-attach moving tracks (which
    # changes the objective itself, so it is not part of a descent)
    rounds: int = 3
    iterations: int = 25  # per round, at most
    # A round ends when an accepted step lowers the objective by less than this (relative)
    tolerance: float = 1e-5
    damping: float = 1e-3
    max_damping: float = 1e10


class _Call(nn.Module):
    """Calls a closure with the rig's parameters swapped (`torch.func.functional_call`)."""

    def __init__(self, rig: Rig) -> None:
        super().__init__()
        self.rig = rig

    def forward(self, fn: Callable[[], object]) -> object:
        return fn()


def supports(rig: Rig) -> bool:
    """Whether every free row is dataset or episode scope (no frame-scope walk)."""

    return not any(bool((c.active & (c.previous >= 0)).any()) for c in (rig.cameras, rig.bodies))


@dataclass(frozen=True, slots=True)
class _Columns:
    """The free unknowns as one vector: camera rows, body rows (6 each), then focals."""

    camera: Tensor[[int, 6]]  # column of each (row, dof), −1 where held
    body: Tensor[[int, 6]]
    focal: Tensor[[int]]
    count: int


def _columns(rig: Rig) -> _Columns:
    device = rig.log_focal.device
    next_column = 0
    tables = []
    for corrections in (rig.cameras, rig.bodies):
        table = torch.full((len(corrections.active), 6), -1, dtype=torch.long, device=device)
        rows = corrections.active.nonzero()[:, 0]
        table[rows] = next_column + torch.arange(6 * len(rows), device=device).reshape(-1, 6)
        next_column += 6 * len(rows)
        tables.append(table)
    focal = torch.full((len(rig.log_focal),), -1, dtype=torch.long, device=device)
    free = (rig.focal_std > 0).nonzero()[:, 0]
    focal[free] = next_column + torch.arange(len(free), device=device)
    return _Columns(tables[0], tables[1], focal, next_column + len(free))


def _apply(rig: Rig, columns: _Columns, x: Tensor[[int]]) -> dict[str, Tensor]:
    """The rig's parameters moved by `x` (differentiable in `x`)."""

    out = {}
    for name, corrections, table in (
        ("cameras.delta", rig.cameras, columns.camera),
        ("bodies.delta", rig.bodies, columns.body),
    ):
        flat = table.flatten()
        used = flat >= 0
        moved = (
            corrections.delta.detach().flatten().index_add(0, used.nonzero()[:, 0], x[flat[used]])
        )
        out[name] = moved.view_as(corrections.delta)
    used = columns.focal >= 0
    out["log_focal"] = rig.log_focal.detach().index_add(
        0, used.nonzero()[:, 0], x[columns.focal[used]]
    )
    return out


def _colors(
    rig: Rig, columns: _Columns, time: Tensor[[int]], track: Tensor[[int]]
) -> tuple[Tensor[[int, int]], Tensor[[int, int]]]:
    """Compressed directions: per color, its tangent `[K, P]` and, per observation, the
    column its entry belongs to `[K, N]` (−1: none)."""

    tangents, owners = [], []
    for corrections, table in ((rig.cameras, columns.camera), (rig.bodies, columns.body)):
        _add_element_colors(corrections, table, time, track, columns.count, tangents, owners)
    for column in columns.focal[columns.focal >= 0].tolist():
        tangent = torch.zeros(columns.count, dtype=rig.log_focal.dtype, device=time.device)
        tangent[column] = 1
        tangents.append(tangent)
        owners.append(torch.full_like(time, column))
    if not tangents:
        empty = torch.zeros(0, columns.count, dtype=rig.log_focal.dtype, device=time.device)
        return empty, torch.zeros(0, len(time), dtype=torch.long, device=time.device)
    return torch.stack(tangents), torch.stack(owners)


def _add_element_colors(
    corrections: Corrections,
    table: Tensor[[int, 6]],
    time: Tensor[[int]],
    track: Tensor[[int]],
    count: int,
    tangents: list[Tensor],
    owners: list[Tensor],
) -> None:
    for element in range(corrections.row.shape[0]):
        rows = (corrections.owner == element) & corrections.active
        if not bool(rows.any()):
            continue
        row_of = corrections.row[element, time]  # each observation's row of this element
        if _spanning(row_of, track):
            # A track sees several rows: compression is unsafe, one product per row and dof
            for row in rows.nonzero()[:, 0].tolist():
                for dof in range(6):
                    tangent = torch.zeros(count, dtype=corrections.delta.dtype, device=time.device)
                    tangent[table[row, dof]] = 1
                    tangents.append(tangent)
                    owners.append(torch.full_like(time, int(table[row, dof])))
            continue
        for dof in range(6):
            tangent = torch.zeros(count, dtype=corrections.delta.dtype, device=time.device)
            tangent[table[rows, dof]] = 1
            tangents.append(tangent)
            owners.append(table[row_of, dof])


def _spanning(row_of: Tensor[[int]], track: Tensor[[int]]) -> bool:
    """Whether any track's observations use more than one row."""

    count = int(track.max()) + 1
    low = torch.full((count,), 1 << 62, device=track.device).scatter_reduce(
        0, track, row_of, "amin", include_self=True
    )
    high = torch.full((count,), -1, device=track.device).scatter_reduce(
        0, track, row_of, "amax", include_self=True
    )
    used = high >= 0
    return bool((low[used] != high[used]).any())


def levenberg_marquardt(
    rig: Rig,
    residual_fn: Callable[[], tuple[Tensor[[int, 2]], Tensor[[int]]]],
    weight_fn: Callable[[Tensor[[int]]], Tensor[[int]]],
    time: Tensor[[int]],
    track: Tensor[[int]],
    steps: int,
    cauchy: float,
    config: LMConfig,
    reassign: Callable[[], None],
) -> Tensor[[int, 3]]:
    """Minimise the bundle objective over `rig`'s free rows in place; returns the history
    (total, data, prior) per accepted step.

    `residual_fn()`: per observation, its 2D residual (pixel units) and validity, computed from
    the rig's current parameters; `weight_fn(valid)`: per observation weight (0 to ignore);
    `time`, `track`: each observation's timestep and track; `steps`: total timesteps (the
    prior's normaliser);
    `reassign()`: re-attaches moving tracks (changes `weight_fn`).
    """

    columns = _columns(rig)
    if columns.count == 0:
        return torch.zeros(0, 3)
    tangents, owners = _colors(rig, columns, time, track)
    zero = torch.zeros(columns.count, dtype=rig.log_focal.dtype, device=time.device)

    call = _Call(rig)

    def swapped(x: Tensor[[int]]) -> dict[str, Tensor]:
        return {f"rig.{k}": v for k, v in _apply(rig, columns, x).items()}

    def at(x: Tensor[[int]]) -> tuple[Tensor[[int, 2]], Tensor[[int]]]:
        return torch.func.functional_call(call, swapped(x), (residual_fn,))

    def prior_at(x: Tensor[[int]]) -> Tensor[[int]]:
        return torch.func.functional_call(call, swapped(x), (rig.prior_residuals,))

    def objective(x: Tensor[[int]]) -> Tensor[[3]]:
        with torch.no_grad():
            difference, valid = at(x)
            w = weight_fn(valid)
            s = (difference**2).sum(-1) / cauchy**2
            data = (torch.log1p(s) * w).sum() / w.sum().clamp_min(1e-12)
            prior = (prior_at(x) ** 2).sum() / steps
            return torch.stack((data + prior, data, prior))

    history = []
    for round_ in range(config.rounds):
        if round_:
            reassign()
        history.append(objective(zero))
        damping = config.damping
        for _ in range(config.iterations):
            hessian, gradient = _normal_equations(
                at, prior_at, weight_fn, tangents, owners, zero, cauchy, steps
            )
            accepted = _damped_step(hessian, gradient, damping, history[-1][0], objective, config)
            if accepted is None:
                break
            step, value, damping = accepted
            _commit(rig, columns, step)
            improvement = float((history[-1][0] - value[0]) / history[-1][0].abs().clamp_min(1e-12))
            history.append(value)
            if improvement < config.tolerance:
                break
    return torch.stack(history).cpu()


def _damped_step(
    hessian: Tensor[[int, int]],
    gradient: Tensor[[int]],
    damping: float,
    current: Tensor[[]],
    objective: Callable[[Tensor], Tensor],
    config: LMConfig,
) -> tuple[Tensor[[int]], Tensor[[3]], float] | None:
    """The first damping (from `damping` up) whose step lowers the objective: (step, its
    objective, the next damping); None if none does."""

    diagonal = torch.diag(hessian.diagonal().clamp_min(1e-12))
    while damping <= config.max_damping:
        candidate = torch.linalg.solve(hessian + damping * diagonal, -gradient)
        value = objective(candidate)
        if bool(torch.isfinite(value[0])) and value[0] < current:
            return candidate, value, max(damping / 3, 1e-12)
        damping *= 4
    return None


def _normal_equations(
    at: Callable[[Tensor], tuple[Tensor, Tensor]],
    prior_at: Callable[[Tensor], Tensor],
    weight_fn: Callable[[Tensor], Tensor],
    tangents: Tensor[[int, int]],
    owners: Tensor[[int, int]],
    zero: Tensor[[int]],
    cauchy: float,
    steps: int,
) -> tuple[Tensor[[int, int]], Tensor[[int]]]:
    """IRLS normal equations at the current parameters (see module doc)."""

    count = len(zero)
    with torch.no_grad():
        difference, valid = at(zero)
        w = weight_fn(valid)
        s = (difference**2).sum(-1) / cauchy**2
        omega = w / (1 + s) / w.sum().clamp_min(1e-12)  # IRLS weight, normalised as the mean
    # One forward-mode product per color: [K, N, 2]
    products = torch.stack(
        [torch.func.jvp(lambda x: at(x)[0], (zero,), (t,))[1].detach() for t in tangents]
    )
    root = (omega.sqrt() / cauchy)[None, :, None]
    weighted = products * root  # √ω J / δ
    residual = difference * root[0]  # √ω r / δ
    hessian = torch.zeros(count, count, dtype=zero.dtype, device=zero.device)
    gradient = torch.zeros(count, dtype=zero.dtype, device=zero.device)
    # Observations sharing the same columns (one episode) form one dense block
    signatures, group = torch.unique(owners, dim=1, return_inverse=True)
    for g in range(signatures.shape[1]):
        cols = signatures[:, g]
        keep = (cols >= 0).nonzero()[:, 0]
        if len(keep) == 0:
            continue
        mine = group == g
        block = weighted[keep][:, mine].flatten(1)  # [k, n·2]
        index = cols[keep]
        hessian.index_put_((index[:, None], index[None, :]), block @ block.T, accumulate=True)
        gradient.index_add_(0, index, block @ residual[mine].flatten())
    # Priors: few residuals, an exact Jacobian
    prior_jacobian = torch.func.jacfwd(prior_at)(zero).detach()
    prior_residual = prior_at(zero).detach()
    hessian += prior_jacobian.T @ prior_jacobian / steps
    gradient += prior_jacobian.T @ prior_residual / steps
    return hessian, gradient


def _commit(rig: Rig, columns: _Columns, x: Tensor[[int]]) -> None:
    with torch.no_grad():
        moved = _apply(rig, columns, x)
        rig.cameras.delta.copy_(moved["cameras.delta"])
        rig.bodies.delta.copy_(moved["bodies.delta"])
        rig.log_focal.copy_(moved["log_focal"])
