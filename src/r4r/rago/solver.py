"""Levenberg–Marquardt for the kinematic bundle: few unknowns, many observations.

The objective is the bundle's (`bundle.bundle_adjust`): the mean robust reprojection cost
`ρ(s) = log(1 + s)`, `s = |r|² / δ²`, plus the Gaussian priors. Each step linearises the 2D
residuals `r` (points re-triangulated in closed form inside them, as in the Adam loop) and
solves the reweighted normal equations (IRLS: weight `w / (1 + s)`, the robust cost's slope):

    (H + λ diag H) Δ = −g,   H = Σ ω Jᵀ J + Jₚᵀ Jₚ / T,   g = Σ ω Jᵀ r + Jₚᵀ rₚ / T

with `ω` the IRLS weights `ρ'(s) · w / Σ w`, `J = ∂r/∂x`, `(Jₚ, rₚ)` the priors' Jacobian
and residuals (`rig.prior_residuals`) and `T` = `steps`, the prior's normaliser. Δ is accepted
only if the true robust objective decreases, and then λ ← λ/3; otherwise λ ← 4λ and the step
is retried (up to `max_damping`). Tens of steps replace the Adam loop's ~1000.

The Jacobian of the residuals comes from forward-mode products, compressed: an observation
depends on a single row of each scoped element (the dataset row, or its episode's row: tracks
never span episodes), so all rows of one (element, degree of freedom) share one product, and
each observation's entry is credited to its own row's column. The cost is one product per
(element, dof) and free dataset-wide value (focal, joint offset, time offset), whatever the
number of episodes. Frame-scope rows (a mobile base per frame) break this (a track spans
frames): `supports` says no, and the caller uses Adam.

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
    from shape_extensions import IntVar
    from torch import Tensor

    from r4r.types import UV, VxVyVzWxWyWz

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

    def __init__(self, rig: Rig[int, int, int]) -> None:
        super().__init__()
        self.rig = rig

    def forward[R](self, fn: Callable[[], R]) -> R:
        return fn()


def supports(rig: Rig[int, int, int]) -> bool:
    """Whether every free row is dataset or episode scope (no frame-scope walk)."""

    return not any(bool((c.active & (c.previous >= 0)).any()) for c in (rig.cameras, rig.bodies))


@dataclass(frozen=True, slots=True)
class _Columns:
    """The free unknowns as one vector: camera rows, body rows (6 each), then the free entries
    of the dataset-wide vectors (`VECTORS`)."""

    camera: Tensor[[int, *VxVyVzWxWyWz]]  # column of each (row, dof), −1 where held
    body: Tensor[[int, *VxVyVzWxWyWz]]
    vectors: dict[str, Tensor[[int]]]  # per vector: column of each entry, −1 where held
    count: int


# Dataset-wide parameter vectors of the rig and the std buffer saying which entries are free
VECTORS = {
    "log_focal": "focal_std",
    "joint_offset": "joint_std",
    "time_offset": "time_std",
    "distortion": "distortion_std",  # [C, k], absent for ideal pinholes
}


def _vectors(rig: Rig[int, int, int]) -> list[str]:
    return [name for name in VECTORS if getattr(rig, name, None) is not None]


def _columns(rig: Rig[int, int, int]) -> _Columns:
    device = rig.log_focal.device
    next_column = 0
    tables = []
    for corrections in (rig.cameras, rig.bodies):
        table = torch.full((len(corrections.active), 6), -1, dtype=torch.long, device=device)
        free = corrections.free_dof
        table[free] = next_column + torch.arange(int(free.sum()), device=device)
        next_column += int(free.sum())
        tables.append(table)
    vectors = {}
    for name in _vectors(rig):
        std = VECTORS[name]
        column = torch.full((getattr(rig, name).numel(),), -1, dtype=torch.long, device=device)
        free = (getattr(rig, std).flatten() > 0).nonzero()[:, 0]
        column[free] = next_column + torch.arange(len(free), device=device)
        next_column += len(free)
        vectors[name] = column
    return _Columns(tables[0], tables[1], vectors, next_column)


def _apply(rig: Rig[int, int, int], columns: _Columns, x: Tensor[[int]]) -> dict[str, Tensor]:
    """The rig's parameters moved by `x` (differentiable in `x`), by name (each its own rank)."""

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
    for name, column in columns.vectors.items():
        used = column >= 0
        value = getattr(rig, name).detach()
        moved = value.flatten().index_add(0, used.nonzero()[:, 0], x[column[used]])
        out[name] = moved.view_as(value)
    return out


def _colors[M: IntVar](
    rig: Rig[int, int, int], columns: _Columns, time: Tensor[[M]], track: Tensor[[M]]
) -> tuple[Tensor[[int, int]], Tensor[[int, M]]]:
    """Compressed directions: per color, its tangent `[K, P]` and, per observation, the
    column its entry belongs to `[K, N]` (−1: none)."""

    tangents, owners = [], []
    for corrections, table in ((rig.cameras, columns.camera), (rig.bodies, columns.body)):
        _add_element_colors(corrections, table, time, track, columns.count, tangents, owners)
    vector_columns = torch.cat(list(columns.vectors.values()))
    for column in vector_columns[vector_columns >= 0].tolist():
        tangent = torch.zeros(columns.count, dtype=rig.log_focal.dtype, device=time.device)
        tangent[column] = 1
        tangents.append(tangent)
        owners.append(torch.full_like(time, column))
    if not tangents:
        empty = torch.zeros(0, columns.count, dtype=rig.log_focal.dtype, device=time.device)
        return empty, torch.zeros(0, len(time), dtype=torch.long, device=time.device)
    return torch.stack(tangents), torch.stack(owners)


def _add_element_colors[M: IntVar](
    corrections: Corrections[int],
    table: Tensor[[int, *VxVyVzWxWyWz]],
    time: Tensor[[M]],
    track: Tensor[[M]],
    count: int,
    tangents: list[Tensor[[int]]],
    owners: list[Tensor[[M]]],
) -> None:
    for element in range(corrections.row.shape[0]):
        rows = (corrections.owner == element) & corrections.active
        if not bool(rows.any()):
            continue
        row_of = corrections.row[element, time]  # each observation's row of this element
        if _spanning(row_of, track):
            # A track sees several rows: compression is unsafe, one product per row and dof
            for row in rows.nonzero()[:, 0].tolist():
                for dof in (table[row] >= 0).nonzero()[:, 0].tolist():
                    tangent = torch.zeros(count, dtype=corrections.delta.dtype, device=time.device)
                    tangent[table[row, dof]] = 1
                    tangents.append(tangent)
                    owners.append(torch.full_like(time, int(table[row, dof])))
            continue
        for dof in range(6):
            columns = table[rows, dof]
            if not bool((columns >= 0).any()):
                continue
            tangent = torch.zeros(count, dtype=corrections.delta.dtype, device=time.device)
            tangent[columns[columns >= 0]] = 1
            tangents.append(tangent)
            owners.append(table[row_of, dof])


def _spanning[M: IntVar](row_of: Tensor[[M]], track: Tensor[[M]]) -> bool:
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


def levenberg_marquardt[M: IntVar](
    rig: Rig[int, int, int],
    residual_fn: Callable[[], tuple[Tensor[[M, *UV]], Tensor[[M]]]],
    weight_fn: Callable[[Tensor[[M]]], Tensor[[M]]],
    time: Tensor[[M]],
    track: Tensor[[M]],
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

    def at(x: Tensor[[int]]) -> tuple[Tensor[[M, *UV]], Tensor[[M]]]:
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
        if not bool(torch.isfinite(history[-1]).all()):
            raise ValueError(
                f"bundle objective not finite at the start of round {round_}: "
                f"(total, data, prior) = {history[-1].tolist()}"
            )
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
    objective: Callable[[Tensor[[int]]], Tensor[[3]]],
    config: LMConfig,
) -> tuple[Tensor[[int]], Tensor[[3]], float] | None:
    """The first damping (from `damping` up, ×4 per rejection) whose step
    `Δ = −(H + λ diag H)⁻¹ g` lowers the objective: (Δ, its objective, λ/3 for the next step);
    None if none does up to `max_damping`."""

    # Parameters the data do not constrain (zero rows) keep a damping relative to the others,
    # so the damped system stays solvable
    scale = hessian.diagonal().abs().max().clamp_min(1e-12)
    diagonal = torch.diag(hessian.diagonal().clamp_min(1e-9 * scale))
    while damping <= config.max_damping:
        candidate, info = torch.linalg.solve_ex(hessian + damping * diagonal, -gradient)
        if int(info) != 0 or not bool(torch.isfinite(candidate).all()):
            damping *= 4
            continue
        value = objective(candidate)
        if bool(torch.isfinite(value[0])) and value[0] < current:
            return candidate, value, max(damping / 3, 1e-12)
        damping *= 4
    return None


def _normal_equations[M: IntVar](
    at: Callable[[Tensor[[int]]], tuple[Tensor[[M, *UV]], Tensor[[M]]]],
    prior_at: Callable[[Tensor[[int]]], Tensor[[int]]],
    weight_fn: Callable[[Tensor[[M]]], Tensor[[M]]],
    tangents: Tensor[[int, int]],
    owners: Tensor[[int, M]],
    zero: Tensor[[int]],
    cauchy: float,
    steps: int,
) -> tuple[Tensor[[int, int]], Tensor[[int]]]:
    """IRLS normal equations `(H, g)` at the current parameters (see module doc)."""

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


def _commit(rig: Rig[int, int, int], columns: _Columns, x: Tensor[[int]]) -> None:
    with torch.no_grad():
        moved = _apply(rig, columns, x)
        rig.cameras.delta.copy_(moved["cameras.delta"])
        rig.bodies.delta.copy_(moved["bodies.delta"])
        for name in columns.vectors:
            getattr(rig, name).copy_(moved[name])
