"""Batched robust relative rotations from two-view correspondences (essential matrices).

Minimal-sample hypotheses (8-point, as in RANSAC) pick robust starting weights, then weighted
8-point inside IRLS on the Sampson error (Cauchy weights), projected on the essential
manifold, decomposed with a cheirality vote. All pairs and hypotheses are solved at once.

Epipolar constraint on normalised rays `x = K⁻¹ (u, v, 1)`: `x_bᵀ E x_a = 0`, `E = [t]× R`.

Ref: Hartley & Zisserman, *Multiple View Geometry*, §9.6 and §11.4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from einops import rearrange

if TYPE_CHECKING:
    from shape_extensions import IntTuple, IntVar
    from torch import Tensor

    from r4r.types import UV, XYZ, Mat3x3

# Batched eigh/svd on the GPU fail on very large batches: solve in chunks
CHUNK = 4096
IRLS_ITERATIONS = 12
HYPOTHESES = 64


def relative_rotations[M: IntVar, N: IntVar](
    pixels_a: Tensor[[M, N, *UV]],
    pixels_b: Tensor[[M, N, *UV]],
    mask: Tensor[[M, N]],
    intrinsics: Tensor[[*Mat3x3]],
    cauchy: float = 1.0,
) -> Tensor[[M, *Mat3x3]]:
    """Per pair: the camera rotation `b_from_a` (x_b ∝ R x_a + t), robust to outliers.

    `cauchy` is the Sampson-error scale in pixels.
    """

    inverse = torch.linalg.inv(intrinsics.to(pixels_a))
    ones = torch.ones_like(pixels_a[..., :1])
    xa = torch.cat((pixels_a, ones), -1) @ inverse.mT
    xb = torch.cat((pixels_b, ones), -1) @ inverse.mT
    weight = mask.to(pixels_a.dtype)
    essential, weight = _essential_irls(xa, xb, weight, float(intrinsics[0, 0]), cauchy)
    return _rotation(essential, xa, xb, weight)


def _essential_irls[M: IntVar, N: IntVar](
    xa: Tensor[[M, N, *XYZ]],
    xb: Tensor[[M, N, *XYZ]],
    weight: Tensor[[M, N]],
    focal: float,
    cauchy: float,
) -> tuple[Tensor[[M, *Mat3x3]], Tensor[[M, N]]]:
    """Weighted 8-point inside IRLS: `E = argmin_{‖E‖=1} Σ w_i (x_bᵢᵀ E x_aᵢ)²`, then
    `w_i = mask_i / (1 + (e_i / c)²)` from the Sampson errors `e_i` (pixels), `c = cauchy`.
    Returns the essential matrices and the final robust weights.
    """

    # xbᵀ E xa = rows · vec E
    rows = rearrange(xb[..., :, None] * xa[..., None, :], "m n i j -> m n (i j)")

    # Starting weights from the best of HYPOTHESES minimal samples per pair (most inliers)
    generator = torch.Generator(device=xa.device).manual_seed(0)
    draw = torch.rand(len(xa), HYPOTHESES, xa.shape[1], generator=generator, device=xa.device)
    sample = (draw + (weight[:, None] <= 0)).topk(8, dim=-1, largest=False).indices  # valid first
    picked = torch.gather(
        rows[:, None].expand(-1, HYPOTHESES, -1, -1), 2, sample[..., None].expand(-1, -1, -1, 9)
    )
    candidates = _project(_smallest(picked.mT @ picked))  # [M, HYPOTHESES, 3, 3]
    error = _sampson(candidates, xa[:, None], xb[:, None], focal)  # [M, HYPOTHESES, N]
    inliers = ((error < cauchy) * weight[:, None]).sum(-1)
    best = inliers.argmax(-1)
    error = error[torch.arange(len(xa), device=xa.device), best]
    robust = weight / (1 + (error / cauchy) ** 2)

    essential = candidates[torch.arange(len(xa), device=xa.device), best]
    for _ in range(IRLS_ITERATIONS):
        essential = _project(_smallest((rows * robust[..., None]).mT @ rows))
        robust = weight / (1 + (_sampson(essential, xa, xb, focal) / cauchy) ** 2)
    return essential, robust


def _smallest[Bs: IntTuple](normal: Tensor[[*Bs, 9, 9]]) -> Tensor[[*Bs, 3, 3]]:
    """Unit eigenvector of the smallest eigenvalue of `AᵀA` (the least-squares null vector),
    as a 3 × 3 matrix."""

    flat = normal.reshape(-1, 9, 9)
    vector = torch.cat([torch.linalg.eigh(c)[1][..., 0] for c in flat.split(CHUNK)])
    # Unflattening the batch back to *Bs is beyond the stubs
    return cast("Tensor[[*Bs, 3, 3]]", vector.reshape(*normal.shape[:-2], 3, 3))


def _project[Bs: IntTuple](matrix: Tensor[[*Bs, 3, 3]]) -> Tensor[[*Bs, 3, 3]]:
    """Closest essential matrix in Frobenius norm: `U diag(1, 1, 0) Vᵀ` for `M = U S Vᵀ`."""

    flat = matrix.reshape(-1, 3, 3)
    u, _, vh = _svd(flat)
    project = torch.diag(flat.new_tensor([1.0, 1.0, 0.0]))
    return cast("Tensor[[*Bs, 3, 3]]", (u @ project @ vh).reshape(matrix.shape))


def _sampson[Bs: IntTuple, Ps: IntTuple, N: IntVar](
    essential: Tensor[[*Bs, 3, 3]], xa: Tensor[[*Ps, N, 3]], xb: Tensor[[*Ps, N, 3]], focal: float
) -> Tensor[[*Bs, N]]:
    """Sampson distance in pixels: `f · |x_bᵀ E x_a| / ‖(E x_a)₁,₂, (Eᵀ x_b)₁,₂‖`.

    The rays' batch `Ps` broadcasts against `essential`'s `Bs` (hypotheses share one set of rays).
    """

    ex = xa @ essential.mT
    etx = xb @ essential
    residual = (xb * ex).sum(-1)
    norm = ex[..., 0] ** 2 + ex[..., 1] ** 2 + etx[..., 0] ** 2 + etx[..., 1] ** 2
    # Broadcasting *Ps against *Bs is beyond the stubs
    return cast("Tensor[[*Bs, N]]", residual.abs() / norm.clamp_min(1e-18).sqrt() * focal)


def _rotation[M: IntVar, N: IntVar](
    essential: Tensor[[M, *Mat3x3]],
    xa: Tensor[[M, N, *XYZ]],
    xb: Tensor[[M, N, *XYZ]],
    weight: Tensor[[M, N]],
) -> Tensor[[M, *Mat3x3]]:
    """The rotation of the four decompositions `R ∈ {U W Vᵀ, U Wᵀ Vᵀ}`, `t = ±u₃` with most
    weight in front of both views (cheirality: both depths `z_a`, `z_b` positive)."""

    u, _, vh = _svd(essential)
    u = u * torch.det(u)[..., None, None].sign()
    vh = vh * torch.det(vh)[..., None, None].sign()
    w = essential.new_tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    translation = u[..., :, 2]
    best_score = essential.new_full(essential.shape[:-2], -1.0)
    best = u @ w @ vh
    for rotation in (u @ w @ vh, u @ w.mT @ vh):
        rotated = xa @ rotation.mT
        for sign in (1.0, -1.0):
            t = sign * translation
            # Depths (z_a, z_b) with z_b xb = z_a R xa + t, by least squares
            a11 = (rotated * rotated).sum(-1)
            a12 = -(rotated * xb).sum(-1)
            a22 = (xb * xb).sum(-1)
            b1 = -(rotated * t[..., None, :]).sum(-1)
            b2 = (xb * t[..., None, :]).sum(-1)
            det = a11 * a22 - a12 * a12
            za = (a22 * b1 - a12 * b2) / det
            zb = (a11 * b2 - a12 * b1) / det
            score = (weight * ((za > 0) & (zb > 0)).to(weight.dtype)).sum(-1)
            better = score > best_score
            best_score = torch.where(better, score, best_score)
            best = torch.where(better[..., None, None], rotation, best)
    return best


def _svd[M: IntVar](
    m: Tensor[[M, *Mat3x3]],
) -> tuple[Tensor[[M, *Mat3x3]], Tensor[[M, *XYZ]], Tensor[[M, *Mat3x3]]]:
    """Batched SVD in chunks of `CHUNK` (large GPU batches fail)."""
    parts = [torch.linalg.svd(c) for c in m.split(CHUNK)]
    u, s, vh = (torch.cat(x) for x in zip(*parts, strict=True))
    return u, s, vh
