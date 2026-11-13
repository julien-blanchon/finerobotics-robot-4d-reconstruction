"""Batched robust relative rotations from two-view correspondences (essential matrices).

Minimal-sample hypotheses (8-point, as in RANSAC) pick robust starting weights, then weighted
8-point inside IRLS on the Sampson error (Cauchy weights), projected on the essential
manifold, decomposed with a cheirality vote. All pairs and hypotheses are solved at once.

Ref: Hartley & Zisserman, *Multiple View Geometry*, §9.6 and §11.4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

# Batched eigh/svd on the GPU fail on very large batches: solve in chunks
CHUNK = 4096
IRLS_ITERATIONS = 12
HYPOTHESES = 64


def relative_rotations[P: IntVar, N: IntVar](
    pixels_a: Tensor[[P, N, 2]],
    pixels_b: Tensor[[P, N, 2]],
    mask: Tensor[[P, N]],
    intrinsics: Tensor[[3, 3]],
    cauchy: float = 1.0,
) -> Tensor[[P, 3, 3]]:
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


def _essential_irls[P: IntVar, N: IntVar](
    xa: Tensor[[P, N, 3]],
    xb: Tensor[[P, N, 3]],
    weight: Tensor[[P, N]],
    focal: float,
    cauchy: float,
) -> tuple[Tensor[[P, 3, 3]], Tensor[[P, N]]]:
    rows = (xb[..., :, None] * xa[..., None, :]).flatten(-2)  # [P, N, 9]: xbᵀ E xa = rows · vec E

    # Starting weights from the best of HYPOTHESES minimal samples per pair (most inliers)
    generator = torch.Generator(device=xa.device).manual_seed(0)
    draw = torch.rand(len(xa), HYPOTHESES, xa.shape[1], generator=generator, device=xa.device)
    sample = (draw + (weight[:, None] <= 0)).topk(8, dim=-1, largest=False).indices  # valid first
    picked = torch.gather(
        rows[:, None].expand(-1, HYPOTHESES, -1, -1), 2, sample[..., None].expand(-1, -1, -1, 9)
    )
    candidates = _project(_smallest(picked.mT @ picked))  # [P, H, 3, 3]
    error = _sampson(candidates, xa[:, None], xb[:, None], focal)  # [P, H, N]
    inliers = ((error < cauchy) * weight[:, None]).sum(-1)
    best = inliers.argmax(-1)
    error = error[torch.arange(len(xa), device=xa.device), best]
    robust = weight / (1 + (error / cauchy) ** 2)

    essential = candidates[torch.arange(len(xa), device=xa.device), best]
    for _ in range(IRLS_ITERATIONS):
        essential = _project(_smallest((rows * robust[..., None]).mT @ rows))
        robust = weight / (1 + (_sampson(essential, xa, xb, focal) / cauchy) ** 2)
    return essential, robust


def _smallest(normal: Tensor) -> Tensor:
    """Unit eigenvector of the smallest eigenvalue of [..., 9, 9] matrices, as [..., 3, 3]."""

    flat = normal.reshape(-1, 9, 9)
    vector = torch.cat([torch.linalg.eigh(c)[1][..., 0] for c in flat.split(CHUNK)])
    return vector.reshape(*normal.shape[:-2], 3, 3)


def _project(matrix: Tensor) -> Tensor:
    """Closest essential matrix (singular values 1, 1, 0) of [..., 3, 3] matrices."""

    flat = matrix.reshape(-1, 3, 3)
    u, _, vh = _svd(flat)
    project = torch.diag(flat.new_tensor([1.0, 1.0, 0.0]))
    return (u @ project @ vh).reshape(matrix.shape)


def _sampson(essential: Tensor, xa: Tensor, xb: Tensor, focal: float) -> Tensor:
    """Sampson distance in pixels of correspondences [..., N, 3] under [..., 3, 3] matrices."""

    ex = xa @ essential.mT
    etx = xb @ essential
    residual = (xb * ex).sum(-1)
    norm = ex[..., 0] ** 2 + ex[..., 1] ** 2 + etx[..., 0] ** 2 + etx[..., 1] ** 2
    return residual.abs() / norm.clamp_min(1e-18).sqrt() * focal


def _rotation[P: IntVar, N: IntVar](
    essential: Tensor[[P, 3, 3]],
    xa: Tensor[[P, N, 3]],
    xb: Tensor[[P, N, 3]],
    weight: Tensor[[P, N]],
) -> Tensor[[P, 3, 3]]:
    """The rotation of the four (R, ±t) decompositions with most weight in front of both views."""

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


def _svd[P: IntVar](
    m: Tensor[[P, 3, 3]],
) -> tuple[Tensor[[P, 3, 3]], Tensor[[P, 3]], Tensor[[P, 3, 3]]]:
    parts = [torch.linalg.svd(c) for c in m.split(CHUNK)]
    u, s, vh = (torch.cat(x) for x in zip(*parts, strict=True))
    return u, s, vh
