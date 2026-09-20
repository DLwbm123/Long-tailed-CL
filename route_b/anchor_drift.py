"""Differentiable visual-anchor drift fit used by ACTM."""
from __future__ import annotations

from typing import Any

import numpy as np


def fit_anchor_drift(
    v_teacher: Any,
    v_student_delta: Any,
    *,
    rho: float | None = None,
    conditional: bool = True,
    norm_cap_B: float = 0.25,
    norm_cap_b: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit ``d ~= v B + b`` and return B (512x1536), b (1536).

    This NumPy implementation is for the support snapshot.  During the
    differentiable episode callers should compute the same equation with
    torch tensors; no detached student feature is accepted by this helper.
    """
    v = np.asarray(v_teacher, dtype=np.float64)
    d = np.asarray(v_student_delta, dtype=np.float64)
    if v.ndim != 2 or d.ndim != 2 or v.shape[0] != d.shape[0] or not np.isfinite(v).all() or not np.isfinite(d).all():
        raise ValueError("BLOCKED_DRIFT_INPUT")
    vc = v - v.mean(0, keepdims=True)
    dc = d - d.mean(0, keepdims=True)
    Cv = vc.T @ vc / len(v)
    if rho is None:
        rho = max(1e-4, 0.1 * float(np.trace(Cv)))
    if conditional:
        B = np.linalg.solve(Cv + float(rho) * np.eye(v.shape[1]), vc.T @ dc / len(v))
        n = np.linalg.norm(B)
        if n > norm_cap_B:
            B *= norm_cap_B / n
        b = d.mean(0) - v.mean(0) @ B
    else:
        B = np.zeros((v.shape[1], d.shape[1]), dtype=np.float64)
        b = d.mean(0)
    bn = np.linalg.norm(b)
    if bn > norm_cap_b:
        b = b * norm_cap_b / bn
    return B, b, {"rho": float(rho), "conditional": conditional,
                  "B_norm": float(np.linalg.norm(B)), "b_norm": float(np.linalg.norm(b))}


def torch_anchor_drift(v_teacher: Any, d_student: Any, *, rho: float | None = None, conditional: bool = True,
                       norm_cap_B: float = 0.25, norm_cap_b: float = 0.05):
    """Differentiable support fit; gradients flow through ``d_student``."""
    import torch
    v = v_teacher
    d = d_student
    if not isinstance(v, torch.Tensor) or not isinstance(d, torch.Tensor) or d.requires_grad is False:
        raise ValueError("BLOCKED_STUDENT_GRADIENT")
    vc = v - v.mean(0, keepdim=True)
    dc = d - d.mean(0, keepdim=True)
    Cv = vc.T @ vc / v.shape[0]
    rr = max(1e-4, 0.1 * float(torch.trace(Cv).detach())) if rho is None else float(rho)
    if conditional:
        B = torch.linalg.solve(Cv + rr * torch.eye(v.shape[1], device=v.device, dtype=v.dtype), vc.T @ dc / v.shape[0])
        B = B * torch.clamp(norm_cap_B / torch.linalg.vector_norm(B).clamp_min(1e-12), max=1.0)
        b = d.mean(0) - v.mean(0) @ B
    else:
        B = d.new_zeros((v.shape[1], d.shape[1]))
        b = d.mean(0)
    b = b * torch.clamp(norm_cap_b / torch.linalg.vector_norm(b).clamp_min(1e-12), max=1.0)
    return B, b
