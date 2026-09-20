"""Differentiable visual-anchor drift fit used by ACTM."""
from __future__ import annotations

from typing import Any

import numpy as np


def _weights(labels, n: int, sample_weights=None, group_data=None) -> np.ndarray:
    if sample_weights is not None:
        w = np.asarray(sample_weights, dtype=np.float64)
        if w.shape != (n,) or not np.isfinite(w).all() or np.any(w < 0):
            raise ValueError("BLOCKED_DRIFT_WEIGHTS")
        if not w.sum() > 0:
            raise ValueError("BLOCKED_DRIFT_WEIGHTS")
        return w / w.sum()
    if labels is None:
        # The real route passes class labels. For a label-free synthetic
        # snapshot, identical anchor rows form deterministic support groups;
        # weighting these groups equally avoids silently reverting to image
        # frequency weighting.
        if group_data is None:
            return np.full(n, 1.0 / n, dtype=np.float64)
        _, inverse, counts = np.unique(np.asarray(group_data), axis=0, return_inverse=True, return_counts=True)
        w = 1.0 / (len(counts) * counts[inverse])
        return w
    y = np.asarray(labels)
    if y.shape != (n,):
        raise ValueError("BLOCKED_DRIFT_LABELS")
    classes, counts = np.unique(y, return_counts=True)
    if not len(classes):
        raise ValueError("BLOCKED_DRIFT_LABELS")
    w = np.zeros(n, dtype=np.float64)
    for c, count in zip(classes, counts):
        w[y == c] = 1.0 / (len(classes) * count)
    return w


def fit_anchor_drift(
    v_teacher: Any,
    v_student_delta: Any,
    *,
    labels: Any = None,
    sample_weights: Any = None,
    rho: float | None = None,
    conditional: bool = True,
    norm_cap_B: float = 0.25,
    norm_cap_b: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit ``d ~= v B + b`` with class-balanced support weights."""
    v = np.asarray(v_teacher, dtype=np.float64)
    d = np.asarray(v_student_delta, dtype=np.float64)
    if v.ndim != 2 or d.ndim != 2 or v.shape[0] != d.shape[0] or not np.isfinite(v).all() or not np.isfinite(d).all():
        raise ValueError("BLOCKED_DRIFT_INPUT")
    w = _weights(labels, len(v), sample_weights, v)
    vbar, dbar = w @ v, w @ d
    vc, dc = v - vbar, d - dbar
    Cv = v.T @ (w[:, None] * vc)
    Cvd = v.T @ (w[:, None] * dc)
    if rho is None:
        rho = max(1e-4, 0.1 * float(np.trace(Cv)))
    if conditional:
        B = np.linalg.solve(Cv + float(rho) * np.eye(v.shape[1]), Cvd)
        n = np.linalg.norm(B)
        if n > norm_cap_B:
            B *= norm_cap_B / n
        b = dbar - vbar @ B
    else:
        B = np.zeros((v.shape[1], d.shape[1]), dtype=np.float64)
        b = dbar
    bn = np.linalg.norm(b)
    if bn > norm_cap_b:
        b = b * norm_cap_b / bn
    return B, b, {"rho": float(rho), "conditional": conditional, "weights": w.tolist(),
                  "B_norm": float(np.linalg.norm(B)), "b_norm": float(np.linalg.norm(b))}


def torch_anchor_drift(v_teacher: Any, d_student: Any, *, labels: Any = None,
                       sample_weights: Any = None, rho: float | None = None,
                       conditional: bool = True, norm_cap_B: float = 0.25,
                       norm_cap_b: float = 0.05):
    """Differentiable weighted support fit; gradients flow through d_student."""
    import torch
    if not isinstance(v_teacher, torch.Tensor) or not isinstance(d_student, torch.Tensor) or not d_student.requires_grad:
        raise ValueError("BLOCKED_STUDENT_GRADIENT")
    if v_teacher.ndim != 2 or d_student.ndim != 2 or v_teacher.shape[0] != d_student.shape[0]:
        raise ValueError("BLOCKED_DRIFT_INPUT")
    v = v_teacher.detach().to(dtype=torch.float64)
    d = d_student.to(dtype=torch.float64)
    n = d.shape[0]
    if sample_weights is not None:
        w = torch.as_tensor(sample_weights, dtype=torch.float64, device=d.device)
        if w.shape != (n,) or bool((w < 0).any()) or not bool((w.sum() > 0)):
            raise ValueError("BLOCKED_DRIFT_WEIGHTS")
        w = w / w.sum()
    elif labels is not None:
        y = torch.as_tensor(labels, device=d.device).long()
        classes, inv = torch.unique(y, sorted=True, return_inverse=True)
        counts = torch.bincount(inv, minlength=len(classes)).to(torch.float64)
        w = 1.0 / (len(classes) * counts[inv])
    else:
        w = torch.full((n,), 1.0 / n, dtype=torch.float64, device=d.device)
    vbar, dbar = (w[:, None] * v).sum(0), (w[:, None] * d).sum(0)
    vc, dc = v - vbar, d - dbar
    Cv = v.T @ (w[:, None] * vc)
    Cvd = v.T @ (w[:, None] * dc)
    rr = max(1e-4, 0.1 * float(torch.trace(Cv).detach())) if rho is None else float(rho)
    if conditional:
        eye = torch.eye(v.shape[1], dtype=torch.float64, device=d.device)
        B = torch.linalg.solve(Cv + rr * eye, Cvd)
        B = B * torch.clamp(norm_cap_B / torch.linalg.vector_norm(B).clamp_min(1e-12), max=1.0)
        b = dbar - vbar @ B
    else:
        B = d.new_zeros((v.shape[1], d.shape[1]), dtype=torch.float64)
        b = dbar
    b = b * torch.clamp(norm_cap_b / torch.linalg.vector_norm(b).clamp_min(1e-12), max=1.0)
    return B, b
