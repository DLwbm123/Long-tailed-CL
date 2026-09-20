"""Minimal differentiable ACTM episode path for synthetic qualification.

This is deliberately separate from the NumPy snapshot bank. It keeps the
student feature graph alive through drift fitting, affine transport, memory
scores, and the final loss; no image loader or checkpoint side effect is
implicit.
"""
from __future__ import annotations

from typing import Any

import torch

from route_b.anchor_drift import torch_anchor_drift
from route_b.affine_moment_transport import homogeneous_matrix
from route_b.margin_memory_loss import current_query_risk, memory_margin_loss


def actm_episode_loss(
    student_a: torch.Tensor,
    teacher_a: torch.Tensor,
    anchor_u: torch.Tensor,
    labels: torch.Tensor,
    W: torch.Tensor,
    old_mean: torch.Tensor,
    old_cov: torch.Tensor,
    *,
    variant: str,
    fd_loss: torch.Tensor | None = None,
    fd_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute one B00/B01/B10/B11 episode with all differentiable paths."""
    if variant not in {"B00", "B01", "B10", "B11"}:
        raise ValueError("BLOCKED_ACTM_VARIANT")
    if not student_a.requires_grad:
        raise ValueError("BLOCKED_STUDENT_GRADIENT")
    if teacher_a.shape != student_a.shape or anchor_u.shape[0] != student_a.shape[0]:
        raise ValueError("BLOCKED_ACTM_FEATURE_SHAPE")
    conditional = variant in {"B10", "B11"}
    with torch.no_grad():
        teacher = teacher_a.detach() / (2.0 ** .5)
    v = anchor_u / (2.0 ** .5)
    delta = student_a / (2.0 ** .5) - teacher
    B, b = torch_anchor_drift(v, delta, labels=labels, conditional=conditional)
    H = homogeneous_matrix(B, b, dim_a=student_a.shape[1], dim_u=anchor_u.shape[1])
    L = H[:-1, :-1]
    transported_mean = L @ old_mean + H[:-1, -1, None]
    transported_cov = L @ old_cov @ L.T
    h = torch.cat((student_a / (2.0 ** .5), v), dim=1)
    query_logits = h @ W
    query = current_query_risk(query_logits, labels)
    memory = memory_margin_loss(W, transported_mean, transported_cov) if variant in {"B01", "B11"} else query.new_zeros(())
    fd = ((student_a - teacher_a) ** 2).mean() if fd_loss is None else fd_loss
    total = query + memory + fd_weight * fd
    return total, {"query": query, "memory": memory, "fd": fd, "B": B, "b": b}
