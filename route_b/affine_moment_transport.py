"""Exact homogeneous transport of class-balanced moments (NumPy and Torch)."""
from __future__ import annotations

import numpy as np


def homogeneous_matrix(B, b, *, dim_a: int = 1536, dim_u: int = 512):
    """Build H while preserving an autograd path when B/b are torch tensors."""
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(B, torch.Tensor):
        if B.shape != (dim_u, dim_a) or b.shape != (dim_a,):
            raise ValueError("BLOCKED_AFFINE_SHAPE")
        H = torch.eye(dim_a + dim_u + 1, dtype=B.dtype, device=B.device)
        H = H.clone()
        H[:dim_a, dim_a:dim_a + dim_u] = B.T
        H[:dim_a, -1] = b
        return H
    B = np.asarray(B, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if B.shape != (dim_u, dim_a) or b.shape != (dim_a,):
        raise ValueError("BLOCKED_AFFINE_SHAPE")
    H = np.eye(dim_a + dim_u + 1, dtype=np.float64)
    H[:dim_a, dim_a:dim_a + dim_u] = B.T
    H[:dim_a, -1] = b
    return H


def transport_homogeneous(bar_S, bar_M, H):
    """Apply H to [h;1], retaining all a-u cross moments."""
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(H, torch.Tensor):
        if bar_S.shape[0] != H.shape[0] or bar_S.shape[0] != bar_S.shape[1] or bar_M.shape[0] != H.shape[0]:
            raise ValueError("BLOCKED_TRANSPORT_SHAPE")
        outS = H @ bar_S @ H.T
        outM = H @ bar_M
        return outS, outM
    S = np.asarray(bar_S, dtype=np.float64); M = np.asarray(bar_M, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    if S.shape[0] != S.shape[1] or S.shape[0] != H.shape[0] or M.shape[0] != H.shape[0]:
        raise ValueError("BLOCKED_TRANSPORT_SHAPE")
    return H @ S @ H.T, H @ M


def transport_old_bank(S, M, B, b, *, dim_a: int = 1536, dim_u: int = 512):
    """Transport true homogeneous bank moments and return class averages.

    The full unnormalized result is available through
    :func:`transport_homogeneous`; this compatibility helper returns the
    top-left and top rows divided by K for ordinary ridge consumers.
    """
    H = homogeneous_matrix(B, b, dim_a=dim_a, dim_u=dim_u)
    if S.shape[0] != H.shape[0] or M.shape[0] != H.shape[0]:
        raise ValueError("BLOCKED_TRANSPORT_SHAPE")
    newS, newM = transport_homogeneous(S, M, H)
    K = M.shape[1]
    return newS[:-1, :-1] / K, newM[:-1] / K


def transported_score_covariance(L, old_cov, W):
    """Score-space covariance ``W^T L C L^T W`` with NumPy/Torch dispatch."""
    return (L.T @ W).T @ old_cov @ (L.T @ W)
