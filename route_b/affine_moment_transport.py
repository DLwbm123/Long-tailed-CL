"""Exact homogeneous transport of class-balanced moments."""
from __future__ import annotations

import numpy as np


def homogeneous_matrix(B: np.ndarray, b: np.ndarray, *, dim_a: int = 1536, dim_u: int = 512) -> np.ndarray:
    B = np.asarray(B, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if B.shape != (dim_u, dim_a) or b.shape != (dim_a,):
        raise ValueError("BLOCKED_AFFINE_SHAPE")
    H = np.eye(dim_a + dim_u + 1, dtype=np.float64)
    H[:dim_a, dim_a:dim_a + dim_u] = B.T
    H[:dim_a, -1] = b
    return H


def transport_homogeneous(bar_S: np.ndarray, bar_M: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``H`` to [h;1] and retain all a-u cross moments."""
    S = np.asarray(bar_S, dtype=np.float64); M = np.asarray(bar_M, dtype=np.float64)
    if S.shape[0] + 1 != H.shape[0] or S.shape[0] != S.shape[1] or M.shape[0] != S.shape[0]:
        raise ValueError("BLOCKED_TRANSPORT_SHAPE")
    K = M.shape[1]
    augS = np.zeros((S.shape[0] + 1, S.shape[0] + 1), dtype=np.float64)
    augS[:-1, :-1] = S
    augS[:-1, -1] = augS[-1, :-1] = M.mean(1)
    augS[-1, -1] = 1.0
    outS = (H @ augS @ H.T)[:-1, :-1]
    outM = (H @ np.vstack((M, np.ones((1, K)))) )[:-1]
    return (outS + outS.T) / 2, outM


def transport_old_bank(S: np.ndarray, M: np.ndarray, B: np.ndarray, b: np.ndarray,
                       *, dim_a: int = 1536, dim_u: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Transport homogeneous (already class-averaged) bank moments."""
    return transport_homogeneous(S, M, homogeneous_matrix(B, b, dim_a=dim_a, dim_u=dim_u))
