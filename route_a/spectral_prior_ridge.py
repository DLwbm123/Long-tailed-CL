"""Float64 spectral prior ridge solves and the complete A0--A8 readout set."""
from __future__ import annotations

from typing import Mapping

import numpy as np


def ridge(S: np.ndarray, M: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    S = np.asarray(S, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or M.shape[0] != S.shape[0]:
        raise ValueError("BLOCKED_RIDGE_SHAPE")
    return np.linalg.solve((S + S.T) / 2 + lam * np.eye(S.shape[0]), M)


def spectral_prior_ridge(
    S: np.ndarray,
    M: np.ndarray,
    V: np.ndarray,
    gamma: np.ndarray | float,
    *,
    lam: float = 1e-3,
    identity_prior: bool = False,
) -> tuple[np.ndarray, dict]:
    """Solve ``(G+lam I+gamma_c P)w_c=R_c+gamma_c Pv_c`` in eigenspace."""
    S, M, V = map(lambda x: np.asarray(x, dtype=np.float64), (S, M, V))
    K = M.shape[1]
    if V.shape != M.shape or S.shape[0] != S.shape[1] or S.shape[0] != M.shape[0]:
        raise ValueError("BLOCKED_RASP_SHAPE")
    g = np.broadcast_to(np.asarray(gamma, dtype=np.float64), (K,))
    G = (S / K + (S / K).T) / 2
    R = M / K
    sigma, U = np.linalg.eigh(G)
    sigma = np.maximum(sigma, 0.0)
    p = lam / (sigma + lam)
    if identity_prior:
        P = np.eye(G.shape[0])
        # The identity prior is not diagonal in the eigenspace unless the
        # eigensystem is orthonormal (it is), so its spectral coefficients are
        # all one.
        p = np.ones_like(sigma)
    else:
        with np.errstate(all="ignore"):
            P = (U * p) @ U.T
    W = np.empty_like(R)
    for c in range(K):
        rhs = R[:, c] + g[c] * (P @ V[:, c])
        W[:, c] = U @ ((U.T @ rhs) / (sigma + lam + g[c] * p))
    residual = max(float(np.linalg.norm((G + lam * np.eye(G.shape[0]) + g[c] * P) @ W[:, c] -
                                        (R[:, c] + g[c] * P @ V[:, c]))) for c in range(K))
    if not np.isfinite(W).all() or residual > 1e-8:
        raise ValueError("BLOCKED_RASP_SOLVE")
    return W, {"lambda": lam, "gamma": g.tolist(), "eigenvalues": sigma.tolist(),
               "max_residual": float(residual), "P": P, "identity_prior": identity_prior}


def a0_to_a8(S: np.ndarray, M: np.ndarray, text_u: np.ndarray, *, V: np.ndarray | None = None,
             gamma: np.ndarray | float = 0.0, lam: float = 1e-3) -> dict[str, np.ndarray]:
    """Return all mandated readouts; no output is silently selected or dropped."""
    S = np.asarray(S, dtype=np.float64); M = np.asarray(M, dtype=np.float64)
    T = np.asarray(text_u, dtype=np.float64)
    if T.shape[1] != M.shape[1]:
        raise ValueError("BLOCKED_TEXT_CLASS_ORDER")
    K = M.shape[1]
    W0 = ridge(S[:1536, :1536] / K, M[:1536] / K, lam)
    Wu = ridge(S[1536:, 1536:] / K, M[1536:] / K, lam)
    W2 = ridge(S / K, M / K, lam)
    outputs = {"A0": W0, "A1": Wu, "A2": W2}
    outputs["A3"] = np.asarray(T, dtype=np.float64)
    outputs["A3s"] = (T - T.mean(axis=1, keepdims=True))
    if V is None:
        V = np.zeros_like(M)
    outputs["A4"], _ = spectral_prior_ridge(S, M, V, gamma, lam=lam, identity_prior=True)
    outputs["A5"], _ = spectral_prior_ridge(S, M, V, np.ones(M.shape[1]) * 0.001, lam=lam)
    outputs["A6"], diagnostics = spectral_prior_ridge(S, M, V, gamma, lam=lam)
    outputs["A7"] = M / np.maximum(np.linalg.norm(M, axis=0, keepdims=True), 1e-12)
    # A8 is ordinary ridge with an unpenalized intercept.  The class-balanced
    # sufficient statistics are enough for this readout; no sample matrix is
    # reconstructed and no class is silently dropped.
    K = M.shape[1]
    G, R = S / K, M / K
    mbar = M.mean(1)
    aug = np.block([[G, mbar[:, None]], [mbar[None, :], np.ones((1, 1))]])
    rhs = np.vstack((R, np.full((1, K), 1.0 / K)))
    reg = np.diag(np.r_[np.full(S.shape[0], lam), 0.0])
    outputs["A8"] = np.linalg.solve(aug + reg, rhs)
    outputs["A6_diagnostics"] = diagnostics["max_residual"]
    return outputs


def cosine_scores(features: np.ndarray, prototypes: np.ndarray, *, centered: bool = False) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    p = np.asarray(prototypes, dtype=np.float64)
    if centered:
        x = x - x.mean(axis=1, keepdims=True)
        p = p - p.mean(axis=1, keepdims=True)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    p = p / np.maximum(np.linalg.norm(p, axis=0, keepdims=True), 1e-12)
    return x @ p
