"""Float64 spectral prior ridge solves and the complete A0--A8 readout set."""
from __future__ import annotations

import numpy as np


def ridge(S: np.ndarray, M: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    S = np.asarray(S, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    if lam <= 0 or S.ndim != 2 or S.shape[0] != S.shape[1] or M.shape[0] != S.shape[0]:
        raise ValueError("BLOCKED_RIDGE_SHAPE")
    if not np.isfinite(S).all() or not np.isfinite(M).all():
        raise ValueError("BLOCKED_RIDGE_FINITE")
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
    if lam <= 0 or V.shape != M.shape or S.shape[0] != S.shape[1] or S.shape[0] != M.shape[0]:
        raise ValueError("BLOCKED_RASP_SHAPE")
    g = np.broadcast_to(np.asarray(gamma, dtype=np.float64), (K,))
    if not np.isfinite(S).all() or not np.isfinite(M).all() or not np.isfinite(V).all() or not np.isfinite(g).all() or np.any(g < 0):
        raise ValueError("BLOCKED_RASP_FINITE_OR_GAMMA")
    G = (S / K + (S / K).T) / 2
    R = M / K
    sigma, U = np.linalg.eigh(G)
    if sigma.min() < -1e-10:
        raise ValueError("BLOCKED_RASP_NONPSD")
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
    with np.errstate(all="ignore"):
        for c in range(K):
            rhs = R[:, c] + g[c] * (P @ V[:, c])
            W[:, c] = U @ ((U.T @ rhs) / (sigma + lam + g[c] * p))
        residual = max(float(np.linalg.norm((G + lam * np.eye(G.shape[0]) + g[c] * P) @ W[:, c] -
                                            (R[:, c] + g[c] * P @ V[:, c]))) for c in range(K))
    if not np.isfinite(W).all() or residual > 1e-8:
        raise ValueError("BLOCKED_RASP_SOLVE")
    return W, {"lambda": lam, "gamma": g.tolist(), "eigenvalues": sigma.tolist(),
               "max_residual": float(residual), "P": P, "identity_prior": identity_prior}


class RASPReadout:
    """An input-dependent CSE readout, rather than a bare text prototype."""

    def __init__(self, W: np.ndarray, text_u: np.ndarray, a_scale: float, *, centered: bool = False,
                 top_k: int = 5):
        self.W = np.asarray(W, dtype=np.float64)
        self.text_u = np.asarray(text_u, dtype=np.float64)
        self.a_scale = float(a_scale)
        self.centered = bool(centered)
        self.top_k = int(top_k)

    def scores(self, h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=np.float64)
        if h.ndim != 2 or h.shape[1] != self.W.shape[0]:
            raise ValueError("BLOCKED_READOUT_FEATURE_SHAPE")
        scores = np.einsum("nd,dk->nk", h, self.W)
        u = np.sqrt(2.0) * h[:, -self.text_u.shape[0]:]
        text = self.text_u
        # A3 uses raw unit u/T cosine. A3s only centers text across classes;
        # u is never centered and Tc is never renormalized.
        if self.centered:
            text = text - text.mean(axis=1, keepdims=True)
            scale = self.a_scale
        else:
            scale = 1.0
        u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        cosine = scale * np.einsum("nd,dk->nk", u, text)
        k = min(self.top_k, scores.shape[1])
        top = np.argpartition(scores, -k, axis=1)[:, -k:]
        fused = scores.copy()
        rows = np.arange(len(h))[:, None]
        fused[rows, top] += cosine[rows, top]
        return fused

    def predict(self, h: np.ndarray) -> np.ndarray:
        return np.argmax(self.scores(h), axis=1)


def a0_to_a8(S: np.ndarray, M: np.ndarray, text_u: np.ndarray, *, V: np.ndarray | None = None,
             gamma: np.ndarray | float = 0.0, lam: float = 1e-3, a_scale: float = 1.0,
             rarity_gamma: np.ndarray | float | None = None,
             ncomp: np.ndarray | None = None) -> dict[str, object]:
    """Return all mandated readouts; no output is silently selected or dropped."""
    S = np.asarray(S, dtype=np.float64); M = np.asarray(M, dtype=np.float64)
    T = np.asarray(text_u, dtype=np.float64)
    if T.shape[1] != M.shape[1]:
        raise ValueError("BLOCKED_TEXT_CLASS_ORDER")
    K = M.shape[1]
    # h contains a/sqrt(2),u/sqrt(2). Restore the original branch Gram before
    # fitting; A2 deliberately remains in the joint h coordinates.
    W0 = ridge(2.0 * S[:1536, :1536] / K, np.sqrt(2.0) * M[:1536] / K, lam)
    Wu = ridge(2.0 * S[1536:, 1536:] / K, np.sqrt(2.0) * M[1536:] / K, lam)
    W2 = ridge(S / K, M / K, lam)
    outputs = {"A0": W0, "A1": Wu, "A2": W2}
    outputs["A3"] = RASPReadout(W2, T, a_scale, centered=False)
    outputs["A3s"] = RASPReadout(W2, T, a_scale, centered=True)
    if V is None:
        V = np.zeros_like(M)
    outputs["A4"], _ = spectral_prior_ridge(S, M, V, gamma, lam=lam, identity_prior=True)
    # The caller supplies the reliability-free rarity strength. If it is not
    # available, retaining gamma is safer than silently changing the model.
    if a_scale == 0:
        g_rarity = 0.0
    elif rarity_gamma is not None:
        g_rarity = rarity_gamma
    elif ncomp is not None:
        g_rarity = 0.001 * 10.0 / (np.asarray(ncomp, dtype=np.float64) + 10.0)
    else:
        raise ValueError("BLOCKED_A5_RARITY_PARAMETER")
    outputs["A5"], _ = spectral_prior_ridge(S, M, V, g_rarity, lam=lam)
    outputs["A6"], diagnostics = spectral_prior_ridge(S, M, V, gamma, lam=lam)
    outputs["A7"] = M / np.maximum(np.linalg.norm(M, axis=0, keepdims=True), 1e-12)
    # A8 is ordinary ridge with an unpenalized intercept.  The class-balanced
    # sufficient statistics are enough for this readout; no sample matrix is
    # reconstructed and no class is silently dropped.
    G, R = S / K, M / K
    mbar, ybar = M.mean(1), np.full(K, 1.0 / K)
    Wc = np.linalg.solve(G - np.outer(mbar, mbar) + lam * np.eye(S.shape[0]), R - np.outer(mbar, ybar))
    # ``einsum`` keeps the intercept reduction in float64 without dispatching
    # a large BLAS matmul for this one-row product.
    outputs["A8"] = np.vstack((Wc, ybar - np.einsum("i,ij->j", mbar, Wc)))
    return outputs


def readout_scores(outputs: dict[str, object], name: str, h: np.ndarray) -> np.ndarray:
    """Score every readout in its declared coordinate system."""
    h = np.asarray(h, dtype=np.float64)
    if h.ndim != 2 or h.shape[1] != 2048:
        raise ValueError("BLOCKED_READOUT_FEATURE_SHAPE")
    if name == "A0":
        return np.einsum("nd,dk->nk", np.sqrt(2.0) * h[:, :1536], np.asarray(outputs[name]))
    if name == "A1":
        return np.einsum("nd,dk->nk", np.sqrt(2.0) * h[:, 1536:], np.asarray(outputs[name]))
    value = outputs[name]
    if hasattr(value, "scores"):
        return value.scores(h)
    if name == "A8":
        x = np.column_stack((h, np.ones(len(h))))
        return np.einsum("nd,dk->nk", x, np.asarray(value))
    return np.einsum("nd,dk->nk", h, np.asarray(value))


def cosine_scores(features: np.ndarray, prototypes: np.ndarray, *, centered: bool = False) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    p = np.asarray(prototypes, dtype=np.float64)
    if centered:
        x = x - x.mean(axis=1, keepdims=True)
        p = p - p.mean(axis=1, keepdims=True)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    p = p / np.maximum(np.linalg.norm(p, axis=0, keepdims=True), 1e-12)
    return x @ p
