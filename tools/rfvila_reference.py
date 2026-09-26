"""Small CPU reference for NB2-RFVILA-12H-R1; not a production runner.

No network, images, GPU jobs, training, or experiment claims. Production code must
pass these identities and its own provenance/access/budget checks.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
from typing import Sequence
import numpy as np


def matrix(value: np.ndarray, name: str = "matrix") -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError(f"invalid {name}")
    return x


def normalize_rows(value: np.ndarray) -> np.ndarray:
    x = matrix(value)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("zero feature norm")
    return x / norm


def projections(dim_a: int, dim_u: int, width: int, seed: int):
    if min(dim_a, dim_u, width) <= 0 or width % 2:
        raise ValueError("positive dimensions and even width required")
    # Separate substreams: changing dim_a does not silently change the u map.
    ra = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 0]))).standard_normal((dim_a, width))
    ru = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 1]))).standard_normal((dim_u, width))
    return ra, ru


def features(a: np.ndarray, u: np.ndarray, ra: np.ndarray, ru: np.ndarray, kind: str):
    a, u, ra, ru = (matrix(x) for x in (a, u, ra, ru))
    if len(a) != len(u) or a.shape[1] != ra.shape[0] or u.shape[1] != ru.shape[0] or ra.shape[1] != ru.shape[1]:
        raise ValueError("feature/map dimensions")
    m = ra.shape[1]
    if kind == "F.LIN":
        return a.copy()
    if kind == "J.LIN":
        return np.concatenate([a, u], axis=1) / np.sqrt(2)
    if kind == "F.RF":
        return np.sqrt(2 / m) * np.maximum(a @ ra, 0)
    joint = (a @ ra + u @ ru) / np.sqrt(2)
    if kind == "J.RF":
        return np.sqrt(2 / m) * np.maximum(joint, 0)
    if kind == "J.RPLINEAR":
        return joint / np.sqrt(m)
    if kind == "J.SPLIT":
        if m % 2:
            raise ValueError("even width required")
        return np.sqrt(2 / m) * np.concatenate([
            np.maximum(a @ ra[:, :m // 2], 0),
            np.maximum(u @ ru[:, :m // 2], 0)], axis=1)
    raise ValueError(f"unknown feature kind: {kind}")


@dataclass
class MomentBank:
    dim: int
    ids: list[int] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    S: np.ndarray = field(init=False)
    Q: np.ndarray = field(init=False)

    def __post_init__(self):
        if self.dim <= 0:
            raise ValueError("positive dimension required")
        self.S = np.zeros((self.dim, self.dim), dtype=np.float64)
        self.Q = np.empty((self.dim, 0), dtype=np.float64)

    def append_class(self, class_id: int, phi: np.ndarray):
        x = matrix(phi, "features")
        if class_id in self.ids or x.shape[1] != self.dim or len(x) == 0:
            raise ValueError("duplicate/empty/wrong-dimensional class")
        self.S += x.T @ x / len(x)
        self.Q = np.column_stack([self.Q, x.mean(axis=0)])
        self.ids.append(int(class_id))
        self.counts.append(len(x))

    def solve(self, lam: float) -> np.ndarray:
        if not self.ids or not np.isfinite(lam) or lam <= 0:
            raise ValueError("positive lambda and nonempty bank required")
        A = (self.S + self.S.T) / 2 + len(self.ids) * lam * np.eye(self.dim)
        chol = np.linalg.cholesky(A)
        W = np.linalg.solve(chol.T, np.linalg.solve(chol, self.Q))
        residual = np.linalg.norm(A @ W - self.Q) / max(np.linalg.norm(self.Q), 1e-30)
        if not np.isfinite(W).all() or residual > 1e-8:
            raise ArithmeticError(f"ridge residual {residual}")
        return W


def weighted_design(phi: np.ndarray, labels: np.ndarray, class_ids: Sequence[int]):
    x = matrix(phi)
    y = np.asarray(labels)
    ids = list(map(int, class_ids))
    if len(set(ids)) != len(ids) or y.shape != (len(x),) or set(y.tolist()) != set(ids):
        raise ValueError("class coverage/labels")
    lookup = {c: j for j, c in enumerate(ids)}
    n = {c: int(np.sum(y == c)) for c in ids}
    w = np.array([1 / (len(ids) * n[int(c)]) for c in y])
    Y = np.eye(len(ids))[[lookup[int(c)] for c in y]]
    return x * np.sqrt(w[:, None]), Y * np.sqrt(w[:, None])


def batch_ridge(phi, labels, class_ids, lam, *, dual=False):
    X, Y = weighted_design(phi, labels, class_ids)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("positive lambda")
    if dual:
        return X.T @ np.linalg.solve(X @ X.T + lam * np.eye(len(X)), Y)
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


def group_folds(labels, components, *, seed=67026, max_folds=3):
    """Class-wise hashed round-robin groups; None means predeclared fallback.

    A component spanning labels is an unsupported grouping, not permission to
    split that component; production must record the common fallback.
    """
    y = np.asarray(labels)
    comp = np.asarray(components, dtype=str)
    if y.ndim != 1 or comp.shape != y.shape or len(y) == 0:
        raise ValueError("grouping shapes")
    ids = sorted(set(y.tolist()))
    for group in set(comp.tolist()):
        if len(set(y[comp == group].tolist())) != 1:
            return None
    counts = [len(set(comp[y == c].tolist())) for c in ids]
    folds = min(max_folds, min(counts))
    if folds < 2:
        return None
    out = np.full(len(y), -1, dtype=np.int64)
    for c in ids:
        def key(g):
            return hashlib.sha256(f"{seed}|{c}|{g}".encode()).hexdigest()
        groups = sorted(set(comp[y == c].tolist()), key=key)
        for rank, group in enumerate(groups):
            out[comp == group] = rank % folds
    return out


def class_balanced_mse(scores, labels, class_ids):
    s = matrix(scores)
    y = np.asarray(labels)
    ids = list(map(int, class_ids))
    if s.shape != (len(y), len(ids)) or set(y.tolist()) != set(ids):
        raise ValueError("MSE label coverage")
    loss = []
    for j, c in enumerate(ids):
        target = np.zeros(len(ids)); target[j] = 1
        loss.append(float(np.mean(np.sum((s[y == c] - target) ** 2, axis=1))))
    return float(np.mean(loss))


def pick_lambda(grid, losses, *, tolerance=1e-12):
    pairs = list(zip(map(float, grid), map(float, losses)))
    if not pairs or len(pairs) != len(grid) or len(grid) != len(losses):
        raise ValueError("lambda grid lengths")
    if any(l <= 0 or not np.isfinite(l) or not np.isfinite(v) for l, v in pairs):
        raise ValueError("invalid lambda/loss")
    best = min(v for _, v in pairs)
    # Predeclared tie-break, independent of evaluation scores.
    return max(l for l, v in pairs if v <= best + tolerance)


def select_lambda_cv(phi, labels, components, class_ids, grid, *, seed=67026):
    x = matrix(phi); y = np.asarray(labels)
    fold = group_folds(y, components, seed=seed)
    if fold is None:
        return 1e-3, {"status": "TASK1_GROUP_CV_UNAVAILABLE_FIXED_LAMBDA", "folds": 0}
    losses = []
    for lam in grid:
        vals = []
        for f in range(int(fold.max()) + 1):
            tr, va = fold != f, fold == f
            W = batch_ridge(x[tr], y[tr], class_ids, lam, dual=int(tr.sum()) < x.shape[1])
            vals.append(class_balanced_mse(x[va] @ W, y[va], class_ids))
        losses.append(float(np.mean(vals)))
    chosen = pick_lambda(grid, losses)
    return chosen, {"status": "GROUPED_TASK1_READOUT_CV", "folds": int(fold.max()) + 1,
                    "grid": list(grid), "losses": losses}


def cse(scores, cosine, class_ids, *, alpha=1., top_k=5, permute=False):
    s, v = matrix(scores, "scores"), matrix(cosine, "cosine")
    ids = np.asarray(class_ids, dtype=np.int64)
    if s.shape != v.shape or s.shape[1] != len(ids) or len(set(ids.tolist())) != len(ids):
        raise ValueError("CSE shapes/class order")
    if top_k < 1 or not np.isfinite(alpha):
        raise ValueError("CSE coefficient")
    if permute:
        order = np.argsort(ids)
        perm = np.arange(len(ids)); perm[order] = np.roll(order, 1)
        v = v[:, perm]
    out = s.copy()
    for i in range(len(s)):
        keep = np.lexsort((ids, -s[i]))[:min(top_k, len(ids))]
        out[i, keep] += alpha * v[i, keep]
    return out


def predict(scores, class_ids):
    s = matrix(scores)
    ids = np.asarray(class_ids, dtype=np.int64)
    if s.shape[1] != len(ids) or len(set(ids.tolist())) != len(ids):
        raise ValueError("prediction columns")
    return np.asarray([ids[np.lexsort((ids, -r))[0]] for r in s])


def metrics(labels, pred, class_ids):
    y, p = np.asarray(labels), np.asarray(pred)
    ids = list(map(int, class_ids))
    if y.shape != p.shape or y.ndim != 1 or not len(y) or set(y.tolist()) != set(ids) or not set(p.tolist()).issubset(set(ids)):
        raise ValueError("metrics coverage")
    lookup = {c: j for j, c in enumerate(ids)}
    C = np.zeros((len(ids), len(ids)), dtype=np.int64)
    for a, b in zip(y, p):
        C[lookup[int(a)], lookup[int(b)]] += 1
    tp = C.diagonal().astype(float)
    recall = tp / C.sum(1)
    denom = C.sum(1) + C.sum(0)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return {"ba": float(recall.mean()), "macro_f1": float(f1.mean()),
            "accuracy": float(tp.sum() / C.sum()), "zero_recall_ids": [ids[j] for j in np.flatnonzero(tp == 0)],
            "confusion": C}
