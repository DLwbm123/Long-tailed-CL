"""Text prior, semantic scale, and component reliability for RASP."""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


TEMPLATES = {
    "isic": ("a dermoscopic image of {class_name}.", "a clinical image of {class_name}."),
    "hyperkvasir": ("an endoscopic image of {class_name}.", "a gastrointestinal endoscopy image of {class_name}."),
}


def unit_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=0, keepdims=True)
    if x.ndim != 2 or not np.isfinite(x).all() or np.any(n <= 1e-12):
        raise ValueError("BLOCKED_TEXT_PROTOTYPE")
    return x / n


def centered_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean(axis=1, keepdims=True)


def text_prompts(dataset: str, class_names: Iterable[str], template_index: int = 0) -> list[str]:
    names = list(class_names)
    if dataset.lower() not in TEMPLATES or template_index not in (0, 1):
        raise ValueError("BLOCKED_TEXT_TEMPLATE")
    return [TEMPLATES[dataset.lower()][template_index].format(class_name=name) for name in names]


def semantic_scale(text_u: np.ndarray, Gu: np.ndarray, Ru: np.ndarray, lam: float = 1e-3) -> float:
    """Compute the bounded scalar shared by all classes."""
    T = centered_columns(unit_columns(text_u))
    den = float(np.trace(T.T @ Gu @ T) + lam)
    value = float(np.trace(T.T @ Ru) / den)
    return float(np.clip(value, 0.0, 10.0))


def component_reliability(
    text_u: np.ndarray,
    component_means: Mapping[int, np.ndarray],
    component_counts: Mapping[int, int],
    *,
    class_ids: Iterable[int],
    a_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``r`` and ``gamma=.001*10/(n+10)*r`` in class order."""
    T = unit_columns(text_u)
    ids = list(class_ids)
    K = len(ids)
    if a_scale == 0:
        return np.zeros(K), np.zeros(K)
    def stats_for(c):
        raw = component_means.get(c)
        n = int(component_counts.get(c, 0))
        if isinstance(raw, Mapping):
            n = int(raw.get("count", n)); total = np.asarray(raw.get("sum"), dtype=np.float64)
            second = np.asarray(raw.get("second"), dtype=np.float64)
            mean = total / max(n, 1)
            cov = second / max(n, 1) - np.outer(mean, mean)
            return n, mean, (cov + cov.T) / 2
        values = np.asarray(raw, dtype=np.float64)
        if values.ndim != 2 or not len(values):
            return 0, np.zeros(T.shape[0]), np.eye(T.shape[0]) / T.shape[0]
        mean = values.mean(0)
        centered = values - mean
        return len(values), mean, centered.T @ centered / len(values)

    valid = [stats_for(c)[2] for c in ids if stats_for(c)[0] >= 2]
    if valid:
        pool = sum(valid, np.zeros((T.shape[0], T.shape[0]))) / len(valid)
    else:
        pool = np.eye(T.shape[0]) / T.shape[0]
    out = np.zeros(K, dtype=np.float64)
    for j, c in enumerate(ids):
        n, mean, Cc = stats_for(c)
        if n < 2:
            continue
        omega = 10.0 / (n + 10.0)
        Ctilde = (1.0 - omega) * Cc + omega * pool
        own = float(mean @ T[:, j])
        competitors = [float(mean @ T[:, k]) for k in range(K) if k != j]
        if competitors:
            worst = max((k for k in range(K) if k != j), key=lambda k: float(mean @ T[:, k]))
            diff = T[:, j] - T[:, worst]
            m = own - float(mean @ T[:, worst])
        else:
            diff = T[:, j]
            m = own
        v = float(diff.T @ Ctilde @ diff)
        out[j] = np.clip((m - np.sqrt(max(v, 0.0) / n)) /
                         (abs(m) + np.sqrt(max(v, 0.0)) + 1e-8), 0.0, 1.0)
    gamma = 0.001 * 10.0 / (np.asarray([component_counts.get(c, 0) for c in ids]) + 10.0) * out
    return out, gamma


def build_prior(text_u: np.ndarray, a_scale: float, dim_a: int = 1536) -> np.ndarray:
    T = centered_columns(unit_columns(text_u))
    if T.shape[1] == 0:
        raise ValueError("BLOCKED_EMPTY_TEXT")
    return np.vstack((np.zeros((dim_a, T.shape[1])), np.sqrt(2.0) * a_scale * T))
