"""Class-balanced dual moments with the complete 1536x512 cross block."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np


def _matrix(x: np.ndarray, d: int, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != d or not np.isfinite(x).all():
        raise ValueError(f"BLOCKED_{name.upper()}")
    return x


@dataclass
class ClassMoments:
    class_id: int
    task: int
    n: int
    mean: np.ndarray
    second: np.ndarray
    component_ids: tuple[str, ...] = ()
    component_means: dict[str, np.ndarray] = field(default_factory=dict)
    component_seconds: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class DualMomentBank:
    """A sufficient-statistics bank; raw images/features are never retained."""

    dim_a: int = 1536
    dim_u: int = 512
    classes: dict[int, ClassMoments] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return self.dim_a + self.dim_u

    def add_class(
        self,
        class_id: int,
        h: np.ndarray,
        *,
        task: int,
        component_ids: Iterable[str] | None = None,
    ) -> None:
        if class_id in self.classes:
            raise ValueError("BLOCKED_DUPLICATE_CLASS")
        h = _matrix(h, self.dim, "joint_moments")
        if not len(h):
            raise ValueError("BLOCKED_EMPTY_CLASS")
        ids = tuple(str(x) for x in (component_ids if component_ids is not None else ["default"] * len(h)))
        if len(ids) != len(h):
            raise ValueError("BLOCKED_COMPONENT_COUNT")
        mean = h.mean(axis=0)
        # Some NumPy/OpenBLAS builds emit spurious floating-point warnings for
        # large BLAS matmuls; the finite checks below remain the guard.
        with np.errstate(all="ignore"):
            second = h.T @ h / len(h)
        comp_means: dict[str, np.ndarray] = {}
        comp_seconds: dict[str, np.ndarray] = {}
        for comp in sorted(set(ids)):
            x = h[np.asarray(ids) == comp]
            comp_means[comp] = x.mean(axis=0)
            with np.errstate(all="ignore"):
                comp_seconds[comp] = x.T @ x / len(x)
        self.classes[int(class_id)] = ClassMoments(
            int(class_id), int(task), len(h), mean, second, tuple(sorted(comp_means)),
            comp_means, comp_seconds,
        )

    def _ordered(self) -> list[ClassMoments]:
        if not self.classes:
            raise ValueError("BLOCKED_EMPTY_BANK")
        return [self.classes[k] for k in sorted(self.classes)]

    def class_balanced(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``S=sum_c E[h h^T]``, ``M=[mu_c]`` and class ids."""
        rows = self._ordered()
        S = sum((r.second for r in rows), np.zeros((self.dim, self.dim), dtype=np.float64))
        M = np.column_stack([r.mean for r in rows])
        return S, M, np.asarray([r.class_id for r in rows], dtype=np.int64)

    def homogeneous(self) -> tuple[np.ndarray, np.ndarray]:
        """Return homogeneous class-balanced moments ``bar_S`` and ``bar_M``."""
        S, M, _ = self.class_balanced()
        return S / M.shape[1], M / M.shape[1]

    def u_stats(self) -> tuple[np.ndarray, np.ndarray]:
        rows = self._ordered()
        means = np.column_stack([r.mean[self.dim_a :] for r in rows])
        second = sum((r.second[self.dim_a :, self.dim_a :] for r in rows), np.zeros((self.dim_u, self.dim_u)))
        return second / len(rows), means / len(rows)

    def component_stats(self, component: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = self._ordered()
        selected = [(r.component_means[component], r.component_seconds[component])
                    for r in rows if component in r.component_means]
        if not selected:
            return np.zeros((self.dim_u, 0)), np.zeros((self.dim_u, self.dim_u)), np.zeros(0, dtype=np.int64)
        means = np.column_stack([x[0][self.dim_a :] for x in selected])
        # Component count is the number of independent observations available
        # for reliability; the full second moment retains within-component data.
        counts = np.asarray([1] * len(selected), dtype=np.int64)
        cov = sum((x[1][self.dim_a :, self.dim_a :] for x in selected), np.zeros((self.dim_u, self.dim_u))) / len(selected)
        return means, cov, counts

    def component_reliability_inputs(self, component: str) -> tuple[dict[int, np.ndarray], dict[int, int]]:
        """Expose only component means/counts needed by the RASP reliability term."""
        means: dict[int, np.ndarray] = {}
        counts: dict[int, int] = {}
        for row in self._ordered():
            values = [x[self.dim_a :] for key, x in row.component_means.items() if key == component]
            if values:
                means[row.class_id] = np.vstack(values)
                counts[row.class_id] = len(values)
        return means, counts

    def state_dict(self) -> dict:
        return {
            "dim_a": self.dim_a,
            "dim_u": self.dim_u,
            "classes": {
                str(k): {
                    "task": v.task, "n": v.n,
                    "mean": v.mean.tolist(), "second": v.second.tolist(),
                    "component_ids": list(v.component_ids),
                    "component_means": {c: x.tolist() for c, x in v.component_means.items()},
                    "component_seconds": {c: x.tolist() for c, x in v.component_seconds.items()},
                } for k, v in self.classes.items()
            },
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "DualMomentBank":
        bank = cls(int(state["dim_a"]), int(state["dim_u"]))
        for key, raw in state["classes"].items():
            bank.classes[int(key)] = ClassMoments(
                int(key), int(raw["task"]), int(raw["n"]),
                np.asarray(raw["mean"], dtype=np.float64), np.asarray(raw["second"], dtype=np.float64),
                tuple(raw.get("component_ids", ())),
                {c: np.asarray(x, dtype=np.float64) for c, x in raw.get("component_means", {}).items()},
                {c: np.asarray(x, dtype=np.float64) for c, x in raw.get("component_seconds", {}).items()},
            )
        return bank
