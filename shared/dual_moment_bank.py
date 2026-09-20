"""Class-balanced dual moments with a bounded, aggregate component state.

The persistent bank stores one global joint second moment and one mean per
class. Component observations are reduced to per-class 512-D sums/counts and
second sums before the call returns; individual component vectors never enter
``state_dict``.
"""
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
    component_count: int = 0
    component_sum: np.ndarray | None = None
    component_second: np.ndarray | None = None


@dataclass
class DualMomentBank:
    """A sufficient-statistics bank; raw images/features are never retained."""

    dim_a: int = 1536
    dim_u: int = 512
    classes: dict[int, ClassMoments] = field(default_factory=dict)
    class_order: list[int] = field(default_factory=list)
    _S: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.dim_a <= 0 or self.dim_u <= 0:
            raise ValueError("BLOCKED_FEATURE_DIM")
        if self._S is None:
            self._S = np.zeros((self.dim, self.dim), dtype=np.float64)
        else:
            self._S = _matrix(self._S, self.dim, "bank_second")

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
        with np.errstate(all="ignore"):
            second = h.T @ h / len(h)
        if not np.isfinite(second).all():
            raise ValueError("BLOCKED_NONFINITE_MOMENT")

        # Component reliability lives in unit CLIP coordinates u=sqrt(2)*h_u.
        comp_sum = np.zeros(self.dim_u, dtype=np.float64)
        comp_second = np.zeros((self.dim_u, self.dim_u), dtype=np.float64)
        comp_count = 0
        ids_array = np.asarray(ids)
        for comp in sorted(set(ids)):
            x = h[ids_array == comp]
            u_mean = np.sqrt(2.0) * x[:, self.dim_a :].mean(axis=0)
            comp_sum += u_mean
            with np.errstate(all="ignore"):
                comp_second += np.outer(u_mean, u_mean)
            comp_count += 1
        self._S = self._S + second
        self.classes[int(class_id)] = ClassMoments(
            int(class_id), int(task), len(h), mean, comp_count, comp_sum, comp_second,
        )
        self.class_order.append(int(class_id))

    def _ordered(self) -> list[ClassMoments]:
        if not self.classes:
            raise ValueError("BLOCKED_EMPTY_BANK")
        order = self.class_order or sorted(self.classes)
        return [self.classes[k] for k in order]

    def class_balanced(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``S=sum_c E[h h^T]``, ``M=[mu_c]`` and explicit class ids."""
        rows = self._ordered()
        return self._S.copy(), np.column_stack([r.mean for r in rows]), np.asarray(
            [r.class_id for r in rows], dtype=np.int64
        )

    def homogeneous(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the protocol's unnormalized homogeneous ``bar_S, bar_M``.

        ``bar_S`` has bottom-right K and cross term ``M@1``; ``bar_M`` has a
        unit bottom row. Use :func:`transport_homogeneous` for the full form.
        """
        S, M, _ = self.class_balanced()
        K = M.shape[1]
        bar_S = np.zeros((self.dim + 1, self.dim + 1), dtype=np.float64)
        bar_S[:-1, :-1] = S
        bar_S[:-1, -1] = bar_S[-1, :-1] = M @ np.ones(K)
        bar_S[-1, -1] = K
        bar_M = np.vstack((M, np.ones((1, K), dtype=np.float64)))
        return bar_S, bar_M

    def normalized_homogeneous(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``bar_S/K`` and ``bar_M/K`` for class-averaged consumers."""
        bar_S, bar_M = self.homogeneous()
        return bar_S / bar_M.shape[1], bar_M / bar_M.shape[1]

    def u_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Restore unit-CLIP coordinates from h's ``u/sqrt(2)`` subblock."""
        S, M, _ = self.class_balanced()
        K = M.shape[1]
        Gu = 2.0 * S[self.dim_a :, self.dim_a :] / K
        Ru = np.sqrt(2.0) * M[self.dim_a :] / K
        return Gu, Ru

    def component_reliability_inputs(self, component: str | None = None) -> tuple[dict[int, dict], dict[int, int]]:
        """Return aggregate component distributions for every class.

        ``component`` is retained for API compatibility but is intentionally
        not used to select one component: reliability must see all components
        in a class. The returned stats contain only count/sum/second-sum.
        """
        del component
        stats: dict[int, dict] = {}
        counts: dict[int, int] = {}
        for row in self._ordered():
            count = int(row.component_count)
            stats[row.class_id] = {
                "count": count,
                "sum": row.component_sum.copy(),
                "second": row.component_second.copy(),
            }
            counts[row.class_id] = count
        return stats, counts

    def state_dict(self) -> dict:
        """Serialize only aggregate learning state, never component IDs/vectors."""
        return {
            "dim_a": self.dim_a,
            "dim_u": self.dim_u,
            "S": self._S.tolist(),
            "class_order": list(self.class_order),
            "classes": {
                str(k): {
                    "task": v.task, "n": v.n, "mean": v.mean.tolist(),
                    "component_count": v.component_count,
                    "component_sum": v.component_sum.tolist(),
                    "component_second": v.component_second.tolist(),
                } for k, v in self.classes.items()
            },
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "DualMomentBank":
        dim_a, dim_u = int(state["dim_a"]), int(state["dim_u"])
        S = np.asarray(state["S"], dtype=np.float64)
        bank = cls(dim_a, dim_u, _S=S)
        if S.shape != (bank.dim, bank.dim) or not np.isfinite(S).all():
            raise ValueError("BLOCKED_RESTORE_SECOND")
        order = [int(x) for x in state.get("class_order", state["classes"].keys())]
        if len(order) != len(set(order)) or set(order) != {int(x) for x in state["classes"]}:
            raise ValueError("BLOCKED_RESTORE_CLASS_ORDER")
        for key, raw in state["classes"].items():
            cid = int(key)
            mean = np.asarray(raw["mean"], dtype=np.float64)
            csum = np.asarray(raw["component_sum"], dtype=np.float64)
            csecond = np.asarray(raw["component_second"], dtype=np.float64)
            if mean.shape != (bank.dim,) or csum.shape != (dim_u,) or csecond.shape != (dim_u, dim_u):
                raise ValueError("BLOCKED_RESTORE_CLASS_SHAPE")
            if not all(np.isfinite(x).all() for x in (mean, csum, csecond)):
                raise ValueError("BLOCKED_RESTORE_CLASS_FINITE")
            bank.classes[cid] = ClassMoments(cid, int(raw["task"]), int(raw["n"]), mean,
                                             int(raw["component_count"]), csum, csecond)
        bank.class_order = order
        tasks = [bank.classes[c].task for c in order]
        if tasks != sorted(tasks):
            raise ValueError("BLOCKED_RESTORE_TASK_ORDER")
        return bank
