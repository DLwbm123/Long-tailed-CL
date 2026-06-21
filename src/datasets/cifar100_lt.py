"""Deterministic CIFAR-100-LT construction and task datasets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100

CIFAR100_NORMALIZATIONS = {
    "current": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    "ltcil_official": {
        "mean": (0.5071, 0.4866, 0.4409),
        "std": (0.2009, 0.1984, 0.2023),
    },
}


def _normalization_values(name: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if name not in CIFAR100_NORMALIZATIONS:
        raise ValueError("--cifar-normalization must be current or ltcil_official")
    values = CIFAR100_NORMALIZATIONS[name]
    return values["mean"], values["std"]


def cifar_train_transform(normalization: str = "current") -> transforms.Compose:
    mean, std = _normalization_values(normalization)
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def cifar_eval_transform(normalization: str = "current") -> transforms.Compose:
    mean, std = _normalization_values(normalization)
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def long_tail_counts(num_classes: int, max_count: int, rho: float) -> List[int]:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if max_count <= 0:
        raise ValueError("max_count must be positive")
    if not (0 < rho <= 1):
        raise ValueError("rho must satisfy 0 < rho <= 1")
    if num_classes == 1:
        return [max_count]
    counts = []
    for rank in range(num_classes):
        value = int(round(max_count * (rho ** (rank / (num_classes - 1)))))
        counts.append(max(1, value))
    return counts


class CIFARClassSubset(Dataset):
    def __init__(self, base_dataset: CIFAR100, indices: Sequence[int], transform: transforms.Compose):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        image, target = self.base_dataset[self.indices[item]]
        if self.transform is not None:
            image = self.transform(image)
        return image, int(target)


@dataclass
class CIFAR100LTProtocol:
    train_base: CIFAR100
    test_base: CIFAR100
    selected_classes: List[int]
    class_counts: Dict[int, int]
    class_order: List[int]
    tasks: List[List[int]]
    train_indices_by_class: Dict[int, List[int]]
    test_indices_by_class: Dict[int, List[int]]
    train_transform: transforms.Compose
    eval_transform: transforms.Compose
    seed: int
    rho: float
    order: str
    count_assignment: str
    class_order_source: str
    class_order_file: str | None
    cifar_normalization: str
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    max_train_per_class: int

    def train_dataset_for_classes(self, class_ids: Sequence[int]) -> CIFARClassSubset:
        indices: List[int] = []
        for class_id in class_ids:
            indices.extend(self.train_indices_by_class[int(class_id)])
        return CIFARClassSubset(self.train_base, indices, self.train_transform)

    def train_dataset_for_indices(self, indices: Sequence[int]) -> CIFARClassSubset:
        return CIFARClassSubset(self.train_base, indices, self.train_transform)

    def prototype_dataset_for_classes(self, class_ids: Sequence[int]) -> CIFARClassSubset:
        indices: List[int] = []
        for class_id in class_ids:
            indices.extend(self.train_indices_by_class[int(class_id)])
        return CIFARClassSubset(self.train_base, indices, self.eval_transform)

    def test_dataset_for_classes(self, class_ids: Sequence[int]) -> CIFARClassSubset:
        indices: List[int] = []
        for class_id in class_ids:
            indices.extend(self.test_indices_by_class[int(class_id)])
        return CIFARClassSubset(self.test_base, indices, self.eval_transform)

    def preview(self) -> Dict[str, object]:
        task_sizes = [len(task) for task in self.tasks]
        counts = [self.class_counts[class_id] for class_id in self.selected_classes]
        return {
            "dataset": "cifar100_lt",
            "num_classes": len(self.selected_classes),
            "rho": self.rho,
            "order": self.order,
            "count_assignment": self.count_assignment,
            "class_order_source": self.class_order_source,
            "class_order_file": self.class_order_file,
            "cifar_normalization": self.cifar_normalization,
            "normalization_mean": self.normalization_mean,
            "normalization_std": self.normalization_std,
            "seed": self.seed,
            "max_count": max(counts),
            "min_count": min(counts),
            "base_classes": len(self.tasks[0]),
            "incremental_steps": len(self.tasks) - 1,
            "incremental_classes_per_step": task_sizes[1:],
            "task_sizes": task_sizes,
            "class_order": self.class_order,
            "class_counts": {str(k): v for k, v in sorted(self.class_counts.items())},
        }


def _load_class_order_file(path: str | Path) -> List[int]:
    text = Path(path).read_text(encoding="utf-8")
    return [int(item) for item in re.findall(r"-?\d+", text)]


def _resolve_class_order(
    selected_classes: Sequence[int],
    order: str,
    class_order_source: str,
    class_order_file: str | None,
    rng: np.random.Generator,
) -> List[int]:
    selected_classes = [int(class_id) for class_id in selected_classes]
    if class_order_source == "random_seed":
        if order == "ordered":
            return selected_classes.copy()
        if order == "shuffled":
            return rng.permutation(selected_classes).astype(int).tolist()
        raise ValueError("--order must be either ordered or shuffled")
    if class_order_source == "official_ltcil":
        from src.datasets.official_orders import CIFAR100_LTCIL_CLASS_ORDER

        class_order = CIFAR100_LTCIL_CLASS_ORDER[: len(selected_classes)]
    elif class_order_source == "file":
        if not class_order_file:
            raise ValueError("--class-order-source file requires --class-order-file")
        class_order = _load_class_order_file(class_order_file)
    else:
        raise ValueError("--class-order-source must be random_seed, official_ltcil, or file")

    if len(class_order) != len(selected_classes):
        raise ValueError(
            f"class order length {len(class_order)} does not match selected class count {len(selected_classes)}"
        )
    if set(int(class_id) for class_id in class_order) != set(selected_classes):
        raise ValueError("class order must contain each selected class exactly once")
    return [int(class_id) for class_id in class_order]


def build_cifar100_lt_protocol(
    data_root: str | Path,
    rho: float,
    order: str,
    base_classes: int,
    incremental_steps: int,
    seed: int,
    debug_num_classes: int | None = None,
    max_train_per_class: int | None = None,
    count_assignment: str = "original_id",
    class_order_source: str = "random_seed",
    class_order_file: str | None = None,
    cifar_normalization: str = "current",
    download: bool = True,
) -> CIFAR100LTProtocol:
    data_root = Path(data_root)
    train_base = CIFAR100(root=str(data_root), train=True, download=download, transform=None)
    test_base = CIFAR100(root=str(data_root), train=False, download=download, transform=None)

    total_classes = 100 if debug_num_classes is None else int(debug_num_classes)
    if not (1 < total_classes <= 100):
        raise ValueError("--debug-num-classes must be between 2 and 100")
    if base_classes <= 0 or base_classes >= total_classes:
        raise ValueError("--base-classes must be positive and smaller than the selected class count")
    if incremental_steps <= 0:
        raise ValueError("--incremental-steps must be positive")
    if count_assignment not in {"original_id", "shuffled_rank"}:
        raise ValueError("--count-assignment must be either original_id or shuffled_rank")
    if class_order_source not in {"random_seed", "official_ltcil", "file"}:
        raise ValueError("--class-order-source must be random_seed, official_ltcil, or file")

    remaining_classes = total_classes - base_classes
    if remaining_classes % incremental_steps != 0:
        raise ValueError(
            "remaining classes must divide evenly by --incremental-steps "
            f"({remaining_classes} % {incremental_steps} != 0)"
        )

    selected_classes = list(range(total_classes))
    rng = np.random.default_rng(seed)

    available_train: Dict[int, List[int]] = {class_id: [] for class_id in selected_classes}
    for index, target in enumerate(train_base.targets):
        target = int(target)
        if target in available_train:
            available_train[target].append(index)

    available_max = min(len(indices) for indices in available_train.values())
    effective_max = available_max if max_train_per_class is None else min(int(max_train_per_class), available_max)
    counts = long_tail_counts(total_classes, effective_max, rho)

    if count_assignment == "shuffled_rank":
        class_order = _resolve_class_order(
            selected_classes,
            order,
            class_order_source,
            class_order_file,
            rng,
        )
        class_counts = {int(class_id): int(counts[rank]) for rank, class_id in enumerate(class_order)}
    else:
        class_counts = {class_id: int(counts[rank]) for rank, class_id in enumerate(selected_classes)}

    train_indices_by_class: Dict[int, List[int]] = {}
    for class_id in selected_classes:
        indices = np.array(available_train[class_id], dtype=np.int64)
        rng.shuffle(indices)
        requested = class_counts[class_id]
        if requested > len(indices):
            raise RuntimeError(f"Class {class_id} has only {len(indices)} train samples, need {requested}")
        train_indices_by_class[class_id] = indices[:requested].astype(int).tolist()

    test_indices_by_class: Dict[int, List[int]] = {class_id: [] for class_id in selected_classes}
    for index, target in enumerate(test_base.targets):
        target = int(target)
        if target in test_indices_by_class:
            test_indices_by_class[target].append(index)

    if count_assignment == "original_id":
        class_order = _resolve_class_order(
            selected_classes,
            order,
            class_order_source,
            class_order_file,
            rng,
        )

    step_size = remaining_classes // incremental_steps
    tasks = [class_order[:base_classes]]
    offset = base_classes
    for _ in range(incremental_steps):
        tasks.append(class_order[offset : offset + step_size])
        offset += step_size

    normalization_mean, normalization_std = _normalization_values(cifar_normalization)
    return CIFAR100LTProtocol(
        train_base=train_base,
        test_base=test_base,
        selected_classes=selected_classes,
        class_counts=class_counts,
        class_order=class_order,
        tasks=tasks,
        train_indices_by_class=train_indices_by_class,
        test_indices_by_class=test_indices_by_class,
        train_transform=cifar_train_transform(cifar_normalization),
        eval_transform=cifar_eval_transform(cifar_normalization),
        seed=seed,
        rho=rho,
        order=order,
        count_assignment=count_assignment,
        class_order_source=class_order_source,
        class_order_file=class_order_file,
        cifar_normalization=cifar_normalization,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        max_train_per_class=effective_max,
    )
