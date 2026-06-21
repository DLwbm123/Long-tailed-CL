"""Evaluation metrics for LT-CIL smoke runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass
class EvalResult:
    top1: float
    balanced_acc: float
    macro_f1: float
    correct: int
    total: int
    per_class: Dict[int, Dict[str, int]]


def build_label_map(class_to_head_index: Mapping[int, int], device: torch.device) -> torch.Tensor:
    max_class_id = max(class_to_head_index)
    mapping = torch.full((max_class_id + 1,), -1, dtype=torch.long, device=device)
    for class_id, head_index in class_to_head_index.items():
        mapping[int(class_id)] = int(head_index)
    return mapping


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    class_to_head_index: Mapping[int, int],
    device: torch.device,
) -> EvalResult:
    model.eval()
    label_map = build_label_map(class_to_head_index, device)
    correct = 0
    total = 0
    per_class: Dict[int, Dict[str, int]] = {}
    head_to_class = {int(head_index): int(class_id) for class_id, head_index in class_to_head_index.items()}

    for images, labels_global in loader:
        images = images.to(device)
        labels_global = labels_global.to(device)
        labels_head = label_map[labels_global]
        if torch.any(labels_head < 0):
            bad = labels_global[labels_head < 0].detach().cpu().tolist()
            raise RuntimeError(f"Encountered labels outside the seen class map: {bad}")

        logits = model(images)
        predictions = logits.argmax(dim=1)
        matches = predictions.eq(labels_head)
        correct += int(matches.sum().item())
        total += int(labels_head.numel())

        for class_id_tensor in labels_global.unique():
            class_id = int(class_id_tensor.item())
            mask = labels_global == class_id_tensor
            stats = per_class.setdefault(class_id, {"correct": 0, "total": 0, "predicted": 0})
            stats["correct"] += int(matches[mask].sum().item())
            stats["total"] += int(mask.sum().item())
        for pred_head_tensor in predictions.unique():
            pred_head = int(pred_head_tensor.item())
            pred_class = head_to_class[pred_head]
            mask = predictions == pred_head_tensor
            stats = per_class.setdefault(pred_class, {"correct": 0, "total": 0, "predicted": 0})
            stats["predicted"] += int(mask.sum().item())

    top1 = 100.0 * correct / total if total else 0.0
    recalls: List[float] = []
    f1_values: List[float] = []
    for stats in per_class.values():
        class_total = int(stats["total"])
        class_predicted = int(stats.get("predicted", 0))
        class_correct = int(stats["correct"])
        if class_total <= 0:
            continue
        recall = class_correct / class_total
        precision = class_correct / class_predicted if class_predicted > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        recalls.append(recall)
        f1_values.append(f1)
    balanced_acc = 100.0 * float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = 100.0 * float(np.mean(f1_values)) if f1_values else 0.0
    return EvalResult(
        top1=top1,
        balanced_acc=balanced_acc,
        macro_f1=macro_f1,
        correct=correct,
        total=total,
        per_class=per_class,
    )


def group_accuracy(per_class: Mapping[int, Mapping[str, int]], class_ids: Iterable[int]) -> float | None:
    correct = 0
    total = 0
    for class_id in class_ids:
        stats = per_class.get(int(class_id))
        if stats is None:
            continue
        correct += int(stats["correct"])
        total += int(stats["total"])
    if total == 0:
        return None
    return 100.0 * correct / total


def prediction_task_histogram(
    per_class: Mapping[int, Mapping[str, int]],
    tasks: Sequence[Sequence[int]],
) -> Dict[str, int]:
    class_to_task = {
        int(class_id): task_idx
        for task_idx, task_classes in enumerate(tasks)
        for class_id in task_classes
    }
    hist = {f"task_{task_idx}": 0 for task_idx in range(len(tasks))}
    for class_id, stats in per_class.items():
        task_idx = class_to_task.get(int(class_id))
        if task_idx is None:
            continue
        hist[f"task_{task_idx}"] += int(stats.get("predicted", 0))
    return hist


def normalize_histogram(hist: Mapping[str, int], total: int) -> Dict[str, float]:
    if total <= 0:
        return {key: 0.0 for key in hist}
    return {key: float(value) / float(total) for key, value in hist.items()}


def frequency_groups(
    class_counts: Mapping[int, int],
    seen_classes: Sequence[int],
    many_threshold: int = 100,
    few_threshold: int = 20,
) -> Dict[str, List[int]]:
    groups = {"many": [], "medium": [], "few": []}
    for class_id in seen_classes:
        count = int(class_counts[int(class_id)])
        if count > many_threshold:
            groups["many"].append(int(class_id))
        elif count < few_threshold:
            groups["few"].append(int(class_id))
        else:
            groups["medium"].append(int(class_id))
    return groups


def forgetting(acc_matrix: Sequence[Sequence[float | None]], phase: int) -> float:
    if phase <= 0:
        return 0.0
    values: List[float] = []
    for task_idx in range(phase):
        current = acc_matrix[phase][task_idx]
        if current is None:
            continue
        history = [
            acc_matrix[row_idx][task_idx]
            for row_idx in range(task_idx, phase + 1)
            if acc_matrix[row_idx][task_idx] is not None
        ]
        if history:
            values.append(max(history) - current)
    return float(np.mean(values)) if values else 0.0
