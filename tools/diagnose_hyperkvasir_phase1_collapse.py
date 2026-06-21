#!/usr/bin/env python3
"""Evaluate HyperKvasir23 phase-1 old-task collapse from saved checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets import build_protocol
from src.models.resnet_cifar import resnet32
from src.utils.metrics import frequency_groups
from train import apply_method_aliases, build_parser


@dataclass
class EvalBundle:
    total: int
    correct: int
    old_total: int
    old_correct: int
    current_total: int
    current_correct: int
    old_pred_current: int
    current_pred_old: int
    per_class: Dict[int, Dict[str, int]]


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _namespace_from_checkpoint_args(raw_args: Mapping[str, object], data_root: str | None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args([])
    for key, value in raw_args.items():
        setattr(args, key, value)
    if data_root is not None:
        args.data_root = data_root
    args.download = False
    return apply_method_aliases(args)


def _make_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )


def _head_indices(class_ids: Iterable[int], class_to_head_index: Mapping[int, int], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [int(class_to_head_index[int(class_id)]) for class_id in class_ids],
        dtype=torch.long,
        device=device,
    )


def _build_label_map(class_to_head_index: Mapping[int, int], device: torch.device) -> torch.Tensor:
    max_class_id = max(int(class_id) for class_id in class_to_head_index)
    label_map = torch.full((max_class_id + 1,), -1, dtype=torch.long, device=device)
    for class_id, head_index in class_to_head_index.items():
        label_map[int(class_id)] = int(head_index)
    return label_map


def _safe_percent(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return 100.0 * float(numerator) / float(denominator)


def _balanced_and_f1(per_class: Mapping[int, Mapping[str, int]]) -> tuple[float, float]:
    recalls: List[float] = []
    f1_values: List[float] = []
    for stats in per_class.values():
        total = int(stats.get("total", 0))
        predicted = int(stats.get("predicted", 0))
        correct = int(stats.get("correct", 0))
        if total <= 0:
            continue
        recall = correct / total
        precision = correct / predicted if predicted > 0 else 0.0
        recalls.append(recall)
        f1_values.append(2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)
    balanced = 100.0 * sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = 100.0 * sum(f1_values) / len(f1_values) if f1_values else 0.0
    return balanced, macro_f1


def _group_accuracy(per_class: Mapping[int, Mapping[str, int]], class_ids: Sequence[int]) -> float | None:
    total = 0
    correct = 0
    for class_id in class_ids:
        stats = per_class.get(int(class_id))
        if not stats:
            continue
        total += int(stats.get("total", 0))
        correct += int(stats.get("correct", 0))
    return _safe_percent(correct, total)


def _empty_bundle() -> EvalBundle:
    return EvalBundle(
        total=0,
        correct=0,
        old_total=0,
        old_correct=0,
        current_total=0,
        current_correct=0,
        old_pred_current=0,
        current_pred_old=0,
        per_class={},
    )


def _update_per_class(
    per_class: Dict[int, Dict[str, int]],
    labels_global: torch.Tensor,
    predictions_head: torch.Tensor,
    matches: torch.Tensor,
    head_to_class: Mapping[int, int],
) -> None:
    for class_id_tensor in labels_global.unique():
        class_id = int(class_id_tensor.item())
        mask = labels_global == class_id_tensor
        stats = per_class.setdefault(class_id, {"correct": 0, "total": 0, "predicted": 0})
        stats["correct"] += int(matches[mask].sum().item())
        stats["total"] += int(mask.sum().item())
    for pred_head_tensor in predictions_head.unique():
        pred_head = int(pred_head_tensor.item())
        pred_class = int(head_to_class[pred_head])
        mask = predictions_head == pred_head_tensor
        stats = per_class.setdefault(pred_class, {"correct": 0, "total": 0, "predicted": 0})
        stats["predicted"] += int(mask.sum().item())


@torch.no_grad()
def _evaluate_masked(
    model: torch.nn.Module,
    loader: DataLoader,
    class_to_head_index: Mapping[int, int],
    old_classes: Sequence[int],
    current_classes: Sequence[int],
    device: torch.device,
    allowed_classes: Sequence[int] | None = None,
    current_alpha: float | None = None,
    old_beta: float | None = None,
) -> EvalBundle:
    model.eval()
    label_map = _build_label_map(class_to_head_index, device)
    head_to_class = {int(head_index): int(class_id) for class_id, head_index in class_to_head_index.items()}
    old_heads = set(int(class_to_head_index[int(class_id)]) for class_id in old_classes)
    current_heads = set(int(class_to_head_index[int(class_id)]) for class_id in current_classes)
    old_head_tensor = _head_indices(old_classes, class_to_head_index, device)
    current_head_tensor = _head_indices(current_classes, class_to_head_index, device)
    allowed_head_tensor = None
    if allowed_classes is not None:
        allowed_head_tensor = _head_indices(allowed_classes, class_to_head_index, device)

    result = _empty_bundle()
    for images, labels_global in loader:
        images = images.to(device)
        labels_global = labels_global.to(device)
        labels_head = label_map[labels_global]
        if torch.any(labels_head < 0):
            bad = labels_global[labels_head < 0].detach().cpu().tolist()
            raise RuntimeError(f"labels outside seen class map: {bad}")

        logits = model(images).clone()
        if current_alpha is not None:
            logits.index_copy_(1, current_head_tensor, logits.index_select(1, current_head_tensor) * float(current_alpha))
        if old_beta is not None:
            logits.index_copy_(1, old_head_tensor, logits.index_select(1, old_head_tensor) * float(old_beta))
        if allowed_head_tensor is not None:
            masked = torch.full_like(logits, -torch.inf)
            masked.index_copy_(1, allowed_head_tensor, logits.index_select(1, allowed_head_tensor))
            logits = masked
        predictions = logits.argmax(dim=1)
        matches = predictions.eq(labels_head)

        old_label_mask = torch.tensor([int(item) in old_heads for item in labels_head.detach().cpu().tolist()], device=device)
        current_label_mask = torch.tensor(
            [int(item) in current_heads for item in labels_head.detach().cpu().tolist()], device=device
        )
        pred_old_mask = torch.tensor([int(item) in old_heads for item in predictions.detach().cpu().tolist()], device=device)
        pred_current_mask = torch.tensor(
            [int(item) in current_heads for item in predictions.detach().cpu().tolist()], device=device
        )

        result.correct += int(matches.sum().item())
        result.total += int(labels_head.numel())
        result.old_total += int(old_label_mask.sum().item())
        result.old_correct += int(matches[old_label_mask].sum().item())
        result.current_total += int(current_label_mask.sum().item())
        result.current_correct += int(matches[current_label_mask].sum().item())
        result.old_pred_current += int((old_label_mask & pred_current_mask).sum().item())
        result.current_pred_old += int((current_label_mask & pred_old_mask).sum().item())
        _update_per_class(result.per_class, labels_global, predictions, matches, head_to_class)
    return result


def _bundle_row(
    method: str,
    mode: str,
    bundle: EvalBundle,
    class_counts: Mapping[int, int],
    seen_classes: Sequence[int],
    old_classes: Sequence[int],
    current_classes: Sequence[int],
) -> Dict[str, object]:
    balanced, macro_f1 = _balanced_and_f1(bundle.per_class)
    groups = frequency_groups(class_counts, seen_classes, many_threshold=700, few_threshold=70)
    return {
        "method": method,
        "mode": mode,
        "AccT": _safe_percent(bundle.correct, bundle.total),
        "balanced_acc": balanced,
        "macro_f1": macro_f1,
        "old_acc": _safe_percent(bundle.old_correct, bundle.old_total),
        "current_acc": _safe_percent(bundle.current_correct, bundle.current_total),
        "old_to_current_rate": _safe_percent(bundle.old_pred_current, bundle.old_total),
        "current_to_old_rate": _safe_percent(bundle.current_pred_old, bundle.current_total),
        "many_acc": _group_accuracy(bundle.per_class, groups["many"]),
        "medium_acc": _group_accuracy(bundle.per_class, groups["medium"]),
        "few_acc": _group_accuracy(bundle.per_class, groups["few"]),
        "total": bundle.total,
        "old_total": bundle.old_total,
        "current_total": bundle.current_total,
        "old_classes": list(map(int, old_classes)),
        "current_classes": list(map(int, current_classes)),
    }


@torch.no_grad()
def _logit_block_stats(
    model: torch.nn.Module,
    loader: DataLoader,
    class_to_head_index: Mapping[int, int],
    old_classes: Sequence[int],
    current_classes: Sequence[int],
    device: torch.device,
    sample_block: str,
) -> Dict[str, object]:
    model.eval()
    old_heads = _head_indices(old_classes, class_to_head_index, device)
    current_heads = _head_indices(current_classes, class_to_head_index, device)
    head_to_class = {int(head_index): int(class_id) for class_id, head_index in class_to_head_index.items()}
    max_old_values: List[torch.Tensor] = []
    max_current_values: List[torch.Tensor] = []
    current_minus_old: List[torch.Tensor] = []
    pred_hist: Dict[int, int] = {}
    opposite_wins = 0
    total = 0
    for images, _ in loader:
        images = images.to(device)
        logits = model(images)
        old_logits = logits.index_select(1, old_heads)
        current_logits = logits.index_select(1, current_heads)
        max_old, old_arg = old_logits.max(dim=1)
        max_current, current_arg = current_logits.max(dim=1)
        max_old_values.append(max_old.detach().cpu())
        max_current_values.append(max_current.detach().cpu())
        margin = max_current - max_old
        current_minus_old.append(margin.detach().cpu())
        total += int(images.shape[0])
        if sample_block == "old":
            opposite_wins += int((max_current > max_old).sum().item())
            pred_heads = current_heads[current_arg].detach().cpu().tolist()
            mask = (max_current > max_old).detach().cpu().tolist()
        else:
            opposite_wins += int((max_old > max_current).sum().item())
            pred_heads = old_heads[old_arg].detach().cpu().tolist()
            mask = (max_old > max_current).detach().cpu().tolist()
        for use_item, pred_head in zip(mask, pred_heads):
            if use_item:
                pred_class = int(head_to_class[int(pred_head)])
                pred_hist[pred_class] = pred_hist.get(pred_class, 0) + 1

    def mean_or_none(values: List[torch.Tensor]) -> float | None:
        if not values:
            return None
        return float(torch.cat(values).mean().item())

    return {
        "sample_block": sample_block,
        "mean_max_old_logit": mean_or_none(max_old_values),
        "mean_max_current_logit": mean_or_none(max_current_values),
        "mean_current_minus_old_margin": mean_or_none(current_minus_old),
        "frac_current_gt_old" if sample_block == "old" else "frac_old_gt_current": (
            float(opposite_wins) / float(total) if total else None
        ),
        "opposite_block_hist": {str(key): value for key, value in sorted(pred_hist.items())},
        "total": total,
    }


def _classifier_norms(
    model: torch.nn.Module,
    class_to_head_index: Mapping[int, int],
    old_classes: Sequence[int],
    current_classes: Sequence[int],
    device: torch.device,
) -> Dict[str, float | None]:
    if model.classifier is None:
        return {
            "old_head_norm_mean": None,
            "current_head_norm_mean": None,
            "current_old_norm_ratio": None,
        }
    old_heads = _head_indices(old_classes, class_to_head_index, device)
    current_heads = _head_indices(current_classes, class_to_head_index, device)
    weight = model.classifier.weight.detach()
    old_norm = torch.linalg.vector_norm(weight.index_select(0, old_heads), dim=1)
    current_norm = torch.linalg.vector_norm(weight.index_select(0, current_heads), dim=1)
    old_mean = float(old_norm.mean().item()) if old_norm.numel() else None
    current_mean = float(current_norm.mean().item()) if current_norm.numel() else None
    return {
        "old_head_norm_mean": old_mean,
        "current_head_norm_mean": current_mean,
        "current_old_norm_ratio": (
            current_mean / old_mean if current_mean is not None and old_mean not in {None, 0.0} else None
        ),
    }


def _per_class_count_rows(protocol, old_classes: Sequence[int], current_classes: Sequence[int]) -> List[Dict[str, object]]:
    old_set = set(map(int, old_classes))
    current_set = set(map(int, current_classes))
    rows: List[Dict[str, object]] = []
    class_names = getattr(protocol, "class_names", {})
    for class_id in protocol.selected_classes:
        class_id = int(class_id)
        if class_id in old_set:
            split = "old/base"
        elif class_id in current_set:
            split = "current/phase1"
        else:
            split = "future"
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_names.get(class_id, str(class_id)),
                "split": split,
                "train_count": len(protocol.train_indices_by_class.get(class_id, [])),
                "test_count": len(protocol.test_indices_by_class.get(class_id, [])),
            }
        )
    return rows


def _load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = resnet32().to(device)
    seen_classes = [int(class_id) for class_id in checkpoint["seen_classes"]]
    model.expand_classifier(len(seen_classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    class_to_head_index = {int(key): int(value) for key, value in checkpoint["class_to_head_index"].items()}
    return checkpoint, model, seen_classes, class_to_head_index


def _format_float(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def _markdown_table(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format_float(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def _diagnose_method(
    method: str,
    run_root: Path,
    ckpt_root: Path,
    data_root: str | None,
    output_dir: Path,
    device: torch.device,
    batch_size_override: int | None,
    num_workers: int,
    current_alphas: Sequence[float],
    old_betas: Sequence[float],
) -> Dict[str, object]:
    checkpoint_path = ckpt_root / method / "model_phase_1.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    checkpoint, model, seen_classes, class_to_head_index = _load_model(checkpoint_path, device)
    args = _namespace_from_checkpoint_args(checkpoint.get("args", {}), data_root)
    protocol = build_protocol(args)
    old_classes = [int(class_id) for class_id in protocol.tasks[0]]
    current_classes = [int(class_id) for class_id in protocol.tasks[1]]
    expected_seen = old_classes + current_classes
    if seen_classes != expected_seen:
        raise RuntimeError(f"{method}: checkpoint seen_classes {seen_classes} != protocol phase1 seen {expected_seen}")
    if set(class_to_head_index) != set(seen_classes):
        raise RuntimeError(f"{method}: class_to_head_index keys do not match seen classes")

    batch_size = int(batch_size_override or getattr(args, "batch_size", 64))
    old_loader = _make_loader(protocol.test_dataset_for_classes(old_classes), batch_size, num_workers)
    current_loader = _make_loader(protocol.test_dataset_for_classes(current_classes), batch_size, num_workers)
    seen_loader = _make_loader(protocol.test_dataset_for_classes(seen_classes), batch_size, num_workers)

    oracle_rows = [
        _bundle_row(
            method,
            "all_seen",
            _evaluate_masked(model, seen_loader, class_to_head_index, old_classes, current_classes, device),
            protocol.class_counts,
            seen_classes,
            old_classes,
            current_classes,
        ),
        _bundle_row(
            method,
            "old_only_on_old_samples",
            _evaluate_masked(
                model,
                old_loader,
                class_to_head_index,
                old_classes,
                current_classes,
                device,
                allowed_classes=old_classes,
            ),
            protocol.class_counts,
            seen_classes,
            old_classes,
            current_classes,
        ),
        _bundle_row(
            method,
            "current_only_on_current_samples",
            _evaluate_masked(
                model,
                current_loader,
                class_to_head_index,
                old_classes,
                current_classes,
                device,
                allowed_classes=current_classes,
            ),
            protocol.class_counts,
            seen_classes,
            old_classes,
            current_classes,
        ),
    ]

    logit_rows = []
    norms = _classifier_norms(model, class_to_head_index, old_classes, current_classes, device)
    for block, loader in [("old", old_loader), ("current", current_loader)]:
        row = {"method": method, **_logit_block_stats(model, loader, class_to_head_index, old_classes, current_classes, device, block)}
        row.update(norms)
        logit_rows.append(row)

    current_scale_rows: List[Dict[str, object]] = []
    for alpha in current_alphas:
        bundle = _evaluate_masked(
            model,
            seen_loader,
            class_to_head_index,
            old_classes,
            current_classes,
            device,
            current_alpha=float(alpha),
        )
        row = _bundle_row(method, "scale_current_logits", bundle, protocol.class_counts, seen_classes, old_classes, current_classes)
        row["alpha"] = float(alpha)
        current_scale_rows.append(row)

    old_scale_rows: List[Dict[str, object]] = []
    for beta in old_betas:
        bundle = _evaluate_masked(
            model,
            seen_loader,
            class_to_head_index,
            old_classes,
            current_classes,
            device,
            old_beta=float(beta),
        )
        row = _bundle_row(method, "scale_old_logits", bundle, protocol.class_counts, seen_classes, old_classes, current_classes)
        row["beta"] = float(beta)
        old_scale_rows.append(row)

    method_dir = output_dir / method
    _write_csv(method_dir / "oracle_mask_metrics.csv", oracle_rows)
    _write_csv(method_dir / "logit_block_diagnostics.csv", logit_rows)
    _write_csv(method_dir / "calibration_current_scale.csv", current_scale_rows)
    _write_csv(method_dir / "calibration_old_scale.csv", old_scale_rows)

    return {
        "method": method,
        "checkpoint": str(checkpoint_path),
        "run_dir": str(run_root / method),
        "oracle_rows": oracle_rows,
        "logit_rows": logit_rows,
        "current_scale_rows": current_scale_rows,
        "old_scale_rows": old_scale_rows,
        "classifier_norms": norms,
        "args": vars(args),
        "seen_classes": seen_classes,
        "old_classes": old_classes,
        "current_classes": current_classes,
        "protocol_preview": protocol.preview(),
        "class_count_rows": _per_class_count_rows(protocol, old_classes, current_classes),
    }


def _audit_split(protocol, phase: int = 1) -> Dict[str, object]:
    old_classes = [int(class_id) for class_id in protocol.tasks[0]]
    current_classes = [int(class_id) for class_id in protocol.tasks[phase]]
    seen_classes = [int(class_id) for task in protocol.tasks[: phase + 1] for class_id in task]
    all_task_classes = [int(class_id) for task in protocol.tasks for class_id in task]
    train_labels = sorted({int(record.target) for record in protocol.train_records})
    test_labels = sorted({int(record.target) for record in protocol.test_records})
    overlaps = {
        "old_current_overlap": sorted(set(old_classes) & set(current_classes)),
        "seen_future_overlap": sorted(set(seen_classes) & (set(all_task_classes) - set(seen_classes))),
        "missing_from_tasks": sorted(set(protocol.selected_classes) - set(all_task_classes)),
        "duplicates_in_tasks": len(all_task_classes) - len(set(all_task_classes)),
        "train_test_label_symmetric_difference": sorted(set(train_labels) ^ set(test_labels)),
    }
    return {
        "class_to_idx": {str(k): v for k, v in sorted(getattr(protocol, "class_names", {}).items())},
        "class_order": [int(class_id) for class_id in protocol.class_order],
        "base_classes": old_classes,
        "phase1_current_classes": current_classes,
        "old_classes_at_phase1": old_classes,
        "seen_classes_at_phase1": seen_classes,
        "future_classes_after_phase1": [
            int(class_id) for task in protocol.tasks[phase + 1 :] for class_id in task
        ],
        "tasks": [[int(class_id) for class_id in task] for task in protocol.tasks],
        "class_counts": {str(k): int(v) for k, v in sorted(protocol.class_counts.items())},
        "train_counts_per_class": {
            str(class_id): len(protocol.train_indices_by_class[int(class_id)])
            for class_id in sorted(protocol.selected_classes)
        },
        "test_counts_per_class": {
            str(class_id): len(protocol.test_indices_by_class[int(class_id)])
            for class_id in sorted(protocol.selected_classes)
        },
        "overlap_missing_checks": overlaps,
        "eval_seen_mask": {
            "phase1_seen_classes": seen_classes,
            "uses_unseen_future_classes": False,
            "note": "diagnostic evaluator masks logits to checkpoint classifier rows for phase1 seen classes only",
        },
    }


def _build_report(
    output_dir: Path,
    command: str,
    split_audit: Mapping[str, object],
    method_results: Sequence[Mapping[str, object]],
    ran_freeze_bn: bool,
    ran_tiny_replay: bool,
) -> str:
    oracle_rows: List[Mapping[str, object]] = []
    logit_rows: List[Mapping[str, object]] = []
    best_current_scale_rows: List[Mapping[str, object]] = []
    best_old_scale_rows: List[Mapping[str, object]] = []
    conclusion_bits: List[str] = []
    for result in method_results:
        oracle_rows.extend(result["oracle_rows"])
        logit_rows.extend(result["logit_rows"])
        current_rows = list(result["current_scale_rows"])
        old_rows = list(result["old_scale_rows"])
        best_current_scale_rows.extend(
            sorted(current_rows, key=lambda row: (row.get("old_acc") or -1.0, row.get("AccT") or -1.0), reverse=True)[:3]
        )
        best_old_scale_rows.extend(
            sorted(old_rows, key=lambda row: (row.get("old_acc") or -1.0, row.get("AccT") or -1.0), reverse=True)[:3]
        )

        method = str(result["method"])
        all_seen = next(row for row in result["oracle_rows"] if row["mode"] == "all_seen")
        old_only = next(row for row in result["oracle_rows"] if row["mode"] == "old_only_on_old_samples")
        current_only = next(row for row in result["oracle_rows"] if row["mode"] == "current_only_on_current_samples")
        if (all_seen.get("old_acc") or 0.0) <= 1.0 and (old_only.get("AccT") or 0.0) >= 50.0:
            conclusion_bits.append(
                f"- `{method}`: old-only accuracy is high ({old_only.get('AccT'):.2f}) while all-seen old acc is {all_seen.get('old_acc'):.2f}; collapse is primarily task-block/logit calibration bias."
            )
        elif (old_only.get("AccT") or 0.0) <= 5.0:
            conclusion_bits.append(
                f"- `{method}`: old-only accuracy is also near zero ({old_only.get('AccT'):.2f}); representation forgetting or BN/backbone drift is likely."
            )
        if (current_only.get("AccT") or 0.0) <= 50.0:
            conclusion_bits.append(
                f"- `{method}`: current-only accuracy is low ({current_only.get('AccT'):.2f}); phase1 current classes are undertrained or very hard."
            )

    lines = [
        "# HyperKvasir23 Phase1 Collapse Diagnosis",
        "",
        "## Commands Run",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Class Mapping And Split Audit",
        "",
        f"- Class order: `{split_audit['class_order']}`",
        f"- Base classes: `{split_audit['base_classes']}`",
        f"- Phase1 current classes: `{split_audit['phase1_current_classes']}`",
        f"- Seen classes at phase1: `{split_audit['seen_classes_at_phase1']}`",
        f"- Overlap/missing checks: `{split_audit['overlap_missing_checks']}`",
        f"- Evaluation seen mask: `{split_audit['eval_seen_mask']}`",
        "",
        "Per-class counts are saved in `class_split_audit.json` and `class_counts_by_split.csv`.",
        "",
        "## Checkpoint Availability",
        "",
    ]
    for result in method_results:
        lines.append(f"- `{result['method']}`: `{result['checkpoint']}`")
    lines.extend(
        [
            "",
            "## Oracle-Mask Evaluation",
            "",
            _markdown_table(
                oracle_rows,
                [
                    "method",
                    "mode",
                    "AccT",
                    "balanced_acc",
                    "macro_f1",
                    "old_acc",
                    "current_acc",
                    "old_to_current_rate",
                    "current_to_old_rate",
                ],
            ),
            "",
            "## Logit Block Diagnostics",
            "",
            _markdown_table(
                logit_rows,
                [
                    "method",
                    "sample_block",
                    "mean_max_old_logit",
                    "mean_max_current_logit",
                    "mean_current_minus_old_margin",
                    "frac_current_gt_old",
                    "frac_old_gt_current",
                    "old_head_norm_mean",
                    "current_head_norm_mean",
                    "current_old_norm_ratio",
                ],
            ),
            "",
            "Full prediction histograms are saved in each method's `logit_block_diagnostics.csv`.",
            "",
            "## Calibration Sweep Summary",
            "",
            "Top current-logit scaling rows by recovered old accuracy:",
            "",
            _markdown_table(
                best_current_scale_rows,
                [
                    "method",
                    "alpha",
                    "AccT",
                    "balanced_acc",
                    "macro_f1",
                    "old_acc",
                    "current_acc",
                    "old_to_current_rate",
                    "current_to_old_rate",
                ],
            ),
            "",
            "Top old-logit scaling rows by recovered old accuracy:",
            "",
            _markdown_table(
                best_old_scale_rows,
                [
                    "method",
                    "beta",
                    "AccT",
                    "balanced_acc",
                    "macro_f1",
                    "old_acc",
                    "current_acc",
                    "old_to_current_rate",
                    "current_to_old_rate",
                ],
            ),
            "",
            "Complete sweeps are saved under each method directory:",
            "`calibration_current_scale.csv` and `calibration_old_scale.csv`.",
            "",
            "## Freeze / BN Diagnostic",
            "",
            "Not run." if not ran_freeze_bn else "Run; see appended artifacts.",
            "",
            "## Tiny Exemplar Diagnostic",
            "",
            "Not run." if not ran_tiny_replay else "Run; see appended artifacts.",
            "",
            "## Final Conclusion",
            "",
            "\n".join(conclusion_bits) if conclusion_bits else "- Inconclusive from current checkpoints.",
            "",
            "Recommendation: do not expand modules or run full settings yet. If oracle old-only is high, test task-block calibration; if old-only is low, prioritize freeze BN/backbone and then tiny exemplar replay.",
            "",
            "## Artifact Paths",
            "",
            f"- Diagnosis root: `{output_dir}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--ckpt-root", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", nargs="+", default=["finetune", "taconcm_stage3_calib_tailanchor", "taconcm_stage4_full"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--current-alphas",
        default="1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1",
    )
    parser.add_argument("--old-betas", default="1.0,1.1,1.2,1.5,2.0,3.0")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    ckpt_root = Path(args.ckpt_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    current_alphas = [float(item) for item in args.current_alphas.split(",") if item.strip()]
    old_betas = [float(item) for item in args.old_betas.split(",") if item.strip()]
    command = shlex.join([sys.executable, *sys.argv])

    first_checkpoint = ckpt_root / args.methods[0] / "model_phase_1.pt"
    first_payload = torch.load(first_checkpoint, map_location="cpu")
    first_args = _namespace_from_checkpoint_args(first_payload.get("args", {}), args.data_root)
    protocol = build_protocol(first_args)
    split_audit = _audit_split(protocol)
    _write_json(output_dir / "class_split_audit.json", split_audit)
    _write_csv(
        output_dir / "class_counts_by_split.csv",
        _per_class_count_rows(protocol, split_audit["old_classes_at_phase1"], split_audit["phase1_current_classes"]),
    )

    results = []
    for method in args.methods:
        results.append(
            _diagnose_method(
                method=method,
                run_root=run_root,
                ckpt_root=ckpt_root,
                data_root=args.data_root,
                output_dir=output_dir,
                device=device,
                batch_size_override=args.batch_size,
                num_workers=args.num_workers,
                current_alphas=current_alphas,
                old_betas=old_betas,
            )
        )

    all_oracle_rows = [row for result in results for row in result["oracle_rows"]]
    all_logit_rows = [row for result in results for row in result["logit_rows"]]
    all_current_rows = [row for result in results for row in result["current_scale_rows"]]
    all_old_rows = [row for result in results for row in result["old_scale_rows"]]
    _write_csv(output_dir / "oracle_mask_metrics_all.csv", all_oracle_rows)
    _write_csv(output_dir / "logit_block_diagnostics_all.csv", all_logit_rows)
    _write_csv(output_dir / "calibration_current_scale_all.csv", all_current_rows)
    _write_csv(output_dir / "calibration_old_scale_all.csv", all_old_rows)
    summary = {
        "command": command,
        "run_root": str(run_root),
        "ckpt_root": str(ckpt_root),
        "data_root": args.data_root,
        "methods": args.methods,
        "split_audit": split_audit,
        "oracle_mask_metrics": all_oracle_rows,
        "logit_block_diagnostics": all_logit_rows,
        "current_scale_sweep": all_current_rows,
        "old_scale_sweep": all_old_rows,
        "checkpoint_availability": {result["method"]: result["checkpoint"] for result in results},
        "ran_freeze_bn_diagnostic": False,
        "ran_tiny_exemplar_diagnostic": False,
    }
    _write_json(output_dir / "diagnosis_summary.json", summary)
    report = _build_report(
        output_dir=output_dir,
        command=command,
        split_audit=split_audit,
        method_results=results,
        ran_freeze_bn=False,
        ran_tiny_replay=False,
    )
    (output_dir / "HYPERKVASIR23_PHASE1_COLLAPSE_DIAGNOSIS.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
