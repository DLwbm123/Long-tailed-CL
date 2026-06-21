#!/usr/bin/env python3
"""Bounded HyperKvasir23 medical feature-drift matching diagnostic."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets import build_protocol
from src.models.resnet_cifar import resnet32
from src.utils.seed import set_seed
from train import apply_method_aliases, build_parser


@dataclass
class EvalResult:
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


def _write_json(path: Path, payload) -> None:
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


def _format_float(value) -> str:
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


def _make_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args([])
    args.dataset = "hyper_kvasir23"
    args.data_root = cli_args.data_root
    args.output = str(cli_args.output_dir)
    args.ckpt_dir = str(cli_args.ckpt_dir)
    args.cache_dir = str(cli_args.cache_dir) if cli_args.cache_dir else None
    args.method = "finetune"
    args.order = "shuffled"
    args.seed = int(cli_args.seed)
    args.base_classes = 13
    args.incremental_steps = 5
    args.max_phases = 2
    args.epochs = int(cli_args.epochs)
    args.batch_size = int(cli_args.batch_size)
    args.num_workers = int(cli_args.num_workers)
    args.lr = float(cli_args.lr)
    args.scheduler = "cosine"
    args.device = cli_args.device
    args.download = False
    return apply_method_aliases(args)


def _make_loader(dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    def seed_worker(worker_id: int) -> None:
        torch.manual_seed(int(seed) + int(worker_id))

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        worker_init_fn=seed_worker if int(num_workers) > 0 else None,
    )


def _label_map(class_to_head: Mapping[int, int], device: torch.device) -> torch.Tensor:
    mapping = torch.full((max(class_to_head) + 1,), -1, dtype=torch.long, device=device)
    for class_id, head_idx in class_to_head.items():
        mapping[int(class_id)] = int(head_idx)
    return mapping


def _head_indices(class_ids: Iterable[int], class_to_head: Mapping[int, int], device: torch.device) -> torch.Tensor:
    return torch.tensor([int(class_to_head[int(class_id)]) for class_id in class_ids], dtype=torch.long, device=device)


def _set_bn_eval(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def _freeze_backbone_and_bn(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("classifier.")
    _set_bn_eval(model)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if module.weight is not None:
                module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False


def _build_scheduler(optimizer: torch.optim.Optimizer, epochs: int):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))


def _class_to_head_from_seen(seen_classes: Sequence[int]) -> Dict[int, int]:
    return {int(class_id): index for index, class_id in enumerate(seen_classes)}


def _train_ce_phase(
    model: nn.Module,
    loader: DataLoader,
    class_to_head: Mapping[int, int],
    device: torch.device,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    freeze_bn: bool = False,
) -> Dict[str, object]:
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(lr),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
    )
    scheduler = _build_scheduler(optimizer, epochs)
    label_map = _label_map(class_to_head, device)
    epoch_rows: List[Dict[str, float]] = []
    for epoch in range(int(epochs)):
        model.train()
        if freeze_bn:
            _set_bn_eval(model)
        total = 0
        correct = 0
        loss_sum = 0.0
        ce_sum = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            labels_head = label_map[labels]
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            ce_loss = F.cross_entropy(logits, labels_head)
            ce_loss.backward()
            optimizer.step()
            loss_sum += float(ce_loss.detach().cpu().item()) * int(labels.numel())
            ce_sum += float(ce_loss.detach().cpu().item()) * int(labels.numel())
            pred = logits.argmax(dim=1)
            correct += int(pred.eq(labels_head).sum().item())
            total += int(labels.numel())
        scheduler.step()
        epoch_rows.append(
            {
                "epoch": epoch + 1,
                "loss": loss_sum / max(total, 1),
                "ce_loss": ce_sum / max(total, 1),
                "train_acc": 100.0 * correct / max(total, 1),
            }
        )
    return {
        "epoch_rows": epoch_rows,
        "train_loss_last_epoch": epoch_rows[-1]["loss"],
        "train_ce_loss_last_epoch": epoch_rows[-1]["ce_loss"],
        "train_acc_last_epoch": epoch_rows[-1]["train_acc"],
    }


@torch.no_grad()
def _compute_anchors(
    model: nn.Module,
    loader: DataLoader,
    class_ids: Sequence[int],
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    model.eval()
    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {int(class_id): 0 for class_id in class_ids}
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        features = model.extract_features(images).detach()
        for class_id in labels.unique():
            class_int = int(class_id.item())
            mask = labels == class_id
            sums[class_int] = sums.get(class_int, torch.zeros_like(features[0])) + features[mask].sum(dim=0)
            counts[class_int] = counts.get(class_int, 0) + int(mask.sum().item())
    anchors = {}
    for class_id in class_ids:
        class_id = int(class_id)
        if counts.get(class_id, 0) <= 0:
            continue
        anchors[class_id] = (sums[class_id] / counts[class_id]).detach().clone()
    return anchors


def _anchor_stats(anchors: Mapping[int, torch.Tensor]) -> Dict[str, object]:
    if not anchors:
        return {
            "num_anchors": 0,
            "finite": False,
            "norm_mean": None,
            "norm_min": None,
            "norm_max": None,
        }
    matrix = torch.stack([value.detach().float().cpu() for _, value in sorted(anchors.items())])
    norms = torch.linalg.vector_norm(matrix, dim=1)
    return {
        "num_anchors": int(matrix.shape[0]),
        "finite": bool(torch.isfinite(matrix).all().item()),
        "norm_mean": float(norms.mean().item()),
        "norm_min": float(norms.min().item()),
        "norm_max": float(norms.max().item()),
        "class_ids": [int(class_id) for class_id in sorted(anchors)],
    }


def _select_exemplars(protocol, old_classes: Sequence[int], exemplars_per_class: int) -> tuple[List[int], List[Dict[str, object]]]:
    indices: List[int] = []
    rows: List[Dict[str, object]] = []
    for class_id in old_classes:
        class_id = int(class_id)
        selected = list(protocol.train_indices_by_class[class_id])[: int(exemplars_per_class)]
        for index in selected:
            record = protocol.train_records[int(index)]
            indices.append(int(index))
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": getattr(protocol, "class_names", {}).get(class_id, str(class_id)),
                    "train_index": int(index),
                    "path": str(record.path),
                }
            )
    return indices, rows


class _Cycler:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iterator: Iterator | None = None

    def next(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def _fdm_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    anchors: Mapping[int, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    features = model.extract_features(images)
    targets = torch.stack([anchors[int(label.item())].to(device=device, dtype=features.dtype) for label in labels])
    return (1.0 - F.cosine_similarity(features, targets, dim=1)).mean()


def _probe_fdm_grad(
    model: nn.Module,
    loader: DataLoader,
    anchors: Mapping[int, torch.Tensor],
    device: torch.device,
) -> Dict[str, object]:
    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)
    for parameter in model.parameters():
        parameter.grad = None
    loss = _fdm_loss(model, images, labels, anchors, device)
    grad_norm = 0.0
    grad_params = 0
    if loss.requires_grad:
        loss.backward()
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                value = float(parameter.grad.detach().abs().sum().item())
                grad_norm += value
                if value > 0.0:
                    grad_params += 1
    for parameter in model.parameters():
        parameter.grad = None
    return {
        "fdm_probe_loss": float(loss.detach().cpu().item()),
        "fdm_probe_requires_grad": bool(loss.requires_grad),
        "fdm_probe_trainable_grad_l1": grad_norm,
        "fdm_probe_nonzero_grad_params": int(grad_params),
        "fdm_probe_nonzero_grad": bool(grad_norm > 0.0),
    }


def _train_phase1_branch(
    model: nn.Module,
    current_loader: DataLoader,
    exemplar_loader: DataLoader,
    class_to_head: Mapping[int, int],
    anchors: Mapping[int, torch.Tensor],
    device: torch.device,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    fdm_lambda: float,
    use_fdm: bool,
) -> Dict[str, object]:
    _freeze_backbone_and_bn(model)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(lr),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
    )
    scheduler = _build_scheduler(optimizer, epochs)
    label_map = _label_map(class_to_head, device)
    cycler = _Cycler(exemplar_loader)
    totals = {
        "loss": 0.0,
        "ce": 0.0,
        "fdm_raw": 0.0,
        "fdm_weighted": 0.0,
        "samples": 0,
        "fdm_batches": 0,
        "fdm_nonzero_value_batches": 0,
        "fdm_requires_grad_batches": 0,
    }
    epoch_rows: List[Dict[str, float]] = []
    for epoch in range(int(epochs)):
        model.train()
        _set_bn_eval(model)
        total = 0
        correct = 0
        loss_sum = 0.0
        ce_sum = 0.0
        fdm_sum = 0.0
        fdm_weighted_sum = 0.0
        for images, labels in current_loader:
            images = images.to(device)
            labels = labels.to(device)
            labels_head = label_map[labels]
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            ce_loss = F.cross_entropy(logits, labels_head)
            fdm = torch.zeros((), device=device)
            if use_fdm and float(fdm_lambda) > 0.0:
                old_images, old_labels = cycler.next()
                old_images = old_images.to(device)
                old_labels = old_labels.to(device)
                fdm = _fdm_loss(model, old_images, old_labels, anchors, device)
                totals["fdm_batches"] += 1
                if bool(fdm.requires_grad):
                    totals["fdm_requires_grad_batches"] += 1
                if float(fdm.detach().cpu().item()) > 0.0:
                    totals["fdm_nonzero_value_batches"] += 1
            weighted_fdm = float(fdm_lambda) * fdm
            loss = ce_loss + weighted_fdm
            loss.backward()
            optimizer.step()

            batch = int(labels.numel())
            loss_sum += float(loss.detach().cpu().item()) * batch
            ce_sum += float(ce_loss.detach().cpu().item()) * batch
            fdm_sum += float(fdm.detach().cpu().item()) * batch
            fdm_weighted_sum += float(weighted_fdm.detach().cpu().item()) * batch
            pred = logits.argmax(dim=1)
            correct += int(pred.eq(labels_head).sum().item())
            total += batch
        scheduler.step()
        totals["loss"] += loss_sum
        totals["ce"] += ce_sum
        totals["fdm_raw"] += fdm_sum
        totals["fdm_weighted"] += fdm_weighted_sum
        totals["samples"] += total
        epoch_rows.append(
            {
                "epoch": epoch + 1,
                "loss": loss_sum / max(total, 1),
                "ce_loss": ce_sum / max(total, 1),
                "fdm_raw": fdm_sum / max(total, 1),
                "fdm_weighted": fdm_weighted_sum / max(total, 1),
                "fdm_to_ce_ratio": fdm_weighted_sum / ce_sum if ce_sum else None,
                "train_acc": 100.0 * correct / max(total, 1),
            }
        )
    samples = int(totals["samples"])
    ce = totals["ce"] / max(samples, 1)
    fdm_weighted = totals["fdm_weighted"] / max(samples, 1)
    return {
        "epoch_rows": epoch_rows,
        "train_loss": totals["loss"] / max(samples, 1),
        "train_ce_loss": ce,
        "fdm_loss_raw": totals["fdm_raw"] / max(samples, 1),
        "fdm_loss_weighted": fdm_weighted,
        "fdm_to_ce_ratio": fdm_weighted / ce if ce else None,
        "fdm_batches": int(totals["fdm_batches"]),
        "fdm_requires_grad_batches": int(totals["fdm_requires_grad_batches"]),
        "fdm_nonzero_value_batches": int(totals["fdm_nonzero_value_batches"]),
        "train_acc_last_epoch": epoch_rows[-1]["train_acc"],
    }


def _empty_eval() -> EvalResult:
    return EvalResult(0, 0, 0, 0, 0, 0, 0, 0, {})


def _safe_percent(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return 100.0 * float(numerator) / float(denominator)


def _balanced_f1(per_class: Mapping[int, Mapping[str, int]]) -> tuple[float, float]:
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
    return (
        100.0 * sum(recalls) / len(recalls) if recalls else 0.0,
        100.0 * sum(f1_values) / len(f1_values) if f1_values else 0.0,
    )


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_to_head: Mapping[int, int],
    old_classes: Sequence[int],
    current_classes: Sequence[int],
    device: torch.device,
    allowed_classes: Sequence[int] | None = None,
    current_alpha: float | None = None,
) -> EvalResult:
    model.eval()
    label_map = _label_map(class_to_head, device)
    head_to_class = {int(head): int(class_id) for class_id, head in class_to_head.items()}
    old_heads = set(int(class_to_head[int(class_id)]) for class_id in old_classes)
    current_heads = set(int(class_to_head[int(class_id)]) for class_id in current_classes)
    current_head_tensor = _head_indices(current_classes, class_to_head, device)
    allowed_head_tensor = _head_indices(allowed_classes, class_to_head, device) if allowed_classes else None
    result = _empty_eval()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        labels_head = label_map[labels]
        logits = model(images).clone()
        if current_alpha is not None:
            logits.index_copy_(1, current_head_tensor, logits.index_select(1, current_head_tensor) * float(current_alpha))
        if allowed_head_tensor is not None:
            masked = torch.full_like(logits, -torch.inf)
            masked.index_copy_(1, allowed_head_tensor, logits.index_select(1, allowed_head_tensor))
            logits = masked
        preds = logits.argmax(dim=1)
        matches = preds.eq(labels_head)
        old_label_mask = torch.tensor([int(item) in old_heads for item in labels_head.detach().cpu().tolist()], device=device)
        current_label_mask = torch.tensor(
            [int(item) in current_heads for item in labels_head.detach().cpu().tolist()], device=device
        )
        pred_old_mask = torch.tensor([int(item) in old_heads for item in preds.detach().cpu().tolist()], device=device)
        pred_current_mask = torch.tensor([int(item) in current_heads for item in preds.detach().cpu().tolist()], device=device)
        result.correct += int(matches.sum().item())
        result.total += int(labels.numel())
        result.old_total += int(old_label_mask.sum().item())
        result.old_correct += int(matches[old_label_mask].sum().item())
        result.current_total += int(current_label_mask.sum().item())
        result.current_correct += int(matches[current_label_mask].sum().item())
        result.old_pred_current += int((old_label_mask & pred_current_mask).sum().item())
        result.current_pred_old += int((current_label_mask & pred_old_mask).sum().item())
        for class_tensor in labels.unique():
            class_id = int(class_tensor.item())
            mask = labels == class_tensor
            stats = result.per_class.setdefault(class_id, {"correct": 0, "total": 0, "predicted": 0})
            stats["correct"] += int(matches[mask].sum().item())
            stats["total"] += int(mask.sum().item())
        for pred_tensor in preds.unique():
            pred_head = int(pred_tensor.item())
            pred_class = int(head_to_class[pred_head])
            mask = preds == pred_tensor
            stats = result.per_class.setdefault(pred_class, {"correct": 0, "total": 0, "predicted": 0})
            stats["predicted"] += int(mask.sum().item())
    return result


def _eval_row(method: str, mode: str, result: EvalResult, alpha: float | None = None) -> Dict[str, object]:
    bal, macro = _balanced_f1(result.per_class)
    row = {
        "method": method,
        "mode": mode,
        "AccT": _safe_percent(result.correct, result.total),
        "balanced_acc": bal,
        "macro_f1": macro,
        "old_acc": _safe_percent(result.old_correct, result.old_total),
        "current_acc": _safe_percent(result.current_correct, result.current_total),
        "old_to_current_rate": _safe_percent(result.old_pred_current, result.old_total),
        "current_to_old_rate": _safe_percent(result.current_pred_old, result.current_total),
        "total": result.total,
    }
    if alpha is not None:
        row["current_alpha"] = float(alpha)
    return row


@torch.no_grad()
def _feature_anchor_cosine(
    model: nn.Module,
    loader: DataLoader,
    anchors: Mapping[int, torch.Tensor],
    device: torch.device,
) -> Dict[str, float | None]:
    model.eval()
    values: List[torch.Tensor] = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        features = model.extract_features(images)
        targets = torch.stack([anchors[int(label.item())].to(device=device, dtype=features.dtype) for label in labels])
        values.append(F.cosine_similarity(features, targets, dim=1).detach().cpu())
    if not values:
        return {"cos_mean": None, "cos_min": None, "cos_max": None}
    joined = torch.cat(values)
    return {
        "cos_mean": float(joined.mean().item()),
        "cos_min": float(joined.min().item()),
        "cos_max": float(joined.max().item()),
    }


def _classifier_norm_alpha(model: nn.Module, class_to_head: Mapping[int, int], old_classes: Sequence[int], current_classes: Sequence[int], device: torch.device) -> Dict[str, float | None]:
    if model.classifier is None:
        return {"old_norm": None, "current_norm": None, "head_norm_current_alpha": None}
    old_heads = _head_indices(old_classes, class_to_head, device)
    current_heads = _head_indices(current_classes, class_to_head, device)
    weights = model.classifier.weight.detach()
    old_norm = torch.linalg.vector_norm(weights.index_select(0, old_heads), dim=1).mean()
    current_norm = torch.linalg.vector_norm(weights.index_select(0, current_heads), dim=1).mean()
    old_value = float(old_norm.item())
    current_value = float(current_norm.item())
    return {
        "old_norm": old_value,
        "current_norm": current_value,
        "head_norm_current_alpha": old_value / current_value if current_value else None,
    }


def _run_branch_evals(
    method: str,
    model: nn.Module,
    seen_loader: DataLoader,
    old_loader: DataLoader,
    current_loader: DataLoader,
    class_to_head: Mapping[int, int],
    old_classes: Sequence[int],
    current_classes: Sequence[int],
    device: torch.device,
    alphas: Sequence[float],
) -> Dict[str, object]:
    seen_classes = [*old_classes, *current_classes]
    oracle_rows = [
        _eval_row(method, "all_seen", _evaluate(model, seen_loader, class_to_head, old_classes, current_classes, device)),
        _eval_row(
            method,
            "old_only_on_old",
            _evaluate(model, old_loader, class_to_head, old_classes, current_classes, device, allowed_classes=old_classes),
        ),
        _eval_row(
            method,
            "current_only_on_current",
            _evaluate(
                model,
                current_loader,
                class_to_head,
                old_classes,
                current_classes,
                device,
                allowed_classes=current_classes,
            ),
        ),
    ]
    sweep_rows = [
        _eval_row(
            method,
            "scale_current_logits",
            _evaluate(model, seen_loader, class_to_head, old_classes, current_classes, device, current_alpha=alpha),
            alpha=alpha,
        )
        for alpha in alphas
    ]
    norm_info = _classifier_norm_alpha(model, class_to_head, old_classes, current_classes, device)
    head_alpha = norm_info.get("head_norm_current_alpha")
    head_norm_row = None
    if head_alpha is not None:
        head_norm_row = _eval_row(
            method,
            "head_norm_current_alpha",
            _evaluate(
                model,
                seen_loader,
                class_to_head,
                old_classes,
                current_classes,
                device,
                current_alpha=float(head_alpha),
            ),
            alpha=float(head_alpha),
        )
        head_norm_row.update(norm_info)
    per_class_rows = []
    all_seen = _evaluate(model, seen_loader, class_to_head, old_classes, current_classes, device)
    for class_id, stats in sorted(all_seen.per_class.items()):
        total = int(stats.get("total", 0))
        per_class_rows.append(
            {
                "method": method,
                "class_id": int(class_id),
                "recall": 100.0 * int(stats.get("correct", 0)) / total if total else None,
                "correct": int(stats.get("correct", 0)),
                "total": total,
                "predicted": int(stats.get("predicted", 0)),
            }
        )
    return {
        "oracle_rows": oracle_rows,
        "sweep_rows": sweep_rows,
        "head_norm_row": head_norm_row,
        "classifier_norms": norm_info,
        "per_class_rows": per_class_rows,
    }


def _build_report(payload: Mapping[str, object]) -> str:
    baseline = payload["branches"]["freeze_baseline"]
    med = payload["branches"]["med_fdm"]
    oracle_rows = baseline["eval"]["oracle_rows"] + med["eval"]["oracle_rows"]
    best_rows = []
    for branch in [baseline, med]:
        best_rows.extend(sorted(branch["eval"]["sweep_rows"], key=lambda row: (row["AccT"] or -1.0), reverse=True)[:4])
        if branch["eval"]["head_norm_row"] is not None:
            best_rows.append(branch["eval"]["head_norm_row"])
    lines = [
        "# HyperKvasir23 Med-FDM Gate Results",
        "",
        "## Scope",
        "",
        "Bounded `max_phases=2`, seed0, 5-epoch diagnostic. Phase0 is trained once; phase1 branches are forked from the same phase0 state.",
        "",
        "## Exact Command",
        "",
        "```bash",
        str(payload["command"]),
        "```",
        "",
        "## Exemplar List Summary",
        "",
        f"- Exemplars per old class: `{payload['exemplars_per_class']}`",
        f"- Total old exemplars: `{len(payload['exemplar_rows'])}`",
        f"- Exemplar file: `{payload['output_dir']}/old_exemplars.csv`",
        "",
        "## Phase0 Anchor Stats",
        "",
        "```json",
        json.dumps(payload["anchor_stats"], indent=2, sort_keys=True),
        "```",
        "",
        "## Freeze Baseline Comparison",
        "",
        _markdown_table(
            oracle_rows,
            ["method", "mode", "AccT", "balanced_acc", "macro_f1", "old_acc", "current_acc", "old_to_current_rate", "current_to_old_rate"],
        ),
        "",
        "## Calibration Sweep",
        "",
        _markdown_table(
            best_rows,
            ["method", "mode", "current_alpha", "AccT", "balanced_acc", "macro_f1", "old_acc", "current_acc", "old_to_current_rate", "current_to_old_rate"],
        ),
        "",
        "## Feature Matching Diagnostics",
        "",
        _markdown_table(
            [
                {
                    "method": name,
                    **branch["train_stats"],
                    "exemplar_cos_before": branch["feature_cosine_before"]["cos_mean"],
                    "exemplar_cos_after": branch["feature_cosine_after"]["cos_mean"],
                    "old_eval_cos_before": branch["old_eval_cosine_before"]["cos_mean"],
                    "old_eval_cos_after": branch["old_eval_cosine_after"]["cos_mean"],
                }
                for name, branch in payload["branches"].items()
            ],
            [
                "method",
                "train_ce_loss",
                "fdm_loss_raw",
                "fdm_loss_weighted",
                "fdm_to_ce_ratio",
                "fdm_batches",
                "fdm_requires_grad_batches",
                "fdm_nonzero_value_batches",
                "exemplar_cos_before",
                "exemplar_cos_after",
                "old_eval_cos_before",
                "old_eval_cos_after",
            ],
        ),
        "",
        "FDM gradient probe:",
        "",
        "```json",
        json.dumps(payload["fdm_grad_probe"], indent=2, sort_keys=True),
        "```",
        "",
        "## Recommendation",
        "",
    ]
    med_probe = payload["fdm_grad_probe"]
    med_all_seen = next(row for row in med["eval"]["oracle_rows"] if row["mode"] == "all_seen")
    base_all_seen = next(row for row in baseline["eval"]["oracle_rows"] if row["mode"] == "all_seen")
    if not med_probe.get("fdm_probe_nonzero_grad"):
        lines.append(
            "Reject this first Med-FDM gate as an active method under the current frozen-backbone protocol: the FDM loss is finite but has no gradient path to the trainable classifier-only branch."
        )
    elif (med_all_seen.get("AccT") or 0.0) > (base_all_seen.get("AccT") or 0.0):
        lines.append("Keep for a follow-up gate: Med-FDM improves the bounded freeze baseline.")
    else:
        lines.append("Reject or revise: Med-FDM does not improve the bounded freeze baseline.")
    lines.extend(
        [
            "",
            "A meaningful feature-space matching method in this ResNet scaffold requires a trainable feature path during phase1, such as unfrozen late blocks, adapters, or a projector. Under strict backbone freezing, feature matching is diagnostic-only.",
            "",
            "## Artifact Paths",
            "",
            f"- Output dir: `{payload['output_dir']}`",
            f"- Checkpoint dir: `{payload['ckpt_dir']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/dev/shm/wangbomin/LongTailedCL/data/hyper-kvasir23")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--exemplars-per-class", type=int, default=1)
    parser.add_argument("--fdm-lambda", type=float, default=0.01)
    parser.add_argument("--current-alphas", default="1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1")
    parser.add_argument(
        "--branches",
        default="freeze_baseline,med_fdm",
        help="Comma-separated phase1 branches to run: freeze_baseline,med_fdm.",
    )
    cli_args = parser.parse_args()
    requested_branches = [item.strip() for item in cli_args.branches.split(",") if item.strip()]
    valid_branches = {"freeze_baseline", "med_fdm"}
    invalid_branches = sorted(set(requested_branches) - valid_branches)
    if invalid_branches:
        raise ValueError(f"Unknown branch(es): {', '.join(invalid_branches)}")
    if not requested_branches:
        raise ValueError("At least one branch must be requested")

    output_dir = Path(cli_args.output_dir)
    ckpt_dir = Path(cli_args.ckpt_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cli_args.seed))
    args = _make_args(cli_args)
    device = torch.device(cli_args.device)
    protocol = build_protocol(args)
    old_classes = [int(class_id) for class_id in protocol.tasks[0]]
    current_classes = [int(class_id) for class_id in protocol.tasks[1]]
    seen_classes = old_classes + current_classes
    phase0_class_to_head = _class_to_head_from_seen(old_classes)
    phase1_class_to_head = _class_to_head_from_seen(seen_classes)

    phase0_train_loader = _make_loader(
        protocol.train_dataset_for_classes(old_classes),
        cli_args.batch_size,
        shuffle=True,
        seed=int(cli_args.seed),
        num_workers=int(cli_args.num_workers),
    )
    phase0_proto_loader = _make_loader(
        protocol.prototype_dataset_for_classes(old_classes),
        cli_args.batch_size,
        shuffle=False,
        seed=int(cli_args.seed),
        num_workers=int(cli_args.num_workers),
    )
    current_train_dataset = protocol.train_dataset_for_classes(current_classes)
    old_loader = _make_loader(
        protocol.test_dataset_for_classes(old_classes),
        cli_args.batch_size,
        shuffle=False,
        seed=int(cli_args.seed),
        num_workers=int(cli_args.num_workers),
    )
    current_loader = _make_loader(
        protocol.test_dataset_for_classes(current_classes),
        cli_args.batch_size,
        shuffle=False,
        seed=int(cli_args.seed),
        num_workers=int(cli_args.num_workers),
    )
    seen_loader = _make_loader(
        protocol.test_dataset_for_classes(seen_classes),
        cli_args.batch_size,
        shuffle=False,
        seed=int(cli_args.seed),
        num_workers=int(cli_args.num_workers),
    )

    model = resnet32().to(device)
    model.expand_classifier(len(old_classes))
    phase0_train_stats = _train_ce_phase(
        model,
        phase0_train_loader,
        phase0_class_to_head,
        device,
        epochs=int(cli_args.epochs),
        lr=float(cli_args.lr),
        momentum=float(cli_args.momentum),
        weight_decay=float(cli_args.weight_decay),
    )
    phase0_eval = _eval_row(
        "phase0",
        "base_only",
        _evaluate(model, old_loader, phase0_class_to_head, old_classes, [], device, allowed_classes=old_classes),
    )
    anchors = _compute_anchors(model, phase0_proto_loader, old_classes, device)
    anchor_stats = _anchor_stats(anchors)
    exemplar_indices, exemplar_rows = _select_exemplars(protocol, old_classes, int(cli_args.exemplars_per_class))
    exemplar_dataset = protocol.train_dataset_for_indices(exemplar_indices)
    old_exemplar_eval_loader = _make_loader(
        protocol.train_dataset_for_indices(exemplar_indices),
        batch_size=max(1, min(int(cli_args.batch_size), len(exemplar_indices))),
        shuffle=False,
        seed=int(cli_args.seed),
        num_workers=int(cli_args.num_workers),
    )
    _write_csv(output_dir / "old_exemplars.csv", exemplar_rows)
    _write_json(output_dir / "anchor_stats.json", anchor_stats)
    torch.save(
        {
            "model_state": model.state_dict(),
            "old_classes": old_classes,
            "current_classes": current_classes,
            "anchors": {int(k): v.detach().cpu() for k, v in anchors.items()},
            "phase0_train_stats": phase0_train_stats,
            "phase0_eval": phase0_eval,
        },
        ckpt_dir / "phase0_with_anchors.pt",
    )

    phase0_state = copy.deepcopy(model.state_dict())
    expanded_template = resnet32().to(device)
    expanded_template.expand_classifier(len(old_classes))
    expanded_template.load_state_dict(phase0_state)
    expanded_template.expand_classifier(len(seen_classes))
    phase1_expanded_state = copy.deepcopy(expanded_template.state_dict())
    alphas = [float(item) for item in cli_args.current_alphas.split(",") if item.strip()]
    payload = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "output_dir": str(output_dir),
        "ckpt_dir": str(ckpt_dir),
        "data_root": cli_args.data_root,
        "seed": int(cli_args.seed),
        "epochs": int(cli_args.epochs),
        "split": {
            "old_classes": old_classes,
            "current_classes": current_classes,
            "seen_classes": seen_classes,
            "class_order": [int(class_id) for class_id in protocol.class_order],
        },
        "phase0_train_stats": phase0_train_stats,
        "phase0_eval": phase0_eval,
        "anchor_stats": anchor_stats,
        "exemplars_per_class": int(cli_args.exemplars_per_class),
        "exemplar_rows": exemplar_rows,
        "branches": {},
        "fdm_grad_probe": {},
    }

    branch_specs = [("freeze_baseline", False), ("med_fdm", True)]
    for branch_name, use_fdm in [item for item in branch_specs if item[0] in requested_branches]:
        branch_model = resnet32().to(device)
        branch_model.expand_classifier(len(seen_classes))
        branch_model.load_state_dict(phase1_expanded_state)
        before_exemplar_cos = _feature_anchor_cosine(branch_model, old_exemplar_eval_loader, anchors, device)
        before_old_cos = _feature_anchor_cosine(branch_model, old_loader, anchors, device)
        branch_current_loader = _make_loader(
            current_train_dataset,
            cli_args.batch_size,
            shuffle=True,
            seed=int(cli_args.seed) + 1,
            num_workers=int(cli_args.num_workers),
        )
        branch_exemplar_loader = _make_loader(
            exemplar_dataset,
            batch_size=max(1, min(int(cli_args.batch_size), len(exemplar_indices))),
            shuffle=True,
            seed=int(cli_args.seed) + 2000,
            num_workers=int(cli_args.num_workers),
        )
        if use_fdm:
            _freeze_backbone_and_bn(branch_model)
            probe_exemplar_loader = _make_loader(
                exemplar_dataset,
                batch_size=max(1, min(int(cli_args.batch_size), len(exemplar_indices))),
                shuffle=True,
                seed=int(cli_args.seed) + 2000,
                num_workers=int(cli_args.num_workers),
            )
            payload["fdm_grad_probe"] = _probe_fdm_grad(branch_model, probe_exemplar_loader, anchors, device)
        train_stats = _train_phase1_branch(
            branch_model,
            branch_current_loader,
            branch_exemplar_loader,
            phase1_class_to_head,
            anchors,
            device,
            epochs=int(cli_args.epochs),
            lr=float(cli_args.lr),
            momentum=float(cli_args.momentum),
            weight_decay=float(cli_args.weight_decay),
            fdm_lambda=float(cli_args.fdm_lambda),
            use_fdm=use_fdm,
        )
        after_exemplar_cos = _feature_anchor_cosine(branch_model, old_exemplar_eval_loader, anchors, device)
        after_old_cos = _feature_anchor_cosine(branch_model, old_loader, anchors, device)
        eval_payload = _run_branch_evals(
            branch_name,
            branch_model,
            seen_loader,
            old_loader,
            current_loader,
            phase1_class_to_head,
            old_classes,
            current_classes,
            device,
            alphas,
        )
        _write_csv(output_dir / branch_name / "oracle_metrics.csv", eval_payload["oracle_rows"])
        _write_csv(output_dir / branch_name / "current_alpha_sweep.csv", eval_payload["sweep_rows"])
        _write_csv(output_dir / branch_name / "per_class_recall.csv", eval_payload["per_class_rows"])
        _write_json(output_dir / branch_name / "train_stats.json", train_stats)
        torch.save(
            {
                "model_state": branch_model.state_dict(),
                "seen_classes": seen_classes,
                "class_to_head_index": phase1_class_to_head,
                "train_stats": train_stats,
                "eval": eval_payload,
            },
            ckpt_dir / f"{branch_name}_phase1.pt",
        )
        payload["branches"][branch_name] = {
            "use_fdm": use_fdm,
            "train_stats": train_stats,
            "feature_cosine_before": before_exemplar_cos,
            "feature_cosine_after": after_exemplar_cos,
            "old_eval_cosine_before": before_old_cos,
            "old_eval_cosine_after": after_old_cos,
            "eval": eval_payload,
        }

    _write_json(output_dir / "med_fdm_gate_summary.json", payload)
    if set(payload["branches"]) == valid_branches:
        report = _build_report(payload)
        (output_dir / "HYPERKVASIR23_MED_FDM_GATE_RESULTS.md").write_text(report, encoding="utf-8")
        print(report)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
