#!/usr/bin/env python3
"""Current-preserving task-block calibration for HyperKvasir23 phase1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets import build_protocol
from src.models.resnet_cifar import resnet32
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


class ExemplarPathDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping[str, object]], transform):
        self.rows = list(rows)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(str(row["path"])) as image:
            image = image.convert("RGB")
        return self.transform(image), int(row["class_id"])


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


def _make_protocol_args(args: argparse.Namespace) -> argparse.Namespace:
    parser = build_parser()
    parsed = parser.parse_args([])
    parsed.dataset = "hyper_kvasir23"
    parsed.data_root = args.data_root
    parsed.method = "finetune"
    parsed.order = "shuffled"
    parsed.seed = int(args.seed)
    parsed.base_classes = 13
    parsed.incremental_steps = 5
    parsed.max_phases = 2
    parsed.epochs = 5
    parsed.batch_size = int(args.batch_size)
    parsed.num_workers = int(args.num_workers)
    parsed.lr = 0.01
    parsed.scheduler = "cosine"
    parsed.device = args.device
    parsed.download = False
    return apply_method_aliases(parsed)


def _make_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )


def _load_exemplar_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "class_id": int(row["class_id"]),
                    "class_name": row.get("class_name", str(row["class_id"])),
                    "train_index": int(row["train_index"]),
                    "path": row["path"],
                }
            )
    return rows


def _label_map(class_to_head: Mapping[int, int], device: torch.device) -> torch.Tensor:
    mapping = torch.full((max(class_to_head) + 1,), -1, dtype=torch.long, device=device)
    for class_id, head_idx in class_to_head.items():
        mapping[int(class_id)] = int(head_idx)
    return mapping


def _head_indices(class_ids: Iterable[int], class_to_head: Mapping[int, int], device: torch.device) -> torch.Tensor:
    return torch.tensor([int(class_to_head[int(class_id)]) for class_id in class_ids], dtype=torch.long, device=device)


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


def _harmonic(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


@torch.no_grad()
def _block_logits_from_features(
    model: nn.Module,
    features: torch.Tensor,
    old_heads: torch.Tensor,
    current_heads: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    logits = model.classifier(features)
    return {
        "logits": logits,
        "max_old": logits.index_select(1, old_heads).max(dim=1).values,
        "max_current": logits.index_select(1, current_heads).max(dim=1).values,
    }


@torch.no_grad()
def _collect_logits(
    model: nn.Module,
    loader: DataLoader,
    class_to_head: Mapping[int, int],
    old_heads: torch.Tensor,
    current_heads: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    model.eval()
    label_map = _label_map(class_to_head, device)
    logits_list: List[torch.Tensor] = []
    labels_head_list: List[torch.Tensor] = []
    labels_global_list: List[torch.Tensor] = []
    for images, labels_global in loader:
        images = images.to(device)
        labels_global = labels_global.to(device)
        logits = model(images).detach()
        logits_list.append(logits.cpu())
        labels_head_list.append(label_map[labels_global].detach().cpu())
        labels_global_list.append(labels_global.detach().cpu())
    logits_cpu = torch.cat(logits_list, dim=0)
    return {
        "logits": logits_cpu,
        "labels_head": torch.cat(labels_head_list, dim=0),
        "labels_global": torch.cat(labels_global_list, dim=0),
        "max_old": logits_cpu.index_select(1, old_heads.cpu()).max(dim=1).values,
        "max_current": logits_cpu.index_select(1, current_heads.cpu()).max(dim=1).values,
    }


def _current_train_metrics(logits_pack: Mapping[str, torch.Tensor], current_head_set: set[int], alpha: float) -> Dict[str, float]:
    logits = logits_pack["logits"].clone()
    current_heads = sorted(current_head_set)
    logits[:, current_heads] *= float(alpha)
    labels_head = logits_pack["labels_head"]
    preds = logits.argmax(dim=1)
    acc = float(preds.eq(labels_head).float().mean().item())
    max_current_scaled = logits[:, current_heads].max(dim=1).values
    old_heads = [idx for idx in range(logits.shape[1]) if idx not in current_head_set]
    max_old = logits[:, old_heads].max(dim=1).values
    return {
        "current_train_acc": acc,
        "current_train_current_vs_old_margin": float((max_current_scaled - max_old).mean().item()),
    }


def _old_exemplar_metrics(
    logits_pack: Mapping[str, torch.Tensor],
    old_head_set: set[int],
    current_head_set: set[int],
    alpha: float,
) -> Dict[str, float]:
    logits = logits_pack["logits"].clone()
    current_heads = sorted(current_head_set)
    old_heads = sorted(old_head_set)
    logits[:, current_heads] *= float(alpha)
    preds = logits.argmax(dim=1)
    labels_head = logits_pack["labels_head"]
    max_old = logits[:, old_heads].max(dim=1).values
    max_current = logits[:, current_heads].max(dim=1).values
    return {
        "old_exemplar_retention": float(preds.eq(labels_head).float().mean().item()),
        "old_exemplar_old_vs_current_margin": float((max_old - max_current).mean().item()),
    }


@torch.no_grad()
def _evaluate_test(
    model: nn.Module,
    loader: DataLoader,
    class_to_head: Mapping[int, int],
    old_classes: Sequence[int],
    current_classes: Sequence[int],
    device: torch.device,
    alpha: float,
) -> EvalResult:
    model.eval()
    label_map = _label_map(class_to_head, device)
    current_heads = _head_indices(current_classes, class_to_head, device)
    old_head_set = set(int(class_to_head[int(class_id)]) for class_id in old_classes)
    current_head_set = set(int(class_to_head[int(class_id)]) for class_id in current_classes)
    head_to_class = {int(head): int(class_id) for class_id, head in class_to_head.items()}
    result = EvalResult(0, 0, 0, 0, 0, 0, 0, 0, {})
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        labels_head = label_map[labels]
        logits = model(images).clone()
        logits.index_copy_(1, current_heads, logits.index_select(1, current_heads) * float(alpha))
        preds = logits.argmax(dim=1)
        matches = preds.eq(labels_head)
        old_label_mask = torch.tensor([int(item) in old_head_set for item in labels_head.detach().cpu().tolist()], device=device)
        current_label_mask = torch.tensor(
            [int(item) in current_head_set for item in labels_head.detach().cpu().tolist()], device=device
        )
        pred_old_mask = torch.tensor([int(item) in old_head_set for item in preds.detach().cpu().tolist()], device=device)
        pred_current_mask = torch.tensor([int(item) in current_head_set for item in preds.detach().cpu().tolist()], device=device)
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


def _eval_row(rule_type: str, rule: str, alpha: float, result: EvalResult) -> Dict[str, object]:
    balanced, macro = _balanced_f1(result.per_class)
    return {
        "rule_type": rule_type,
        "rule": rule,
        "alpha": float(alpha),
        "AccT": _safe_percent(result.correct, result.total),
        "balanced_acc": balanced,
        "macro_f1": macro,
        "old_acc": _safe_percent(result.old_correct, result.old_total),
        "current_acc": _safe_percent(result.current_correct, result.current_total),
        "old_to_current_rate": _safe_percent(result.old_pred_current, result.old_total),
        "current_to_old_rate": _safe_percent(result.current_pred_old, result.current_total),
    }


def _per_class_rows(rule_type: str, rule: str, alpha: float, result: EvalResult) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for class_id, stats in sorted(result.per_class.items()):
        total = int(stats.get("total", 0))
        rows.append(
            {
                "rule_type": rule_type,
                "rule": rule,
                "alpha": float(alpha),
                "class_id": int(class_id),
                "recall": 100.0 * int(stats.get("correct", 0)) / total if total else None,
                "correct": int(stats.get("correct", 0)),
                "total": total,
                "predicted": int(stats.get("predicted", 0)),
            }
        )
    return rows


def _copy_selection(row: Mapping[str, object], source: str, fallback_used: bool = False, **extra) -> Dict[str, object]:
    selected = dict(row)
    selected["selection_source"] = source
    selected["fallback_used"] = bool(fallback_used)
    selected.update(extra)
    return selected


def _nearest_grid_row(rows: Sequence[Mapping[str, object]], value: float) -> Mapping[str, object]:
    return min(rows, key=lambda row: (abs(float(row["alpha"]) - float(value)), float(row["alpha"])))


def _select_cp_anchor(
    rows: Sequence[Mapping[str, object]],
    current_ref: float,
    rho: float,
    fallback: Mapping[str, object],
) -> Dict[str, object]:
    threshold = float(rho) * float(current_ref)
    valid = [row for row in rows if float(row["current_train_acc"]) >= threshold]
    if not valid:
        return _copy_selection(
            fallback,
            f"fallback_balanced_hmean_anchor_no_alpha_meets_rho{rho:.2f}",
            fallback_used=True,
            current_ref=current_ref,
            current_threshold=threshold,
            rho=rho,
        )
    selected = max(valid, key=lambda row: (float(row["old_anchor_retention"]), -float(row["alpha"])))
    return _copy_selection(
        selected,
        f"max_old_anchor_retention_with_current_train_acc_ge_{rho:.2f}_ref",
        current_ref=current_ref,
        current_threshold=threshold,
        rho=rho,
    )


def _select_cp_exemplar(
    rows: Sequence[Mapping[str, object]],
    current_ref: float,
    rho: float,
    fallback: Mapping[str, object],
) -> Dict[str, object]:
    threshold = float(rho) * float(current_ref)
    valid = [row for row in rows if float(row["current_train_acc"]) >= threshold]
    if not valid:
        return _copy_selection(
            fallback,
            f"fallback_balanced_hmean_exemplar_no_alpha_meets_rho{rho:.2f}",
            fallback_used=True,
            current_ref=current_ref,
            current_threshold=threshold,
            rho=rho,
        )
    selected = max(valid, key=lambda row: (float(row["old_exemplar_retention"]), -float(row["alpha"])))
    return _copy_selection(
        selected,
        f"max_old_exemplar_retention_with_current_train_acc_ge_{rho:.2f}_ref",
        current_ref=current_ref,
        current_threshold=threshold,
        rho=rho,
    )


def _select_rules(grid_rows: Sequence[Mapping[str, object]], num_old_classes: int, num_current_classes: int) -> Dict[str, Dict[str, object]]:
    rows = list(grid_rows)
    selections: Dict[str, Dict[str, object]] = {}
    current_ref = float(next(row["current_train_acc"] for row in rows if abs(float(row["alpha"]) - 1.0) < 1e-9))
    balanced_hmean_anchor = max(
        rows,
        key=lambda row: (float(row["hmean_anchor"]), float(row["alpha"])),
    )
    balanced_hmean_exemplar = max(
        rows,
        key=lambda row: (float(row["hmean_exemplar"]), float(row["alpha"])),
    )
    selections["cp_anchor_rho090"] = _select_cp_anchor(rows, current_ref, 0.90, balanced_hmean_anchor)
    selections["cp_exemplar_rho090"] = _select_cp_exemplar(rows, current_ref, 0.90, balanced_hmean_exemplar)
    selections["cp_anchor_rho085"] = _select_cp_anchor(rows, current_ref, 0.85, balanced_hmean_anchor)
    selections["cp_anchor_rho095"] = _select_cp_anchor(rows, current_ref, 0.95, balanced_hmean_anchor)
    margin_candidates = [row for row in rows if float(row["current_train_acc"]) >= 0.90 * current_ref]
    if margin_candidates:
        selected_margin = min(
            margin_candidates,
            key=lambda row: (
                float(row["cp_margin_balance_abs"]),
                -float(row["old_anchor_retention"]),
                float(row["alpha"]),
            ),
        )
        selections["cp_margin_balance"] = _copy_selection(
            selected_margin,
            "min_abs_mean_old_anchor_margin_minus_mean_current_train_margin_with_current_train_acc_ge_0.90_ref",
            current_ref=current_ref,
            current_threshold=0.90 * current_ref,
            rho=0.90,
        )
    else:
        selections["cp_margin_balance"] = _copy_selection(
            balanced_hmean_anchor,
            "fallback_balanced_hmean_anchor_no_alpha_meets_rho0.90",
            fallback_used=True,
            current_ref=current_ref,
            current_threshold=0.90 * current_ref,
            rho=0.90,
        )
    selections["balanced_hmean_anchor"] = _copy_selection(balanced_hmean_anchor, "max_hmean_old_anchor_retention_current_train_acc")
    selections["balanced_hmean_exemplar"] = _copy_selection(
        balanced_hmean_exemplar,
        "max_hmean_old_exemplar_retention_current_train_acc",
    )
    anchor_candidates = [row for row in rows if float(row["current_train_acc"]) >= 0.80]
    if anchor_candidates:
        selections["constrained_anchor"] = _copy_selection(
            max(
                anchor_candidates,
                key=lambda row: (float(row["old_anchor_retention"]), -float(row["alpha"])),
            ),
            "max_old_anchor_retention_with_current_train_acc_ge_0.80",
        )
    else:
        selections["constrained_anchor"] = _copy_selection(
            balanced_hmean_anchor,
            "fallback_balanced_hmean_anchor_no_alpha_meets_0.80",
            fallback_used=True,
        )
    exemplar_candidates = [row for row in rows if float(row["current_train_acc"]) >= 0.80]
    if exemplar_candidates:
        selections["constrained_exemplar"] = _copy_selection(
            max(
                exemplar_candidates,
                key=lambda row: (float(row["old_exemplar_retention"]), -float(row["alpha"])),
            ),
            "max_old_exemplar_retention_with_current_train_acc_ge_0.80",
        )
    else:
        selections["constrained_exemplar"] = _copy_selection(
            balanced_hmean_exemplar,
            "fallback_balanced_hmean_exemplar_no_alpha_meets_0.80",
            fallback_used=True,
        )
    prior_linear_raw = float(num_current_classes) / float(num_old_classes)
    prior_sqrt_raw = math.sqrt(prior_linear_raw)
    selections["prior_linear"] = _copy_selection(
        _nearest_grid_row(rows, prior_linear_raw),
        "nearest_grid_to_num_current_classes_over_num_old_classes",
        raw_alpha=prior_linear_raw,
    )
    selections["prior_sqrt"] = _copy_selection(
        _nearest_grid_row(rows, prior_sqrt_raw),
        "nearest_grid_to_sqrt_num_current_classes_over_num_old_classes",
        raw_alpha=prior_sqrt_raw,
    )
    return selections


def _build_report(payload: Mapping[str, object]) -> str:
    selected_rows = payload["selected_test_rows"]
    oracle_rows = payload["oracle_test_rows"]
    grid_rows = payload["calibration_grid_rows"]
    selected_alpha_rows = payload["selected_alpha_rows"]
    best_selected = max(selected_rows, key=lambda row: row.get("AccT") or -1.0)
    oracle_03 = next(row for row in oracle_rows if abs(float(row["alpha"]) - 0.3) < 1e-9)
    current_ref = float(payload["current_ref"])
    best_gap = float(oracle_03["AccT"]) - float(best_selected["AccT"])
    lines = [
        "# HyperKvasir23 Current-Preserving Task-Block Calibration Results",
        "",
        "## Scope",
        "",
        "Evaluation-only Current-Preserving Task-Block Calibration (CP-TBC). Alpha selection uses phase0 old anchors, 1-shot old exemplars, and phase1 current-task training samples only. Test accuracy is used only after alpha is selected; oracle sweep is diagnostic only. No training, checkpoints, model weights, classifier heads, backbone, BN, replay, FDM, or ConCM modules are modified.",
        "",
        "## Exact Command",
        "",
        "```bash",
        str(payload["command"]),
        "```",
        "",
        "## Calibration Reference",
        "",
        f"- `current_ref = current_train_acc(alpha=1.0) = {current_ref:.4f}`",
        f"- `rho0.85 threshold = {0.85 * current_ref:.4f}`",
        f"- `rho0.90 threshold = {0.90 * current_ref:.4f}`",
        f"- `rho0.95 threshold = {0.95 * current_ref:.4f}`",
        "",
        "## Calibration-Set Alpha Selection Table",
        "",
        _markdown_table(
            grid_rows,
            [
                "alpha",
                "old_anchor_retention",
                "old_exemplar_retention",
                "current_train_acc",
                "valid_rho085",
                "valid_rho090",
                "valid_rho095",
                "old_anchor_margin",
                "current_train_margin",
                "cp_margin_balance_abs",
                "hmean_anchor",
                "hmean_exemplar",
            ],
        ),
        "",
        "## Selected Alpha By Rule",
        "",
        _markdown_table(
            selected_alpha_rows,
            [
                "rule",
                "alpha",
                "raw_alpha",
                "selection_source",
                "fallback_used",
                "current_train_acc",
                "old_anchor_retention",
                "old_exemplar_retention",
                "cp_margin_balance_abs",
            ],
        ),
        "",
        "## Test-Set Evaluation Of Selected Non-Oracle Rules",
        "",
        _markdown_table(
            selected_rows,
            ["rule", "alpha", "AccT", "balanced_acc", "macro_f1", "old_acc", "current_acc", "old_to_current_rate", "current_to_old_rate"],
        ),
        "",
        "## Oracle Sweep Reference",
        "",
        "Diagnostic only; these alphas are not selected by test performance.",
        "",
        _markdown_table(
            oracle_rows,
            ["rule", "alpha", "AccT", "balanced_acc", "macro_f1", "old_acc", "current_acc", "old_to_current_rate", "current_to_old_rate"],
        ),
        "",
        "## Best Non-Oracle Rule",
        "",
        f"- Best rule: `{best_selected['rule']}`",
        f"- Selected alpha: `{best_selected['alpha']:.4f}`",
        f"- AccT: `{best_selected['AccT']:.4f}`",
        f"- Old/current acc: `{best_selected['old_acc']:.4f}` / `{best_selected['current_acc']:.4f}`",
        "",
        "## Comparison With Oracle Alpha 0.3",
        "",
        f"- Oracle alpha 0.3 AccT: `{oracle_03['AccT']:.4f}`",
        f"- Oracle alpha 0.3 old/current acc: `{oracle_03['old_acc']:.4f}` / `{oracle_03['current_acc']:.4f}`",
        f"- Best non-oracle gap vs oracle AccT: `{best_gap:.4f}`",
        "",
        "## Recommendation",
        "",
    ]
    if (
        0.25 <= float(best_selected["alpha"]) <= 0.35
        and float(best_selected["AccT"]) >= 75.0
        and abs(best_gap) <= 2.0
        and float(best_selected["old_acc"]) >= 75.0
        and float(best_selected["current_acc"]) >= 80.0
    ):
        lines.append("Use as the stable HyperKvasir23 medical CIL baseline. The best non-oracle rule selects an oracle-good alpha from calibration-side statistics and preserves both old and current accuracy.")
    elif float(best_selected["AccT"]) > 70.0 and float(best_selected["current_acc"]) >= 80.0:
        lines.append("Revise calibration before promoting it: the rule improves collapse while preserving current accuracy, but it is not close enough to the oracle alpha 0.3 target.")
    else:
        lines.append("Stop this calibration branch for now; it does not meet the current-preserving recovery gate. Do not proceed to FDM v2 from this result.")
    lines.extend(
        [
            "",
            "Per-class recall is saved in `per_class_recall.csv`.",
            "",
            "## Artifact Paths",
            "",
            f"- Output dir: `{payload['output_dir']}`",
            f"- Final JSON: `{payload['final_json']}`",
            f"- Report path: `{payload['report_path']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--phase0-ckpt", required=True)
    parser.add_argument("--phase1-ckpt", required=True)
    parser.add_argument("--exemplar-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--alpha-grid", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,0.90,1.00")
    parser.add_argument("--oracle-alphas", default="0.10,0.20,0.25,0.30,0.35,0.40,0.50,1.00")
    parser.add_argument("--report-path", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    protocol_args = _make_protocol_args(args)
    protocol = build_protocol(protocol_args)

    phase0_payload = torch.load(args.phase0_ckpt, map_location="cpu")
    phase1_payload = torch.load(args.phase1_ckpt, map_location=device)
    old_classes = [int(class_id) for class_id in phase0_payload["old_classes"]]
    current_classes = [int(class_id) for class_id in phase0_payload["current_classes"]]
    seen_classes = old_classes + current_classes
    class_to_head = {int(class_id): index for index, class_id in enumerate(seen_classes)}
    if [int(class_id) for class_id in phase1_payload["seen_classes"]] != seen_classes:
        raise RuntimeError("phase1 checkpoint seen_classes do not match phase0 metadata")

    model = resnet32().to(device)
    model.expand_classifier(len(seen_classes))
    model.load_state_dict(phase1_payload["model_state"])
    model.eval()
    old_heads = _head_indices(old_classes, class_to_head, device)
    current_heads = _head_indices(current_classes, class_to_head, device)
    old_head_set = set(int(class_to_head[int(class_id)]) for class_id in old_classes)
    current_head_set = set(int(class_to_head[int(class_id)]) for class_id in current_classes)

    anchors = {int(k): v.to(device).float() for k, v in phase0_payload["anchors"].items()}
    anchor_matrix = torch.stack([anchors[int(class_id)] for class_id in old_classes])
    anchor_pack = _block_logits_from_features(model, anchor_matrix, old_heads, current_heads)

    exemplar_rows = []
    with Path(args.exemplar_csv).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            exemplar_rows.append(
                {
                    "class_id": int(row["class_id"]),
                    "class_name": row.get("class_name", str(row["class_id"])),
                    "train_index": int(row["train_index"]),
                    "path": row["path"],
                }
            )
    exemplar_loader = _make_loader(ExemplarPathDataset(exemplar_rows, protocol.eval_transform), args.batch_size, args.num_workers)
    current_train_loader = _make_loader(
        protocol.prototype_dataset_for_classes(current_classes),
        args.batch_size,
        args.num_workers,
    )
    test_seen_loader = _make_loader(
        protocol.test_dataset_for_classes(seen_classes),
        args.batch_size,
        args.num_workers,
    )
    exemplar_logits = _collect_logits(model, exemplar_loader, class_to_head, old_heads, current_heads, device)
    current_train_logits = _collect_logits(model, current_train_loader, class_to_head, old_heads, current_heads, device)

    grid = [float(item) for item in args.alpha_grid.split(",") if item.strip()]
    if not any(abs(alpha - 1.0) < 1e-9 for alpha in grid):
        raise ValueError("--alpha-grid must include 1.0 so current_ref can be computed")
    grid_rows: List[Dict[str, object]] = []
    raw_grid_rows: List[Dict[str, object]] = []
    for alpha in grid:
        old_anchor_margin_values = anchor_pack["max_old"] - float(alpha) * anchor_pack["max_current"]
        old_anchor_retention = float((old_anchor_margin_values >= 0.0).float().mean().item())
        old_exemplar = _old_exemplar_metrics(exemplar_logits, old_head_set, current_head_set, alpha)
        current_train = _current_train_metrics(current_train_logits, current_head_set, alpha)
        old_anchor_margin = float(old_anchor_margin_values.mean().item())
        current_train_margin = float(current_train["current_train_current_vs_old_margin"])
        h_anchor = _harmonic(old_anchor_retention, float(current_train["current_train_acc"]))
        h_exemplar = _harmonic(float(old_exemplar["old_exemplar_retention"]), float(current_train["current_train_acc"]))
        cp_margin_balance_abs = abs(old_anchor_margin - current_train_margin)
        legacy_margin_balance_abs = abs(old_anchor_margin + current_train_margin)
        raw_grid_rows.append(
            {
                "alpha": alpha,
                "old_anchor_retention": old_anchor_retention,
                "old_anchor_margin": old_anchor_margin,
                "old_exemplar_retention": old_exemplar["old_exemplar_retention"],
                "old_exemplar_old_vs_current_margin": old_exemplar["old_exemplar_old_vs_current_margin"],
                "current_train_acc": current_train["current_train_acc"],
                "current_train_current_vs_old_margin": current_train["current_train_current_vs_old_margin"],
                "current_train_margin": current_train_margin,
                "hmean_anchor": h_anchor,
                "hmean_exemplar": h_exemplar,
                "cp_margin_balance_abs": cp_margin_balance_abs,
                "margin_balance_abs": legacy_margin_balance_abs,
            }
        )

    current_ref = float(next(row["current_train_acc"] for row in raw_grid_rows if abs(float(row["alpha"]) - 1.0) < 1e-9))
    for row in raw_grid_rows:
        row = dict(row)
        row["valid_rho085"] = float(row["current_train_acc"]) >= 0.85 * current_ref
        row["valid_rho090"] = float(row["current_train_acc"]) >= 0.90 * current_ref
        row["valid_rho095"] = float(row["current_train_acc"]) >= 0.95 * current_ref
        grid_rows.append(row)

    selections = _select_rules(grid_rows, len(old_classes), len(current_classes))
    selected_rows: List[Dict[str, object]] = []
    selected_alpha_rows: List[Dict[str, object]] = []
    per_class_rows: List[Dict[str, object]] = []
    for rule, selected in selections.items():
        alpha = float(selected["alpha"])
        selected_alpha_rows.append(
            {
                "rule": rule,
                "alpha": alpha,
                "raw_alpha": selected.get("raw_alpha"),
                "selection_source": selected.get("selection_source"),
                "fallback_used": selected.get("fallback_used"),
                "current_train_acc": selected.get("current_train_acc"),
                "current_ref": selected.get("current_ref", current_ref),
                "current_threshold": selected.get("current_threshold"),
                "old_anchor_retention": selected.get("old_anchor_retention"),
                "old_exemplar_retention": selected.get("old_exemplar_retention"),
                "old_anchor_margin": selected.get("old_anchor_margin"),
                "current_train_margin": selected.get("current_train_margin"),
                "cp_margin_balance_abs": selected.get("cp_margin_balance_abs"),
            }
        )
        result = _evaluate_test(model, test_seen_loader, class_to_head, old_classes, current_classes, device, alpha)
        row = _eval_row("non_oracle", rule, alpha, result)
        row.update(
            {
                "selection_source": selected.get("selection_source"),
                "fallback_used": selected.get("fallback_used"),
                "raw_alpha": selected.get("raw_alpha"),
                "selected_hmean_anchor": selected.get("hmean_anchor"),
                "selected_hmean_exemplar": selected.get("hmean_exemplar"),
                "selected_current_train_acc": selected.get("current_train_acc"),
                "selected_old_anchor_retention": selected.get("old_anchor_retention"),
                "selected_old_exemplar_retention": selected.get("old_exemplar_retention"),
                "selected_cp_margin_balance_abs": selected.get("cp_margin_balance_abs"),
            }
        )
        selected_rows.append(row)
        per_class_rows.extend(_per_class_rows("non_oracle", rule, alpha, result))

    oracle_rows: List[Dict[str, object]] = []
    for alpha in [float(item) for item in args.oracle_alphas.split(",") if item.strip()]:
        result = _evaluate_test(model, test_seen_loader, class_to_head, old_classes, current_classes, device, alpha)
        row = _eval_row("oracle_diagnostic", f"oracle_alpha_{alpha:g}", alpha, result)
        oracle_rows.append(row)
        per_class_rows.extend(_per_class_rows("oracle_diagnostic", f"oracle_alpha_{alpha:g}", alpha, result))

    final_json = output_dir / "final_results.json"
    report_path = Path(args.report_path) if args.report_path else output_dir / "HYPERKVASIR23_CURRENT_PRESERVING_CALIBRATION_RESULTS.md"
    payload = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "output_dir": str(output_dir),
        "final_json": str(final_json),
        "report_path": str(report_path),
        "phase0_ckpt": args.phase0_ckpt,
        "phase1_ckpt": args.phase1_ckpt,
        "exemplar_csv": args.exemplar_csv,
        "old_classes": old_classes,
        "current_classes": current_classes,
        "alpha_grid": grid,
        "current_ref": current_ref,
        "current_thresholds": {
            "rho085": 0.85 * current_ref,
            "rho090": 0.90 * current_ref,
            "rho095": 0.95 * current_ref,
        },
        "calibration_grid_rows": grid_rows,
        "selected_rules": selections,
        "selected_alpha_rows": selected_alpha_rows,
        "selected_test_rows": selected_rows,
        "oracle_test_rows": oracle_rows,
    }
    _write_json(final_json, payload)
    _write_csv(output_dir / "calibration_grid_metrics.csv", grid_rows)
    _write_csv(output_dir / "selected_alpha_by_rule.csv", selected_alpha_rows)
    _write_csv(output_dir / "selected_rules_metrics.csv", selected_rows)
    _write_csv(output_dir / "oracle_sweep_metrics.csv", oracle_rows)
    _write_csv(output_dir / "per_class_recall.csv", per_class_rows)
    report = _build_report(payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
