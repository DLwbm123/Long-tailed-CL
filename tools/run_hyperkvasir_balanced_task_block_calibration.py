#!/usr/bin/env python3
"""Balanced non-oracle task-block calibration for HyperKvasir23 phase1."""

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


def _select_rules(grid_rows: Sequence[Mapping[str, object]]) -> Dict[str, Dict[str, object]]:
    rows = list(grid_rows)
    selections: Dict[str, Dict[str, object]] = {}
    selections["balanced_hmean_anchor"] = max(
        rows,
        key=lambda row: (float(row["hmean_anchor"]), float(row["alpha"])),
    )
    selections["balanced_hmean_exemplar"] = max(
        rows,
        key=lambda row: (float(row["hmean_exemplar"]), float(row["alpha"])),
    )
    anchor_candidates = [row for row in rows if float(row["current_train_acc"]) >= 0.80]
    if anchor_candidates:
        selections["constrained_anchor"] = max(
            anchor_candidates,
            key=lambda row: (float(row["old_anchor_retention"]), -float(row["alpha"])),
        )
    else:
        selections["constrained_anchor"] = selections["balanced_hmean_anchor"]
    exemplar_candidates = [row for row in rows if float(row["current_train_acc"]) >= 0.80]
    if exemplar_candidates:
        selections["constrained_exemplar"] = max(
            exemplar_candidates,
            key=lambda row: (float(row["old_exemplar_retention"]), -float(row["alpha"])),
        )
    else:
        selections["constrained_exemplar"] = selections["balanced_hmean_exemplar"]
    margin_candidates = [row for row in rows if float(row["current_train_acc"]) >= 0.75]
    if not margin_candidates:
        margin_candidates = rows
    selections["margin_balance"] = min(
        margin_candidates,
        key=lambda row: (float(row["margin_balance_abs"]), -float(row["current_train_acc"])),
    )
    return selections


def _build_report(payload: Mapping[str, object]) -> str:
    selected_rows = payload["selected_test_rows"]
    oracle_rows = payload["oracle_test_rows"]
    grid_rows = payload["calibration_grid_rows"]
    best_selected = max(selected_rows, key=lambda row: row.get("AccT") or -1.0)
    oracle_03 = next(row for row in oracle_rows if abs(float(row["alpha"]) - 0.3) < 1e-9)
    lines = [
        "# HyperKvasir23 Balanced Task-Block Calibration Results",
        "",
        "## Scope",
        "",
        "Evaluation-only balanced non-oracle calibration. Alpha selection uses phase0 old anchors, 1-shot old exemplars, and phase1 current-task training samples only. Test accuracy is used only after alpha is selected; oracle sweep is diagnostic only.",
        "",
        "## Exact Command",
        "",
        "```bash",
        str(payload["command"]),
        "```",
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
                "old_anchor_margin",
                "old_exemplar_old_vs_current_margin",
                "current_train_current_vs_old_margin",
                "hmean_anchor",
                "hmean_exemplar",
                "margin_balance_abs",
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
        f"- Gap best non-oracle vs oracle AccT: `{(oracle_03['AccT'] - best_selected['AccT']):.4f}`",
        "",
        "## Recommendation",
        "",
    ]
    if (
        0.25 <= float(best_selected["alpha"]) <= 0.35
        and float(best_selected["AccT"]) >= 75.0
        and float(best_selected["current_acc"]) >= 80.0
    ):
        lines.append("Use this balanced calibration as the current stable medical-CIL baseline gate.")
    elif float(best_selected["AccT"]) > 70.0 and float(best_selected["current_acc"]) >= 80.0:
        lines.append("Promising but revise before promotion: the rule improves collapse while preserving current accuracy, but remains below oracle alpha 0.3.")
    else:
        lines.append("Revise calibration; do not proceed to adapter-FDM v2 yet.")
    lines.extend(
        [
            "",
            "Per-class recall is saved in `per_class_recall.csv`.",
            "",
            "## Artifact Paths",
            "",
            f"- Output dir: `{payload['output_dir']}`",
            f"- Final JSON: `{payload['final_json']}`",
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
    parser.add_argument("--oracle-alphas", default="0.10,0.20,0.30,0.40,0.50,0.70,0.90,1.00")
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
    grid_rows: List[Dict[str, object]] = []
    for alpha in grid:
        old_anchor_margin_values = anchor_pack["max_old"] - float(alpha) * anchor_pack["max_current"]
        old_anchor_retention = float((old_anchor_margin_values >= 0.0).float().mean().item())
        old_exemplar = _old_exemplar_metrics(exemplar_logits, old_head_set, current_head_set, alpha)
        current_train = _current_train_metrics(current_train_logits, current_head_set, alpha)
        h_anchor = _harmonic(old_anchor_retention, float(current_train["current_train_acc"]))
        h_exemplar = _harmonic(float(old_exemplar["old_exemplar_retention"]), float(current_train["current_train_acc"]))
        margin_balance_abs = abs(
            float(old_anchor_margin_values.mean().item())
            + float(current_train["current_train_current_vs_old_margin"])
        )
        grid_rows.append(
            {
                "alpha": alpha,
                "old_anchor_retention": old_anchor_retention,
                "old_anchor_margin": float(old_anchor_margin_values.mean().item()),
                "old_exemplar_retention": old_exemplar["old_exemplar_retention"],
                "old_exemplar_old_vs_current_margin": old_exemplar["old_exemplar_old_vs_current_margin"],
                "current_train_acc": current_train["current_train_acc"],
                "current_train_current_vs_old_margin": current_train["current_train_current_vs_old_margin"],
                "hmean_anchor": h_anchor,
                "hmean_exemplar": h_exemplar,
                "margin_balance_abs": margin_balance_abs,
            }
        )

    selections = _select_rules(grid_rows)
    selected_rows: List[Dict[str, object]] = []
    per_class_rows: List[Dict[str, object]] = []
    for rule, selected in selections.items():
        alpha = float(selected["alpha"])
        result = _evaluate_test(model, test_seen_loader, class_to_head, old_classes, current_classes, device, alpha)
        row = _eval_row("non_oracle", rule, alpha, result)
        row.update(
            {
                "selected_hmean_anchor": selected.get("hmean_anchor"),
                "selected_hmean_exemplar": selected.get("hmean_exemplar"),
                "selected_current_train_acc": selected.get("current_train_acc"),
                "selected_old_anchor_retention": selected.get("old_anchor_retention"),
                "selected_old_exemplar_retention": selected.get("old_exemplar_retention"),
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
    payload = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "output_dir": str(output_dir),
        "final_json": str(final_json),
        "phase0_ckpt": args.phase0_ckpt,
        "phase1_ckpt": args.phase1_ckpt,
        "exemplar_csv": args.exemplar_csv,
        "old_classes": old_classes,
        "current_classes": current_classes,
        "alpha_grid": grid,
        "calibration_grid_rows": grid_rows,
        "selected_rules": selections,
        "selected_test_rows": selected_rows,
        "oracle_test_rows": oracle_rows,
    }
    _write_json(final_json, payload)
    _write_csv(output_dir / "calibration_grid_metrics.csv", grid_rows)
    _write_csv(output_dir / "selected_rules_metrics.csv", selected_rows)
    _write_csv(output_dir / "oracle_sweep_metrics.csv", oracle_rows)
    _write_csv(output_dir / "per_class_recall.csv", per_class_rows)
    report = _build_report(payload)
    (output_dir / "HYPERKVASIR23_BALANCED_TASK_BLOCK_CALIBRATION_RESULTS.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
