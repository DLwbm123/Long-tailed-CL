#!/usr/bin/env python3
"""Evaluation-only task-block calibration for HyperKvasir23 phase1."""

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


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_to_head: Mapping[int, int],
    old_classes: Sequence[int],
    current_classes: Sequence[int],
    device: torch.device,
    current_alpha: float,
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
        logits.index_copy_(1, current_heads, logits.index_select(1, current_heads) * float(current_alpha))
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


def _eval_row(rule: str, source: str, alpha: float, result: EvalResult, rule_type: str) -> Dict[str, object]:
    balanced, macro_f1 = _balanced_f1(result.per_class)
    return {
        "rule_type": rule_type,
        "rule": rule,
        "source": source,
        "alpha": float(alpha),
        "AccT": _safe_percent(result.correct, result.total),
        "balanced_acc": balanced,
        "macro_f1": macro_f1,
        "old_acc": _safe_percent(result.old_correct, result.old_total),
        "current_acc": _safe_percent(result.current_correct, result.current_total),
        "old_to_current_rate": _safe_percent(result.old_pred_current, result.old_total),
        "current_to_old_rate": _safe_percent(result.current_pred_old, result.current_total),
    }


def _clip_alpha(value: float, min_alpha: float = 0.05, max_alpha: float = 1.0) -> float:
    if not math.isfinite(float(value)):
        return 1.0
    return max(float(min_alpha), min(float(max_alpha), float(value)))


def _median(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    sorted_values = sorted(float(value) for value in values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[mid]
    return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])


@torch.no_grad()
def _block_scores_from_features(
    model: nn.Module,
    features: torch.Tensor,
    old_heads: torch.Tensor,
    current_heads: torch.Tensor,
) -> Dict[str, object]:
    logits = model.classifier(features)
    old_scores = logits.index_select(1, old_heads).max(dim=1).values.detach().cpu()
    current_scores = logits.index_select(1, current_heads).max(dim=1).values.detach().cpu()
    ratio = old_scores / (current_scores + 1e-6)
    ratio_list = [float(item) for item in ratio.tolist() if math.isfinite(float(item))]
    return {
        "old_scores": old_scores.tolist(),
        "current_scores": current_scores.tolist(),
        "ratios": ratio_list,
        "median_ratio_alpha": _clip_alpha(_median(ratio_list)),
        "mean_ratio_alpha": _clip_alpha(float(old_scores.mean().item()) / (float(current_scores.mean().item()) + 1e-6)),
        "old_score_mean": float(old_scores.mean().item()),
        "current_score_mean": float(current_scores.mean().item()),
        "old_score_median": float(old_scores.median().item()),
        "current_score_median": float(current_scores.median().item()),
    }


@torch.no_grad()
def _block_scores_from_loader(
    model: nn.Module,
    loader: DataLoader,
    old_heads: torch.Tensor,
    current_heads: torch.Tensor,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    old_scores: List[torch.Tensor] = []
    current_scores: List[torch.Tensor] = []
    for images, _ in loader:
        images = images.to(device)
        logits = model(images)
        old_scores.append(logits.index_select(1, old_heads).max(dim=1).values.detach().cpu())
        current_scores.append(logits.index_select(1, current_heads).max(dim=1).values.detach().cpu())
    old_joined = torch.cat(old_scores) if old_scores else torch.empty(0)
    current_joined = torch.cat(current_scores) if current_scores else torch.empty(0)
    ratio = old_joined / (current_joined + 1e-6)
    ratio_list = [float(item) for item in ratio.tolist() if math.isfinite(float(item))]
    return {
        "old_scores": old_joined.tolist(),
        "current_scores": current_joined.tolist(),
        "ratios": ratio_list,
        "median_ratio_alpha": _clip_alpha(_median(ratio_list)),
        "mean_ratio_alpha": _clip_alpha(float(old_joined.mean().item()) / (float(current_joined.mean().item()) + 1e-6)),
        "old_score_mean": float(old_joined.mean().item()),
        "current_score_mean": float(current_joined.mean().item()),
        "old_score_median": float(old_joined.median().item()),
        "current_score_median": float(current_joined.median().item()),
    }


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


def _per_class_rows(rule: str, source: str, alpha: float, result: EvalResult) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for class_id, stats in sorted(result.per_class.items()):
        total = int(stats.get("total", 0))
        rows.append(
            {
                "rule": rule,
                "source": source,
                "alpha": float(alpha),
                "class_id": int(class_id),
                "recall": 100.0 * int(stats.get("correct", 0)) / total if total else None,
                "correct": int(stats.get("correct", 0)),
                "total": total,
                "predicted": int(stats.get("predicted", 0)),
            }
        )
    return rows


def _build_report(payload: Mapping[str, object]) -> str:
    rows = payload["evaluation_rows"]
    non_oracle_rows = [row for row in rows if row["rule_type"] == "non_oracle"]
    oracle_rows = [row for row in rows if row["rule_type"] == "oracle_diagnostic"]
    best_non_oracle = max(non_oracle_rows, key=lambda row: row.get("AccT") or -1.0)
    lines = [
        "# HyperKvasir23 Task-Block Calibration Results",
        "",
        "## Scope",
        "",
        "Evaluation-only non-oracle task-block calibration on the bounded HyperKvasir23 phase1 freeze-backbone+BN checkpoint. No full phases, multi-seed runs, old module matrix, GUIDE, or FDM v2 were run.",
        "",
        "## Exact Command",
        "",
        "```bash",
        str(payload["command"]),
        "```",
        "",
        "## Alpha Selection Rules",
        "",
        "- `old_anchor_median_ratio`: median over phase0 old-class anchors of `max_old_logit / (max_current_logit + eps)`, clipped to `[0.05, 1.0]`.",
        "- `old_anchor_mean_stat`: mean-stat matching on phase0 old-class anchors, `mean(max_old_logit) / mean(max_current_logit)`, clipped to `[0.05, 1.0]`.",
        "- `old_exemplar_median_ratio`: median over 1-shot old exemplars of `max_old_logit / (max_current_logit + eps)`, clipped to `[0.05, 1.0]`.",
        "- `old_exemplar_mean_stat`: mean-stat matching on 1-shot old exemplars, clipped to `[0.05, 1.0]`.",
        "- `prior_linear`: `num_current_classes / num_old_classes`.",
        "- `prior_sqrt`: `sqrt(num_current_classes / num_old_classes)`.",
        "",
        "## Selection Statistics",
        "",
        "```json",
        json.dumps(payload["selection_stats"], indent=2, sort_keys=True),
        "```",
        "",
        "## Non-Oracle Calibration",
        "",
        _markdown_table(
            non_oracle_rows,
            ["rule", "source", "alpha", "AccT", "balanced_acc", "macro_f1", "old_acc", "current_acc", "old_to_current_rate", "current_to_old_rate"],
        ),
        "",
        "## Oracle Sweep Reference",
        "",
        "Diagnostic only; these alphas are not selected by test performance in the proposed baseline.",
        "",
        _markdown_table(
            oracle_rows,
            ["rule", "source", "alpha", "AccT", "balanced_acc", "macro_f1", "old_acc", "current_acc", "old_to_current_rate", "current_to_old_rate"],
        ),
        "",
        "## Recommendation",
        "",
    ]
    if (best_non_oracle.get("AccT") or 0.0) >= 70.0 and (best_non_oracle.get("current_acc") or 0.0) >= 80.0:
        lines.append(
            f"Use task-block calibration as the current medical-CIL baseline candidate. Best non-oracle rule is `{best_non_oracle['rule']}` with alpha `{best_non_oracle['alpha']:.4f}`, AccT `{best_non_oracle['AccT']:.4f}`, old `{best_non_oracle['old_acc']:.4f}`, current `{best_non_oracle['current_acc']:.4f}`."
        )
    else:
        lines.append(
            f"Revise calibration before promoting it. Best non-oracle rule is `{best_non_oracle['rule']}` with AccT `{best_non_oracle.get('AccT'):.4f}`, which is still not close enough to the oracle-good alpha range."
        )
    lines.extend(
        [
            "",
            "Per-class recall is saved in `per_class_recall.csv`.",
            "",
            "## Artifact Paths",
            "",
            f"- Output dir: `{payload['output_dir']}`",
            f"- Phase0 checkpoint: `{payload['phase0_ckpt']}`",
            f"- Phase1 checkpoint: `{payload['phase1_ckpt']}`",
            f"- Exemplar CSV: `{payload['exemplar_csv']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/dev/shm/wangbomin/LongTailedCL/data/hyper-kvasir23")
    parser.add_argument("--phase0-ckpt", required=True)
    parser.add_argument("--phase1-ckpt", required=True)
    parser.add_argument("--exemplar-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--oracle-alphas", default="0.1,0.2,0.3,0.4,0.5,0.7,0.9,1.0")
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

    anchors = {int(k): v.to(device).float() for k, v in phase0_payload["anchors"].items()}
    anchor_matrix = torch.stack([anchors[int(class_id)] for class_id in old_classes])
    anchor_scores = _block_scores_from_features(model, anchor_matrix, old_heads, current_heads)

    exemplar_rows = _load_exemplar_rows(Path(args.exemplar_csv))
    exemplar_loader = _make_loader(ExemplarPathDataset(exemplar_rows, protocol.eval_transform), args.batch_size, args.num_workers)
    exemplar_scores = _block_scores_from_loader(model, exemplar_loader, old_heads, current_heads, device)

    seen_loader = _make_loader(protocol.test_dataset_for_classes(seen_classes), args.batch_size, args.num_workers)
    rules = [
        ("old_anchor_median_ratio", "phase0_anchors", anchor_scores["median_ratio_alpha"], "non_oracle"),
        ("old_anchor_mean_stat", "phase0_anchors", anchor_scores["mean_ratio_alpha"], "non_oracle"),
        ("old_exemplar_median_ratio", "old_1shot_exemplars", exemplar_scores["median_ratio_alpha"], "non_oracle"),
        ("old_exemplar_mean_stat", "old_1shot_exemplars", exemplar_scores["mean_ratio_alpha"], "non_oracle"),
        ("prior_linear", "class_count_prior", _clip_alpha(len(current_classes) / len(old_classes)), "non_oracle"),
        ("prior_sqrt", "class_count_prior", _clip_alpha(math.sqrt(len(current_classes) / len(old_classes))), "non_oracle"),
    ]
    rules.extend(
        [
            (f"oracle_alpha_{alpha:g}", "test_sweep_reference", float(alpha), "oracle_diagnostic")
            for alpha in [float(item) for item in args.oracle_alphas.split(",") if item.strip()]
        ]
    )

    evaluation_rows: List[Dict[str, object]] = []
    per_class_rows: List[Dict[str, object]] = []
    for rule, source, alpha, rule_type in rules:
        result = _evaluate(model, seen_loader, class_to_head, old_classes, current_classes, device, float(alpha))
        evaluation_rows.append(_eval_row(rule, source, float(alpha), result, rule_type))
        per_class_rows.extend(_per_class_rows(rule, source, float(alpha), result))

    payload = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "output_dir": str(output_dir),
        "data_root": args.data_root,
        "phase0_ckpt": str(args.phase0_ckpt),
        "phase1_ckpt": str(args.phase1_ckpt),
        "exemplar_csv": str(args.exemplar_csv),
        "old_classes": old_classes,
        "current_classes": current_classes,
        "seen_classes": seen_classes,
        "selection_stats": {
            "anchor": {
                key: value
                for key, value in anchor_scores.items()
                if key not in {"old_scores", "current_scores", "ratios"}
            },
            "exemplar": {
                key: value
                for key, value in exemplar_scores.items()
                if key not in {"old_scores", "current_scores", "ratios"}
            },
            "num_old_classes": len(old_classes),
            "num_current_classes": len(current_classes),
            "prior_linear_alpha": len(current_classes) / len(old_classes),
            "prior_sqrt_alpha": math.sqrt(len(current_classes) / len(old_classes)),
        },
        "evaluation_rows": evaluation_rows,
    }
    _write_json(output_dir / "task_block_calibration_summary.json", payload)
    _write_csv(output_dir / "task_block_calibration_metrics.csv", evaluation_rows)
    _write_csv(output_dir / "per_class_recall.csv", per_class_rows)
    _write_json(
        output_dir / "alpha_selection_raw_scores.json",
        {
            "anchor": anchor_scores,
            "exemplar": exemplar_scores,
        },
    )
    report = _build_report(payload)
    (output_dir / "HYPERKVASIR23_TASK_BLOCK_CALIBRATION_RESULTS.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
