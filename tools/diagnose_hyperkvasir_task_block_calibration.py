#!/usr/bin/env python3
"""Diagnostic-only analysis for HyperKvasir23 task-block calibration artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
from collections import defaultdict
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


def _label_map(class_to_head: Mapping[int, int], device: torch.device) -> torch.Tensor:
    mapping = torch.full((max(class_to_head) + 1,), -1, dtype=torch.long, device=device)
    for class_id, head_idx in class_to_head.items():
        mapping[int(class_id)] = int(head_idx)
    return mapping


def _head_indices(class_ids: Iterable[int], class_to_head: Mapping[int, int], device: torch.device) -> torch.Tensor:
    return torch.tensor([int(class_to_head[int(class_id)]) for class_id in class_ids], dtype=torch.long, device=device)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_to_head: Mapping[int, int],
    old_classes: Sequence[int],
    current_classes: Sequence[int],
    device: torch.device,
    alpha: float = 1.0,
) -> Dict[str, object]:
    model.eval()
    label_map = _label_map(class_to_head, device)
    current_heads = _head_indices(current_classes, class_to_head, device) if current_classes else None
    old_head_set = set(int(class_to_head[int(class_id)]) for class_id in old_classes)
    current_head_set = set(int(class_to_head[int(class_id)]) for class_id in current_classes)
    head_to_class = {int(head): int(class_id) for class_id, head in class_to_head.items()}
    stats = {
        "total": 0,
        "correct": 0,
        "old_total": 0,
        "old_correct": 0,
        "current_total": 0,
        "current_correct": 0,
        "old_pred_current": 0,
        "current_pred_old": 0,
        "per_class": {},
    }
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        labels_head = label_map[labels]
        logits = model(images).clone()
        if current_heads is not None and abs(float(alpha) - 1.0) > 0.0:
            logits.index_copy_(1, current_heads, logits.index_select(1, current_heads) * float(alpha))
        preds = logits.argmax(dim=1)
        matches = preds.eq(labels_head)
        old_label_mask = torch.tensor([int(item) in old_head_set for item in labels_head.detach().cpu().tolist()], device=device)
        current_label_mask = torch.tensor(
            [int(item) in current_head_set for item in labels_head.detach().cpu().tolist()], device=device
        )
        pred_old_mask = torch.tensor([int(item) in old_head_set for item in preds.detach().cpu().tolist()], device=device)
        pred_current_mask = torch.tensor([int(item) in current_head_set for item in preds.detach().cpu().tolist()], device=device)
        stats["correct"] += int(matches.sum().item())
        stats["total"] += int(labels.numel())
        stats["old_total"] += int(old_label_mask.sum().item())
        stats["old_correct"] += int(matches[old_label_mask].sum().item())
        stats["current_total"] += int(current_label_mask.sum().item())
        stats["current_correct"] += int(matches[current_label_mask].sum().item())
        stats["old_pred_current"] += int((old_label_mask & pred_current_mask).sum().item())
        stats["current_pred_old"] += int((current_label_mask & pred_old_mask).sum().item())
        for class_tensor in labels.unique():
            class_id = int(class_tensor.item())
            mask = labels == class_tensor
            item = stats["per_class"].setdefault(class_id, {"correct": 0, "total": 0, "predicted": 0})
            item["correct"] += int(matches[mask].sum().item())
            item["total"] += int(mask.sum().item())
        for pred_tensor in preds.unique():
            pred_head = int(pred_tensor.item())
            pred_class = int(head_to_class[pred_head])
            item = stats["per_class"].setdefault(pred_class, {"correct": 0, "total": 0, "predicted": 0})
            item["predicted"] += int((preds == pred_tensor).sum().item())
    balanced, macro = _balanced_f1(stats["per_class"])
    return {
        "alpha": float(alpha),
        "AccT": _safe_percent(stats["correct"], stats["total"]),
        "balanced_acc": balanced,
        "macro_f1": macro,
        "old_acc": _safe_percent(stats["old_correct"], stats["old_total"]),
        "current_acc": _safe_percent(stats["current_correct"], stats["current_total"]),
        "old_to_current_rate": _safe_percent(stats["old_pred_current"], stats["old_total"]),
        "current_to_old_rate": _safe_percent(stats["current_pred_old"], stats["current_total"]),
        "total": int(stats["total"]),
        "per_class": {int(k): v for k, v in stats["per_class"].items()},
    }


def _classwise_summary(metrics: Mapping[str, object]) -> Dict[str, float | int | None]:
    recalls = []
    for item in metrics["per_class"].values():
        total = int(item.get("total", 0))
        if total > 0:
            recalls.append(100.0 * int(item.get("correct", 0)) / total)
    return {
        "class_acc_mean": sum(recalls) / len(recalls) if recalls else None,
        "class_acc_min": min(recalls) if recalls else None,
        "class_acc_max": max(recalls) if recalls else None,
        "num_classes": len(recalls),
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


def _check_exemplar_files(rows: Sequence[Mapping[str, object]], old_classes: Sequence[int]) -> Dict[str, object]:
    counts = {int(class_id): 0 for class_id in old_classes}
    readable = 0
    missing = 0
    unreadable = 0
    row_summaries = []
    for row in rows:
        class_id = int(row["class_id"])
        counts[class_id] = counts.get(class_id, 0) + 1
        path = Path(str(row["path"]))
        exists = path.exists()
        image_ok = False
        error = ""
        if exists:
            try:
                with Image.open(path) as image:
                    image.verify()
                image_ok = True
                readable += 1
            except Exception as exc:  # diagnostic surface only
                unreadable += 1
                error = str(exc)
        else:
            missing += 1
        row_summaries.append(
            {
                "class_id": class_id,
                "class_name": row.get("class_name"),
                "path": str(path),
                "exists": exists,
                "readable": image_ok,
                "error": error,
            }
        )
    return {
        "counts": counts,
        "all_old_classes_have_exemplar": all(counts.get(int(class_id), 0) > 0 for class_id in old_classes),
        "balanced_one_per_old_class": all(counts.get(int(class_id), 0) == 1 for class_id in old_classes),
        "num_rows": len(rows),
        "readable": readable,
        "missing": missing,
        "unreadable": unreadable,
        "rows": row_summaries,
    }


@torch.no_grad()
def _exemplar_reliability(
    model: nn.Module,
    rows: Sequence[Mapping[str, object]],
    transform,
    class_to_head: Mapping[int, int],
    old_classes: Sequence[int],
    current_classes: Sequence[int],
    device: torch.device,
    alpha: float,
    batch_size: int,
    num_workers: int,
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    dataset = ExemplarPathDataset(rows, transform)
    loader = _make_loader(dataset, batch_size=max(1, min(batch_size, len(rows))), num_workers=num_workers)
    metrics = _evaluate(model, loader, class_to_head, old_classes, current_classes, device, alpha=alpha)
    per_class_rows = []
    recalls = []
    for class_id in sorted(set(int(row["class_id"]) for row in rows)):
        item = metrics["per_class"].get(class_id, {"correct": 0, "total": 0, "predicted": 0})
        total = int(item.get("total", 0))
        recall = 100.0 * int(item.get("correct", 0)) / total if total else None
        if recall is not None:
            recalls.append(recall)
        per_class_rows.append(
            {
                "class_id": class_id,
                "total": total,
                "correct": int(item.get("correct", 0)),
                "recall": recall,
                "predicted": int(item.get("predicted", 0)),
            }
        )
    summary = {
        "alpha": float(alpha),
        "exemplar_acc": metrics["AccT"],
        "exemplar_old_to_current_rate": metrics["old_to_current_rate"],
        "per_class_recall_mean": sum(recalls) / len(recalls) if recalls else None,
        "per_class_recall_min": min(recalls) if recalls else None,
        "per_class_recall_max": max(recalls) if recalls else None,
        "failed_classes": [row["class_id"] for row in per_class_rows if row["recall"] is not None and row["recall"] <= 0.0],
    }
    return summary, per_class_rows


def _compact_metrics(row: Mapping[str, object], extra: Mapping[str, object] | None = None) -> Dict[str, object]:
    keys = [
        "alpha",
        "AccT",
        "balanced_acc",
        "macro_f1",
        "old_acc",
        "current_acc",
        "old_to_current_rate",
        "current_to_old_rate",
    ]
    payload = {key: row.get(key) for key in keys}
    if extra:
        payload.update(extra)
    return payload


def _best_guardrail(rows: Sequence[Mapping[str, object]]) -> Dict[str, object] | None:
    candidates = [
        row
        for row in rows
        if (row.get("current_acc") is not None and row.get("current_to_old_rate") is not None)
        and float(row["current_acc"]) >= 85.0
        and float(row["current_to_old_rate"]) < 15.0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (float(row["old_acc"] or 0.0), float(row["AccT"] or 0.0)))


def _landscape_summary(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    return {
        "best_AccT": _compact_metrics(max(rows, key=lambda row: float(row["AccT"] or -1.0))),
        "best_balanced_acc": _compact_metrics(max(rows, key=lambda row: float(row["balanced_acc"] or -1.0))),
        "best_macro_f1": _compact_metrics(max(rows, key=lambda row: float(row["macro_f1"] or -1.0))),
        "best_guardrail": _compact_metrics(_best_guardrail(rows)) if _best_guardrail(rows) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--phase0-ckpt", required=True)
    parser.add_argument("--phase1-ckpt", required=True)
    parser.add_argument("--exemplar-csv", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--alpha-grid", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.75,1.00")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    protocol = build_protocol(_make_protocol_args(args))
    phase0_payload = torch.load(args.phase0_ckpt, map_location="cpu")
    phase1_payload = torch.load(args.phase1_ckpt, map_location=device)
    old_classes = [int(class_id) for class_id in phase0_payload["old_classes"]]
    current_classes = [int(class_id) for class_id in phase0_payload["current_classes"]]
    seen_classes = old_classes + current_classes
    phase0_class_to_head = {int(class_id): index for index, class_id in enumerate(old_classes)}
    phase1_class_to_head = {int(class_id): index for index, class_id in enumerate(seen_classes)}
    calibration_payload = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))
    selected_rows = calibration_payload["selected_test_rows"]
    selected_by_rule = {row["rule"]: row for row in selected_rows}

    phase0_model = resnet32().to(device)
    phase0_model.expand_classifier(len(old_classes))
    phase0_model.load_state_dict(phase0_payload["model_state"])
    phase0_loader = _make_loader(
        protocol.test_dataset_for_classes(old_classes),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    phase0_metrics = _evaluate(
        phase0_model,
        phase0_loader,
        phase0_class_to_head,
        old_classes,
        [],
        device,
        alpha=1.0,
    )
    phase0_summary = {**_compact_metrics(phase0_metrics), **_classwise_summary(phase0_metrics)}
    anchors = {int(k): v for k, v in phase0_payload["anchors"].items()}
    anchor_norms = [float(tensor.float().norm().item()) for tensor in anchors.values()]
    phase0_summary["anchor_stats"] = {
        "num_anchors": len(anchor_norms),
        "finite": all(math.isfinite(value) for value in anchor_norms),
        "norm_mean": sum(anchor_norms) / len(anchor_norms) if anchor_norms else None,
        "norm_min": min(anchor_norms) if anchor_norms else None,
        "norm_max": max(anchor_norms) if anchor_norms else None,
    }

    phase1_model = resnet32().to(device)
    phase1_model.expand_classifier(len(seen_classes))
    phase1_model.load_state_dict(phase1_payload["model_state"])
    seen_loader = _make_loader(
        protocol.test_dataset_for_classes(seen_classes),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    raw_freeze = _evaluate(phase1_model, seen_loader, phase1_class_to_head, old_classes, current_classes, device, alpha=1.0)
    raw_freeze_summary = _compact_metrics(raw_freeze)
    raw_freeze_summary["raw_freeze_alpha_equals_one_confirmed"] = True
    raw_freeze_summary["confirmation"] = "The evaluator multiplies only current-head logits by alpha; alpha=1.0 leaves logits unchanged."

    alpha_values = [float(item) for item in args.alpha_grid.split(",") if item.strip()]
    alpha_rows = [
        _compact_metrics(
            _evaluate(phase1_model, seen_loader, phase1_class_to_head, old_classes, current_classes, device, alpha=alpha),
            {"seed": int(args.seed)},
        )
        for alpha in alpha_values
    ]
    landscape = _landscape_summary(alpha_rows)

    exemplar_rows = _load_exemplar_rows(Path(args.exemplar_csv))
    exemplar_files = _check_exemplar_files(exemplar_rows, old_classes)
    selected_exemplar_alpha = float(selected_by_rule["balanced_hmean_exemplar"]["alpha"])
    exemplar_selected_summary, exemplar_selected_rows = _exemplar_reliability(
        phase1_model,
        exemplar_rows,
        protocol.eval_transform,
        phase1_class_to_head,
        old_classes,
        current_classes,
        device,
        alpha=selected_exemplar_alpha,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    exemplar_raw_summary, exemplar_raw_rows = _exemplar_reliability(
        phase1_model,
        exemplar_rows,
        protocol.eval_transform,
        phase1_class_to_head,
        old_classes,
        current_classes,
        device,
        alpha=1.0,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )

    rule_rows = []
    oracle_best = landscape["best_AccT"]
    oracle_guardrail = landscape["best_guardrail"]
    for rule_name in [
        "balanced_hmean_exemplar",
        "constrained_anchor",
        "constrained_exemplar",
        "margin_balance",
    ]:
        selected = selected_by_rule[rule_name]
        rule_rows.append(
            {
                "seed": int(args.seed),
                "rule": rule_name,
                "selected_alpha": selected["alpha"],
                "selected_AccT": selected["AccT"],
                "selected_old_acc": selected["old_acc"],
                "selected_current_acc": selected["current_acc"],
                "selected_current_to_old_rate": selected["current_to_old_rate"],
                "oracle_best_alpha": oracle_best["alpha"],
                "oracle_best_AccT": oracle_best["AccT"],
                "alpha_gap": float(selected["alpha"]) - float(oracle_best["alpha"]),
                "AccT_gap": float(oracle_best["AccT"]) - float(selected["AccT"]),
                "oracle_guardrail_alpha": oracle_guardrail["alpha"] if oracle_guardrail else None,
                "oracle_guardrail_AccT": oracle_guardrail["AccT"] if oracle_guardrail else None,
            }
        )

    per_class_rows = []
    for split, metrics in [("phase0", phase0_metrics), ("raw_freeze", raw_freeze)]:
        for class_id, item in sorted(metrics["per_class"].items()):
            total = int(item.get("total", 0))
            per_class_rows.append(
                {
                    "seed": int(args.seed),
                    "split": split,
                    "class_id": int(class_id),
                    "recall": 100.0 * int(item.get("correct", 0)) / total if total else None,
                    "correct": int(item.get("correct", 0)),
                    "total": total,
                    "predicted": int(item.get("predicted", 0)),
                }
            )

    payload = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "seed": int(args.seed),
        "paths": {
            "data_root": args.data_root,
            "phase0_ckpt": args.phase0_ckpt,
            "phase1_ckpt": args.phase1_ckpt,
            "exemplar_csv": args.exemplar_csv,
            "calibration_json": args.calibration_json,
            "output_dir": str(output_dir),
        },
        "old_classes": old_classes,
        "current_classes": current_classes,
        "raw_freeze_metrics": raw_freeze_summary,
        "phase0_quality": phase0_summary,
        "alpha_landscape": alpha_rows,
        "alpha_landscape_summary": landscape,
        "selected_rules": selected_rows,
        "rule_selection_vs_oracle": rule_rows,
        "exemplar_stability": {
            "file_check": exemplar_files,
            "selected_alpha_reliability": exemplar_selected_summary,
            "raw_alpha1_reliability": exemplar_raw_summary,
        },
    }
    _write_json(output_dir / "diagnostics.json", payload)
    _write_json(output_dir / "raw_freeze_metrics.json", raw_freeze_summary)
    _write_json(output_dir / "phase0_metrics.json", phase0_summary)
    _write_csv(output_dir / "alpha_landscape.csv", alpha_rows)
    _write_csv(output_dir / "selected_rules.csv", selected_rows)
    _write_csv(output_dir / "rule_selection_vs_oracle.csv", rule_rows)
    _write_csv(output_dir / "per_class_metrics.csv", per_class_rows)
    _write_csv(output_dir / "exemplar_file_check.csv", exemplar_files["rows"])
    _write_csv(output_dir / "exemplar_selected_alpha_per_class.csv", exemplar_selected_rows)
    _write_csv(output_dir / "exemplar_raw_alpha1_per_class.csv", exemplar_raw_rows)
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
