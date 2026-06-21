#!/usr/bin/env python3
"""Summarize Finetune vs Finetune+GPA reproduction diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_METHOD_DIRS = ["finetune", "finetune_gpa"]

DIAGNOSTIC_KEYS = [
    "paper_acc_include_base",
    "paper_acc_exclude_base",
    "paper_accT",
    "base_train_acc_last_epoch",
    "base_test_acc",
    "old_head_weight_max_delta_before_after_phase",
    "old_head_bias_max_delta_before_after_phase",
    "predicted_task_hist_over_all_seen",
    "gpa_anchor_loss_raw",
    "gpa_anchor_loss_scaled",
    "ce_loss",
    "anchor_to_ce_ratio",
    "feature_norm_mean",
    "feature_norm_std",
    "classifier_weight_norm_by_task",
    "classifier_bias_mean_by_task",
]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _finite_or_none(value: Any) -> bool:
    if value is None:
        return True
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _discover_runs(args: argparse.Namespace) -> List[Path]:
    if args.runs:
        return [Path(run) for run in args.runs]
    root = Path(args.root)
    if (root / "summary.json").exists():
        return [root]
    return [root / method for method in DEFAULT_METHOD_DIRS if (root / method).exists()]


def _validate_run(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json: {run_dir}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics.jsonl: {run_dir}")

    summary = _read_json(summary_path)
    rows = _read_jsonl(metrics_path)
    if not rows:
        raise RuntimeError(f"empty metrics.jsonl: {run_dir}")
    final = rows[-1]

    problems: List[str] = []
    method = str(summary.get("method") or final.get("method"))
    if method not in {"finetune", "finetune_gpa"}:
        problems.append(f"unexpected method {method!r}")

    for key in DIAGNOSTIC_KEYS:
        if key not in final and key not in summary:
            problems.append(f"missing diagnostic field {key}")

    for key in [
        "train_loss",
        "train_ce_loss",
        "gpa_anchor_loss",
        "method_extra_loss",
        "paper_acc_include_base",
        "paper_accT",
        "anchor_to_ce_ratio",
    ]:
        value = final.get(key, summary.get(key))
        if not _finite_or_none(value):
            problems.append(f"non-finite {key}: {value!r}")

    if method == "finetune" and bool(summary.get("gpa_freeze_old_head", final.get("gpa_freeze_old_head"))):
        problems.append("finetune should not default to gpa_freeze_old_head=True")
    if method == "finetune_gpa" and not bool(summary.get("gpa_freeze_old_head", final.get("gpa_freeze_old_head"))):
        problems.append("finetune_gpa should default to gpa_freeze_old_head=True")

    status = "ok" if not problems else "fail"
    row = {
        "method": method,
        "status": status,
        "run_dir": str(run_dir),
        "problems": "; ".join(problems),
        "seed": summary.get("seed"),
        "rho": summary.get("rho"),
        "order": summary.get("order"),
        "final_acc": summary.get("final_accuracy"),
        "avg_inc_acc": summary.get("average_incremental_accuracy"),
        "paper_acc_include_base": summary.get("paper_acc_include_base", final.get("paper_acc_include_base")),
        "paper_acc_exclude_base": summary.get("paper_acc_exclude_base", final.get("paper_acc_exclude_base")),
        "paper_accT": summary.get("paper_accT", final.get("paper_accT")),
        "forgetting": summary.get("forgetting"),
        "base_train_acc_last_epoch": summary.get("base_train_acc_last_epoch", final.get("base_train_acc_last_epoch")),
        "base_test_acc": summary.get("base_test_acc", final.get("base_test_acc")),
        "gpa_freeze_old_head": summary.get("gpa_freeze_old_head", final.get("gpa_freeze_old_head")),
        "gpa_restore_old_head_after_step": summary.get(
            "gpa_restore_old_head_after_step",
            final.get("gpa_restore_old_head_after_step"),
        ),
        "gpa_anchor_reduction": summary.get("gpa_anchor_reduction", final.get("gpa_anchor_reduction")),
        "old_head_weight_delta": summary.get(
            "old_head_weight_max_delta_before_after_phase",
            final.get("old_head_weight_max_delta_before_after_phase"),
        ),
        "old_head_bias_delta": summary.get(
            "old_head_bias_max_delta_before_after_phase",
            final.get("old_head_bias_max_delta_before_after_phase"),
        ),
        "new_weight_cos_mean": summary.get(
            "new_weight_cos_to_frozen_proto_mean",
            final.get("new_weight_cos_to_frozen_proto_mean"),
        ),
        "new_weight_cos_min": summary.get(
            "new_weight_cos_to_frozen_proto_min",
            final.get("new_weight_cos_to_frozen_proto_min"),
        ),
        "anchor_to_ce_ratio": summary.get("anchor_to_ce_ratio", final.get("anchor_to_ce_ratio")),
        "predicted_task_hist_over_all_seen": json.dumps(
            summary.get("predicted_task_hist_over_all_seen", final.get("predicted_task_hist_over_all_seen")),
            sort_keys=True,
        ),
    }
    if problems:
        row["_raise"] = RuntimeError(f"{run_dir}: " + "; ".join(problems))
    return row


def _print_table(rows: List[Dict[str, Any]]) -> None:
    columns = [
        "method",
        "status",
        "final_acc",
        "paper_acc_include_base",
        "paper_acc_exclude_base",
        "paper_accT",
        "forgetting",
        "base_train_acc_last_epoch",
        "base_test_acc",
        "gpa_freeze_old_head",
        "gpa_restore_old_head_after_step",
        "old_head_weight_delta",
        "anchor_to_ce_ratio",
        "new_weight_cos_mean",
    ]
    widths = {column: len(column) for column in columns}
    for row in rows:
        for column in columns:
            widths[column] = max(widths[column], len(str(row.get(column, ""))))
    print(" | ".join(column.ljust(widths[column]) for column in columns))
    print("-+-".join("-" * widths[column] for column in columns))
    for row in rows:
        print(" | ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Finetune/GPA reproduction diagnostics.")
    parser.add_argument("--root", default="runs/finetune_gpa_diag")
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()

    run_dirs = _discover_runs(args)
    if not run_dirs:
        raise FileNotFoundError(f"no runs found under {args.root}")
    rows = [_validate_run(run_dir) for run_dir in run_dirs]
    rows.sort(key=lambda row: DEFAULT_METHOD_DIRS.index(row["method"]) if row["method"] in DEFAULT_METHOD_DIRS else 99)
    _print_table(rows)

    output_csv = Path(args.output_csv) if args.output_csv else Path(args.root) / "finetune_gpa_repro_summary.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in rows[0] if key != "_raise"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key != "_raise"})
    print(f"summary_csv={output_csv}")

    failures = [row["_raise"] for row in rows if "_raise" in row]
    if failures and not args.allow_failures:
        raise failures[0]


if __name__ == "__main__":
    main()
