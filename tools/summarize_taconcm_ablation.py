#!/usr/bin/env python3
"""Summarize and validate TaConCM-GPA ablation runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

EXPECTED_STAGES = [
    "finetune",
    "finetune_gpa",
    "taconcm_stage1_calib_init_only",
    "taconcm_stage2_calib_anchor",
    "taconcm_stage3_calib_tailanchor",
    "taconcm_stage4_full",
]

REQUIRED_RUN_METADATA_KEYS = [
    "git_commit_hash",
    "git_is_dirty",
    "git_status_short",
    "method",
    "method_name",
    "ablation_stage",
    "seed",
    "dataset",
    "rho",
    "imbalance_ratio",
    "task_split",
    "order",
    "base_classes",
    "incremental_steps",
    "debug_num_classes",
    "max_train_per_class",
    "lambda_gpa",
    "gpa_anchor_start_phase",
    "gpa_init_bias",
    "gpa_flags",
    "use_ltconcm",
    "ltconcm_calibrate_prototypes",
    "ltconcm_anchor_target",
    "ltconcm_tail_anchor",
    "ltconcm_anchor_gamma",
    "ltconcm_anchor_max_weight",
    "ltconcm_memory_topk",
    "ltconcm_alpha_a",
    "ltconcm_alpha_b",
    "ltconcm_alpha_min",
    "ltconcm_alpha_max",
    "ltconcm_use_tdsm",
    "ltconcm_use_match_loss",
    "ltconcm_match_lambda",
    "taconcm_flags",
]

REQUIRED_SUMMARY_REPRO_KEYS = [
    "reproduction_command",
    "argv",
    "python_executable",
    "resolved_data_root",
    "resolved_output_dir",
    "config",
    "class_order",
    "class_counts",
]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _is_finite(value: Any) -> bool:
    if value is None:
        return True
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _has_alpha_diagnostics(row: Dict[str, Any]) -> bool:
    return any(
        key in row
        for key in (
            "ltconcm_alpha_many_mean",
            "ltconcm_alpha_medium_mean",
            "ltconcm_alpha_few_mean",
        )
    )


def _check_metadata(source: str, payload: Dict[str, Any], problems: List[str]) -> None:
    for key in REQUIRED_RUN_METADATA_KEYS:
        if key not in payload:
            problems.append(f"{source} missing reproducibility field {key}")
    task_split = payload.get("task_split")
    if not isinstance(task_split, dict):
        problems.append(f"{source} task_split should be a dictionary")
    else:
        for key in ["base_classes", "incremental_steps", "num_phases", "task_sizes", "tasks"]:
            if key not in task_split:
                problems.append(f"{source} task_split missing {key}")
    gpa_flags = payload.get("gpa_flags")
    if not isinstance(gpa_flags, dict):
        problems.append(f"{source} gpa_flags should be a dictionary")
    else:
        for key in ["lambda_gpa", "gpa_anchor_start_phase", "gpa_init_bias"]:
            if key not in gpa_flags:
                problems.append(f"{source} gpa_flags missing {key}")
    taconcm_flags = payload.get("taconcm_flags")
    if not isinstance(taconcm_flags, dict):
        problems.append(f"{source} taconcm_flags should be a dictionary")
    else:
        for key in [
            "use_ltconcm",
            "ltconcm_calibrate_prototypes",
            "ltconcm_anchor_target",
            "ltconcm_tail_anchor",
            "ltconcm_anchor_gamma",
            "ltconcm_anchor_max_weight",
            "ltconcm_memory_topk",
            "ltconcm_alpha_a",
            "ltconcm_alpha_b",
            "ltconcm_alpha_min",
            "ltconcm_alpha_max",
            "ltconcm_use_tdsm",
            "ltconcm_use_match_loss",
            "ltconcm_match_lambda",
        ]:
            if key not in taconcm_flags:
                problems.append(f"{source} taconcm_flags missing {key}")


def _stage_expectations(stage: str) -> Dict[str, Any]:
    return {
        "finetune": {
            "uses_gpa": False,
            "use_ltconcm": False,
            "ltconcm_anchor_target": "gpa_raw",
            "ltconcm_tail_anchor": False,
            "ltconcm_use_tdsm": False,
            "ltconcm_use_match_loss": False,
        },
        "finetune_gpa": {
            "uses_gpa": True,
            "use_ltconcm": False,
            "ltconcm_anchor_target": "gpa_raw",
            "ltconcm_tail_anchor": False,
            "ltconcm_use_tdsm": False,
            "ltconcm_use_match_loss": False,
        },
        "taconcm_stage1_calib_init_only": {
            "uses_gpa": True,
            "use_ltconcm": True,
            "ltconcm_calibrate_prototypes": True,
            "ltconcm_anchor_target": "gpa_raw",
            "ltconcm_tail_anchor": False,
            "ltconcm_use_tdsm": False,
            "ltconcm_use_match_loss": False,
        },
        "taconcm_stage2_calib_anchor": {
            "uses_gpa": True,
            "use_ltconcm": True,
            "ltconcm_calibrate_prototypes": True,
            "ltconcm_anchor_target": "calibrated",
            "ltconcm_tail_anchor": False,
            "ltconcm_use_tdsm": False,
            "ltconcm_use_match_loss": False,
        },
        "taconcm_stage3_calib_tailanchor": {
            "uses_gpa": True,
            "use_ltconcm": True,
            "ltconcm_calibrate_prototypes": True,
            "ltconcm_anchor_target": "calibrated",
            "ltconcm_tail_anchor": True,
            "ltconcm_use_tdsm": False,
            "ltconcm_use_match_loss": False,
        },
        "taconcm_stage4_full": {
            "uses_gpa": True,
            "use_ltconcm": True,
            "ltconcm_calibrate_prototypes": True,
            "ltconcm_anchor_target": "tdsm",
            "ltconcm_tail_anchor": True,
            "ltconcm_use_tdsm": True,
            "ltconcm_use_match_loss": True,
        },
    }[stage]


def _discover_runs(args: argparse.Namespace) -> List[Path]:
    if args.runs:
        return [Path(run) for run in args.runs]
    root = Path(args.root)
    return [root / stage for stage in EXPECTED_STAGES]


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
    stage = str(summary.get("ablation_stage") or summary.get("method"))
    if stage not in EXPECTED_STAGES:
        raise RuntimeError(f"unexpected ablation stage {stage!r} in {run_dir}")

    problems: List[str] = []
    for key in ["train_loss", "train_ce_loss", "gpa_anchor_loss", "ltconcm_match_loss", "method_extra_loss"]:
        for row in rows:
            if not _is_finite(row.get(key)):
                problems.append(f"non-finite {key} at phase {row.get('phase')}")

    _check_metadata("summary.json", summary, problems)
    _check_metadata("metrics.jsonl final row", final, problems)
    for key in REQUIRED_SUMMARY_REPRO_KEYS:
        if key not in summary:
            problems.append(f"summary.json missing command reproduction field {key}")

    expected = _stage_expectations(stage)
    for key, expected_value in expected.items():
        actual = summary.get(key, final.get(key))
        if actual != expected_value:
            problems.append(f"{key} expected {expected_value!r}, got {actual!r}")

    start_phase = int(summary.get("gpa_anchor_start_phase", 1))
    for row in rows:
        phase = int(row.get("phase", -1))
        if phase < start_phase:
            anchor_loss = float(row.get("gpa_anchor_loss", 0.0) or 0.0)
            if abs(anchor_loss) > 1e-12:
                problems.append(f"phase {phase} has anchor loss before start phase")
            if row.get("gpa_anchor_active") is not False:
                problems.append(f"phase {phase} gpa_anchor_active should be false")

    if stage == "finetune":
        if any(float(row.get("method_extra_loss", 0.0) or 0.0) != 0.0 for row in rows):
            problems.append("finetune has method_extra_loss")
    elif stage == "finetune_gpa":
        if _has_alpha_diagnostics(final):
            problems.append("finetune_gpa should not report TaConCM alpha diagnostics")
    elif stage == "taconcm_stage1_calib_init_only":
        if not _has_alpha_diagnostics(final):
            problems.append("stage1 missing calibration alpha diagnostics")
        if final.get("ltconcm_anchor_target") not in {None, "gpa_raw"}:
            problems.append("stage1 should use gpa_raw anchor target")
        if any(key.startswith("tdsm_") for key in final):
            problems.append("stage1 should not report T-DSM diagnostics")
    elif stage == "taconcm_stage2_calib_anchor":
        if final.get("ltconcm_anchor_target") != "calibrated":
            problems.append("stage2 should use calibrated anchor target")
        if final.get("ltconcm_anchor_weight_few_mean") is not None:
            few_weight = float(final["ltconcm_anchor_weight_few_mean"])
            if abs(few_weight - 1.0) > 1e-6:
                problems.append("stage2 should not tail-weight anchors")
    elif stage == "taconcm_stage3_calib_tailanchor":
        if final.get("ltconcm_anchor_target") != "calibrated":
            problems.append("stage3 should use calibrated anchor target")
        if final.get("ltconcm_anchor_weight_few_mean") is not None:
            few_weight = float(final["ltconcm_anchor_weight_few_mean"])
            if abs(few_weight - 1.0) <= 1e-6:
                problems.append("stage3 tail anchor weights did not differ from 1")
    elif stage == "taconcm_stage4_full":
        if final.get("ltconcm_anchor_target") != "tdsm":
            problems.append("stage4 should use T-DSM anchor target")
        if not any(key.startswith("tdsm_") for key in final):
            problems.append("stage4 missing T-DSM diagnostics")
        if float(final.get("ltconcm_match_loss", 0.0) or 0.0) == 0.0:
            problems.append("stage4 match loss is zero")

    status = "ok" if not problems else "fail"
    row = {
        "stage": stage,
        "run_dir": str(run_dir),
        "status": status,
        "problems": "; ".join(problems),
        "seed": summary.get("seed"),
        "rho": summary.get("rho"),
        "base_classes": summary.get("base_classes"),
        "incremental_steps": summary.get("incremental_steps"),
        "final_acc": summary.get("final_accuracy"),
        "avg_inc_acc": summary.get("average_incremental_accuracy"),
        "forgetting": summary.get("forgetting"),
        "many_acc": summary.get("many_acc"),
        "medium_acc": summary.get("medium_acc"),
        "few_acc": summary.get("few_acc"),
        "head_tail_gap": summary.get("head_tail_gap"),
        "gpa_anchor_start_phase": summary.get("gpa_anchor_start_phase"),
        "anchor_target": summary.get("ltconcm_anchor_target"),
        "tail_anchor": summary.get("ltconcm_tail_anchor"),
        "tdsm": summary.get("ltconcm_use_tdsm"),
        "match_loss_enabled": summary.get("ltconcm_use_match_loss"),
        "final_train_loss": final.get("train_loss"),
        "final_anchor_loss": final.get("gpa_anchor_loss"),
        "final_match_loss": final.get("ltconcm_match_loss"),
        "final_extra_loss": final.get("method_extra_loss"),
    }
    if problems:
        row["_raise"] = RuntimeError(f"{run_dir}: " + "; ".join(problems))
    return row


def _print_table(rows: List[Dict[str, Any]]) -> None:
    columns = [
        "stage",
        "status",
        "final_acc",
        "avg_inc_acc",
        "forgetting",
        "final_anchor_loss",
        "final_match_loss",
        "anchor_target",
        "tail_anchor",
        "tdsm",
        "match_loss_enabled",
    ]
    widths = {column: len(column) for column in columns}
    for row in rows:
        for column in columns:
            widths[column] = max(widths[column], len(str(row.get(column, ""))))
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("-+-".join("-" * widths[column] for column in columns))
    for row in rows:
        print(" | ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TaConCM ablation runs.")
    parser.add_argument("--root", default="runs/taconcm_ablation_smoke_s0")
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()

    rows = [_validate_run(run_dir) for run_dir in _discover_runs(args)]
    rows.sort(key=lambda row: EXPECTED_STAGES.index(row["stage"]))
    _print_table(rows)

    output_csv = Path(args.output_csv) if args.output_csv else Path(args.root) / "ablation_summary.csv"
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
