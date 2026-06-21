#!/usr/bin/env python3
"""Summarize HyperKvasir23 module validation runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


FIELDS = [
    "method",
    "final_accuracy",
    "average_incremental_accuracy",
    "paper_acc_include_base",
    "paper_acc_exclude_base",
    "paper_accT",
    "forgetting",
    "many_acc",
    "medium_acc",
    "few_acc",
    "balanced_acc",
    "macro_f1",
    "base_test_acc",
    "phase1_task0_acc_after_train",
    "phase1_latest_pred_rate_after_train",
    "old_head_weight_max_delta_before_after_phase",
    "old_head_bias_max_delta_before_after_phase",
    "anchor_to_ce_ratio",
    "feature_drift_old_eval_mean_cos",
    "bn_running_mean_max_delta_after_phase",
    "bn_running_var_max_delta_after_phase",
    "output_dir",
]


def _load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Run root containing one directory per method.")
    args = parser.parse_args()

    root = Path(args.root)
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("*/summary.json")):
        summary = _load_summary(summary_path)
        row = {field: summary.get(field) for field in FIELDS}
        row["method"] = summary.get("method") or summary.get("method_name") or summary_path.parent.name
        row["output_dir"] = str(summary_path.parent)
        rows.append(row)

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


if __name__ == "__main__":
    main()
