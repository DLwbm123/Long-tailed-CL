#!/usr/bin/env python3
"""Summarize complete HyperKvasir23 full-ConCM runs across seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Mapping, Sequence


CORE = [
    "AccT",
    "old_acc",
    "current_acc",
    "HM_old_current",
    "old_to_current_rate",
    "current_to_old_rate",
    "BER",
    "macro_balanced_acc",
    "macro_f1",
    "worst_class_recall",
    "head_accuracy",
    "mid_accuracy",
    "tail_accuracy",
    "ETF_residual",
    "SMR_old",
    "SMR_current",
    "own_anchor_margin",
]


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, group_keys):
    grouped = {}
    for row in rows:
        key = tuple(row[value] for value in group_keys)
        grouped.setdefault(key, []).append(row)
    output = []
    for key, values in sorted(grouped.items()):
        item = dict(zip(group_keys, key))
        item["n"] = len(values)
        for metric in CORE:
            numbers = [float(value[metric]) for value in values if value.get(metric) not in {None, ""}]
            if numbers:
                item[f"{metric}_mean"] = statistics.mean(numbers)
                item[f"{metric}_std"] = statistics.stdev(numbers) if len(numbers) > 1 else 0.0
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    csv_dir = root / "csv"
    json_dir = root / "json"
    csv_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    run_status = []
    for seed in (1, 2, 3):
        for method in ("locked_nc", "full_dynamic", "nc_anchored"):
            run = root / "runs" / f"seed{seed}_{method}"
            result = run / "json" / "final_results.json"
            if not result.is_file():
                run_status.append({"seed": seed, "method": method, "status": "MISSING", "path": str(result)})
                continue
            payload = json.loads(result.read_text())
            run_status.append(
                {
                    "seed": seed,
                    "method": method,
                    "status": "COMPLETE" if payload.get("completed_full_five_session") else "PARTIAL",
                    "successful_session_count": payload.get("successful_session_count"),
                    "path": str(result),
                }
            )
            rows.extend(payload["session_metrics"])
    write_csv(csv_dir / "all_session_metrics.csv", rows)
    final_rows = [row for row in rows if int(row["session"]) == 5]
    write_csv(csv_dir / "final_session_rows.csv", final_rows)
    final_mean_std = aggregate(final_rows, ["method"])
    write_csv(csv_dir / "final_session_mean_std.csv", final_mean_std)

    incremental_rows = [row for row in rows if int(row["session"]) > 0]
    per_seed_average = aggregate(incremental_rows, ["seed", "method"])
    write_csv(csv_dir / "five_session_per_seed_average.csv", per_seed_average)
    normalized_average_rows = []
    for row in per_seed_average:
        normalized = {"seed": row["seed"], "method": row["method"]}
        for metric in CORE:
            if f"{metric}_mean" in row:
                normalized[metric] = row[f"{metric}_mean"]
        normalized_average_rows.append(normalized)
    five_session_mean_std = aggregate(normalized_average_rows, ["method"])
    write_csv(csv_dir / "five_session_average_mean_std.csv", five_session_mean_std)

    lookup = {(int(row["seed"]), row["method"], int(row["session"])): row for row in rows}
    deltas = []
    for seed in (1, 2, 3):
        for method in ("full_dynamic", "nc_anchored"):
            for session in range(6):
                candidate = lookup.get((seed, method, session))
                baseline = lookup.get((seed, "locked_nc", session))
                if not candidate or not baseline:
                    continue
                item = {"seed": seed, "method": method, "session": session}
                for metric in CORE:
                    if candidate.get(metric) is not None and baseline.get(metric) is not None:
                        item[f"delta_{metric}"] = float(candidate[metric]) - float(baseline[metric])
                deltas.append(item)
    write_csv(csv_dir / "paired_delta_vs_locked.csv", deltas)

    collapse_rows = [
        row
        for row in rows
        if int(row["session"]) > 0
        and (float(row.get("current_to_old_rate") or 0.0) >= 30.0 or float(row.get("current_acc") or 100.0) < 50.0)
    ]
    write_csv(csv_dir / "collapse_rows.csv", collapse_rows)
    summary = {
        "run_status": run_status,
        "successful_run_count": sum(value["status"] == "COMPLETE" for value in run_status),
        "expected_run_count": 9,
        "collapse_count_by_method": {
            method: sum(row["method"] == method for row in collapse_rows)
            for method in ("locked_nc", "full_dynamic", "nc_anchored")
        },
        "worst_seed_final_HM": {
            method: min(
                (row for row in final_rows if row["method"] == method),
                key=lambda row: float(row["HM_old_current"]),
                default=None,
            )
            for method in ("locked_nc", "full_dynamic", "nc_anchored")
        },
        "final_session_mean_std": final_mean_std,
        "five_session_average_mean_std": five_session_mean_std,
    }
    (json_dir / "multiseed_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "runs": len(run_status), "complete": summary["successful_run_count"]}, indent=2))


if __name__ == "__main__":
    main()
