import argparse
import ast
import json
import math
import re
from pathlib import Path


SEED_RE = re.compile(r"\[trainer.py\] => seed: (\d+)")
CNN_RE = re.compile(r"\[trainer.py\] => CNN: (\{.*\})")
CURVE_RE = re.compile(r"\[trainer.py\] => CNN top1 curve: (\[.*\])")
AVG_RE = re.compile(r"Average Accuracy \(CNN\): ([0-9.]+)")
CAL_RE = re.compile(
    r"ConCMStage1EvalCalibration rule=(?P<rule>\S+) alpha=(?P<alpha>[0-9.]+) "
    r"old_norm=(?P<old_norm>[0-9.]+) new_norm=(?P<new_norm>[0-9.]+) "
    r"uncalibrated_total=(?P<uncalibrated_total>[0-9.]+) "
    r"uncalibrated_old_acc=(?P<uncalibrated_old_acc>[0-9.]+) "
    r"uncalibrated_new_acc=(?P<uncalibrated_new_acc>[0-9.]+) "
    r"uncalibrated_new_eval_pred_old_rate=(?P<uncalibrated_new_eval_pred_old_rate>[0-9.]+) "
    r"uncalibrated_old_eval_pred_new_rate=(?P<uncalibrated_old_eval_pred_new_rate>[0-9.]+) "
    r"calibrated_total=(?P<calibrated_total>[0-9.]+) "
    r"calibrated_old_acc=(?P<calibrated_old_acc>[0-9.]+) "
    r"calibrated_new_acc=(?P<calibrated_new_acc>[0-9.]+) "
    r"calibrated_new_eval_pred_old_rate=(?P<calibrated_new_eval_pred_old_rate>[0-9.]+) "
    r"calibrated_old_eval_pred_new_rate=(?P<calibrated_old_eval_pred_new_rate>[0-9.]+)"
)


def _float(value):
    return None if value is None else float(value)


def parse_log(path):
    path = Path(path)
    text = path.read_text(errors="replace")
    current_seed = None
    by_seed = {}
    for line in text.splitlines():
        seed_match = SEED_RE.search(line)
        if seed_match:
            current_seed = int(seed_match.group(1))
            by_seed.setdefault(current_seed, {"seed": current_seed})
            continue
        if current_seed is None:
            continue

        cnn_match = CNN_RE.search(line)
        if cnn_match:
            by_seed[current_seed]["cnn_grouped"] = ast.literal_eval(cnn_match.group(1))
            continue
        curve_match = CURVE_RE.search(line)
        if curve_match:
            by_seed[current_seed]["curve"] = ast.literal_eval(curve_match.group(1))
            continue
        avg_match = AVG_RE.search(line)
        if avg_match:
            by_seed[current_seed]["avg_acc"] = float(avg_match.group(1))
            continue
        cal_match = CAL_RE.search(line)
        if cal_match:
            by_seed[current_seed]["calibration"] = {
                key: (value if key == "rule" else float(value))
                for key, value in cal_match.groupdict().items()
            }

    error_counts = {
        "Traceback": text.count("Traceback"),
        "RuntimeError": text.count("RuntimeError"),
        "CUDA OOM": text.count("CUDA out of memory"),
    }
    return {"path": str(path), "error_counts": error_counts, "seeds": by_seed}


def flatten_run(parsed, method):
    rows = {}
    for seed, record in parsed["seeds"].items():
        grouped = record.get("cnn_grouped", {})
        curve = record.get("curve", [])
        calibration = record.get("calibration", {})
        rows[int(seed)] = {
            "method": method,
            "seed": int(seed),
            "curve": curve,
            "avg_acc": _float(record.get("avg_acc")),
            "task1_accT": _float(grouped.get("total")),
            "old_acc": _float(grouped.get("old")),
            "new_acc": _float(grouped.get("new")),
            "new_eval_pred_old_rate": _float(grouped.get("new_eval_pred_old_rate")),
            "old_eval_pred_new_rate": _float(grouped.get("old_eval_pred_new_rate")),
            "alpha": _float(grouped.get("eval_calibration_alpha")),
            "old_effective_norm": _float(grouped.get("eval_calibration_old_norm")),
            "new_effective_norm": _float(grouped.get("eval_calibration_new_norm")),
            "uncalibrated_task1_accT": _float(grouped.get("uncalibrated_total", calibration.get("uncalibrated_total"))),
            "uncalibrated_old_acc": _float(grouped.get("uncalibrated_old_acc", calibration.get("uncalibrated_old_acc"))),
            "uncalibrated_new_acc": _float(grouped.get("uncalibrated_new_acc", calibration.get("uncalibrated_new_acc"))),
            "uncalibrated_new_eval_pred_old_rate": _float(grouped.get("uncalibrated_new_eval_pred_old_rate", calibration.get("uncalibrated_new_eval_pred_old_rate"))),
            "uncalibrated_old_eval_pred_new_rate": _float(grouped.get("uncalibrated_old_eval_pred_new_rate", calibration.get("uncalibrated_old_eval_pred_new_rate"))),
        }
    return rows


def mean_std(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def complete(row):
    required = ["avg_acc", "task1_accT", "old_acc", "new_acc"]
    return len(row.get("curve") or []) >= 2 and all(row.get(key) is not None for key in required)


def summarize(rows):
    metrics = [
        "avg_acc",
        "task1_accT",
        "old_acc",
        "new_acc",
        "new_eval_pred_old_rate",
        "old_eval_pred_new_rate",
        "alpha",
    ]
    out = {}
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method and complete(row)]
        out[method] = {}
        for metric in metrics:
            mean, std = mean_std(row.get(metric) for row in subset)
            out[method][metric] = {"mean": mean, "std": std}
    return out


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-log", required=True)
    parser.add_argument("--calibrated-log", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    baseline = parse_log(args.baseline_log)
    calibrated = parse_log(args.calibrated_log)
    baseline_rows = flatten_run(baseline, "baseline")
    calibrated_rows = flatten_run(calibrated, "stage1_headnorm")
    seeds = sorted(set(baseline_rows) | set(calibrated_rows))

    rows = []
    deltas = []
    for seed in seeds:
        if seed in baseline_rows:
            rows.append(baseline_rows[seed])
        if seed in calibrated_rows:
            rows.append(calibrated_rows[seed])
        if (
            seed in baseline_rows
            and seed in calibrated_rows
            and complete(baseline_rows[seed])
            and complete(calibrated_rows[seed])
        ):
            base = baseline_rows[seed]
            cal = calibrated_rows[seed]
            deltas.append(
                {
                    "seed": seed,
                    "delta_avg_acc": cal["avg_acc"] - base["avg_acc"],
                    "delta_task1_accT": cal["task1_accT"] - base["task1_accT"],
                    "delta_old_acc": cal["old_acc"] - base["old_acc"],
                    "delta_new_acc": cal["new_acc"] - base["new_acc"],
                    "delta_new_eval_pred_old_rate": (
                        None
                        if base["new_eval_pred_old_rate"] is None
                        else cal["new_eval_pred_old_rate"] - base["new_eval_pred_old_rate"]
                    ),
                }
            )

    result = {
        "baseline_log": args.baseline_log,
        "calibrated_log": args.calibrated_log,
        "baseline_error_counts": baseline["error_counts"],
        "calibrated_error_counts": calibrated["error_counts"],
        "rows": rows,
        "summary": summarize(rows),
        "deltas": deltas,
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2))

    per_seed_rows = []
    for row in rows:
        per_seed_rows.append(
            [
                str(row["seed"]),
                row["method"],
                str(row["curve"]),
                fmt(row["avg_acc"]),
                fmt(row["task1_accT"], 2),
                fmt(row["old_acc"], 2),
                fmt(row["new_acc"], 2),
                fmt(row["new_eval_pred_old_rate"], 2),
                fmt(row["old_eval_pred_new_rate"], 2),
                fmt(row["alpha"], 6),
                fmt(row["old_effective_norm"], 6),
                fmt(row["new_effective_norm"], 6),
            ]
        )
    summary_rows = []
    for method, metrics in result["summary"].items():
        summary_rows.append(
            [
                method,
                f"{fmt(metrics['avg_acc']['mean'])} +/- {fmt(metrics['avg_acc']['std'])}",
                f"{fmt(metrics['task1_accT']['mean'])} +/- {fmt(metrics['task1_accT']['std'])}",
                f"{fmt(metrics['old_acc']['mean'])} +/- {fmt(metrics['old_acc']['std'])}",
                f"{fmt(metrics['new_acc']['mean'])} +/- {fmt(metrics['new_acc']['std'])}",
                f"{fmt(metrics['new_eval_pred_old_rate']['mean'])} +/- {fmt(metrics['new_eval_pred_old_rate']['std'])}",
                f"{fmt(metrics['alpha']['mean'], 6)} +/- {fmt(metrics['alpha']['std'], 6)}",
            ]
        )
    delta_rows = [
        [
            str(row["seed"]),
            fmt(row["delta_avg_acc"]),
            fmt(row["delta_task1_accT"], 2),
            fmt(row["delta_old_acc"], 2),
            fmt(row["delta_new_acc"], 2),
            fmt(row["delta_new_eval_pred_old_rate"], 2),
        ]
        for row in deltas
    ]
    md = "# APART Stage1 Head-Norm 3-Seed Phase1 Summary\n\n"
    md += "## Per-Seed Results\n\n"
    md += table(
        [
            "seed",
            "method",
            "curve",
            "Avg Acc",
            "task1 AccT",
            "old acc",
            "new acc",
            "new->old",
            "old->new",
            "alpha",
            "old norm",
            "new norm",
        ],
        per_seed_rows,
    )
    md += "\n## Mean +/- Std\n\n"
    md += table(
        ["method", "Avg Acc", "task1 AccT", "old acc", "new acc", "new->old", "alpha"],
        summary_rows,
    )
    md += "\n## Deltas: calibrated - baseline\n\n"
    md += table(["seed", "Avg Acc", "task1 AccT", "old acc", "new acc", "new->old"], delta_rows)
    md += "\n## Error Counts\n\n"
    md += "```json\n" + json.dumps(
        {
            "baseline": baseline["error_counts"],
            "stage1_headnorm": calibrated["error_counts"],
        },
        indent=2,
    ) + "\n```\n"
    Path(args.output_md).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
