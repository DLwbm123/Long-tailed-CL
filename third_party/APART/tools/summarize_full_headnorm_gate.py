import argparse
import ast
import json
import math
import re
from pathlib import Path


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


def parse_log(path):
    text = Path(path).read_text(errors="replace")
    cnn_rows = []
    curves = []
    avgs = []
    calibrations = []
    for line in text.splitlines():
        match = CNN_RE.search(line)
        if match:
            cnn_rows.append(ast.literal_eval(match.group(1)))
            continue
        match = CURVE_RE.search(line)
        if match:
            curves.append(ast.literal_eval(match.group(1)))
            continue
        match = AVG_RE.search(line)
        if match:
            avgs.append(float(match.group(1)))
            continue
        match = CAL_RE.search(line)
        if match:
            calibrations.append(
                {
                    key: (value if key == "rule" else float(value))
                    for key, value in match.groupdict().items()
                }
            )

    phases = []
    for idx, grouped in enumerate(cnn_rows):
        phase = {
            "phase": idx,
            "total": float(grouped.get("total")),
            "old": float(grouped.get("old", 0)),
            "new": float(grouped.get("new", 0)),
            "new_eval_pred_old_rate": grouped.get("new_eval_pred_old_rate"),
            "old_eval_pred_new_rate": grouped.get("old_eval_pred_new_rate"),
            "h_m_f": grouped.get("h-m-f"),
            "grouped": grouped,
        }
        if idx > 0 and idx - 1 < len(calibrations):
            phase["calibration"] = calibrations[idx - 1]
        phases.append(phase)

    return {
        "path": str(path),
        "curve": curves[-1] if curves else [],
        "avg_acc": avgs[-1] if avgs else None,
        "phases": phases,
        "error_counts": {
            "Traceback": text.count("Traceback"),
            "RuntimeError": text.count("RuntimeError"),
            "CUDA OOM": text.count("CUDA out of memory"),
        },
    }


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def mean_std(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--method-log", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    baseline = json.loads(Path(args.baseline_json).read_text())
    method = parse_log(args.method_log)
    baseline_curve = baseline["curve"]
    method_curve = method["curve"]
    curve_rows = []
    for idx in range(max(len(baseline_curve), len(method_curve))):
        base = baseline_curve[idx] if idx < len(baseline_curve) else None
        meth = method_curve[idx] if idx < len(method_curve) else None
        curve_rows.append(
            [
                str(idx),
                fmt(base, 2),
                fmt(meth, 2),
                fmt(None if base is None or meth is None else meth - base, 2),
            ]
        )

    phase_rows = []
    alpha_values = []
    for phase in method["phases"]:
        calibration = phase.get("calibration", {})
        if "alpha" in calibration:
            alpha_values.append(calibration["alpha"])
        phase_rows.append(
            [
                str(phase["phase"]),
                fmt(phase["total"], 2),
                fmt(phase["old"], 2),
                fmt(phase["new"], 2),
                fmt(phase["new_eval_pred_old_rate"], 2),
                fmt(phase["old_eval_pred_new_rate"], 2),
                fmt(calibration.get("alpha"), 6),
                fmt(calibration.get("old_norm"), 6),
                fmt(calibration.get("new_norm"), 6),
                fmt(calibration.get("uncalibrated_total"), 2),
                fmt(calibration.get("uncalibrated_new_eval_pred_old_rate"), 2),
            ]
        )

    alpha_mean, alpha_std = mean_std(alpha_values)
    final_phase = method["phases"][-1] if method["phases"] else {}
    result = {
        "baseline": baseline,
        "method": method,
        "delta_avg_acc": None if method["avg_acc"] is None else method["avg_acc"] - baseline["avg_acc"],
        "delta_accT": None if not method_curve else method_curve[-1] - baseline_curve[-1],
        "alpha_mean": alpha_mean,
        "alpha_std": alpha_std,
        "alpha_min": min(alpha_values) if alpha_values else None,
        "alpha_max": max(alpha_values) if alpha_values else None,
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2))

    md = "# APART Stage1 Head-Norm Full Gate Summary\n\n"
    md += "## Curve Comparison\n\n"
    md += table(["phase", "baseline", "stage1_headnorm", "delta"], curve_rows)
    md += "\n## Method Per-Task Diagnostics\n\n"
    md += table(
        [
            "phase",
            "AccT",
            "old acc",
            "new/current acc",
            "new->old",
            "old->new",
            "alpha",
            "old norm",
            "new norm",
            "uncal AccT",
            "uncal new->old",
        ],
        phase_rows,
    )
    md += "\n## Summary\n\n"
    md += f"- Baseline Avg Acc: {fmt(baseline['avg_acc'])}\n"
    md += f"- Method Avg Acc: {fmt(method['avg_acc'])}\n"
    md += f"- Delta Avg Acc: {fmt(result['delta_avg_acc'])}\n"
    md += f"- Baseline AccT: {fmt(baseline_curve[-1], 2)}\n"
    md += f"- Method AccT: {fmt(method_curve[-1] if method_curve else None, 2)}\n"
    md += f"- Delta AccT: {fmt(result['delta_accT'], 2)}\n"
    md += f"- Final h/m/f: {final_phase.get('h_m_f')}\n"
    md += f"- Alpha mean/std/min/max: {fmt(alpha_mean, 6)} / {fmt(alpha_std, 6)} / {fmt(result['alpha_min'], 6)} / {fmt(result['alpha_max'], 6)}\n"
    md += "\n## Error Counts\n\n"
    md += "```json\n" + json.dumps(method["error_counts"], indent=2) + "\n```\n"

    Path(args.output_md).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
