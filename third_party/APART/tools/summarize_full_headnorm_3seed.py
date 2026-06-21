import argparse
import json
import math
from pathlib import Path

from summarize_full_headnorm_gate import parse_log


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def mean_std(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(var)


def final_hmf(parsed):
    if not parsed["phases"]:
        return [None, None, None]
    hmf = parsed["phases"][-1].get("h_m_f")
    return list(hmf) if hmf else [None, None, None]


def per_seed_summary(seed, baseline_log, method_log):
    baseline = parse_log(baseline_log)
    method = parse_log(method_log)
    baseline_curve = baseline["curve"]
    method_curve = method["curve"]
    baseline_hmf = final_hmf(baseline)
    method_hmf = final_hmf(method)
    alpha_values = [
        phase.get("calibration", {}).get("alpha")
        for phase in method["phases"]
        if phase.get("calibration", {}).get("alpha") is not None
    ]
    alpha_mean, alpha_std = mean_std(alpha_values)
    return {
        "seed": int(seed),
        "baseline_log": str(baseline_log),
        "method_log": str(method_log),
        "baseline": {
            "curve": baseline_curve,
            "avg_acc": baseline["avg_acc"],
            "accT": baseline_curve[-1] if baseline_curve else None,
            "h_m_f": baseline_hmf,
            "phases": baseline["phases"],
            "error_counts": baseline["error_counts"],
        },
        "method": {
            "curve": method_curve,
            "avg_acc": method["avg_acc"],
            "accT": method_curve[-1] if method_curve else None,
            "h_m_f": method_hmf,
            "phases": method["phases"],
            "error_counts": method["error_counts"],
            "alpha_values": alpha_values,
            "alpha_mean": alpha_mean,
            "alpha_std": alpha_std,
            "alpha_min": min(alpha_values) if alpha_values else None,
            "alpha_max": max(alpha_values) if alpha_values else None,
        },
        "delta": {
            "avg_acc": None if baseline["avg_acc"] is None or method["avg_acc"] is None else method["avg_acc"] - baseline["avg_acc"],
            "accT": None if not baseline_curve or not method_curve else method_curve[-1] - baseline_curve[-1],
            "many": None if baseline_hmf[0] is None or method_hmf[0] is None else method_hmf[0] - baseline_hmf[0],
            "medium": None if baseline_hmf[1] is None or method_hmf[1] is None else method_hmf[1] - baseline_hmf[1],
            "few": None if baseline_hmf[2] is None or method_hmf[2] is None else method_hmf[2] - baseline_hmf[2],
        },
    }


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def build_report(results):
    rows = []
    for result in results:
        rows.append(
            [
                str(result["seed"]),
                fmt(result["baseline"]["avg_acc"]),
                fmt(result["method"]["avg_acc"]),
                fmt(result["delta"]["avg_acc"]),
                fmt(result["baseline"]["accT"], 2),
                fmt(result["method"]["accT"], 2),
                fmt(result["delta"]["accT"], 2),
                fmt(result["baseline"]["h_m_f"][2], 2),
                fmt(result["method"]["h_m_f"][2], 2),
                fmt(result["delta"]["few"], 2),
                fmt(result["method"]["alpha_mean"], 6),
                fmt(result["method"]["alpha_min"], 6),
                fmt(result["method"]["alpha_max"], 6),
            ]
        )

    metrics = ["avg_acc", "accT"]
    means = {}
    for side in ["baseline", "method"]:
        for metric in metrics:
            mean, std = mean_std([result[side][metric] for result in results])
            means[f"{side}_{metric}"] = (mean, std)
        for idx, metric in enumerate(["many", "medium", "few"]):
            mean, std = mean_std([result[side]["h_m_f"][idx] for result in results])
            means[f"{side}_{metric}"] = (mean, std)
    delta_mean_std = {
        key: mean_std([result["delta"][key] for result in results])
        for key in ["avg_acc", "accT", "many", "medium", "few"]
    }

    mean_rows = [
        [
            "Avg Acc",
            f"{fmt(means['baseline_avg_acc'][0])} +- {fmt(means['baseline_avg_acc'][1])}",
            f"{fmt(means['method_avg_acc'][0])} +- {fmt(means['method_avg_acc'][1])}",
            f"{fmt(delta_mean_std['avg_acc'][0])} +- {fmt(delta_mean_std['avg_acc'][1])}",
        ],
        [
            "AccT",
            f"{fmt(means['baseline_accT'][0], 2)} +- {fmt(means['baseline_accT'][1], 2)}",
            f"{fmt(means['method_accT'][0], 2)} +- {fmt(means['method_accT'][1], 2)}",
            f"{fmt(delta_mean_std['accT'][0], 2)} +- {fmt(delta_mean_std['accT'][1], 2)}",
        ],
        [
            "few",
            f"{fmt(means['baseline_few'][0], 2)} +- {fmt(means['baseline_few'][1], 2)}",
            f"{fmt(means['method_few'][0], 2)} +- {fmt(means['method_few'][1], 2)}",
            f"{fmt(delta_mean_std['few'][0], 2)} +- {fmt(delta_mean_std['few'][1], 2)}",
        ],
    ]
    for metric in ["many", "medium"]:
        mean_rows.append(
            [
                metric,
                f"{fmt(means[f'baseline_{metric}'][0], 2)} +- {fmt(means[f'baseline_{metric}'][1], 2)}",
                f"{fmt(means[f'method_{metric}'][0], 2)} +- {fmt(means[f'method_{metric}'][1], 2)}",
                f"{fmt(delta_mean_std[metric][0], 2)} +- {fmt(delta_mean_std[metric][1], 2)}",
            ]
        )

    curve_rows = []
    max_phase = max(len(result["baseline"]["curve"]) for result in results)
    for phase in range(max_phase):
        row = [str(phase)]
        for result in results:
            base = result["baseline"]["curve"][phase] if phase < len(result["baseline"]["curve"]) else None
            meth = result["method"]["curve"][phase] if phase < len(result["method"]["curve"]) else None
            delta = None if base is None or meth is None else meth - base
            row.append(f"{fmt(base, 2)}->{fmt(meth, 2)} ({fmt(delta, 2)})")
        curve_rows.append(row)

    alpha_rows = []
    bias_rows = []
    for result in results:
        for run_name in ["baseline", "method"]:
            for phase in result[run_name]["phases"]:
                bias_rows.append(
                    [
                        str(result["seed"]),
                        run_name,
                        str(phase["phase"]),
                        fmt(phase.get("total"), 2),
                        fmt(phase.get("old"), 2),
                        fmt(phase.get("new"), 2),
                        fmt(phase.get("new_eval_pred_old_rate"), 2),
                        fmt(phase.get("old_eval_pred_new_rate"), 2),
                    ]
                )
        for phase in result["method"]["phases"]:
            calibration = phase.get("calibration", {})
            if not calibration:
                continue
            alpha_rows.append(
                [
                    str(result["seed"]),
                    str(phase["phase"]),
                    fmt(calibration.get("alpha"), 6),
                    fmt(calibration.get("old_norm"), 6),
                    fmt(calibration.get("new_norm"), 6),
                    fmt(phase.get("new_eval_pred_old_rate"), 2),
                    fmt(phase.get("old_eval_pred_new_rate"), 2),
                    fmt(calibration.get("uncalibrated_new_eval_pred_old_rate"), 2),
                ]
            )

    report = "# APART Stage1 Head-Norm 3-Seed Full Summary\n\n"
    report += "## Per-Seed Summary\n\n"
    report += table(
        [
            "seed",
            "base Avg",
            "method Avg",
            "delta Avg",
            "base AccT",
            "method AccT",
            "delta AccT",
            "base few",
            "method few",
            "delta few",
            "alpha mean",
            "alpha min",
            "alpha max",
        ],
        rows,
    )
    report += "\n## Mean +/- Std\n\n"
    report += table(["metric", "baseline", "method", "delta"], mean_rows)
    report += "\n## Curve Deltas\n\n"
    report += table(["phase"] + [f"seed{result['seed']}" for result in results], curve_rows)
    report += "\n## Per-Task Old/New Diagnostics\n\n"
    report += table(
        ["seed", "run", "phase", "AccT", "old acc", "new acc", "new->old", "old->new"],
        bias_rows,
    )
    report += "\n## Alpha And Bias Diagnostics\n\n"
    report += table(
        ["seed", "phase", "alpha", "old norm", "new norm", "new->old", "old->new", "uncal new->old"],
        alpha_rows,
    )
    report += "\n## Error Counts\n\n"
    report += "```json\n"
    report += json.dumps(
        {
            result["seed"]: {
                "baseline": result["baseline"]["error_counts"],
                "method": result["method"]["error_counts"],
            }
            for result in results
        },
        indent=2,
    )
    report += "\n```\n"
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("SEED", "BASELINE_LOG", "METHOD_LOG"),
        required=True,
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    results = [per_seed_summary(seed, baseline_log, method_log) for seed, baseline_log, method_log in args.pair]
    result = {"seeds": results}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2))
    report = build_report(results)
    Path(args.output_md).write_text(report)
    print(report)


if __name__ == "__main__":
    main()
