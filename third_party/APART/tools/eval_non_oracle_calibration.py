import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _accuracy(preds, labels):
    return round(float((preds == labels).sum() * 100.0 / len(labels)), 2)


def _metrics_for_alpha(logits, labels, known_classes, previous_top1, alpha):
    calibrated = logits.copy()
    calibrated[:, :known_classes] *= alpha
    preds = calibrated.argmax(axis=1)

    old_mask = labels < known_classes
    new_mask = labels >= known_classes
    total_acc = _accuracy(preds, labels)
    old_acc = _accuracy(preds[old_mask], labels[old_mask]) if old_mask.any() else 0.0
    new_acc = _accuracy(preds[new_mask], labels[new_mask]) if new_mask.any() else 0.0
    new_eval_pred_old_rate = (
        round(float((preds[new_mask] < known_classes).sum() * 100.0 / new_mask.sum()), 2)
        if new_mask.any()
        else 0.0
    )
    old_eval_pred_new_rate = (
        round(float((preds[old_mask] >= known_classes).sum() * 100.0 / old_mask.sum()), 2)
        if old_mask.any()
        else 0.0
    )
    avg_acc = round(float((previous_top1 + total_acc) / 2.0), 3)

    return {
        "alpha": float(alpha),
        "avg_acc": avg_acc,
        "task1_accT": total_acc,
        "old_acc": old_acc,
        "new_acc": new_acc,
        "new_eval_pred_old_rate": new_eval_pred_old_rate,
        "old_eval_pred_new_rate": old_eval_pred_new_rate,
    }


def _find_head_weight(state_dict, suffix):
    matches = [(key, value) for key, value in state_dict.items() if key.endswith(suffix)]
    matches = [(key, value) for key, value in matches if value.ndim == 2]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one state_dict key ending with {suffix}, got {[key for key, _ in matches]}")
    return matches[0]


def _head_norm_alphas(checkpoint_path, known_classes, total_classes):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    head_key, head_weight = _find_head_weight(state_dict, "head.weight")
    head_few_key, head_few_weight = _find_head_weight(state_dict, "head_few.weight")

    head_weight = head_weight[:total_classes].float()
    head_few_weight = head_few_weight[:total_classes].float()
    effective_weight = head_weight + head_few_weight

    def ratio(weight):
        old_norm = weight[:known_classes].norm(dim=1).mean().item()
        new_norm = weight[known_classes:total_classes].norm(dim=1).mean().item()
        return {
            "old_mean_norm": old_norm,
            "new_mean_norm": new_norm,
            "alpha": new_norm / old_norm,
        }

    return {
        "head_key": head_key,
        "head_few_key": head_few_key,
        "head": ratio(head_weight),
        "head_few": ratio(head_few_weight),
        "effective_sum": ratio(effective_weight),
    }


def _markdown_table(rows):
    header = (
        "| rule | selected alpha | Avg Acc | task1 AccT | old acc | new acc | "
        "new_eval_pred_old_rate | old_eval_pred_new_rate | decision |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---|\n"
    )
    body = []
    for row in rows:
        body.append(
            "| {rule} | {alpha:.6f} | {avg_acc:.3f} | {task1_accT:.2f} | {old_acc:.2f} | {new_acc:.2f} | {new_eval_pred_old_rate:.2f} | {old_eval_pred_new_rate:.2f} | {decision} |".format(
                **row
            )
        )
    return header + "\n".join(body) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Evaluate non-oracle old-logit calibration rules from saved APART logits.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--logits-npz", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--fixed-alpha", type=float, nargs="*", default=[0.8, 0.75, 0.7])
    parser.add_argument("--baseline-avg-acc", type=float, default=89.560)
    parser.add_argument("--baseline-acct", type=float, default=88.70)
    parser.add_argument("--baseline-old-acc", type=float, default=89.40)
    parser.add_argument("--baseline-new-acc", type=float, default=85.20)
    args = parser.parse_args()

    metrics_json = Path(args.metrics_json)
    with metrics_json.open() as f:
        previous_metrics = json.load(f)

    logits_data = np.load(args.logits_npz)
    logits = logits_data["logits"].astype(np.float32)
    labels = logits_data["labels"].astype(np.int64)
    known_classes = int(previous_metrics["known_classes"])
    total_classes = int(previous_metrics["total_classes"])
    previous_top1 = float(previous_metrics["previous_eval_top1"])

    norm_info = _head_norm_alphas(args.checkpoint, known_classes, total_classes)
    head_norm_alpha = float(norm_info["effective_sum"]["alpha"])

    rules = [("head_norm_effective_sum", head_norm_alpha)]
    for alpha in args.fixed_alpha:
        rules.append((f"fixed_{alpha:g}", float(alpha)))

    rows = []
    for rule, alpha in rules:
        metrics = _metrics_for_alpha(logits, labels, known_classes, previous_top1, alpha)
        promising = (
            metrics["task1_accT"] > args.baseline_acct
            and metrics["avg_acc"] > args.baseline_avg_acc
            and metrics["new_acc"] >= args.baseline_new_acc
            and metrics["old_acc"] >= args.baseline_old_acc
        )
        metrics["rule"] = rule
        metrics["decision"] = "promising" if promising else "inconclusive_or_negative"
        metrics["close_to_oracle_good_range_0.7_0.8"] = 0.7 <= alpha <= 0.8
        rows.append(metrics)

    result = {
        "checkpoint": args.checkpoint,
        "logits_npz": args.logits_npz,
        "source_metrics_json": args.metrics_json,
        "known_classes": known_classes,
        "total_classes": total_classes,
        "previous_top1": previous_top1,
        "baseline_reference": {
            "avg_acc": args.baseline_avg_acc,
            "task1_accT": args.baseline_acct,
            "old_acc": args.baseline_old_acc,
            "new_acc": args.baseline_new_acc,
        },
        "head_norm_info": norm_info,
        "train_val_calibration": {
            "status": "not_run",
            "reason": "The APART Stage1 run is exemplar-free (memory_size=0) and no separate saved validation split exists at task1; using old-class train data or test data to select alpha would violate the requested non-oracle gate.",
        },
        "rows": rows,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(result, f, indent=2)

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with output_md.open("w") as f:
        f.write("# Non-Oracle Calibration Evaluation\n\n")
        f.write(_markdown_table(rows))
        f.write("\n## Head Norm Info\n\n")
        f.write("```json\n")
        f.write(json.dumps(norm_info, indent=2))
        f.write("\n```\n")
        f.write("\n## Train/Val Calibration\n\n")
        f.write(result["train_val_calibration"]["reason"] + "\n")

    print(_markdown_table(rows))
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")


if __name__ == "__main__":
    main()
