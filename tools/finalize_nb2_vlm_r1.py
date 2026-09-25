"""Summarize the fixed NB2 matrix without changing any trained or fitted state."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

CONTRASTS = (("A6", "A2"), ("A2", "A0"), ("A6", "A0"),
             ("A6", "A5"), ("A6", "A4"), ("A6", "A3s"))
METHODS = ("P", "A0", "A1", "A2", "A3", "A3s", "A4", "A5", "A6", "A7", "A8")
TASKS = {"ISIC": 4, "HK": 11}


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("BLOCKED_EMPTY_FINALIZER_TABLE")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + ".part")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _load_predictions(out: Path, dataset: str) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int, str], np.ndarray]]:
    path = out / "private" / f"{dataset}_VAL_PREDICTIONS.npz"
    with np.load(path, allow_pickle=False) as data:
        labels = data["labels"].astype(int)
        components = data["components"].astype(str)
        pred = data["predictions"].astype(int)
        keys = [json.loads(str(item)) for item in data["index"]]
    if len(keys) != len(pred) or len(keys) != 3 * TASKS[dataset] * len(METHODS):
        raise ValueError("BLOCKED_PREDICTION_COVERAGE")
    result = {(int(key["seed"]), int(key["task"]), key["method"]): value
              for key, value in zip(keys, pred)}
    if len(result) != len(keys):
        raise ValueError("BLOCKED_PREDICTION_KEYS")
    return labels, components, result


def _bootstrap(out: Path) -> list[dict[str, Any]]:
    rows = []
    for dataset, final_task in TASKS.items():
        labels, components, pred = _load_predictions(out, dataset)
        rng = np.random.default_rng(57002)
        groups = {}
        for cid in np.unique(labels):
            groups[int(cid)] = [np.flatnonzero((labels == cid) & (components == comp))
                                for comp in np.unique(components[labels == cid])]
        draws = {cid: [rng.integers(0, len(items), size=len(items)) for _ in range(2000)]
                 for cid, items in groups.items()}
        for candidate, comparator in CONTRASTS:
            values = np.empty(2000, dtype=np.float64)
            per_seed = []
            for seed in (1993, 1994, 1995):
                left, right = pred[seed, final_task, candidate], pred[seed, final_task, comparator]
                mean = np.mean([np.mean(left[labels == cid] == cid) -
                                np.mean(right[labels == cid] == cid) for cid in groups])
                per_seed.append(100 * float(mean))
            for sample in range(2000):
                class_diffs = []
                for cid, items in groups.items():
                    selected = np.concatenate([items[i] for i in draws[cid][sample]])
                    if not len(selected):
                        raise ValueError("BLOCKED_BOOTSTRAP_DENOMINATOR")
                    for seed in (1993, 1994, 1995):
                        left = pred[seed, final_task, candidate]
                        right = pred[seed, final_task, comparator]
                        class_diffs.append(np.mean(left[selected] == cid) -
                                           np.mean(right[selected] == cid))
                values[sample] = 100 * np.mean(class_diffs)
            low, high = np.quantile(values, [.025, .975])
            rows.append({"dataset": dataset, "candidate": candidate, "comparator": comparator,
                         "final_ba_gain_pp": float(np.mean(per_seed)),
                         "three_order_sd_pp": float(np.std(per_seed, ddof=1)),
                         "conditional_low_pp": float(low), "conditional_high_pp": float(high),
                         "crosses_zero": bool(low <= 0 <= high), "bootstrap_replicates": 2000,
                         "bootstrap_seed": 57002, "unit": "class_stratified_identity_component",
                         "fixed_model_seeds": 3, "minimum_class_val_n":
                             min(int(np.sum(labels == cid)) for cid in groups)})
    return rows


def _gate(stage: dict[tuple[str, int, int, str], dict[str, str]],
          classes: dict[tuple[str, int, int, str, int], dict[str, str]],
          summaries: dict[tuple[str, int, str], dict[str, float]],
          dataset: str, candidate: str, comparator: str) -> dict[str, Any]:
    t = TASKS[dataset]
    paired = [(stage[dataset, seed, t, candidate], stage[dataset, seed, t, comparator])
              for seed in (1993, 1994, 1995)]
    final_gain = [100 * (_float(left, "ba") - _float(right, "ba")) for left, right in paired]
    inc_gain = [100 * (summaries[dataset, seed, candidate]["avg_ba_inc"] -
                       summaries[dataset, seed, comparator]["avg_ba_inc"])
                for seed in (1993, 1994, 1995)]
    tail_gain = [100 * (_float(left, "tail_ba") - _float(right, "tail_ba"))
                 for left, right in paired]
    old_loss = [100 * (_float(right, "old_ba") - _float(left, "old_ba"))
                for left, right in paired]
    current_loss = [100 * (_float(right, "current_ba") - _float(left, "current_ba"))
                    for left, right in paired]
    new_zero = []
    for (ds, seed, task, method, cid), row in classes.items():
        if ds != dataset or method != candidate:
            continue
        other = classes[ds, seed, task, comparator, cid]
        if _float(row, "recall") == 0 and _float(other, "recall") > 0:
            new_zero.append([seed, task, cid])
    criteria = {
        "final_ba_mean_gain_ge_1pp": np.mean(final_gain) >= 1.0,
        "at_least_two_orders_improve": sum(value > 0 for value in final_gain) >= 2,
        "avg_ba_inc_nonnegative": np.mean(inc_gain) >= 0,
        "final_tail_nonnegative": np.mean(tail_gain) >= 0,
        "mean_old_loss_le_2pp": np.mean(old_loss) <= 2,
        "mean_current_loss_le_2pp": np.mean(current_loss) <= 2,
        "each_old_loss_le_5pp": max(old_loss) <= 5,
        "each_current_loss_le_5pp": max(current_loss) <= 5,
        "no_new_zero_recall": not new_zero,
    }
    return {"dataset": dataset, "candidate": candidate, "comparator": comparator,
            "status": "PASS" if all(criteria.values()) else "FAIL",
            "criteria": {k: bool(v) for k, v in criteria.items()},
            "final_ba_gain_pp_by_seed": final_gain,
            "avg_ba_inc_gain_pp_by_seed": inc_gain,
            "final_tail_gain_pp_by_seed": tail_gain,
            "final_old_loss_pp_by_seed": old_loss,
            "final_current_loss_pp_by_seed": current_loss,
            "new_zero_recall": new_zero}


def finalize(out: Path) -> dict[str, Any]:
    stage_rows = _csv(out / "stage_metrics.csv")
    class_rows = _csv(out / "class_metrics.csv")
    if len(stage_rows) != 495 or len(class_rows) != 5049:
        raise ValueError("BLOCKED_FINALIZER_COVERAGE")
    stage = {(r["dataset"], int(r["seed"]), int(r["task"]), r["method"]): r for r in stage_rows}
    classes = {(r["dataset"], int(r["seed"]), int(r["task"]), r["method"], int(r["class_id"])): r
               for r in class_rows}
    if len(stage) != 495 or len(classes) != 5049:
        raise ValueError("BLOCKED_FINALIZER_KEYS")
    summaries = {}
    for dataset, t in TASKS.items():
        for seed in (1993, 1994, 1995):
            for method in METHODS:
                vals = [float(stage[dataset, seed, task, method]["ba"]) for task in range(1, t + 1)]
                summaries[dataset, seed, method] = {
                    "avg_ba_all": float(np.mean(vals)),
                    "avg_ba_inc": float(np.mean(vals[1:])),
                    "final_ba": vals[-1],
                    "final_macro_f1": float(stage[dataset, seed, t, method]["macro_f1"]),
                    "final_tail": float(stage[dataset, seed, t, method]["tail_ba"]),
                }
    _write_csv(out / "summary_by_seed.csv", [
        {"dataset": ds, "seed": seed, "method": method, **values}
        for (ds, seed, method), values in summaries.items()
    ])
    paired = []
    for dataset, t in TASKS.items():
        for seed in (1993, 1994, 1995):
            for task in range(1, t + 1):
                for candidate, comparator in CONTRASTS:
                    left, right = stage[dataset, seed, task, candidate], stage[dataset, seed, task, comparator]
                    paired.append({
                        "dataset": dataset, "seed": seed, "task": task,
                        "candidate": candidate, "comparator": comparator,
                        **{f"{key}_difference_pp": 100 * (_float(left, key) - _float(right, key))
                           for key in ("ba", "macro_f1", "accuracy", "tail_ba",
                                       "old_ba", "current_ba", "hm")},
                        "zero_recall_difference": int(left["zero_recall"]) - int(right["zero_recall"]),
                    })
    _write_csv(out / "paired_differences.csv", paired)
    forgetting = []
    for dataset, t in TASKS.items():
        for seed in (1993, 1994, 1995):
            for method in METHODS:
                by_class: dict[int, list[tuple[int, float]]] = defaultdict(list)
                for task in range(1, t + 1):
                    for (ds, sd, tk, md, cid), row in classes.items():
                        if (ds, sd, tk, md) == (dataset, seed, task, method):
                            by_class[cid].append((task, _float(row, "recall")))
                for cid, series in by_class.items():
                    first, final = series[0][1], series[-1][1]
                    forgetting.append({"dataset": dataset, "seed": seed, "method": method,
                                       "class_id": cid, "first_task": series[0][0],
                                       "first_recall": first, "final_recall": final,
                                       "signed_forgetting": first - final,
                                       "maximum_forgetting": max(v for _, v in series) - final})
    _write_csv(out / "forgetting.csv", forgetting)
    confusion = []
    for dataset, t in TASKS.items():
        labels, _, predictions = _load_predictions(out, dataset)
        for seed in (1993, 1994, 1995):
            for method in METHODS:
                pred = predictions[seed, t, method]
                for true in np.unique(labels):
                    for predicted in np.unique(labels):
                        confusion.append({"dataset": dataset, "seed": seed, "task": t,
                                          "method": method, "true_class": int(true),
                                          "predicted_class": int(predicted),
                                          "n": int(np.sum((labels == true) & (pred == predicted)))})
    _write_csv(out / "final_confusion.csv", confusion)
    bootstrap = _bootstrap(out)
    _write_csv(out / "bootstrap_intervals.csv", bootstrap)
    gates = [_gate(stage, classes, summaries, dataset, candidate, comparator)
             for dataset in TASKS for candidate, comparator in (("A6", "A2"), ("A6", "A0"), ("A2", "A0"))]
    overall = all(item["status"] == "PASS" for item in gates if item["candidate"] == "A6")
    result = {"status": "EXPERIMENT_COMPLETE", "utility_gate": "PASS" if overall else "FAIL",
              "gates": gates, "new_training_epochs": 0, "optimizer_steps": 0,
              "test_access": 0, "reserved_access": 0,
              "next_decision": "STOP", "independent_confirmation": False}
    _write(out / "NEXT_DECISION.json", {"decision": "STOP", "reason": "frozen_matrix_finished"})
    _write(out / "BACKUP_ACK.json", {"status": "INDEPENDENT_BACKUP_UNVERIFIED",
                                    "primary_storage": str(out), "large_files_other_storage": False})
    _write(out / "FINAL_REPORT.json", result)
    lines = ["# NB2-VLM-R1 最终报告", "",
             "状态：实验矩阵完成；效用门槛 " + result["utility_gate"] + "。下一步 STOP。", "",
             "本轮是既有开发数据上的受控分析，不是独立确认或临床验证。新增神经训练 epoch=0，optimizer step=0；test/reserved 访问=0。", "",
             "## 方法成绩", "",
             "| 数据集 | 方法 | Final BA均值 | Final Macro-F1均值 | AvgBA_inc均值 |",
             "|---|---|---:|---:|---:|"]
    for dataset in TASKS:
        for method in METHODS:
            values = [summaries[dataset, seed, method] for seed in (1993, 1994, 1995)]
            lines.append(f"| {dataset} | {method} | {100*np.mean([v['final_ba'] for v in values]):.3f} | "
                         f"{100*np.mean([v['final_macro_f1'] for v in values]):.3f} | "
                         f"{100*np.mean([v['avg_ba_inc'] for v in values]):.3f} |")
    lines += ["", "## 固定比较与门槛", ""]
    for item in gates:
        gain = np.mean(item["final_ba_gain_pp_by_seed"])
        lines.append(f"- {item['dataset']} {item['candidate']}−{item['comparator']}: "
                     f"Final BA {gain:+.3f} pp；门槛 {item['status']}。")
    lines += ["", "条件区间见 bootstrap_intervals.csv；原始逐阶段与逐类值见 CSV。"
              " 大型私有状态仅在 remote-home，独立第二存储备份未验证。", ""]
    (out / "FINAL_REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    return result
