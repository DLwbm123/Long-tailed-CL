#!/usr/bin/env python3
"""Summarize HyperKvasir23 locked NC-DSM auxiliary gate outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


METRICS = ["AccT", "old_acc", "current_acc", "HM_old_current", "old_to_current", "current_to_old"]
DISPLAY_COLUMNS = ["seed", "method", "lambda_dsm", *METRICS, "output_path"]
PAIR_SPECS = [
    ("locked_nc_dsm_aux lambda=0.1 vs Locked NC-ConCM", ("locked_nc_dsm_aux", 0.1), ("Locked NC-ConCM", 0.0)),
    ("locked_nc_dsm_aux lambda=0.2 vs Locked NC-ConCM", ("locked_nc_dsm_aux", 0.2), ("Locked NC-ConCM", 0.0)),
    ("Pure DSM-ConCM vs locked_nc_dsm_aux lambda=0.1", ("Pure DSM-ConCM", None), ("locked_nc_dsm_aux", 0.1)),
    ("Pure DSM-ConCM vs Locked NC-ConCM", ("Pure DSM-ConCM", None), ("Locked NC-ConCM", 0.0)),
]


def _load_payload(path: Path) -> Mapping[str, object]:
    if path.is_dir():
        path = path / "final_results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    return float(value)


def _hm(left, right) -> float | None:
    left_f = _as_float(left)
    right_f = _as_float(right)
    if left_f is None or right_f is None:
        return None
    if left_f + right_f <= 0.0:
        return 0.0
    return 2.0 * left_f * right_f / (left_f + right_f)


def _lambda_value(row: Mapping[str, object]) -> float | None:
    if row.get("lambda_dsm") is not None:
        return float(row["lambda_dsm"])
    if row.get("lambda_value") is not None and row.get("method") in {"locked_NCConCM", None}:
        return 0.0
    return None


def _canonical_method(row: Mapping[str, object]) -> str | None:
    method = row.get("method")
    group = str(row.get("group") or "")
    if method in {"locked_NCConCM_recomputed", "locked_NCConCM_reference_json", "locked_NCConCM"}:
        return "Locked NC-ConCM"
    if group in {"C_locked_NCConCM", "C_task_block_nc_concm"}:
        return "Locked NC-ConCM"
    if method in {"pure_DSM_ConCM", "pure_DSM_ConCM_reused", "pure_DSM_aux_projector"}:
        return "Pure DSM-ConCM"
    if method == "locked_nc_dsm_aux":
        return "locked_nc_dsm_aux"
    return None


def _row_priority(row: Mapping[str, object]) -> int:
    method = row.get("method")
    if method == "pure_DSM_ConCM":
        return 50
    if method == "locked_NCConCM_recomputed":
        return 50
    if method == "locked_nc_dsm_aux":
        return 40
    if method == "locked_NCConCM_reference_json":
        return 30
    if method == "pure_DSM_aux_projector":
        return 30
    return 10


def _normalize_row(row: Mapping[str, object], payload: Mapping[str, object], source: str, source_priority: int) -> Dict[str, object] | None:
    method = _canonical_method(row)
    if method is None:
        return None
    seed = row.get("seed", payload.get("seed"))
    if seed is None:
        seed = payload.get("seed")
    lambda_dsm = _lambda_value(row)
    old_to_current = row.get("old_to_current", row.get("old_to_current_rate"))
    current_to_old = row.get("current_to_old", row.get("current_to_old_rate"))
    normalized = {
        "seed": int(seed) if seed is not None else None,
        "method": method,
        "lambda_dsm": lambda_dsm,
        "AccT": _as_float(row.get("AccT")),
        "old_acc": _as_float(row.get("old_acc")),
        "current_acc": _as_float(row.get("current_acc")),
        "HM_old_current": _as_float(row.get("HM_old_current"), _hm(row.get("old_acc"), row.get("current_acc"))),
        "old_to_current": _as_float(old_to_current),
        "current_to_old": _as_float(current_to_old),
        "output_path": source,
        "_priority": int(source_priority) * 100 + _row_priority(row),
    }
    return normalized


def _collect_rows(paths: Sequence[Path]) -> List[Dict[str, object]]:
    dedup: Dict[Tuple[int | None, str, float | None], Dict[str, object]] = {}
    for source_priority, path in enumerate(paths):
        payload = _load_payload(path)
        source = str(path if path.is_file() else path / "final_results.json")
        for row in payload.get("summary_rows", []):
            normalized = _normalize_row(row, payload, source, source_priority)
            if normalized is None:
                continue
            key = (normalized["seed"], normalized["method"], normalized["lambda_dsm"])
            current = dedup.get(key)
            if current is None or int(normalized["_priority"]) > int(current["_priority"]):
                dedup[key] = normalized
    rows = list(dedup.values())
    for row in rows:
        row.pop("_priority", None)
    return sorted(rows, key=lambda r: (int(r["seed"] or -1), str(r["method"]), -1.0 if r["lambda_dsm"] is None else float(r["lambda_dsm"])))


def _mean_std(values: Sequence[float]) -> tuple[float | None, float | None]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return None, None
    mean = statistics.fmean(clean)
    std = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return mean, std


def _mean_std_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, float | None], List[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), row.get("lambda_dsm")), []).append(row)
    out = []
    for (method, lambda_dsm), items in sorted(groups.items(), key=lambda x: (x[0][0], -1 if x[0][1] is None else float(x[0][1]))):
        result: Dict[str, object] = {"method": method, "lambda_dsm": lambda_dsm, "n": len(items)}
        for metric in METRICS:
            mean, std = _mean_std([item.get(metric) for item in items])
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std
            result[metric] = _format_mean_std(mean, std)
        out.append(result)
    return out


def _index(rows: Sequence[Mapping[str, object]]) -> Dict[Tuple[int, str, float | None], Mapping[str, object]]:
    return {(int(row["seed"]), str(row["method"]), row.get("lambda_dsm")): row for row in rows if row.get("seed") is not None}


def _delta_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    by_key = _index(rows)
    seeds = sorted({int(row["seed"]) for row in rows if row.get("seed") is not None})
    out: List[Dict[str, object]] = []
    for seed in seeds:
        for label, left_key, right_key in PAIR_SPECS:
            left = by_key.get((seed, left_key[0], left_key[1]))
            right = by_key.get((seed, right_key[0], right_key[1]))
            if left is None or right is None:
                continue
            row: Dict[str, object] = {"seed": seed, "comparison": label}
            for metric in METRICS:
                row[f"delta_{metric}"] = float(left[metric]) - float(right[metric])
            out.append(row)
    return out


def _delta_mean_std_rows(delta_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[str, List[Mapping[str, object]]] = {}
    for row in delta_rows:
        groups.setdefault(str(row["comparison"]), []).append(row)
    out = []
    for comparison, items in groups.items():
        result: Dict[str, object] = {"comparison": comparison, "n": len(items)}
        for metric in METRICS:
            key = f"delta_{metric}"
            mean, std = _mean_std([item.get(key) for item in items])
            result[f"{key}_mean"] = mean
            result[f"{key}_std"] = std
            result[key] = _format_mean_std(mean, std)
        out.append(result)
    return out


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


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_mean_std(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return ""
    return f"{mean:.4f}+/-{std:.4f}"


def _markdown_table(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def _write_markdown(path: Path, per_seed: Sequence[Mapping[str, object]], mean_std: Sequence[Mapping[str, object]], deltas: Sequence[Mapping[str, object]], delta_mean_std: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mean_columns = ["method", "lambda_dsm", "n", *METRICS]
    delta_columns = ["seed", "comparison", *[f"delta_{metric}" for metric in METRICS]]
    delta_mean_columns = ["comparison", "n", *[f"delta_{metric}" for metric in METRICS]]
    content = "\n\n".join(
        [
            "# HyperKvasir23 Locked NC-DSM Auxiliary Multi-Seed Summary",
            "## Per-Seed Table\n\n" + _markdown_table(per_seed, DISPLAY_COLUMNS),
            "## Mean/Std Table\n\n" + _markdown_table(mean_std, mean_columns),
            "## Delta Table\n\n" + _markdown_table(deltas, delta_columns),
            "## Delta Mean/Std Table\n\n" + _markdown_table(delta_mean_std, delta_mean_columns),
        ]
    )
    path.write_text(content + "\n", encoding="utf-8")


def _print_table(rows: Sequence[Mapping[str, object]], title: str, columns: Sequence[str]) -> None:
    print(f"\n## {title}")
    print(_markdown_table(rows, columns))


def _default_output_dir(paths: Sequence[Path]) -> Path:
    first = paths[0]
    if first.is_dir():
        return first
    return first.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="final_results.json files or run directories.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None, help="Backward-compatible alias for sorted per-seed CSV.")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    out_dir = args.output_dir or _default_output_dir(args.paths)
    per_seed = _collect_rows(args.paths)
    mean_std = _mean_std_rows(per_seed)
    deltas = _delta_rows(per_seed)
    delta_mean_std = _delta_mean_std_rows(deltas)

    per_seed_csv = out_dir / "locked_nc_dsm_aux_per_seed.csv"
    mean_std_csv = out_dir / "locked_nc_dsm_aux_mean_std.csv"
    delta_csv = out_dir / "locked_nc_dsm_aux_deltas.csv"
    delta_mean_std_csv = out_dir / "locked_nc_dsm_aux_delta_mean_std.csv"
    markdown = out_dir / "locked_nc_dsm_aux_multiseed_summary.md"
    _write_csv(per_seed_csv, per_seed)
    _write_csv(mean_std_csv, mean_std)
    _write_csv(delta_csv, deltas)
    _write_csv(delta_mean_std_csv, delta_mean_std)
    _write_markdown(markdown, per_seed, mean_std, deltas, delta_mean_std)
    if args.output_csv is not None:
        by_acct = sorted(per_seed, key=lambda row: _as_float(row.get("AccT"), -1.0), reverse=True)
        _write_csv(args.output_csv, by_acct)

    _print_table(per_seed, "Per-Seed Table", DISPLAY_COLUMNS)
    _print_table(mean_std, "Mean/Std Table", ["method", "lambda_dsm", "n", *METRICS])
    _print_table(deltas, "Delta Table", ["seed", "comparison", *[f"delta_{metric}" for metric in METRICS]])
    _print_table(delta_mean_std, "Delta Mean/Std Table", ["comparison", "n", *[f"delta_{metric}" for metric in METRICS]])
    print(f"\nWrote {per_seed_csv}")
    print(f"Wrote {mean_std_csv}")
    print(f"Wrote {delta_csv}")
    print(f"Wrote {delta_mean_std_csv}")
    print(f"Wrote {markdown}")


if __name__ == "__main__":
    main()
