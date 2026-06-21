"""Small file IO helpers for reproducible run artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_yaml(path: str | Path, payload: Any) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(json_safe(payload), handle, sort_keys=True)


def append_jsonl(path: str | Path, payload: Any) -> None:
    path = Path(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(payload), sort_keys=True) + "\n")


def write_acc_matrix_csv(path: str | Path, matrix: Sequence[Sequence[float | None]]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        width = len(matrix[0]) if matrix else 0
        writer.writerow(["phase"] + [f"task_{idx}" for idx in range(width)])
        for phase, row in enumerate(matrix):
            writer.writerow([phase] + ["" if value is None else f"{value:.6f}" for value in row])


def write_summary_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    keys = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(json_safe(row))
