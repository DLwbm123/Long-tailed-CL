#!/usr/bin/env python3
"""Freeze a test-free HyperKvasir official-5-fold validation protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.medical_lt import medical_eval_transform


TARGETS = (
    {"seed": 2, "session": 5, "role": "stable_control", "output": "seed2/session5_control"},
    {"seed": 1, "session": 2, "role": "collapse_target", "output": "seed1/session2"},
    {"seed": 3, "session": 5, "role": "collapse_target", "output": "seed3/session5"},
    {"seed": 1, "session": 5, "role": "collapse_target", "output": "seed1/session5"},
)
UPSTREAM_COMMIT = "21cc366e78c0cb4e180a26a0e441d6c0d5171da9"
UPSTREAM_REPOSITORY = "https://github.com/simula/hyper-kvasir"
COLLAPSE_SOURCE_SHA256 = "84195d9b3246d13f256d79f064d2d653f90f5bcba9e145ea90a277c509c22fa7"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    text = json.dumps(list(rows), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or [])
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def read_official(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    required = {"file-name", "class-name", "split-index"}
    if not required.issubset(columns):
        raise RuntimeError(f"Official CSV columns are invalid: {columns}")
    by_name: dict[str, dict[str, str]] = {}
    duplicates = []
    for row in rows:
        basename = Path(row["file-name"]).name
        if basename in by_name:
            duplicates.append(basename)
        by_name[basename] = row
    if duplicates:
        raise RuntimeError(f"Official CSV has duplicate basenames: {duplicates[:10]}")
    return rows, columns, by_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--official-csv", required=True)
    args = parser.parse_args()
    existing = Path(args.existing_root).resolve()
    output = Path(args.output_root).resolve()
    official_path = Path(args.official_csv).resolve()
    if not official_path.is_file():
        raise FileNotFoundError(official_path)

    official_rows, official_columns, official_by_name = read_official(official_path)
    official_names = set(official_by_name)
    caches = {
        seed: torch.load(existing / "cache" / f"seed{seed}_feature_bank.pt", map_location="cpu", weights_only=False)
        for seed in (1, 2, 3)
    }
    project_by_name: dict[str, list[dict[str, object]]] = {}
    alignment_rows = []
    unmatched_project = []
    mismatch_rows = []
    for seed, cache in caches.items():
        for membership in ("train", "test"):
            for class_id, paths in cache[f"{membership}_paths"].items():
                for raw_path in paths:
                    path = Path(raw_path)
                    basename = path.name
                    local_class = path.parent.name
                    project_row = {
                        "seed": seed,
                        "membership": membership,
                        "class_id": int(class_id),
                        "class_name": local_class,
                        "path": str(path),
                        "basename": basename,
                    }
                    project_by_name.setdefault(basename, []).append(project_row)
                    official = official_by_name.get(basename)
                    if official is None:
                        unmatched_project.append(project_row)
                        continue
                    class_match = local_class == official["class-name"]
                    aligned = {
                        **project_row,
                        "official_class_name": official["class-name"],
                        "official_split_index": int(official["split-index"]),
                        "class_name_exact_match": class_match,
                    }
                    alignment_rows.append(aligned)
                    if not class_match:
                        mismatch_rows.append(aligned)

    unmatched_official = [
        {"basename": name, **official_by_name[name]}
        for name in sorted(official_names - set(project_by_name))
    ]
    all_project_names = set(project_by_name)
    if unmatched_project or unmatched_official:
        alignment_status = "PARTIAL"
    else:
        alignment_status = "COMPLETE"

    coverage_rows = []
    fold_ok: dict[int, bool] = {}
    target_details: dict[tuple[int, int], dict[str, object]] = {}
    for target in TARGETS:
        seed, session = int(target["seed"]), int(target["session"])
        cache = caches[seed]
        current_classes = [int(value) for value in cache["tasks"][session]]
        target_details[(seed, session)] = {**target, "current_classes": current_classes}

    for fold in range(5):
        valid_fold = True
        for (seed, session), target in target_details.items():
            cache = caches[seed]
            test_names = {Path(path).name for paths in cache["test_paths"].values() for path in paths}
            for class_id in target["current_classes"]:
                train_paths = [Path(path) for path in cache["train_paths"][class_id]]
                validation = [
                    path for path in train_paths
                    if path.name in official_by_name and int(official_by_name[path.name]["split-index"]) == fold
                ]
                fit = [path for path in train_paths if path not in set(validation)]
                unique = len({path.name for path in validation}) == len(validation)
                test_overlap = len({path.name for path in validation} & test_names)
                class_ok = len(validation) >= 1 and len(fit) >= 2 and unique and test_overlap == 0
                valid_fold = valid_fold and class_ok
                coverage_rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "session": session,
                        "role": target["role"],
                        "class_id": class_id,
                        "class_name": train_paths[0].parent.name,
                        "project_train_count": len(train_paths),
                        "validation_count": len(validation),
                        "fit_count": len(fit),
                        "validation_test_overlap": test_overlap,
                        "validation_unique": unique,
                        "class_coverage_ok": class_ok,
                    }
                )
        fold_ok[fold] = valid_fold

    eligible_folds = [fold for fold in range(5) if fold_ok[fold]]
    selected_fold = min(eligible_folds) if eligible_folds else None
    split_root = output / "splits" / "hyperkvasir23_official5fold_val_v2"
    write_csv(split_root / "filename_alignment.csv", alignment_rows)
    write_csv(split_root / "unmatched_project_images.csv", unmatched_project,
              ("seed", "membership", "class_id", "class_name", "path", "basename"))
    write_csv(split_root / "unmatched_official_images.csv", unmatched_official,
              ("basename", "file-name", "class-name", "split-index"))
    write_csv(split_root / "class_name_mismatch.csv", mismatch_rows)
    write_csv(split_root / "split_coverage.csv", coverage_rows)

    source = {
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_path": "official_splits/5_fold_split.csv",
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "local_path": str(official_path),
        "sha256": file_sha256(official_path),
        "delimiter": ";",
        "original_columns": official_columns,
        "row_count": len(official_rows),
        "official_basename_unique": True,
    }
    write_json(output / "official_split_source.json", source)
    write_json(
        output / "selected_official_fold.json",
        {
            "status": "SELECTED" if selected_fold is not None else "OFFICIAL_FOLD_COVERAGE_UNAVAILABLE",
            "selected_fold": selected_fold,
            "eligible_folds": eligible_folds,
            "fold_coverage": {str(key): value for key, value in fold_ok.items()},
            "selection_rule": "smallest fold satisfying all target current-class count and disjointness constraints",
            "performance_metrics_accessed": False,
        },
    )
    if selected_fold is None:
        write_csv(output / "coverage_by_run_and_class.csv", coverage_rows)
        raise SystemExit("OFFICIAL_FOLD_COVERAGE_UNAVAILABLE")

    validation_rows = []
    fit_rows = []
    selected_coverage = [row for row in coverage_rows if int(row["fold"]) == selected_fold]
    for (seed, session), target in target_details.items():
        cache = caches[seed]
        test_names = {Path(path).name for paths in cache["test_paths"].values() for path in paths}
        for class_id in target["current_classes"]:
            train_paths = [Path(path) for path in cache["train_paths"][class_id]]
            validation = [
                path for path in train_paths
                if int(official_by_name[path.name]["split-index"]) == selected_fold
            ]
            val_names = {path.name for path in validation}
            if val_names & test_names:
                raise RuntimeError("Validation/test overlap detected")
            for path in validation:
                validation_rows.append(
                    {"seed": seed, "session": session, "role": target["role"], "class_id": class_id,
                     "class_name": path.parent.name, "basename": path.name, "path": str(path),
                     "official_fold": selected_fold, "project_membership": "train"}
                )
            for path in train_paths:
                if path.name not in val_names:
                    fit_rows.append(
                        {"seed": seed, "session": session, "role": target["role"], "class_id": class_id,
                         "class_name": path.parent.name, "basename": path.name, "path": str(path),
                         "official_fold": int(official_by_name[path.name]["split-index"]),
                         "project_membership": "train"}
                    )

    transform = medical_eval_transform(image_size=224, normalization="imagenet")
    dry_run = []
    for target in TARGETS:
        rows = [row for row in validation_rows if row["seed"] == target["seed"] and row["session"] == target["session"]]
        sample = rows[0]
        with Image.open(sample["path"]) as image:
            tensor = transform(image.convert("RGB"))
        if tuple(tensor.shape) != (3, 224, 224) or not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"Validation loader dry-run failed for {target}")
        dry_run.append({"seed": target["seed"], "session": target["session"], "sample": sample["path"],
                        "shape": list(tensor.shape), "finite": True})

    write_csv(output / "validation_manifest.csv", validation_rows)
    write_csv(output / "fit_manifest.csv", fit_rows)
    write_csv(output / "coverage_by_run_and_class.csv", selected_coverage)
    manifest_hashes = {
        "official_csv_sha256": file_sha256(official_path),
        "validation_manifest_content_sha256": rows_sha256(validation_rows),
        "fit_manifest_content_sha256": rows_sha256(fit_rows),
        "coverage_content_sha256": rows_sha256(selected_coverage),
    }
    write_json(output / "manifest_sha256.json", manifest_hashes)
    protocol = {
        "status": "FROZEN_READY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_official_fold": selected_fold,
        "fold_selection_used_model_metrics": False,
        "alignment_status": alignment_status,
        "unmatched_project_count": len(unmatched_project),
        "unmatched_official_count": len(unmatched_official),
        "class_name_mismatch_count": len(mismatch_rows),
        "validation_source_membership": "project_train_only",
        "validation_test_overlap": 0,
        "validation_scope": "current_session_classes_only",
        "fit_excludes_validation": True,
        "test_used_by_selector": False,
        "targets": list(target_details.values()),
        "collapse_predicate": {
            "source_file": "code/tools/run_hyperkvasir_full_dynamic_bounded_rescue.py",
            "function_name": "is_collapse",
            "current_accuracy_min": 50.0,
            "current_to_old_max": 30.0,
            "source_sha256": COLLAPSE_SOURCE_SHA256,
        },
        "eligibility": ["all metrics finite", "full-state restore passes", "current-side predicate false"],
        "ranking_key": ["val_current_macro_accuracy", "-val_current_to_old", "val_current_micro_accuracy",
                        "-val_max_old_absorption_ratio", "p10_current_vs_old_margin",
                        "old_prototype_self_accuracy", "-epoch"],
        "epochs": [0, 1, 2, 3, 4, 5],
        "gpu": {"physical": 0, "cuda_visible_devices": "0", "shared_mode": True, "initial_min_free_mib": 12000},
        "loader_dry_run": dry_run,
        "manifest_hashes": manifest_hashes,
        "raw_prototype_rescue_allowed": False,
    }
    write_json(output / "protocol_frozen_v2.json", protocol)
    print(json.dumps({"status": "FROZEN_READY", "selected_fold": selected_fold,
                      "validation_rows": len(validation_rows), "fit_rows": len(fit_rows),
                      "eligible_folds": eligible_folds}, indent=2))


if __name__ == "__main__":
    main()
