#!/usr/bin/env python3
"""Inventory existing HyperKvasir23 DSM artifacts without modifying them."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path: Path) -> dict:
    item = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        stat = path.stat()
        item.update(
            {
                "bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "sha256": sha256(path),
            }
        )
    return item


def checkpoint_metadata(path: Path) -> dict | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "keys": sorted(payload),
        "old_classes": payload.get("old_classes"),
        "current_classes": payload.get("current_classes"),
        "seen_classes": payload.get("seen_classes"),
        "has_model_state": "model_state" in payload,
        "has_geometry": "geometry_vectors" in payload,
        "has_prototype_repository": "prototype_repository" in payload,
        "has_covariance": any("cov" in key.lower() for key in payload),
    }


def json_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "keys": sorted(payload),
        "seed": payload.get("seed"),
        "method": payload.get("method"),
        "split": payload.get("split"),
        "artifact_paths": payload.get("artifact_paths"),
        "base_projector_epochs": payload.get("base_projector_epochs"),
        "increment_projector_epochs": payload.get("increment_projector_epochs"),
        "sample_num_old": payload.get("sample_num_old"),
        "sample_num_current": payload.get("sample_num_current"),
        "cont_weight": payload.get("cont_weight"),
    }


def projector_metadata(path: Path) -> dict | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    geometry = payload.get("geometry_vectors")
    return {
        "keys": sorted(payload),
        "projector_config": payload.get("projector_config"),
        "seen_classes": payload.get("seen_classes"),
        "class_to_head": payload.get("class_to_head"),
        "geometry_shape": list(geometry.shape) if isinstance(geometry, torch.Tensor) else None,
        "has_geometry": isinstance(geometry, torch.Tensor),
        "has_prototype_repository": "prototype_repository" in payload,
        "has_covariance": any("cov" in key.lower() for key in payload),
        "has_optimizer": any("optim" in key.lower() for key in payload),
    }


def seed_paths(base: Path, seed: int) -> dict[str, Path]:
    runs = base / "runs"
    checkpoints = base / "checkpoints" / f"hyperkvasir23_med_fdm_gate_fair_20260709_s{seed}_5ep"
    train = runs / f"hyperkvasir23_med_fdm_gate_fair_20260709_s{seed}_5ep"
    aux = runs / f"hyperkvasir23_locked_nc_dsm_aux_20260709_s{seed}"
    pure = runs / f"hyperkvasir23_pure_dsm_concm_gate_20260709_s{seed}"
    return {
        "phase0_checkpoint": checkpoints / "phase0_with_anchors.pt",
        "phase1_checkpoint": checkpoints / "freeze_baseline_phase1.pt",
        "exemplar_manifest": train / "old_exemplars.csv",
        "training_summary": train / "med_fdm_gate_summary.json",
        "calibration_results": runs / f"hyperkvasir23_balanced_task_block_calibration_20260709_s{seed}/final_results.json",
        "diagnostics_results": runs / f"hyperkvasir23_task_block_diag_20260709_s{seed}/diagnostics.json",
        "locked_results": runs / f"hyperkvasir23_ncconcm_rule_lock_20260709_s{seed}/final_results.json",
        "aux_config": aux / "dsm_aux_config.json",
        "aux_results": aux / "final_results.json",
        "aux_projector": aux / "projector_dsm_aux.pt",
        "aux_geometry_stats": aux / "geometry_stats_dsm_aux.json",
        "aux_train_trace": aux / "train_trace_dsm_aux.csv",
        "pure_config": pure / "dsm_config.json",
        "pure_results": pure / "final_results.json",
        "pure_projector": pure / "projector.pt",
        "pure_geometry_stats": pure / "geometry_stats.json",
        "pure_train_trace": pure / "train_trace.csv",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="1,2,3")
    args = parser.parse_args()
    base = Path(args.base_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing inventory: {output}")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    inventory = {
        "base_root": str(base),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_contract": {
            "dataset": "hyper_kvasir23",
            "base_classes": 13,
            "incremental_steps": 5,
            "full_task_count": 6,
            "bounded_max_phases": 2,
            "available_incremental_sessions": [1],
            "missing_incremental_sessions": [2, 3, 4, 5],
        },
        "seeds": {},
    }
    for seed in seeds:
        paths = seed_paths(base, seed)
        seed_item = {"files": {name: describe(path) for name, path in paths.items()}}
        seed_item["phase0_metadata"] = checkpoint_metadata(paths["phase0_checkpoint"])
        seed_item["phase1_metadata"] = checkpoint_metadata(paths["phase1_checkpoint"])
        seed_item["aux_results_summary"] = json_summary(paths["aux_results"])
        seed_item["pure_results_summary"] = json_summary(paths["pure_results"])
        seed_item["aux_projector_metadata"] = projector_metadata(paths["aux_projector"])
        seed_item["pure_projector_metadata"] = projector_metadata(paths["pure_projector"])
        inventory["seeds"][str(seed)] = seed_item
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output), "seeds": seeds}, indent=2))


if __name__ == "__main__":
    main()
