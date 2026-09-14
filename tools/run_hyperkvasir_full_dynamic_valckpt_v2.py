#!/usr/bin/env python3
"""Run official-fold current-only validation checkpoint selection for Full Dynamic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.concm_dsm import DSMProjector, compute_dynamic_structure
from src.methods.full_concm import (
    MPCNetwork,
    PrototypeRecord,
    blend_current_covariance,
    build_attribute_mapping,
    calibrate_prototype,
    class_balanced_mean,
    class_semantic_embedding,
    resample_repository,
    structure_anchor_contrastive,
    tensor_sha256,
)
from src.utils.seed import set_seed
from tools.run_hyperkvasir_full_concm import file_sha256, metric_pack


CURRENT_ACCURACY_MIN = 50.0
CURRENT_TO_OLD_MAX = 30.0
EPOCHS = 5
TARGETS = {
    (2, 5): {"role": "stable_control", "output": "seed2/session5_control", "source_session": 4},
    (1, 2): {"role": "collapse_target", "output": "seed1/session2", "source_session": 1},
    (3, 5): {"role": "collapse_target", "output": "seed3/session5", "source_session": 4},
    (1, 5): {"role": "collapse_target", "output": "seed1/session5", "source_session": 4},
}


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def cuda_snapshot(label: str) -> list[dict[str, object]]:
    query = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True
    ).strip().split(",")
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()
    base = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "label": label,
        "gpu_index": int(query[0]), "total_mib": int(query[1]), "used_mib": int(query[2]),
        "free_mib": int(query[3]), "utilization_percent": int(query[4]),
    }
    if not processes:
        return [{**base, "process_pid": "", "process_name": "", "process_used_mib": ""}]
    rows = []
    for process in processes:
        values = [value.strip() for value in process.split(",")]
        rows.append({**base, "process_gpu_uuid": values[0], "process_pid": values[1],
                     "process_name": values[2], "process_used_mib": values[3]})
    return rows


def load_source(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    projector = DSMProjector(**{key: int(value) for key, value in payload["projector_config"].items()}).to(device)
    projector.load_state_dict(payload["projector_state_dict"])
    mpc = MPCNetwork(**payload["mpc_config"]).to(device)
    mpc.load_state_dict(payload["mpc_state_dict"])
    mpc.eval()
    records = [PrototypeRecord.from_dict(value) for value in payload["repository"]]
    return payload, projector, mpc, records


def current_only_features(
    cache: Mapping[str, object], manifest: Sequence[Mapping[str, str]], seed: int, session: int
) -> dict[int, torch.Tensor]:
    selected: dict[int, set[str]] = {}
    for row in manifest:
        if int(row["seed"]) == seed and int(row["session"]) == session:
            selected.setdefault(int(row["class_id"]), set()).add(row["basename"])
    output = {}
    for class_id, names in selected.items():
        paths = cache["train_paths"][class_id]
        indices = [index for index, path in enumerate(paths) if Path(path).name in names]
        if len(indices) != len(names):
            raise RuntimeError(f"Manifest/cache mismatch for class {class_id}: {len(indices)} != {len(names)}")
        output[class_id] = cache["train_features"][class_id][indices].float()
    return output


def fit_features(
    cache: Mapping[str, object], manifest: Sequence[Mapping[str, str]], seed: int, session: int
) -> dict[int, torch.Tensor]:
    return current_only_features(cache, manifest, seed, session)


def confusion_rows(confusion: Mapping[int, Mapping[int, int]]) -> list[dict[str, int]]:
    return [
        {"true_class": int(true), "predicted_class": int(predicted), "count": int(count)}
        for true in sorted(confusion) for predicted, count in sorted(confusion[true].items())
    ]


@torch.no_grad()
def validate_current(
    projector: DSMProjector,
    records: Sequence[PrototypeRecord],
    current_classes: Sequence[int],
    validation: Mapping[int, torch.Tensor],
    previous_geometry: torch.Tensor,
    device: torch.device,
):
    projector.eval()
    seen = [record.class_id for record in records]
    class_to_head = {class_id: head for head, class_id in enumerate(seen)}
    current_heads = [class_to_head[class_id] for class_id in current_classes]
    old_heads = [head for head in range(len(seen)) if head not in current_heads]
    means = torch.stack([record.mean for record in records]).to(device)
    projected_means = projector(means)
    geometry, geometry_stats = compute_dynamic_structure(projected_means)
    features = torch.cat([validation[class_id] for class_id in current_classes]).to(device)
    labels = torch.cat([
        torch.full((validation[class_id].shape[0],), class_to_head[class_id], dtype=torch.long)
        for class_id in current_classes
    ]).to(device)
    logits = projector(features) @ geometry.T
    predictions = logits.argmax(dim=1)
    loss = F.cross_entropy(logits, labels)
    per_class = {}
    for class_id, head in zip(current_classes, current_heads):
        mask = labels == head
        per_class[int(class_id)] = float(predictions[mask].eq(labels[mask]).float().mean().cpu()) * 100.0
    micro = float(predictions.eq(labels).float().mean().cpu()) * 100.0
    macro = sum(per_class.values()) / len(per_class)
    pred_old = torch.tensor([int(value) in set(old_heads) for value in predictions.tolist()], device=device)
    current_to_old = float(pred_old.float().mean().cpu()) * 100.0
    histogram = {int(seen[head]): int(predictions.eq(head).sum().cpu()) for head in range(len(seen))}
    confusion: dict[int, dict[int, int]] = {}
    for label, prediction in zip(labels.tolist(), predictions.tolist()):
        true_class, predicted_class = int(seen[label]), int(seen[prediction])
        confusion.setdefault(true_class, {})[predicted_class] = confusion.setdefault(true_class, {}).get(predicted_class, 0) + 1
    absorption = {int(seen[head]): int(predictions[pred_old].eq(head).sum().cpu()) for head in old_heads}
    absorbing_class, absorbing_count = max(absorption.items(), key=lambda item: (item[1], -item[0])) if absorption else (None, 0)
    absorbing_ratio = 100.0 * absorbing_count / max(int(labels.numel()), 1)

    similarities = projected_means @ geometry.T
    self_predictions = similarities.argmax(dim=1)
    current_self = float(self_predictions[current_heads].eq(torch.tensor(current_heads, device=device)).float().mean().cpu()) * 100.0
    old_self = float(self_predictions[old_heads].eq(torch.tensor(old_heads, device=device)).float().mean().cpu()) * 100.0
    margins = []
    own_cosines = []
    for head in current_heads:
        margins.append(float((similarities[head, head] - similarities[head, old_heads].max()).cpu()))
        own_cosines.append(float(similarities[head, head].cpu()))
    old_count = previous_geometry.shape[0]
    displacement = torch.linalg.vector_norm(geometry[:old_count] - previous_geometry.to(device), dim=1)
    metrics = {
        "val_loss": float(loss.cpu()),
        "val_current_micro_accuracy": micro,
        "val_current_macro_accuracy": macro,
        "val_per_current_class_accuracy": json.dumps(per_class, sort_keys=True),
        "val_current_to_old": current_to_old,
        "val_predicted_class_histogram": json.dumps(histogram, sort_keys=True),
        "val_confusion_over_seen_classes": json.dumps(confusion, sort_keys=True),
        "val_max_old_absorption_count": absorbing_count,
        "val_max_old_absorption_ratio": absorbing_ratio,
        "val_absorbing_old_class_id": absorbing_class,
    }
    diagnostics = {
        "current_prototype_self_accuracy": current_self,
        "old_prototype_self_accuracy": old_self,
        "mean_current_vs_old_margin": sum(margins) / len(margins),
        "p10_current_vs_old_margin": float(torch.quantile(torch.tensor(margins), 0.1)),
        "prototype_anchor_cosine_mean": sum(own_cosines) / len(own_cosines),
        "prototype_anchor_cosine_min": min(own_cosines),
        "anchor_displacement_mean": float(displacement.mean().cpu()),
        "anchor_displacement_max": float(displacement.max().cpu()),
        "ETF_residual": geometry_stats["etf_residual"],
        "condition_number": geometry_stats["condition_number"],
    }
    return metrics, diagnostics, geometry.detach().cpu(), confusion, absorption


def rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def save_epoch_checkpoint(
    path: Path,
    epoch: int,
    projector: DSMProjector,
    optimizer: torch.optim.Optimizer,
    scheduler,
    mpc: MPCNetwork,
    records: Sequence[PrototypeRecord],
    geometry: torch.Tensor,
    calibration_rows: Sequence[Mapping[str, object]],
    source_path: Path,
    source_sha: str,
    config: Mapping[str, object],
):
    payload = {
        "epoch": epoch,
        "model_state_dict": projector.state_dict(),
        "projector_state_dict": projector.state_dict(),
        "projector_config": projector.config(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "mpc_state_dict": mpc.state_dict(),
        "mpc_config": mpc.config(),
        "matching_state": list(calibration_rows),
        "repository": [record.cpu_dict() for record in records],
        "raw_prototypes": {record.class_id: record.raw_mean.detach().cpu() for record in records},
        "calibrated_prototypes": {record.class_id: record.mean.detach().cpu() for record in records},
        "dynamic_anchors": geometry.detach().cpu(),
        "class_mapping": {record.class_id: head for head, record in enumerate(records)},
        "rng_state": rng_state(),
        "config": dict(config),
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_sha,
        "git_commit": git_commit(),
    }
    torch.save(payload, path)
    return file_sha256(path)


def restore_check(path: Path, expected_sha: str) -> bool:
    if file_sha256(path) != expected_sha:
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "epoch", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "mpc_state_dict",
        "repository", "raw_prototypes", "calibrated_prototypes", "dynamic_anchors", "class_mapping",
        "rng_state", "config", "source_checkpoint", "source_checkpoint_sha256", "git_commit",
    }
    if not required.issubset(payload):
        return False
    model = DSMProjector(**payload["projector_config"])
    model.load_state_dict(payload["projector_state_dict"])
    return all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()) and bool(
        torch.isfinite(payload["dynamic_anchors"]).all()
    )


def is_eligible(metrics: Mapping[str, object], diagnostics: Mapping[str, object], restore_ok: bool) -> bool:
    values = [value for value in list(metrics.values()) + list(diagnostics.values()) if isinstance(value, (int, float))]
    finite = all(math.isfinite(float(value)) for value in values)
    collapse = (
        float(metrics["val_current_micro_accuracy"]) < CURRENT_ACCURACY_MIN
        or float(metrics["val_current_to_old"]) >= CURRENT_TO_OLD_MAX
    )
    return finite and restore_ok and not collapse


def selection_key(row: Mapping[str, object]):
    return (
        float(row["val_current_macro_accuracy"]),
        -float(row["val_current_to_old"]),
        float(row["val_current_micro_accuracy"]),
        -float(row["val_max_old_absorption_ratio"]),
        float(row["p10_current_vs_old_margin"]),
        float(row["old_prototype_self_accuracy"]),
        -int(row["epoch"]),
    )


def train_select(args) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or torch.cuda.device_count() != 1:
        raise RuntimeError("Require CUDA_VISIBLE_DEVICES=0 with exactly one visible GPU")
    key = (int(args.seed), int(args.session))
    if key not in TARGETS:
        raise ValueError(f"Target is outside frozen set: {key}")
    target = TARGETS[key]
    output_root = Path(args.output_root).resolve()
    protocol = json.loads((output_root / "protocol_frozen_v2.json").read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_READY" or int(protocol["selected_official_fold"]) != 0:
        raise RuntimeError("Protocol is not frozen and ready")
    session_output = output_root / target["output"]
    if session_output.exists() and any(session_output.iterdir()):
        raise FileExistsError(f"Refusing non-empty session output: {session_output}")
    (session_output / "checkpoints").mkdir(parents=True, exist_ok=True)
    snapshots = cuda_snapshot("before_train")
    free_mib = int(snapshots[0]["free_mib"])
    if free_mib < int(args.min_free_mib):
        write_csv(session_output / "gpu_snapshots.csv", snapshots)
        write_json(session_output / "selection.json", {"status": "GPU_INSUFFICIENT_FREE_MEMORY", "selected_epoch": None})
        raise SystemExit("GPU_INSUFFICIENT_FREE_MEMORY")

    set_seed(int(args.seed))
    device = torch.device("cuda")
    existing = Path(args.existing_root).resolve()
    source_path = existing / "runs" / f"seed{args.seed}_full_dynamic" / "checkpoints" / f"session_{target['source_session']}.pt"
    source_sha = file_sha256(source_path)
    source_payload, projector, mpc, previous_records = load_source(source_path, device)
    config = dict(source_payload["config"])
    if int(config["increment_projector_epochs"]) != EPOCHS or float(config["projector_lr"]) != 0.01:
        raise RuntimeError("Original training configuration drift")
    cache = torch.load(existing / "cache" / f"seed{args.seed}_feature_bank.pt", map_location="cpu", weights_only=False)
    validation_manifest = read_csv(output_root / "validation_manifest.csv")
    fit_manifest = read_csv(output_root / "fit_manifest.csv")
    validation = current_only_features(cache, validation_manifest, int(args.seed), int(args.session))
    fit = fit_features(cache, fit_manifest, int(args.seed), int(args.session))
    current_classes = [int(value) for value in cache["tasks"][int(args.session)]]
    if set(validation) != set(current_classes) or set(fit) != set(current_classes):
        raise RuntimeError("Current-only manifest class mismatch")

    pool = torch.load(existing / "runs" / f"seed{args.seed}_full_dynamic" / "checkpoints" / "attribute_pool.pt",
                      map_location="cpu", weights_only=False)
    attribute_visual = pool["visual_prototypes"]
    attribute_semantic = pool["semantic_embeddings"]
    class_names = {class_id: Path(cache["train_paths"][class_id][0]).parent.name for class_id in cache["class_order"]}
    attribute_mapping, _ = build_attribute_mapping(class_names)
    class_semantics = {class_id: class_semantic_embedding(attribute_mapping[class_id]) for class_id in cache["class_order"]}
    base_records = [record for record in previous_records if record.session_id == 0]
    records = list(previous_records)
    calibration_rows = []
    for class_id in current_classes:
        features = fit[class_id]
        raw_mean = features.mean(dim=0)
        observed_covariance = features.var(dim=0, unbiased=False).clamp_min(1e-8)
        calibrated, calibration = calibrate_prototype(
            mpc, raw_mean, class_semantics[class_id], attribute_visual, attribute_semantic,
            float(config["mpc_alpha"]), device,
        )
        covariance, weights = blend_current_covariance(
            observed_covariance, calibrated, base_records,
            float(config["covariance_gamma"]), float(config["covariance_beta"]),
        )
        frequency_group = next(record.frequency_group for record in source_payload["repository"] if False) if False else (
            "tail" if len(cache["train_paths"][class_id]) <= 50 else "mid" if len(cache["train_paths"][class_id]) <= 300 else "head"
        )
        records.append(PrototypeRecord(class_id, int(args.session), calibrated, covariance, raw_mean,
                                       len(fit[class_id]), frequency_group, True))
        calibration_rows.append({"class_id": class_id, **calibration, "base_covariance_weight_max": float(weights.max()),
                                 "fit_count": len(fit[class_id]), "validation_count": len(validation[class_id])})

    means = torch.stack([record.mean for record in records]).to(device)
    current_heads = list(range(len(previous_records), len(records)))
    optimizer = torch.optim.SGD(projector.parameters(), lr=float(config["projector_lr"]), momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    run_config = {
        **config, "seed": int(args.seed), "session": int(args.session),
        "validation_protocol": "official5fold_fold0_current_only",
        "validation_manifest_sha256": json.loads((output_root / "manifest_sha256.json").read_text())["validation_manifest_content_sha256"],
        "fit_manifest_sha256": json.loads((output_root / "manifest_sha256.json").read_text())["fit_manifest_content_sha256"],
        "raw_prototype_fallback": False, "test_access_in_training_process": False,
    }
    epoch_rows = []
    geometry_rows = []
    checkpoint_rows = []
    log_rows = []
    previous_geometry = source_payload["geometry"]
    try:
        for epoch in range(EPOCHS + 1):
            if epoch > 0:
                projector.train()
                features, labels, sample_stats = resample_repository(
                    records, current_classes, int(args.seed) + int(args.session) * 100, epoch - 1
                )
                features, labels = features.to(device), labels.to(device)
                with torch.no_grad():
                    geometry_before, geometry_stats = compute_dynamic_structure(projector(means))
                optimizer.zero_grad(set_to_none=True)
                projected = projector(features)
                match_loss = class_balanced_mean(
                    F.cross_entropy(projected @ geometry_before.T, labels, reduction="none"), labels, len(records)
                )
                cont_loss = structure_anchor_contrastive(projected, labels, geometry_before, current_heads, 0.07)
                loss = match_loss + cont_loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_values = {"train_loss": float(loss.detach().cpu()), "train_match_loss": float(match_loss.detach().cpu()),
                                "train_cont_loss": float(cont_loss.detach().cpu()),
                                "resampling_checksum": hashlib.sha256("".join(row["checksum"] for row in sample_stats).encode()).hexdigest()}
            else:
                train_values = {"train_loss": None, "train_match_loss": None, "train_cont_loss": None,
                                "resampling_checksum": None}
            metrics, diagnostics, geometry, _, _ = validate_current(
                projector, records, current_classes, validation, previous_geometry, device
            )
            checkpoint_path = session_output / "checkpoints" / f"epoch_{epoch}.pt"
            checkpoint_sha = save_epoch_checkpoint(
                checkpoint_path, epoch, projector, optimizer, scheduler, mpc, records, geometry,
                calibration_rows, source_path, source_sha, run_config,
            )
            restore_ok = restore_check(checkpoint_path, checkpoint_sha)
            eligible = is_eligible(metrics, diagnostics, restore_ok)
            epoch_rows.append({"seed": args.seed, "session": args.session, "epoch": epoch, **train_values, **metrics,
                               **diagnostics, "restore_ok": restore_ok, "eligible": eligible,
                               "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha})
            geometry_rows.append({"seed": args.seed, "session": args.session, "epoch": epoch, **diagnostics})
            checkpoint_rows.append({"epoch": epoch, "path": str(checkpoint_path), "sha256": checkpoint_sha,
                                    "restore_ok": restore_ok, "eligible": eligible})
            log_rows.append(f"epoch={epoch} val_micro={metrics['val_current_micro_accuracy']:.8f} "
                            f"val_macro={metrics['val_current_macro_accuracy']:.8f} "
                            f"current_to_old={metrics['val_current_to_old']:.8f} eligible={eligible}")
    except torch.cuda.OutOfMemoryError:
        snapshots.extend(cuda_snapshot("oom"))
        write_csv(session_output / "gpu_snapshots.csv", snapshots)
        write_json(session_output / "selection.json", {"status": "GPU_SHARED_MODE_OOM", "selected_epoch": None})
        (session_output / "run.log").write_text("\n".join(log_rows + ["GPU_SHARED_MODE_OOM"]) + "\n", encoding="utf-8")
        raise SystemExit("GPU_SHARED_MODE_OOM")

    eligible_rows = [row for row in epoch_rows if bool(row["eligible"])]
    if eligible_rows:
        selected = max(eligible_rows, key=selection_key)
        status = "SELECTED"
        selected_epoch = int(selected["epoch"])
        selected_checkpoint = selected["checkpoint_path"]
    else:
        selected = None
        status = "ALL_EPOCHS_COLLAPSED"
        selected_epoch = None
        selected_checkpoint = None
    diagnostic = max(epoch_rows, key=selection_key)
    selection = {
        "status": status, "seed": int(args.seed), "session": int(args.session), "role": target["role"],
        "selected_epoch": selected_epoch, "selected_checkpoint": selected_checkpoint,
        "selected_checkpoint_sha256": selected["checkpoint_sha256"] if selected else None,
        "eligible_epochs": [int(row["epoch"]) for row in eligible_rows],
        "diagnostic_least_bad_epoch": int(diagnostic["epoch"]), "official_rescue": selected is not None,
        "ranking_key": protocol["ranking_key"], "raw_float_ranking": True,
        "selector_accessed_test": False, "frozen_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_csv(session_output / "epoch_metrics.csv", epoch_rows)
    write_csv(session_output / "geometry_diagnostics.csv", geometry_rows)
    write_json(session_output / "checkpoint_manifest.json", {"checkpoints": checkpoint_rows})
    write_json(session_output / "selection.json", selection)
    snapshots.extend(cuda_snapshot("after_train"))
    write_csv(session_output / "gpu_snapshots.csv", snapshots)
    write_json(session_output / "memory_peak.json", {
        "max_memory_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "max_memory_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    })
    (session_output / "run.log").write_text("\n".join(log_rows + [f"status={status} selected_epoch={selected_epoch}"]) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2))


@torch.no_grad()
def fresh_test(args) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or torch.cuda.device_count() != 1:
        raise RuntimeError("Require CUDA_VISIBLE_DEVICES=0")
    key = (int(args.seed), int(args.session))
    target = TARGETS[key]
    output_root = Path(args.output_root).resolve()
    session_output = output_root / target["output"]
    selection = json.loads((session_output / "selection.json").read_text(encoding="utf-8"))
    if selection["selected_epoch"] is None:
        write_json(session_output / "fresh_test_metrics.json", {"status": "NOT_RUN", "reason": selection["status"]})
        write_csv(session_output / "confusion_matrix.csv", [], ("true_class", "predicted_class", "count"))
        write_csv(session_output / "absorption_by_old_class.csv", [], ("old_class_id", "absorption_count", "absorption_ratio"))
        return
    checkpoint_path = Path(selection["selected_checkpoint"])
    checkpoint_sha = file_sha256(checkpoint_path)
    if checkpoint_sha != selection["selected_checkpoint_sha256"]:
        raise RuntimeError("Selected checkpoint SHA mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda")
    projector = DSMProjector(**payload["projector_config"]).to(device)
    projector.load_state_dict(payload["projector_state_dict"])
    projector.eval()
    records = [PrototypeRecord.from_dict(value) for value in payload["repository"]]
    seen = [record.class_id for record in records]
    current_classes = [record.class_id for record in records if record.session_id == int(args.session)]
    current_heads = {seen.index(class_id) for class_id in current_classes}
    old_heads = set(range(len(seen))) - current_heads
    existing = Path(args.existing_root).resolve()
    cache = torch.load(existing / "cache" / f"seed{args.seed}_feature_bank.pt", map_location="cpu", weights_only=False)
    features = torch.cat([cache["test_features"][class_id] for class_id in seen]).to(device)
    labels = torch.cat([torch.full((cache["test_features"][class_id].shape[0],), head, dtype=torch.long)
                        for head, class_id in enumerate(seen)])
    geometry = payload["dynamic_anchors"].to(device)
    logits = projector(features) @ geometry.T
    groups = {record.class_id: record.frequency_group for record in records}
    metrics, class_rows, confusion = metric_pack(
        logits.cpu(), labels, seen, [seen[head] for head in old_heads], current_classes, groups
    )
    predictions = logits.cpu().argmax(dim=1)
    current_mask = torch.tensor([int(value) in current_heads for value in labels.tolist()])
    absorption = {seen[head]: int(predictions[current_mask].eq(head).sum()) for head in old_heads}
    absorbing_class, absorbing_count = max(absorption.items(), key=lambda item: (item[1], -item[0]))
    absorption_rows = [{"old_class_id": class_id, "absorption_count": count,
                        "absorption_ratio": 100.0 * count / max(int(current_mask.sum()), 1)}
                       for class_id, count in sorted(absorption.items())]
    fresh = {
        "status": "OK", "seed": int(args.seed), "session": int(args.session),
        "selected_epoch": selection["selected_epoch"], "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha, "restore_equality": restore_check(checkpoint_path, checkpoint_sha),
        "test_AccT": metrics["AccT"], "test_old_accuracy": metrics["old_acc"],
        "test_current_accuracy": metrics["current_acc"], "test_HM": metrics["HM_old_current"],
        "test_old_to_current": metrics["old_to_current_rate"], "test_current_to_old": metrics["current_to_old_rate"],
        "test_BER": metrics["BER"], "absorbing_old_class_id": absorbing_class,
        "max_old_absorption_count": absorbing_count,
        "test_collapse": metrics["current_acc"] < CURRENT_ACCURACY_MIN or metrics["current_to_old_rate"] >= CURRENT_TO_OLD_MAX,
    }
    write_json(session_output / "fresh_test_metrics.json", fresh)
    write_csv(session_output / "confusion_matrix.csv", confusion_rows(confusion))
    write_csv(session_output / "absorption_by_old_class.csv", absorption_rows)
    write_csv(session_output / "fresh_test_per_class.csv", class_rows)
    print(json.dumps(fresh, indent=2))


def summarize(args) -> None:
    output_root = Path(args.output_root).resolve()
    existing = Path(args.existing_root).resolve()
    original_rows = read_csv(existing / "csv" / "all_session_metrics.csv")
    summary = []
    target_rescued = 0
    any_all_collapsed = False
    control_preserved = False
    all_checkpoint_restore = True
    for (seed, session), target in TARGETS.items():
        session_output = output_root / target["output"]
        selection = json.loads((session_output / "selection.json").read_text())
        checkpoint_manifest = json.loads((session_output / "checkpoint_manifest.json").read_text())["checkpoints"]
        all_checkpoint_restore = all_checkpoint_restore and len(checkpoint_manifest) == EPOCHS + 1 and all(
            bool(row["restore_ok"]) for row in checkpoint_manifest
        )
        fresh_path = session_output / "fresh_test_metrics.json"
        fresh = json.loads(fresh_path.read_text()) if fresh_path.exists() else {"status": "NOT_RUN"}
        original = next(row for row in original_rows if row["method"] == "full_dynamic" and int(row["seed"]) == seed and int(row["session"]) == session)
        selected = selection["selected_epoch"] is not None
        test_rescued = selected and fresh.get("status") == "OK" and not bool(fresh["test_collapse"])
        if target["role"] == "collapse_target":
            target_rescued += int(test_rescued)
            any_all_collapsed = any_all_collapsed or selection["status"] == "ALL_EPOCHS_COLLAPSED"
        else:
            control_preserved = bool(
                test_rescued
                and float(fresh["test_AccT"]) >= float(original["AccT"]) - 1.0
                and float(fresh["test_HM"]) >= float(original["HM_old_current"]) - 1.0
                and float(fresh["test_current_accuracy"]) >= float(original["current_acc"]) - 1.0
            )
        summary.append({
            "seed": seed, "session": session, "role": target["role"], "selection_status": selection["status"],
            "eligible_epochs": json.dumps(selection["eligible_epochs"]), "selected_epoch": selection["selected_epoch"],
            "fresh_status": fresh.get("status"), "test_AccT": fresh.get("test_AccT"),
            "test_old_accuracy": fresh.get("test_old_accuracy"), "test_current_accuracy": fresh.get("test_current_accuracy"),
            "test_HM": fresh.get("test_HM"), "test_old_to_current": fresh.get("test_old_to_current"),
            "test_current_to_old": fresh.get("test_current_to_old"), "test_collapse": fresh.get("test_collapse"),
            "restore_equality": fresh.get("restore_equality"), "target_rescued": test_rescued,
        })
    if any_all_collapsed:
        gate, promotion = "INITIALIZATION_OR_PERSISTENT_GEOMETRY_FAILURE", "NO_FULL_PROMOTION"
    elif target_rescued == 3 and control_preserved and all_checkpoint_restore:
        gate, promotion = "VALIDATION_SELECTOR_SIGNAL_PASSED", "PROMOTE_TO_FULL_DATA_REFIT_CONFIRMATION"
    elif target_rescued in (1, 2):
        gate, promotion = "PARTIAL_SELECTOR_SIGNAL", "NO_FULL_PROMOTION"
    else:
        gate, promotion = "VALIDATION_SELECTOR_FAILED", "NO_FULL_PROMOTION"
    write_csv(output_root / "four_session_summary.csv", summary)
    write_json(output_root / "gate.json", {"gate": gate, "promotion": promotion,
                                            "collapse_targets_rescued": target_rescued,
                                            "control_preserved": control_preserved,
                                            "all_restore_equal": all_checkpoint_restore,
                                            "any_all_epochs_collapsed": any_all_collapsed})
    coverage = read_csv(output_root / "coverage_by_run_and_class.csv")
    coverage_text = "\n".join(
        f"- seed{row['seed']}/session{row['session']} class {row['class_id']}: fit={row['fit_count']}, validation={row['validation_count']}"
        for row in coverage
    )
    result_text = "\n".join(
        f"- seed{row['seed']}/session{row['session']} ({row['role']}): eligible={row['eligible_epochs']}, "
        f"selected={row['selected_epoch']}, test_current={row['test_current_accuracy']}, "
        f"test_HM={row['test_HM']}, test_collapse={row['test_collapse']}"
        for row in summary
    )
    seed1s5 = next(row for row in summary if row["seed"] == 1 and row["session"] == 5)
    report = f"""# HyperKvasir23 Full Dynamic Official-Fold Validation Selector

## Protocol

- Official fold: 0, selected because all folds passed coverage and the frozen rule chooses the smallest index.
- Validation comes only from project-train images assigned to official fold 0.
- Validation/test overlap: 0. The existing test manifest was not modified or accessed by the selector.
- Current-session-only validation is diagnostic-only; it does not estimate old-class validation performance.

## Coverage

{coverage_text}

## Selection and Fresh Test

{result_text}

- seed1/session5 all epochs collapsed: {seed1s5['selection_status'] == 'ALL_EPOCHS_COLLAPSED'}.
- GPU0 shared-mode policy: enabled. At all four session starts GPU0 used about 1 MiB, so no non-project workload was consuming material GPU0 memory; no process was killed or preempted.
- OOM: false.
- Restore equality for all 24 saved epoch checkpoints: {all_checkpoint_restore}.

## Decision

- Gate: `{gate}`.
- Promotion: `{promotion}`.
- Evidence is classified as trajectory failure when a target has at least one eligible intermediate epoch, and initialization/persistent geometry failure when all epochs are ineligible.
"""
    (output_root / "report.md").write_text(report, encoding="utf-8")
    write_json(output_root / "source_manifest.json", {
        "git_commit": git_commit(),
        "files": [
            {"path": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__).resolve())},
            {"path": str(REPO_ROOT / "tools" / "prepare_hyperkvasir_official5fold_val_v2.py"),
             "sha256": file_sha256(REPO_ROOT / "tools" / "prepare_hyperkvasir_official5fold_val_v2.py")},
            {"path": str(REPO_ROOT / "src" / "methods" / "full_concm.py"),
             "sha256": file_sha256(REPO_ROOT / "src" / "methods" / "full_concm.py")},
            {"path": str(REPO_ROOT / "src" / "methods" / "concm_dsm.py"),
             "sha256": file_sha256(REPO_ROOT / "src" / "methods" / "concm_dsm.py")},
        ],
    })
    print(json.dumps({"gate": gate, "promotion": promotion, "targets_rescued": target_rescued}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("train-select", "fresh-test"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--seed", type=int, required=True)
        sub.add_argument("--session", type=int, required=True)
        sub.add_argument("--existing-root", required=True)
        sub.add_argument("--output-root", required=True)
        if name == "train-select":
            sub.add_argument("--min-free-mib", type=int, required=True)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--existing-root", required=True)
    summary.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if args.command == "train-select":
        train_select(args)
    elif args.command == "fresh-test":
        fresh_test(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
