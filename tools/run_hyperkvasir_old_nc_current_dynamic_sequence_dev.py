#!/usr/bin/env python3
"""Run the fold0-clean matched OLCD sequence-development gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets import build_protocol
from src.datasets.medical_lt import MedicalImageSubset
from src.methods.concm_dsm import DSMProjector, compute_dynamic_structure
from src.methods.full_concm import (
    MPCNetwork,
    PrototypeRecord,
    blend_current_covariance,
    build_attribute_mapping,
    build_visual_attribute_pool,
    calibrate_prototype,
    class_balanced_mean,
    class_semantic_embedding,
    fixed_nc_geometry,
    resample_repository,
    structure_anchor_contrastive,
    tensor_sha256,
    train_mpc_episodic,
    train_projector_session,
)
from src.methods.old_nc_current_dynamic import assemble_old_nc_current_dynamic_anchors
from src.models.resnet_cifar import resnet32
from src.utils.seed import set_seed
from tools.run_hyperkvasir_med_fdm_gate import _make_loader, _train_ce_phase
from train import apply_method_aliases, build_parser


METHODS = ("locked_nc_matched", "old_nc_current_dynamic_v1")
COLLAPSE_CURRENT_MIN = 50.0
COLLAPSE_CURRENT_TO_OLD_MAX = 30.0
MIN_FREE_MIB = 12000


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return str(value)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields)
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    text = json.dumps(list(rows), sort_keys=True, separators=(",", ":"), default=json_default)
    return hashlib.sha256(text.encode()).hexdigest()


def state_dict_sha256(state: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        value = state[key]
        if isinstance(value, torch.Tensor):
            digest.update(value.detach().contiguous().cpu().numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, default=json_default).encode())
    return digest.hexdigest()


def protocol_args(args, seed: int):
    parsed = build_parser().parse_args([])
    parsed.dataset = "hyper_kvasir23"
    parsed.data_root = args.data_root
    parsed.method = "finetune"
    parsed.order = "shuffled"
    parsed.seed = int(seed)
    parsed.base_classes = 13
    parsed.incremental_steps = 5
    parsed.max_phases = 6
    parsed.batch_size = 32
    parsed.num_workers = 4
    parsed.medical_image_size = 224
    parsed.medical_normalization = "imagenet"
    parsed.device = "cuda"
    parsed.download = False
    return apply_method_aliases(parsed)


def read_official(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    required = {"file-name", "class-name", "split-index"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError("Invalid official HyperKvasir 5-fold CSV")
    result = {}
    for row in rows:
        name = Path(row["file-name"]).name
        if name in result:
            raise RuntimeError(f"Duplicate official basename: {name}")
        result[name] = row
    return result


def build_clean_split(protocol, official: Mapping[str, Mapping[str, str]], seed: int):
    fit: dict[int, list[int]] = {}
    validation: dict[int, list[int]] = {}
    fit_rows = []
    val_rows = []
    for class_id in protocol.class_order:
        class_id = int(class_id)
        fit[class_id] = []
        validation[class_id] = []
        for index in protocol.train_indices_by_class[class_id]:
            record = protocol.train_records[index]
            row = official.get(record.path.name)
            if row is None:
                raise RuntimeError(f"Project-train image missing from official CSV: {record.path.name}")
            target = validation if int(row["split-index"]) == 0 else fit
            target[class_id].append(int(index))
            manifest_row = {
                "seed": seed,
                "class_id": class_id,
                "class_name": record.path.parent.name,
                "basename": record.path.name,
                "path": str(record.path),
                "official_fold": int(row["split-index"]),
                "project_membership": "train",
                "role": "validation" if target is validation else "fit_train",
            }
            (val_rows if target is validation else fit_rows).append(manifest_row)
        if not fit[class_id]:
            raise RuntimeError(f"Fold0-clean split lacks fit samples for class {class_id}")
    fit_names = {row["basename"] for row in fit_rows}
    val_names = {row["basename"] for row in val_rows}
    if fit_names & val_names:
        raise RuntimeError("Fit/validation overlap")
    return fit, validation, fit_rows, val_rows


@torch.no_grad()
def extract_features(model, dataset, device: torch.device) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    chunks = []
    model.eval()
    for images, _ in loader:
        chunks.append(model.extract_features(images.to(device, non_blocking=True)).detach().cpu().float())
    if not chunks:
        raise RuntimeError("Empty feature extraction dataset")
    return torch.cat(chunks)


def class_statistics(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return features.mean(0), features.var(0, unbiased=False).clamp_min(1e-8)


def frequency_groups(protocol) -> dict[int, str]:
    ordered = sorted(
        ((int(class_id), int(protocol.class_counts[class_id])) for class_id in protocol.class_order),
        key=lambda item: (item[1], item[0]),
    )
    groups = {}
    for rank, (class_id, _) in enumerate(ordered):
        groups[class_id] = ("tail", "mid", "head")[min(2, (3 * rank) // len(ordered))]
    return groups


def build_feature_bank(protocol, fit, validation, model, device: torch.device):
    fit_features = {}
    validation_features = {}
    paths = {"fit": {}, "validation": {}}
    for class_id in protocol.class_order:
        class_id = int(class_id)
        fit_features[class_id] = extract_features(
            model, MedicalImageSubset(protocol.train_records, fit[class_id], protocol.eval_transform), device
        )
        if validation[class_id]:
            validation_features[class_id] = extract_features(
                model, MedicalImageSubset(protocol.train_records, validation[class_id], protocol.eval_transform), device
            )
        else:
            validation_features[class_id] = torch.empty((0, 64), dtype=torch.float32)
        paths["fit"][class_id] = [str(protocol.train_records[index].path) for index in fit[class_id]]
        paths["validation"][class_id] = [str(protocol.train_records[index].path) for index in validation[class_id]]
    return {"fit_features": fit_features, "validation_features": validation_features, "paths": paths}


def anchor_matrix(method: str, nc: torch.Tensor, dynamic: torch.Tensor, class_order, current_classes):
    if method == "locked_nc_matched":
        matrix = F.normalize(nc[: len(class_order)].detach(), dim=1)
        current = set(map(int, current_classes))
        metadata = {
            "method": method,
            "class_order": list(map(int, class_order)),
            "current_class_ids": sorted(current),
            "old_source": "canonical fixed_nc_geometry rows",
            "current_source": "canonical fixed_nc_geometry rows",
            "normalized": True,
            "requires_grad": False,
            "assembled_sha256": tensor_sha256(matrix),
        }
        return matrix, metadata
    if method != "old_nc_current_dynamic_v1":
        raise ValueError(method)
    return assemble_old_nc_current_dynamic_anchors(
        nc[: len(class_order)], dynamic, class_order, current_classes
    )


def metric_pack(logits: torch.Tensor, labels: torch.Tensor, seen, current):
    predictions = logits.argmax(1)
    current_set = set(map(int, current))
    current_heads = {head for head, class_id in enumerate(seen) if int(class_id) in current_set}
    old_heads = set(range(len(seen))) - current_heads
    confusion = torch.zeros((len(seen), len(seen)), dtype=torch.long)
    for label, prediction in zip(labels.tolist(), predictions.tolist()):
        confusion[int(label), int(prediction)] += 1
    per_class = []
    recalls = []
    for head, class_id in enumerate(seen):
        total = int(confusion[head].sum())
        correct = int(confusion[head, head])
        recall = 100.0 * correct / total if total else None
        if recall is not None:
            recalls.append(recall)
        per_class.append({"head_idx": head, "class_id": int(class_id), "block": "current" if head in current_heads else "old",
                          "total": total, "correct": correct, "recall": recall})
    label_old = torch.tensor([int(value) in old_heads for value in labels.tolist()])
    label_current = ~label_old
    pred_old = torch.tensor([int(value) in old_heads for value in predictions.tolist()])
    pred_current = ~pred_old
    matches = predictions.eq(labels)

    def accuracy(mask):
        return 100.0 * float(matches[mask].float().mean()) if bool(mask.any()) else None

    old_acc = accuracy(label_old)
    current_acc = accuracy(label_current)
    hm = 0.0 if old_acc is None or current_acc is None or old_acc + current_acc == 0 else 2 * old_acc * current_acc / (old_acc + current_acc)
    old_to_current = 100.0 * float((label_old & pred_current).sum()) / max(int(label_old.sum()), 1)
    current_to_old = 100.0 * float((label_current & pred_old).sum()) / max(int(label_current.sum()), 1)
    current_recalls = [per_class[head]["recall"] for head in current_heads if per_class[head]["recall"] is not None]
    old_absorption = []
    if bool(label_current.any()):
        for head in old_heads:
            old_absorption.append((int(seen[head]), int(predictions[label_current].eq(head).sum())))
    absorbing_class, absorbing_count = max(old_absorption, key=lambda item: (item[1], -item[0])) if old_absorption else (None, 0)
    current_total = max(int(label_current.sum()), 1)
    metrics = {
        "AccT": 100.0 * float(matches.float().mean()),
        "old_accuracy": old_acc,
        "current_accuracy": current_acc,
        "HM": hm,
        "old_to_current": old_to_current,
        "current_to_old": current_to_old,
        "BER": (old_to_current + current_to_old) / 2.0,
        "macro_balanced_accuracy": sum(recalls) / len(recalls) if recalls else None,
        "current_macro_accuracy": sum(current_recalls) / len(current_recalls) if current_recalls else None,
        "worst_class_recall": min(recalls) if recalls else None,
        "max_absorbing_old_class": absorbing_class,
        "max_old_absorption_count": absorbing_count,
        "max_old_absorption_ratio": 100.0 * absorbing_count / current_total,
    }
    return metrics, per_class, confusion, predictions


@torch.no_grad()
def evaluate(projector, anchors, features_by_class, eval_classes, seen, current, device):
    features = torch.cat([features_by_class[int(class_id)] for class_id in eval_classes]).to(device)
    mapping = {int(class_id): head for head, class_id in enumerate(seen)}
    labels = torch.cat([
        torch.full((features_by_class[int(class_id)].shape[0],), mapping[int(class_id)], dtype=torch.long)
        for class_id in eval_classes
    ])
    projected = projector(features)
    logits = projected @ F.normalize(anchors.to(device), dim=1).T
    metrics, per_class, confusion, predictions = metric_pack(logits.cpu(), labels, seen, current)
    return metrics, per_class, confusion, predictions.cpu(), logits.cpu()


def is_collapse(metrics: Mapping[str, object]) -> bool:
    values = [value for value in metrics.values() if isinstance(value, (int, float))]
    return (
        not all(math.isfinite(float(value)) for value in values)
        or float(metrics["current_accuracy"]) < COLLAPSE_CURRENT_MIN
        or float(metrics["current_to_old"]) >= COLLAPSE_CURRENT_TO_OLD_MAX
    )


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def save_epoch_checkpoint(path: Path, *, seed, session, epoch, method, projector, mpc, optimizer, scheduler,
                          records, anchors, dynamic, anchor_metadata, common_base_path, common_base_sha, config):
    if path.exists():
        raise FileExistsError(path)
    payload = {
        "full_state": True,
        "seed": int(seed), "session": int(session), "epoch": int(epoch), "method": method,
        "projector_config": projector.config(), "projector_state_dict": projector.state_dict(),
        "mpc_config": mpc.config(), "mpc_state_dict": mpc.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
        "repository": [record.cpu_dict() for record in records],
        "class_order": [record.class_id for record in records],
        "current_classes": list(map(int, config["current_classes"])),
        "anchor_matrix": anchors.detach().cpu(), "original_dynamic_anchors": dynamic.detach().cpu(),
        "anchor_metadata": anchor_metadata,
        "anchor_matrix_sha256": tensor_sha256(anchors),
        "common_base_checkpoint": str(common_base_path), "common_base_checkpoint_sha256": common_base_sha,
        "rng_state": rng_state(), "config": dict(config),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return file_sha256(path)


def restore_epoch_checkpoint(path: Path, expected_sha: str, device: torch.device):
    if file_sha256(path) != expected_sha:
        raise RuntimeError(f"Checkpoint SHA mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    projector = DSMProjector(**payload["projector_config"]).to(device)
    projector.load_state_dict(payload["projector_state_dict"])
    mpc = MPCNetwork(**payload["mpc_config"]).to(device)
    mpc.load_state_dict(payload["mpc_state_dict"])
    records = [PrototypeRecord.from_dict(row) for row in payload["repository"]]
    optimizer = torch.optim.SGD(projector.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    return payload, projector, mpc, records, optimizer, scheduler


def train_one_epoch(projector, optimizer, scheduler, records, current_classes, fixed_geometry, method, seed, session, epoch, device):
    means = torch.stack([record.mean for record in records]).to(device)
    class_order = [record.class_id for record in records]
    with torch.no_grad():
        dynamic, dynamic_stats = compute_dynamic_structure(projector(means))
        anchors, anchor_metadata = anchor_matrix(method, fixed_geometry.to(device), dynamic, class_order, current_classes)
    features, labels, sample_stats = resample_repository(records, current_classes, seed + session * 100, epoch - 1)
    features, labels = features.to(device), labels.to(device)
    current_heads = [class_order.index(int(class_id)) for class_id in current_classes]
    optimizer.zero_grad(set_to_none=True)
    z = projector(features)
    losses = F.cross_entropy(z @ anchors.T, labels, reduction="none")
    match_loss = class_balanced_mean(losses, labels, len(records))
    cont_loss = structure_anchor_contrastive(z, labels, anchors, current_heads, 0.07)
    loss = match_loss + cont_loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    with torch.no_grad():
        dynamic_after, stats_after = compute_dynamic_structure(projector(means))
        anchors_after, metadata_after = anchor_matrix(
            method, fixed_geometry.to(device), dynamic_after, class_order, current_classes
        )
    trace = {
        "session": session, "epoch": epoch, "loss": float(loss.detach().cpu()),
        "match_loss": float(match_loss.detach().cpu()), "cont_loss": float(cont_loss.detach().cpu()),
        "lr": float(optimizer.param_groups[0]["lr"]), "etf_residual_before": dynamic_stats["etf_residual"],
        "etf_residual_after": stats_after["etf_residual"],
        "resampling_checksum": hashlib.sha256("".join(row["checksum"] for row in sample_stats).encode()).hexdigest(),
        "anchor_requires_grad": bool(anchors.requires_grad),
    }
    return anchors_after.detach(), dynamic_after.detach(), metadata_after, trace


def prepare_records(previous_records, current_classes, feature_bank, mpc, class_semantics,
                    attribute_visual, attribute_semantic, base_records, protocol, groups, device):
    records = [PrototypeRecord.from_dict(record.cpu_dict()) for record in previous_records]
    rows = []
    for class_id in current_classes:
        raw, observed = class_statistics(feature_bank["fit_features"][class_id])
        calibrated, calibration = calibrate_prototype(
            mpc, raw, class_semantics[class_id], attribute_visual, attribute_semantic, 0.75, device
        )
        covariance, weights = blend_current_covariance(observed, calibrated, base_records, 16.0, 0.6)
        records.append(PrototypeRecord(class_id, -1, calibrated, covariance, raw, int(protocol.class_counts[class_id]), groups[class_id], True))
        rows.append({"class_id": class_id, **calibration, "covariance_norm": float(torch.linalg.vector_norm(covariance)),
                     "base_covariance_weight_max": float(weights.max())})
    return records, rows


def transition_audit(previous_payload, previous_projector, current_payload, current_projector,
                     validation_features, previous_current, current_classes, method, seed, session, device):
    previous_seen = list(map(int, previous_payload["class_order"]))
    current_seen = list(map(int, current_payload["class_order"]))
    previous_anchors = previous_payload["anchor_matrix"].float()
    current_anchors = current_payload["anchor_matrix"].float()
    rows = []
    deltas = []
    for class_id in previous_current:
        features = validation_features[int(class_id)].to(device)
        if features.shape[0] == 0:
            rows.append({
                "seed": seed, "method": method, "transition": f"{session - 1}->{session}",
                "class_id": int(class_id), "validation_count": 0, "coverage_status": "UNAVAILABLE_IN_PROJECT_TRAIN_FOLD0",
                "before_accuracy": None, "after_accuracy": None, "accuracy_delta": None,
            })
            continue
        with torch.no_grad():
            before_logits = previous_projector(features) @ previous_anchors.to(device).T
            after_logits = current_projector(features) @ current_anchors.to(device).T
        before_ids = torch.tensor([previous_seen[index] for index in before_logits.argmax(1).cpu().tolist()])
        after_ids = torch.tensor([current_seen[index] for index in after_logits.argmax(1).cpu().tolist()])
        before_acc = 100.0 * float(before_ids.eq(int(class_id)).float().mean())
        after_acc = 100.0 * float(after_ids.eq(int(class_id)).float().mean())
        delta = after_acc - before_acc
        deltas.append(delta)
        old_row = previous_seen.index(int(class_id))
        dynamic = F.normalize(previous_payload["original_dynamic_anchors"][old_row].float(), dim=0)
        nc = fixed_nc_geometry(23, dynamic.numel())[old_row]
        cosine = float(F.cosine_similarity(dynamic, nc, dim=0))
        rows.append({
            "seed": seed, "method": method, "transition": f"{session - 1}->{session}", "class_id": int(class_id),
            "validation_count": int(features.shape[0]), "coverage_status": "AVAILABLE",
            "before_accuracy": before_acc, "after_accuracy": after_acc, "accuracy_delta": delta,
            "prediction_flip_count": int(before_ids.ne(after_ids).sum()),
            "dynamic_to_nc_cosine": cosine, "dynamic_to_nc_angle_deg": math.degrees(math.acos(max(-1.0, min(1.0, cosine)))),
            "anchor_l2_distance": float(torch.linalg.vector_norm(dynamic - nc)),
            "before_old_predictions": int(sum(value not in set(previous_current) for value in before_ids.tolist())),
            "before_current_predictions": int(sum(value in set(previous_current) for value in before_ids.tolist())),
            "after_old_predictions": int(sum(value not in set(current_classes) for value in after_ids.tolist())),
            "after_current_predictions": int(sum(value in set(current_classes) for value in after_ids.tolist())),
        })
    for row in rows:
        row["transition_mean_delta"] = sum(deltas) / len(deltas) if deltas else None
        row["transition_worst_delta"] = min(deltas) if deltas else None
    return rows


def make_common_base(seed_dir: Path, seed: int, args, protocol, fit, validation, fit_rows, val_rows, device):
    common = seed_dir / "common_base"
    common.mkdir(parents=True)
    write_csv(common / "fit_manifest.csv", fit_rows)
    write_csv(common / "validation_manifest.csv", val_rows)
    base_classes = list(map(int, protocol.tasks[0]))
    set_seed(seed)
    model = resnet32().to(device)
    model.expand_classifier(len(base_classes))
    base_indices = [index for class_id in base_classes for index in fit[class_id]]
    loader = _make_loader(
        MedicalImageSubset(protocol.train_records, base_indices, protocol.train_transform),
        32, True, seed, 4,
    )
    base_trace = _train_ce_phase(
        model, loader, {class_id: head for head, class_id in enumerate(base_classes)}, device,
        epochs=5, lr=0.01, momentum=0.9, weight_decay=5e-4,
    )
    backbone_path = common / "fold0_clean_backbone.pt"
    torch.save({"model_state": model.state_dict(), "base_classes": base_classes, "train_trace": base_trace,
                "fit_manifest_sha256": rows_sha256(fit_rows), "validation_manifest_sha256": rows_sha256(val_rows),
                "test_accessed": False}, backbone_path)
    backbone_sha = file_sha256(backbone_path)
    for parameter in model.parameters():
        parameter.requires_grad = False
    feature_bank = build_feature_bank(protocol, fit, validation, model, device)
    feature_path = common / "fold0_clean_feature_bank.pt"
    torch.save({**feature_bank, "seed": seed, "backbone_sha256": backbone_sha, "test_features": "FORBIDDEN_NOT_CREATED"}, feature_path)
    groups = frequency_groups(protocol)
    attribute_mapping, coverage = build_attribute_mapping(protocol.class_names)
    base_means = {class_id: feature_bank["fit_features"][class_id].mean(0) for class_id in base_classes}
    attribute_names, attribute_visual, attribute_semantic, associations = build_visual_attribute_pool(
        base_classes, base_means, attribute_mapping
    )
    class_semantics = {class_id: class_semantic_embedding(attribute_mapping[class_id]) for class_id in protocol.class_order}
    mpc = MPCNetwork(64, 64, 128).to(device)
    mpc_trace = train_mpc_episodic(
        mpc, {class_id: feature_bank["fit_features"][class_id] for class_id in base_classes},
        class_semantics, attribute_visual, attribute_semantic, 20, 0.01, seed, device, 5,
    )
    records = []
    for class_id in base_classes:
        mean, covariance = class_statistics(feature_bank["fit_features"][class_id])
        records.append(PrototypeRecord(class_id, 0, mean, covariance, mean.clone(), len(fit[class_id]), groups[class_id], False))
    fixed = fixed_nc_geometry(23, 128)
    projector = DSMProjector(64, 2048, 128).to(device)
    _, projector_trace, augmentation = train_projector_session(
        projector, records, base_classes, fixed, "locked_nc", 5, 0.01, seed, 0, device, 0.2
    )
    common_path = common / "common_base_full_state.pt"
    payload = {
        "full_state": True, "seed": seed, "session": 0, "method": "common_locked_nc_base",
        "backbone_checkpoint": str(backbone_path), "backbone_sha256": backbone_sha,
        "feature_bank_checkpoint": str(feature_path), "feature_bank_sha256": file_sha256(feature_path),
        "projector_config": projector.config(), "projector_state_dict": projector.state_dict(),
        "mpc_config": mpc.config(), "mpc_state_dict": mpc.state_dict(),
        "repository": [record.cpu_dict() for record in records],
        "class_order": base_classes, "fixed_nc_sha256": tensor_sha256(fixed),
        "attribute_mapping": attribute_mapping, "attribute_coverage": coverage,
        "attribute_names": attribute_names, "attribute_visual": attribute_visual,
        "attribute_semantic": attribute_semantic, "attribute_associations": associations,
        "class_semantics": class_semantics, "mpc_trace": mpc_trace,
        "projector_trace": projector_trace, "augmentation_trace": augmentation,
        "fit_manifest_sha256": rows_sha256(fit_rows), "validation_manifest_sha256": rows_sha256(val_rows),
        "test_accessed": False,
    }
    torch.save(payload, common_path)
    common_sha = file_sha256(common_path)
    write_json(common / "common_base_manifest.json", {
        "seed": seed, "path": str(common_path), "sha256": common_sha,
        "backbone_path": str(backbone_path), "backbone_sha256": backbone_sha,
        "feature_bank_path": str(feature_path), "feature_bank_sha256": file_sha256(feature_path),
        "fit_count": len(fit_rows), "validation_count": len(val_rows), "test_accessed": False,
        "fit_manifest_sha256": rows_sha256(fit_rows),
        "validation_manifest_sha256": rows_sha256(val_rows),
        "validation_coverage_gaps": [
            int(class_id) for class_id in protocol.class_order if not validation[int(class_id)]
        ],
    })
    return common_path, common_sha, payload, feature_bank, groups


def load_common(path: Path, expected_sha: str, device):
    if file_sha256(path) != expected_sha:
        raise RuntimeError("Common base SHA mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    projector = DSMProjector(**payload["projector_config"]).to(device)
    projector.load_state_dict(payload["projector_state_dict"])
    mpc = MPCNetwork(**payload["mpc_config"]).to(device)
    mpc.load_state_dict(payload["mpc_state_dict"])
    records = [PrototypeRecord.from_dict(row) for row in payload["repository"]]
    return payload, projector, mpc, records


def run_method(seed_dir: Path, seed: int, method: str, protocol, common_path, common_sha,
               common_payload, feature_bank, groups, device):
    method_dir = seed_dir / method
    method_dir.mkdir(parents=True)
    payload, projector, mpc, records = load_common(common_path, common_sha, device)
    attribute_visual = payload["attribute_visual"]
    attribute_semantic = payload["attribute_semantic"]
    class_semantics = {int(key): value for key, value in payload["class_semantics"].items()}
    base_records = [PrototypeRecord.from_dict(row) for row in payload["repository"]]
    fixed = fixed_nc_geometry(23, 128)
    checkpoint_rows, restore_rows, hash_rows, val_rows, selected_rows = [], [], [], [], []
    transition_rows, trace_rows, calibration_rows = [], [], []
    previous_selected = None
    previous_current = None
    for session in range(1, 6):
        current_classes = list(map(int, protocol.tasks[session]))
        records, calibration = prepare_records(
            records, current_classes, feature_bank, mpc, class_semantics, attribute_visual,
            attribute_semantic, base_records, protocol, groups, device,
        )
        for record in records:
            if record.class_id in current_classes:
                record.session_id = session
        calibration_rows.extend({"seed": seed, "method": method, "session": session, **row} for row in calibration)
        class_order = [record.class_id for record in records]
        optimizer = torch.optim.SGD(projector.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        means = torch.stack([record.mean for record in records]).to(device)
        with torch.no_grad():
            dynamic, _ = compute_dynamic_structure(projector(means))
            anchors, metadata = anchor_matrix(method, fixed.to(device), dynamic, class_order, current_classes)
        session_dir = method_dir / f"session{session}"
        epoch_rows = []
        for epoch in range(0, 6):
            if epoch > 0:
                anchors, dynamic, metadata, trace = train_one_epoch(
                    projector, optimizer, scheduler, records, current_classes, fixed,
                    method, seed, session, epoch, device,
                )
                trace_rows.append({"seed": seed, "method": method, **trace})
            config = {
                "method": method, "seed": seed, "session": session, "epoch": epoch,
                "current_classes": current_classes, "epochs": 5, "projector_lr": 0.01,
                "optimizer": "SGD(momentum=0.9,weight_decay=1e-4)", "scheduler": "CosineAnnealingLR(T_max=5)",
                "mpc_alpha": 0.75, "covariance_gamma": 16.0, "covariance_beta": 0.6,
                "sample_num_old": 100, "sample_num_current": 50, "temperature": 0.07,
                "loss": "class_balanced_LMatch_plus_existing_LCont", "lambda_dsm": 0.2,
                "classifier": "single normalized cosine scale1 no_bias no_temperature",
                "validation": "official_fold0_current_only", "test_accessed": False,
            }
            current_metrics, _, _, _, reference_logits = evaluate(
                projector, anchors, feature_bank["validation_features"], current_classes,
                class_order, current_classes, device,
            )
            current_metrics = {
                "current_micro_accuracy": current_metrics["current_accuracy"],
                "current_macro_accuracy": current_metrics["current_macro_accuracy"],
                "current_to_old": current_metrics["current_to_old"],
                "max_old_absorption_ratio": current_metrics["max_old_absorption_ratio"],
                "max_absorbing_old_class": current_metrics["max_absorbing_old_class"],
                "max_old_absorption_count": current_metrics["max_old_absorption_count"],
            }
            checkpoint = session_dir / "checkpoints" / f"epoch_{epoch}.pt"
            checkpoint_sha = save_epoch_checkpoint(
                checkpoint, seed=seed, session=session, epoch=epoch, method=method,
                projector=projector, mpc=mpc, optimizer=optimizer, scheduler=scheduler,
                records=records, anchors=anchors, dynamic=dynamic, anchor_metadata=metadata,
                common_base_path=common_path, common_base_sha=common_sha, config=config,
            )
            restored_payload, restored_projector, _, _, _, _ = restore_epoch_checkpoint(checkpoint, checkpoint_sha, device)
            restored_metrics, _, _, restored_predictions, restored_logits = evaluate(
                restored_projector, restored_payload["anchor_matrix"], feature_bank["validation_features"],
                current_classes, class_order, current_classes, device,
            )
            restored_current = {
                "current_micro_accuracy": restored_metrics["current_accuracy"],
                "current_macro_accuracy": restored_metrics["current_macro_accuracy"],
                "current_to_old": restored_metrics["current_to_old"],
                "max_old_absorption_ratio": restored_metrics["max_old_absorption_ratio"],
            }
            metric_diff = max(abs(float(current_metrics[key]) - float(restored_current[key])) for key in restored_current)
            logit_diff = float((reference_logits - restored_logits).abs().max())
            restore_ok = metric_diff <= 1e-8 and logit_diff <= 1e-7 and bool(restored_predictions.eq(reference_logits.argmax(1)).all())
            collapse = is_collapse({"current_accuracy": current_metrics["current_micro_accuracy"],
                                    "current_to_old": current_metrics["current_to_old"]})
            row = {"seed": seed, "method": method, "session": session, "epoch": epoch,
                   **current_metrics, "collapse": collapse, "restore_ok": restore_ok,
                   "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha}
            epoch_rows.append(row)
            val_rows.append(row)
            checkpoint_rows.append({"seed": seed, "method": method, "session": session, "epoch": epoch,
                                    "path": str(checkpoint), "sha256": checkpoint_sha, "full_state": True})
            restore_rows.append({"seed": seed, "method": method, "session": session, "epoch": epoch,
                                 "checkpoint_sha256": checkpoint_sha, "max_abs_logit_diff": logit_diff,
                                 "prediction_mismatch_count": int(restored_predictions.ne(reference_logits.argmax(1)).sum()),
                                 "metric_diff": metric_diff, "passed": restore_ok})
            current_set = set(current_classes)
            canonical_matrix = F.normalize(fixed[: len(class_order)].to(anchors.device), dim=1)
            for head, class_id in enumerate(class_order):
                source = "original_dynamic" if method == "old_nc_current_dynamic_v1" and class_id in current_set else "canonical_nc"
                expected = dynamic[head] if source == "original_dynamic" else fixed[head]
                row_hash = tensor_sha256(anchors[head])
                expected_hash = tensor_sha256(F.normalize(expected.detach(), dim=0))
                canonical_hash = tensor_sha256(canonical_matrix[head])
                hash_rows.append({"seed": seed, "method": method, "session": session, "epoch": epoch,
                                  "class_id": class_id, "block": "current" if class_id in current_set else "old",
                                  "source": source, "anchor_sha256": row_hash, "expected_sha256": expected_hash,
                                  "canonical_nc_sha256": canonical_hash,
                                  "differs_from_canonical_nc": row_hash != canonical_hash,
                                  "hash_match": row_hash == expected_hash, "requires_grad": bool(anchors.requires_grad),
                                  "in_optimizer": False})
        write_csv(session_dir / "current_validation_epochs.csv", epoch_rows)
        eligible = [row for row in epoch_rows if not row["collapse"] and row["restore_ok"]]
        selected = max(
            eligible,
            key=lambda row: (float(row["current_macro_accuracy"]), -float(row["current_to_old"]),
                             float(row["current_micro_accuracy"]), -float(row["max_old_absorption_ratio"]), -int(row["epoch"])),
        ) if eligible else None
        selection = {
            "seed": seed, "method": method, "session": session,
            "status": "SELECTED" if selected else "ALL_EPOCHS_COLLAPSED",
            "selector_scope": "official_fold0_current_classes_only", "test_accessed": False,
            "collapse_predicate": {"current_accuracy_min": 50.0, "current_to_old_max": 30.0},
            "ranking_key": ["current_macro_accuracy", "-current_to_old", "current_micro_accuracy",
                            "-max_old_absorption_ratio", "-epoch"],
            "selected_epoch": selected["epoch"] if selected else None,
            "selected_checkpoint": selected["checkpoint"] if selected else None,
            "selected_checkpoint_sha256": selected["checkpoint_sha256"] if selected else None,
        }
        write_json(session_dir / "selection.json", selection)
        if selected is None:
            break
        selected_payload, selected_projector, mpc, records, _, _ = restore_epoch_checkpoint(
            Path(selected["checkpoint"]), selected["checkpoint_sha256"], device
        )
        full_metrics, per_class, confusion, _, _ = evaluate(
            selected_projector, selected_payload["anchor_matrix"], feature_bank["validation_features"],
            class_order, class_order, current_classes, device,
        )
        full_row = {"seed": seed, "method": method, "session": session, "selected_epoch": selected["epoch"],
                    "selected_checkpoint_sha256": selected["checkpoint_sha256"], **full_metrics,
                    "collapse": is_collapse(full_metrics), "test_accessed": False}
        selected_rows.append(full_row)
        write_json(session_dir / "full_seen_fold0_audit.json", {
            "selection_frozen_before_audit": True, "metrics": full_metrics,
            "per_class": per_class, "confusion_matrix": confusion.tolist(), "test_accessed": False,
        })
        if previous_selected is not None:
            prev_payload, prev_projector, _, _, _, _ = restore_epoch_checkpoint(
                Path(previous_selected["checkpoint"]), previous_selected["checkpoint_sha256"], device
            )
            epoch0 = epoch_rows[0]
            epoch0_payload, epoch0_projector, _, _, _, _ = restore_epoch_checkpoint(
                Path(epoch0["checkpoint"]), epoch0["checkpoint_sha256"], device
            )
            transition_rows.extend(transition_audit(
                prev_payload, prev_projector, epoch0_payload, epoch0_projector,
                feature_bank["validation_features"], previous_current, current_classes,
                method, seed, session, device,
            ))
        previous_selected = selected
        previous_current = current_classes
        projector = selected_projector
    write_csv(method_dir / "checkpoint_manifest.csv", checkpoint_rows)
    write_csv(method_dir / "checkpoint_restore_audit.csv", restore_rows)
    write_csv(method_dir / "old_anchor_hash_audit.csv", hash_rows)
    write_csv(method_dir / "validation_epoch_metrics.csv", val_rows)
    write_csv(method_dir / "selected_full_seen_metrics.csv", selected_rows)
    write_csv(method_dir / "transition_audit.csv", transition_rows)
    write_csv(method_dir / "training_trace.csv", trace_rows)
    write_csv(method_dir / "prototype_calibration.csv", calibration_rows)
    return {"checkpoint_rows": checkpoint_rows, "restore_rows": restore_rows, "hash_rows": hash_rows,
            "selected_rows": selected_rows, "transition_rows": transition_rows, "validation_rows": val_rows}


def seed_gate(seed: int, locked, olcd):
    locked_by_session = {int(row["session"]): row for row in locked["selected_rows"]}
    olcd_by_session = {int(row["session"]): row for row in olcd["selected_rows"]}
    paired = sorted(set(locked_by_session) & set(olcd_by_session))
    def mean(rows, key):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return sum(values) / len(values) if values else float("nan")
    comparisons = []
    for session in paired:
        baseline, method = locked_by_session[session], olcd_by_session[session]
        comparisons.append({"seed": seed, "session": session,
                            "AccT_delta": float(method["AccT"]) - float(baseline["AccT"]),
                            "HM_delta": float(method["HM"]) - float(baseline["HM"]),
                            "old_accuracy_delta": float(method["old_accuracy"]) - float(baseline["old_accuracy"]),
                            "current_accuracy_delta": float(method["current_accuracy"]) - float(baseline["current_accuracy"]),
                            "current_to_old_delta": float(method["current_to_old"]) - float(baseline["current_to_old"])})
    locked_transition = mean(locked["transition_rows"], "accuracy_delta")
    olcd_transition = mean(olcd["transition_rows"], "accuracy_delta")
    old_rows = [row for row in olcd["hash_rows"] if row["block"] == "old"]
    old_hashes_by_class = {
        int(class_id): {row["anchor_sha256"] for row in old_rows if int(row["class_id"]) == int(class_id)}
        for class_id in {int(row["class_id"]) for row in old_rows}
    }
    old_hash_ok = bool(old_rows) and all(
        row["source"] == "canonical_nc" and not bool(row["requires_grad"]) and not bool(row["in_optimizer"])
        for row in old_rows
    ) and all(len(values) == 1 for values in old_hashes_by_class.values())
    restore_ok = all(bool(row["passed"]) for row in locked["restore_rows"] + olcd["restore_rows"])
    dynamic_rows = [row for row in olcd["hash_rows"] if row["block"] == "current"]
    dynamic_active = any(
        row["source"] == "original_dynamic" and bool(row["differs_from_canonical_nc"])
        for row in dynamic_rows
    )
    checks = {
        "olcd_selected_5_of_5": len(olcd_by_session) == 5,
        "olcd_collapse_0_of_5": len(olcd_by_session) == 5 and sum(bool(row["collapse"]) for row in olcd_by_session.values()) == 0,
        "mean_HM_drop_within_1pp": len(comparisons) == 5 and mean(comparisons, "HM_delta") >= -1.0,
        "mean_old_accuracy_drop_within_1pp": len(comparisons) == 5 and mean(comparisons, "old_accuracy_delta") >= -1.0,
        "per_session_HM_drop_within_2pp": len(comparisons) == 5 and min(row["HM_delta"] for row in comparisons) >= -2.0,
        "per_session_old_accuracy_drop_within_2pp": len(comparisons) == 5 and min(row["old_accuracy_delta"] for row in comparisons) >= -2.0,
        "seed1_session5_current_to_old_below_30": seed != 1 or (5 in olcd_by_session and float(olcd_by_session[5]["current_to_old"]) < 30.0),
        "transition_not_extra_worse_than_locked_by_1pp": bool(locked["transition_rows"] and olcd["transition_rows"]) and olcd_transition - locked_transition >= -1.0,
        "old_anchor_hash_frozen": old_hash_ok,
        "all_checkpoint_fresh_restore": restore_ok,
        "current_dynamic_path_active": dynamic_active,
    }
    passed = all(checks.values())
    return {
        "seed": seed, "passed": passed, "checks": checks, "comparisons": comparisons,
        "mean_deltas": {key: mean(comparisons, key) for key in ("AccT_delta", "HM_delta", "old_accuracy_delta", "current_accuracy_delta", "current_to_old_delta")},
        "locked_transition_mean_delta": locked_transition, "olcd_transition_mean_delta": olcd_transition,
        "status": f"OLCD_SEED{seed}_PASSED" if passed else f"OLD_NC_CURRENT_DYNAMIC_SEED{seed}_FAILED",
    }


def aggregate(output: Path, seed_results, gates, final_status: str):
    common_rows, run_rows, metric_rows, transition_rows, hash_rows, restore_rows, comparisons = [], [], [], [], [], [], []
    seed_summary = []
    for seed, result in sorted(seed_results.items()):
        common_rows.append(read_json(output / f"seed{seed}/common_base/common_base_manifest.json"))
        gate = gates[seed]
        seed_summary.append({"seed": seed, "passed": gate["passed"], "status": gate["status"], **gate["mean_deltas"]})
        comparisons.extend(gate["comparisons"])
        for method in METHODS:
            values = result[method]
            metric_rows.extend(values["selected_rows"])
            transition_rows.extend(values["transition_rows"])
            hash_rows.extend(values["hash_rows"])
            restore_rows.extend(values["restore_rows"])
            for row in values["selected_rows"]:
                run_rows.append({"seed": seed, "method": method, "session": row["session"],
                                 "selected_epoch": row["selected_epoch"], "checkpoint_sha256": row["selected_checkpoint_sha256"],
                                 "fit_manifest_sha256": common_rows[-1]["fit_manifest_sha256"], "test_accessed": False})
    write_csv(output / "common_base_manifest.csv", common_rows)
    write_csv(output / "matched_run_manifest.csv", run_rows)
    write_csv(output / "per_session_metrics.csv", metric_rows)
    write_csv(output / "per_seed_summary.csv", seed_summary)
    write_csv(output / "transition_audit.csv", transition_rows)
    write_csv(output / "old_anchor_hash_audit.csv", hash_rows)
    write_csv(output / "checkpoint_restore_audit.csv", restore_rows)
    write_csv(output / "three_seed_comparison.csv", comparisons)
    gate_payload = {"status": final_status, "seed_gates": gates, "test_accessed": False, "full_data_refit": False,
                    "fixed_old_hard_margin_added": False}
    write_json(output / "gate.json", gate_payload)
    impl = read_json(output / "implementation_equivalence.json")
    report = f"""# Old NC / Current Dynamic Sequence Development

- Final status: `{final_status}`.
- Stage 0 exact CF5 reproduction: `{impl['status']}`.
- Seeds executed: `{sorted(seed_results)}`.
- Project test accessed: false.
- Full-data refit: false.
- Fixed-old hard-margin added: false.

## Required Questions

1. Exact CF5 reproduction: {'yes' if impl['status'] == 'CF5_IMPLEMENTATION_EQUIVALENT' else 'no'}.
2. Old anchors bitwise frozen: {all(g['checks']['old_anchor_hash_frozen'] for g in gates.values())}.
3. Current dynamic path active: {all(g['checks']['current_dynamic_path_active'] for g in gates.values())}.
4. Transition cliff: see `transition_audit.csv`; the frozen gate result is {all(g['checks']['transition_not_extra_worse_than_locked_by_1pp'] for g in gates.values())}.
5. Seed1/session5 collapse: {not gates.get(1, {}).get('checks', {}).get('seed1_session5_current_to_old_below_30', False)}.
6. Old accuracy/HM protected versus matched Locked NC: {all(g['checks']['mean_HM_drop_within_1pp'] and g['checks']['mean_old_accuracy_drop_within_1pp'] for g in gates.values())}.
7. Stable current-side benefit: see `three_seed_comparison.csv` and frozen gate.
8. Benefit in at least 2/3 seeds: {final_status == 'OLCD_SEQUENCE_DEV_PASSED'}.
9. Freeze for confirmatory evaluation: {final_status == 'OLCD_SEQUENCE_DEV_PASSED'}.
10. Failure localization is encoded by the failed seed checks in `gate.json`: current collapse, transition degradation, or old-class HM/accuracy degradation.
"""
    (output / "report.md").write_text(report, encoding="utf-8")


def artifact_hashes(output: Path):
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_sha256.json":
            rows.append({"path": str(path.relative_to(output)), "sha256": file_sha256(path), "size": path.stat().st_size})
    write_json(output / "artifact_sha256.json", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--official-csv", required=True)
    parser.add_argument("--implcheck-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError(output)
    implcheck = Path(args.implcheck_root).resolve()
    implementation = read_json(implcheck / "implementation_equivalence.json")
    if implementation["status"] != "CF5_IMPLEMENTATION_EQUIVALENT":
        raise SystemExit("TRAINING_NOT_STARTED")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Require CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly physical GPU0 to be visible")
    free_bytes, _ = torch.cuda.mem_get_info(0)
    free_mib = free_bytes / 1024**2
    if free_mib < MIN_FREE_MIB:
        output.mkdir(parents=True)
        write_json(output / "gate.json", {"status": "GPU_INSUFFICIENT_FREE_MEMORY", "free_mib": free_mib,
                                           "required_mib": MIN_FREE_MIB, "training_started": False})
        raise SystemExit("GPU_INSUFFICIENT_FREE_MEMORY")
    output.mkdir(parents=True)
    write_json(output / "implementation_equivalence.json", implementation)
    official_path = Path(args.official_csv).resolve()
    method_frozen = {
        "method": "old_nc_current_dynamic_v1", "branch_type": "independent_not_full_dynamic_repair",
        "full_dynamic_closure_immutable": True, "old_anchor_source": "canonical fixed_nc_geometry",
        "current_anchor_source": "original compute_dynamic_structure", "single_anchor_matrix": True,
        "classifier": {"normalized_cosine": True, "shared_scale": 1, "bias": False, "temperature": False,
                       "block_scale": False, "residual": False, "interpolation": False, "gate": False},
        "transition_rule": "current class switches to canonical NC when it becomes old",
        "only_matched_difference": "anchor matrix assembly", "epochs": [0, 1, 2, 3, 4, 5],
        "selector": ["current_macro_accuracy", "-current_to_old", "current_micro_accuracy",
                     "-max_old_absorption_ratio", "-epoch"],
        "collapse_predicate": {"current_accuracy_min": 50.0, "current_to_old_max": 30.0},
        "fold": 0, "test_forbidden": True, "gpu_physical": 0, "shared_mode": True,
        "training": {"base_epochs": 5, "session_epochs": 5, "batch_size": 32, "image_size": 224,
                     "projector_lr": 0.01, "mpc_epochs": 20, "mpc_lr": 0.01, "lambda_dsm": 0.2},
    }
    write_json(output / "method_frozen.json", method_frozen)
    official = read_official(official_path)
    seed_results = {}
    gates = {}
    final_status = "NOT_EVALUABLE"
    for seed in (1, 2, 3):
        seed_dir = output / f"seed{seed}"
        seed_dir.mkdir()
        protocol = build_protocol(protocol_args(args, seed))
        fit, validation, fit_rows, val_rows = build_clean_split(protocol, official, seed)
        common_path, common_sha, common_payload, feature_bank, groups = make_common_base(
            seed_dir, seed, args, protocol, fit, validation, fit_rows, val_rows, torch.device("cuda")
        )
        result = {}
        for method in METHODS:
            result[method] = run_method(
                seed_dir, seed, method, protocol, common_path, common_sha,
                common_payload, feature_bank, groups, torch.device("cuda")
            )
        seed_results[seed] = result
        gates[seed] = seed_gate(seed, result["locked_nc_matched"], result["old_nc_current_dynamic_v1"])
        write_json(seed_dir / "seed_gate.json", gates[seed])
        if not gates[seed]["passed"]:
            if seed == 1:
                final_status = "OLD_NC_CURRENT_DYNAMIC_SEED1_FAILED / CLOSE_OLCD_BRANCH / NO_SEED2_OR_SEED3"
            elif seed == 2:
                final_status = "OLD_NC_CURRENT_DYNAMIC_SEED2_FAILED / CLOSE_OLCD_BRANCH / NO_SEED3"
            else:
                final_status = "OLCD_SEQUENCE_DEV_FAILED / CLOSE_OLCD_BRANCH / LOCKED_NC_REMAINS_MAIN_METHOD"
            break
    if len(seed_results) == 3 and all(gate["passed"] for gate in gates.values()):
        comparisons = [row for gate in gates.values() for row in gate["comparisons"]]
        mean = lambda key: sum(float(row[key]) for row in comparisons) / len(comparisons)
        benefit_seeds = sum(
            gate["mean_deltas"]["current_accuracy_delta"] >= 1.0
            or gate["mean_deltas"]["current_to_old_delta"] <= -2.0
            or gate["mean_deltas"]["HM_delta"] >= 1.0
            for gate in gates.values()
        )
        promotion = (
            mean("AccT_delta") >= -1.0 and mean("HM_delta") >= -1.0 and mean("old_accuracy_delta") >= -1.0
            and (mean("current_accuracy_delta") >= 1.0 or mean("current_to_old_delta") <= -2.0 or mean("HM_delta") >= 1.0)
            and benefit_seeds >= 2
        )
        final_status = "OLCD_SEQUENCE_DEV_PASSED / FREEZE_METHOD_FOR_CONFIRMATORY_EVALUATION" if promotion else "OLCD_SEQUENCE_DEV_FAILED / CLOSE_OLCD_BRANCH / LOCKED_NC_REMAINS_MAIN_METHOD"
    aggregate(output, seed_results, gates, final_status)
    source_files = [Path(__file__).resolve(), REPO_ROOT / "src/methods/old_nc_current_dynamic.py"]
    write_json(output / "source_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "test_accessed": False,
        "official_csv": str(official_path), "official_csv_sha256": file_sha256(official_path),
        "implementation_check": str(implcheck),
        "sources": [{"path": str(path), "sha256": file_sha256(path)} for path in source_files],
    })
    artifact_hashes(output)
    print(json.dumps({"status": final_status, "seeds_executed": sorted(seed_results), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
