#!/usr/bin/env python3
"""Run full five-session ConCM-style HyperKvasir23 CIL experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets import build_protocol
from src.methods.concm_dsm import DSMProjector, compute_dynamic_structure
from src.methods.full_concm import (
    MPCNetwork,
    PrototypeRecord,
    blend_current_covariance,
    build_attribute_mapping,
    build_visual_attribute_pool,
    calibrate_prototype,
    class_semantic_embedding,
    fixed_nc_geometry,
    tensor_sha256,
    train_mpc_episodic,
    train_projector_session,
    validate_continuity,
)
from src.models.resnet_cifar import resnet32
from src.utils.seed import set_seed
from train import apply_method_aliases, build_parser


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return str(value)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_args(args):
    parsed = build_parser().parse_args([])
    parsed.dataset = "hyper_kvasir23"
    parsed.data_root = args.data_root
    parsed.method = "finetune"
    parsed.order = "shuffled"
    parsed.seed = int(args.seed)
    parsed.base_classes = 13
    parsed.incremental_steps = 5
    parsed.max_phases = 6
    parsed.batch_size = int(args.batch_size)
    parsed.num_workers = int(args.num_workers)
    parsed.device = args.device
    parsed.download = False
    return apply_method_aliases(parsed)


@torch.no_grad()
def extract_dataset_features(model, dataset, device: torch.device, batch_size: int, num_workers: int) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    rows = []
    model.eval()
    for images, _ in loader:
        rows.append(model.extract_features(images.to(device, non_blocking=True)).detach().cpu().float())
    if not rows:
        raise RuntimeError("Cannot extract features from an empty dataset")
    return torch.cat(rows)


def build_feature_bank(args, protocol, checkpoint_path: Path, cache_path: Path, device: torch.device):
    checkpoint_hash = file_sha256(checkpoint_path)
    if cache_path.is_file():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache.get("seed") != int(args.seed) or cache.get("checkpoint_sha256") != checkpoint_hash:
            raise RuntimeError(f"Feature cache provenance mismatch: {cache_path}")
        return cache
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = resnet32().to(device)
    model.expand_classifier(len(checkpoint["old_classes"]))
    model.load_state_dict(checkpoint["model_state"])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    train_features = {}
    test_features = {}
    train_paths = {}
    test_paths = {}
    for class_id in protocol.class_order:
        class_id = int(class_id)
        train_features[class_id] = extract_dataset_features(
            model,
            protocol.prototype_dataset_for_classes([class_id]),
            device,
            int(args.batch_size),
            int(args.num_workers),
        )
        test_features[class_id] = extract_dataset_features(
            model,
            protocol.test_dataset_for_classes([class_id]),
            device,
            int(args.batch_size),
            int(args.num_workers),
        )
        train_paths[class_id] = [
            str(protocol.train_records[index].path) for index in protocol.train_indices_by_class[class_id]
        ]
        test_paths[class_id] = [
            str(protocol.test_records[index].path) for index in protocol.test_indices_by_class[class_id]
        ]
    cache = {
        "seed": int(args.seed),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "class_order": list(map(int, protocol.class_order)),
        "tasks": [list(map(int, task)) for task in protocol.tasks],
        "train_features": train_features,
        "test_features": test_features,
        "train_paths": train_paths,
        "test_paths": test_paths,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        raise FileExistsError(f"Refusing to overwrite feature cache: {cache_path}")
    torch.save(cache, cache_path)
    return cache


def frequency_groups(protocol) -> Dict[int, str]:
    ordered = sorted(((int(class_id), int(protocol.class_counts[class_id])) for class_id in protocol.class_order), key=lambda x: (x[1], x[0]))
    result = {}
    for rank, (class_id, _) in enumerate(ordered):
        bucket = min(2, (3 * rank) // len(ordered))
        result[class_id] = ("tail", "mid", "head")[bucket]
    return result


def class_statistics(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return features.mean(dim=0), features.var(dim=0, unbiased=False).clamp_min(1e-8)


def metric_pack(
    logits: torch.Tensor,
    labels: torch.Tensor,
    seen_classes: Sequence[int],
    old_classes: Sequence[int],
    current_classes: Sequence[int],
    groups: Mapping[int, str],
) -> tuple[Dict[str, object], list[Dict[str, object]], Dict[int, Dict[int, int]]]:
    predictions = logits.argmax(dim=1)
    old_heads = {index for index, class_id in enumerate(seen_classes) if class_id in set(old_classes)}
    current_heads = {index for index, class_id in enumerate(seen_classes) if class_id in set(current_classes)}
    confusion: Dict[int, Dict[int, int]] = {}
    class_rows = []
    recalls = []
    f1_values = []
    for head, class_id in enumerate(seen_classes):
        mask = labels == head
        total = int(mask.sum())
        correct = int(predictions[mask].eq(labels[mask]).sum())
        predicted_count = int(predictions.eq(head).sum())
        recall = 100.0 * correct / total if total else 0.0
        precision = 100.0 * correct / predicted_count if predicted_count else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1_values.append(f1)
        top_old_class = None
        top_old_count = 0
        if head in current_heads and old_heads:
            counts = [(old_head, int(predictions[mask].eq(old_head).sum())) for old_head in old_heads]
            best_head, top_old_count = max(counts, key=lambda item: (item[1], -item[0]))
            top_old_class = int(seen_classes[best_head])
        class_rows.append(
            {
                "head_idx": head,
                "class_id": int(class_id),
                "block": "current" if head in current_heads else "old",
                "frequency_group": groups[int(class_id)],
                "total": total,
                "correct": correct,
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "top_predicted_old_class": top_old_class,
                "top_predicted_old_count": top_old_count,
            }
        )
    for label, prediction in zip(labels.tolist(), predictions.tolist()):
        true_class = int(seen_classes[label])
        predicted_class = int(seen_classes[prediction])
        confusion.setdefault(true_class, {})[predicted_class] = confusion.setdefault(true_class, {}).get(predicted_class, 0) + 1
    old_mask = torch.tensor([int(value) in old_heads for value in labels.tolist()])
    current_mask = ~old_mask
    pred_old = torch.tensor([int(value) in old_heads for value in predictions.tolist()])
    pred_current = ~pred_old
    matches = predictions.eq(labels)

    def accuracy(mask):
        return 100.0 * float(matches[mask].float().mean()) if bool(mask.any()) else None

    old_accuracy = accuracy(old_mask)
    current_accuracy = accuracy(current_mask)
    hm = 0.0 if old_accuracy is None or current_accuracy is None or old_accuracy + current_accuracy == 0 else 2 * old_accuracy * current_accuracy / (old_accuracy + current_accuracy)
    old_to_current = 100.0 * float((old_mask & pred_current).sum()) / max(int(old_mask.sum()), 1)
    current_to_old = 100.0 * float((current_mask & pred_old).sum()) / max(int(current_mask.sum()), 1)
    group_accuracy = {}
    for group in ("head", "mid", "tail"):
        heads = {index for index, class_id in enumerate(seen_classes) if groups[int(class_id)] == group}
        mask = torch.tensor([int(value) in heads for value in labels.tolist()])
        group_accuracy[f"{group}_accuracy"] = accuracy(mask)
    metrics = {
        "AccT": 100.0 * float(matches.float().mean()),
        "old_acc": old_accuracy,
        "current_acc": current_accuracy,
        "HM_old_current": hm,
        "old_to_current_rate": old_to_current,
        "current_to_old_rate": current_to_old,
        "BER": (old_to_current + current_to_old) / 2.0,
        "macro_balanced_acc": sum(recalls) / len(recalls),
        "macro_f1": sum(f1_values) / len(f1_values),
        "worst_class_recall": min(recalls),
        **group_accuracy,
    }
    return metrics, class_rows, confusion


@torch.no_grad()
def evaluate_session(
    mode: str,
    projector: DSMProjector,
    records: Sequence[PrototypeRecord],
    current_classes: Sequence[int],
    test_features: Mapping[int, torch.Tensor],
    fixed_geometry: torch.Tensor,
    dynamic_geometry: torch.Tensor,
    groups: Mapping[int, str],
    previous_geometry: torch.Tensor | None,
    device: torch.device,
    lambda_dsm: float,
):
    seen_classes = [record.class_id for record in records]
    old_classes = [value for value in seen_classes if value not in set(current_classes)]
    features = torch.cat([test_features[class_id] for class_id in seen_classes]).to(device)
    labels = torch.cat(
        [torch.full((test_features[class_id].shape[0],), head, dtype=torch.long) for head, class_id in enumerate(seen_classes)]
    )
    projected = projector(features)
    fixed_logits = projected @ fixed_geometry[: len(records)].to(device).T
    dynamic_logits = projected @ dynamic_geometry.to(device).T
    if mode == "locked_nc":
        logits = fixed_logits
    elif mode == "full_dynamic":
        logits = dynamic_logits
    else:
        logits = F.normalize(fixed_logits, dim=1) + float(lambda_dsm) * F.normalize(dynamic_logits, dim=1)
    metrics, class_rows, confusion = metric_pack(logits.cpu(), labels, seen_classes, old_classes, current_classes, groups)
    means = torch.stack([record.mean for record in records]).to(device)
    projected_means = projector(means)
    rebuilt, geometry_stats = compute_dynamic_structure(projected_means)
    matching = (projected_means * rebuilt).sum(dim=1)
    similarities = projected_means @ rebuilt.T
    ranks = []
    for head in range(len(records)):
        order = similarities[head].argsort(descending=True)
        ranks.append(int((order == head).nonzero(as_tuple=False)[0].item()) + 1)
    current_heads = [seen_classes.index(class_id) for class_id in current_classes]
    old_heads = [head for head in range(len(records)) if head not in current_heads]
    current_margin = []
    for head in current_heads:
        own = similarities[head, head]
        nearest_old = similarities[head, old_heads].max() if old_heads else similarities.new_tensor(0.0)
        current_margin.append(float((own - nearest_old).cpu()))
    old_drift = None
    if previous_geometry is not None:
        old_drift = float((1.0 - F.cosine_similarity(previous_geometry.to(device), rebuilt[: previous_geometry.shape[0]], dim=1)).mean().cpu())
    attraction = 0
    if 3 in old_classes:
        class3_head = seen_classes.index(3)
        current_mask = torch.tensor([int(value) in current_heads for value in labels.tolist()])
        attraction = int(logits.cpu().argmax(dim=1)[current_mask].eq(class3_head).sum())
    health = {
        "ETF_residual": geometry_stats["etf_residual"],
        "SMR_all": float(matching.sum().cpu()),
        "SMR_old": float(matching[old_heads].sum().cpu()) if old_heads else 0.0,
        "SMR_current": float(matching[current_heads].sum().cpu()),
        "singular_values": geometry_stats["singular_values"],
        "condition_number": geometry_stats["condition_number"],
        "own_anchor_margin": sum(current_margin) / len(current_margin),
        "own_anchor_rank": sum(ranks[head] for head in current_heads) / len(current_heads),
        "old_anchor_drift": old_drift,
        "seed3_old_class3_current_absorption": attraction,
    }
    return metrics, class_rows, confusion, health, rebuilt.detach().cpu()


def save_checkpoint(
    path: Path,
    session: int,
    mode: str,
    projector: DSMProjector,
    mpc: MPCNetwork,
    records: Sequence[PrototypeRecord],
    geometry: torch.Tensor,
    mapping: Mapping[int, int],
    exemplar_state: Mapping[int, str],
    previous_sha: str | None,
    backbone_path: Path,
    backbone_sha: str,
    config: Mapping[str, object],
) -> str:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    projector.eval()
    with torch.no_grad():
        delta_prime = projector(torch.stack([record.mean for record in records]).to(next(projector.parameters()).device))
        _, geometry_stats = compute_dynamic_structure(delta_prime)
    payload = {
        "session": int(session),
        "mode": mode,
        "projector_state_dict": projector.state_dict(),
        "projector_config": projector.config(),
        "mpc_state_dict": mpc.state_dict(),
        "mpc_config": mpc.config(),
        "repository": [record.cpu_dict() for record in records],
        "geometry": geometry.detach().cpu(),
        "delta_prime": delta_prime.detach().cpu(),
        "geometry_stats": geometry_stats,
        "class_to_head": {int(key): int(value) for key, value in mapping.items()},
        "exemplar_state": {int(key): str(value) for key, value in exemplar_state.items()},
        "seen_classes": [record.class_id for record in records],
        "previous_checkpoint_sha256": previous_sha,
        "backbone_checkpoint": str(backbone_path),
        "backbone_checkpoint_sha256": backbone_sha,
        "optimizer_rule": "rebuild SGD and cosine scheduler at each session; inherit projector weights",
        "config": dict(config),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return file_sha256(path)


def restore_checkpoint(path: Path, expected_sha: str, device: torch.device):
    actual = file_sha256(path)
    if actual != expected_sha:
        raise RuntimeError(f"Checkpoint continuity SHA mismatch for {path}: {actual} != {expected_sha}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    projector = DSMProjector(**{
        "input_dim": int(payload["projector_config"]["input_dim"]),
        "hidden_dim": int(payload["projector_config"]["hidden_dim"]),
        "output_dim": int(payload["projector_config"]["output_dim"]),
    }).to(device)
    projector.load_state_dict(payload["projector_state_dict"])
    mpc = MPCNetwork(**payload["mpc_config"]).to(device)
    mpc.load_state_dict(payload["mpc_state_dict"])
    mpc.eval()
    records = [PrototypeRecord.from_dict(value) for value in payload["repository"]]
    return payload, projector, mpc, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["locked_nc", "full_dynamic", "nc_anchored"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-session", type=int, default=0)
    parser.add_argument("--end-session", type=int, default=5)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--projector-hidden", type=int, default=2048)
    parser.add_argument("--projector-dim", type=int, default=128)
    parser.add_argument("--base-projector-epochs", type=int, default=5)
    parser.add_argument("--increment-projector-epochs", type=int, default=5)
    parser.add_argument("--mpc-epochs", type=int, default=20)
    parser.add_argument("--mpc-lr", type=float, default=0.01)
    parser.add_argument("--projector-lr", type=float, default=0.01)
    parser.add_argument("--mpc-alpha", type=float, default=0.75)
    parser.add_argument("--covariance-gamma", type=float, default=16.0)
    parser.add_argument("--covariance-beta", type=float, default=0.6)
    parser.add_argument("--lambda-dsm", type=float, default=0.2)
    args = parser.parse_args()
    if not (0 <= args.start_session <= args.end_session <= 5):
        raise ValueError("Require 0 <= start-session <= end-session <= 5")
    if args.start_session > 0 and not args.resume:
        raise ValueError("--resume is required when --start-session > 0")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty: {output}")
    for subdir in ("checkpoints", "configs", "csv", "json", "logs", "manifests"):
        (output / subdir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible GPU; set CUDA_VISIBLE_DEVICES=2")
    set_seed(int(args.seed))
    protocol = build_protocol(protocol_args(args))
    if len(protocol.tasks) != 6:
        raise RuntimeError(f"Expected base plus five incremental sessions, got {len(protocol.tasks)} tasks")
    base_checkpoint = Path(args.base_checkpoint).resolve()
    if not base_checkpoint.is_file():
        raise FileNotFoundError(base_checkpoint)
    feature_cache = build_feature_bank(args, protocol, base_checkpoint, Path(args.feature_cache).resolve(), device)
    groups = frequency_groups(protocol)
    config = vars(args).copy()
    config.update(
        {
            "command": shlex.join([sys.executable, *sys.argv]),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "full_task_count": len(protocol.tasks),
            "optimizer_rule": "rebuild per session; projector and MPC state inherited",
            "semantic_embedding": "deterministic_signed_hash_fallback",
            "k_shot": 5,
            "sample_num_old": 100,
            "sample_num_current": 50,
            "temperature": 0.07,
        }
    )
    write_json(output / "configs" / "config.json", config)
    (output / "logs" / "command.log").write_text(config["command"] + "\n", encoding="utf-8")
    split_payload = {
        "seed": args.seed,
        "class_order": list(map(int, protocol.class_order)),
        "tasks": [list(map(int, task)) for task in protocol.tasks],
        "class_names": {int(k): v for k, v in protocol.class_names.items()},
        "class_counts": {int(k): int(v) for k, v in protocol.class_counts.items()},
        "train_paths": feature_cache["train_paths"],
        "test_paths": feature_cache["test_paths"],
    }
    split_text = json.dumps(split_payload, sort_keys=True, default=json_default)
    split_payload["split_manifest_sha256"] = hashlib.sha256(split_text.encode()).hexdigest()
    write_json(output / "manifests" / "split_manifest.json", split_payload)
    attribute_mapping, coverage = build_attribute_mapping(protocol.class_names)
    write_json(output / "manifests" / "attribute_mapping.json", {"coverage": coverage, "mapping": attribute_mapping})
    backbone_sha = file_sha256(base_checkpoint)
    fixed_geometry = fixed_nc_geometry(len(protocol.class_order), int(args.projector_dim))
    session_metrics = []
    per_class_rows = []
    health_rows = []
    augmentation_rows = []
    calibration_rows = []
    continuity_rows = []
    checkpoint_rows = []
    exemplar_state: Dict[int, str] = {}
    previous_geometry = None
    previous_path = None
    previous_sha = None

    if args.resume:
        previous_path = Path(args.resume).resolve()
        previous_sha = file_sha256(previous_path)
        payload, projector, mpc, records = restore_checkpoint(previous_path, previous_sha, device)
        if int(payload["session"]) != int(args.start_session) - 1:
            raise RuntimeError("Resume checkpoint session does not precede start-session")
        previous_geometry = payload["geometry"]
        exemplar_state = {int(key): str(value) for key, value in payload["exemplar_state"].items()}
        attribute_names, attribute_visual, attribute_semantic, associations = build_visual_attribute_pool(
            protocol.tasks[0], {class_id: feature_cache["train_features"][class_id].mean(0) for class_id in protocol.tasks[0]}, attribute_mapping
        )
        next_session = int(args.start_session)
    else:
        base_classes = list(map(int, protocol.tasks[0]))
        base_means = {class_id: class_statistics(feature_cache["train_features"][class_id])[0] for class_id in base_classes}
        attribute_names, attribute_visual, attribute_semantic, associations = build_visual_attribute_pool(
            base_classes, base_means, attribute_mapping
        )
        class_semantics = {class_id: class_semantic_embedding(attribute_mapping[class_id]) for class_id in protocol.class_order}
        torch.save(
            {
                "attribute_names": attribute_names,
                "visual_prototypes": attribute_visual,
                "semantic_embeddings": attribute_semantic,
                "associations": associations,
            },
            output / "checkpoints" / "attribute_pool.pt",
        )
        mpc = MPCNetwork(feature_dim=64, semantic_dim=64, hidden_dim=128).to(device)
        mpc_trace = train_mpc_episodic(
            mpc,
            {class_id: feature_cache["train_features"][class_id] for class_id in base_classes},
            class_semantics,
            attribute_visual,
            attribute_semantic,
            int(args.mpc_epochs),
            float(args.mpc_lr),
            int(args.seed),
            device,
            k_shot=5,
        )
        write_csv(output / "csv" / "mpc_train_trace.csv", mpc_trace)
        records = []
        for class_id in base_classes:
            mean, covariance = class_statistics(feature_cache["train_features"][class_id])
            records.append(PrototypeRecord(class_id, 0, mean, covariance, mean.clone(), int(protocol.class_counts[class_id]), groups[class_id], False))
            exemplar_state[class_id] = feature_cache["train_paths"][class_id][0]
        projector = DSMProjector(64, int(args.projector_hidden), int(args.projector_dim)).to(device)
        geometry, trace, augmentation = train_projector_session(
            projector,
            records,
            base_classes,
            fixed_geometry,
            args.mode,
            int(args.base_projector_epochs),
            float(args.projector_lr),
            int(args.seed),
            0,
            device,
            float(args.lambda_dsm),
        )
        augmentation_rows.extend(augmentation)
        write_csv(output / "csv" / "projector_train_trace_session0.csv", trace)
        metrics, classes, _, health, rebuilt = evaluate_session(
            args.mode, projector, records, base_classes, feature_cache["test_features"], fixed_geometry, geometry, groups, None, device, float(args.lambda_dsm)
        )
        session_metrics.append({"seed": args.seed, "method": args.mode, "session": 0, "seen_classes": len(records), **metrics, **health})
        per_class_rows.extend({"seed": args.seed, "method": args.mode, "session": 0, **row} for row in classes)
        health_rows.append({"seed": args.seed, "method": args.mode, "session": 0, **health})
        mapping = {record.class_id: head for head, record in enumerate(records)}
        write_json(output / "manifests" / "class_mapping_session0.json", mapping)
        write_json(output / "manifests" / "exemplar_state_session0.json", exemplar_state)
        write_json(
            output / "manifests" / "repository_session0.json",
            [
                {
                    "class_id": record.class_id,
                    "session_id": record.session_id,
                    "frequency": record.frequency,
                    "frequency_group": record.frequency_group,
                    "calibrated": record.calibrated,
                    "mean_sha256": tensor_sha256(record.mean),
                    "covariance_sha256": tensor_sha256(record.covariance_diag),
                    "covariance_norm": float(torch.linalg.vector_norm(record.covariance_diag)),
                }
                for record in records
            ],
        )
        checkpoint_path = output / "checkpoints" / "session_0.pt"
        previous_sha = save_checkpoint(checkpoint_path, 0, args.mode, projector, mpc, records, rebuilt, mapping, exemplar_state, None, base_checkpoint, backbone_sha, config)
        previous_path = checkpoint_path
        previous_geometry = rebuilt
        checkpoint_rows.append({"session": 0, "path": str(checkpoint_path), "sha256": previous_sha, "previous_sha256": None})
        continuity_rows.append({"session": 0, "status": "base_created", "checkpoint_sha256": previous_sha})
        next_session = 1

    class_semantics = {class_id: class_semantic_embedding(attribute_mapping[class_id]) for class_id in protocol.class_order}
    base_records = [record for record in records if record.session_id == 0]
    for session in range(next_session, int(args.end_session) + 1):
        payload, projector, mpc, records = restore_checkpoint(previous_path, previous_sha, device)
        exemplar_state = {int(key): str(value) for key, value in payload["exemplar_state"].items()}
        previous_seen = [record.class_id for record in records]
        previous_mapping = {int(k): int(v) for k, v in payload["class_to_head"].items()}
        current_classes = list(map(int, protocol.tasks[session]))
        for class_id in current_classes:
            raw_mean, observed_covariance = class_statistics(feature_cache["train_features"][class_id])
            if args.mode == "locked_nc":
                calibrated = raw_mean.clone()
                covariance = observed_covariance.clone()
                weights = torch.zeros(len(base_records))
                calibration = {"displacement": 0.0, "raw_calibrated_cosine": 1.0}
            else:
                calibrated, calibration = calibrate_prototype(
                    mpc,
                    raw_mean,
                    class_semantics[class_id],
                    attribute_visual,
                    attribute_semantic,
                    float(args.mpc_alpha),
                    device,
                )
                covariance, weights = blend_current_covariance(
                    observed_covariance,
                    calibrated,
                    base_records,
                    float(args.covariance_gamma),
                    float(args.covariance_beta),
                )
            records.append(PrototypeRecord(class_id, session, calibrated, covariance, raw_mean, int(protocol.class_counts[class_id]), groups[class_id], True))
            exemplar_state[class_id] = feature_cache["train_paths"][class_id][0]
            calibration_rows.append(
                {
                    "seed": args.seed,
                    "method": args.mode,
                    "session": session,
                    "class_id": class_id,
                    **calibration,
                    "raw_full_center_cosine": 1.0,
                    "calibrated_full_center_cosine": float(F.cosine_similarity(calibrated, raw_mean, dim=0)),
                    "base_covariance_weight_max": float(weights.max()) if weights.numel() else 0.0,
                    "mpc_applied": args.mode != "locked_nc",
                }
            )
        current_mapping = {record.class_id: head for head, record in enumerate(records)}
        validate_continuity(previous_seen, [record.class_id for record in records], previous_mapping, current_mapping)
        geometry, trace, augmentation = train_projector_session(
            projector,
            records,
            current_classes,
            fixed_geometry,
            args.mode,
            int(args.increment_projector_epochs),
            float(args.projector_lr),
            int(args.seed),
            session,
            device,
            float(args.lambda_dsm),
        )
        augmentation_rows.extend(augmentation)
        write_csv(output / "csv" / f"projector_train_trace_session{session}.csv", trace)
        metrics, classes, _, health, rebuilt = evaluate_session(
            args.mode,
            projector,
            records,
            current_classes,
            feature_cache["test_features"],
            fixed_geometry,
            geometry,
            groups,
            previous_geometry,
            device,
            float(args.lambda_dsm),
        )
        session_metrics.append({"seed": args.seed, "method": args.mode, "session": session, "seen_classes": len(records), **metrics, **health})
        per_class_rows.extend({"seed": args.seed, "method": args.mode, "session": session, **row} for row in classes)
        health_rows.append({"seed": args.seed, "method": args.mode, "session": session, **health})
        checkpoint_path = output / "checkpoints" / f"session_{session}.pt"
        write_json(output / "manifests" / f"class_mapping_session{session}.json", current_mapping)
        write_json(output / "manifests" / f"exemplar_state_session{session}.json", exemplar_state)
        write_json(
            output / "manifests" / f"repository_session{session}.json",
            [
                {
                    "class_id": record.class_id,
                    "session_id": record.session_id,
                    "frequency": record.frequency,
                    "frequency_group": record.frequency_group,
                    "calibrated": record.calibrated,
                    "mean_sha256": tensor_sha256(record.mean),
                    "covariance_sha256": tensor_sha256(record.covariance_diag),
                    "covariance_norm": float(torch.linalg.vector_norm(record.covariance_diag)),
                }
                for record in records
            ],
        )
        checkpoint_sha = save_checkpoint(
            checkpoint_path, session, args.mode, projector, mpc, records, rebuilt, current_mapping, exemplar_state, previous_sha, base_checkpoint, backbone_sha, config
        )
        checkpoint_rows.append({"session": session, "path": str(checkpoint_path), "sha256": checkpoint_sha, "previous_sha256": previous_sha})
        continuity_rows.append(
            {
                "session": session,
                "status": "passed",
                "loaded_checkpoint": str(previous_path),
                "loaded_sha256": previous_sha,
                "output_checkpoint": str(checkpoint_path),
                "output_sha256": checkpoint_sha,
                "seen_classes_before": len(previous_seen),
                "seen_classes_after": len(records),
                "mapping_prefix_preserved": True,
                "repository_prefix_preserved": True,
                "projector_inherited": True,
            }
        )
        previous_path = checkpoint_path
        previous_sha = checkpoint_sha
        previous_geometry = rebuilt

    write_csv(output / "csv" / "per_session_metrics.csv", session_metrics)
    write_csv(output / "csv" / "per_class_metrics.csv", per_class_rows)
    write_csv(output / "csv" / "module_health.csv", health_rows)
    write_csv(output / "csv" / "augmentation_health.csv", augmentation_rows)
    write_csv(output / "csv" / "prototype_calibration.csv", calibration_rows)
    write_csv(output / "manifests" / "checkpoint_manifest.csv", checkpoint_rows)
    write_csv(output / "manifests" / "continuity_checks.csv", continuity_rows)
    write_json(
        output / "json" / "final_results.json",
        {
            "config": config,
            "attribute_coverage": coverage,
            "attribute_names": attribute_names,
            "attribute_associations": associations,
            "session_metrics": session_metrics,
            "module_health": health_rows,
            "checkpoint_manifest": checkpoint_rows,
            "continuity_checks": continuity_rows,
            "successful_session_count": len(session_metrics),
            "completed_full_five_session": bool(session_metrics and session_metrics[-1]["session"] == 5),
            "final_checkpoint": str(previous_path),
            "final_checkpoint_sha256": previous_sha,
        },
    )
    print(json.dumps({"status": "ok", "mode": args.mode, "seed": args.seed, "sessions": len(session_metrics), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
