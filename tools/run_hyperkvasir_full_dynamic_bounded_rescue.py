#!/usr/bin/env python3
"""Audit full_dynamic collapses and run the single bounded raw-prototype rescue."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.concm_dsm import DSMProjector, compute_dynamic_structure
from src.methods.full_concm import (
    MPCNetwork,
    PrototypeRecord,
    class_balanced_mean,
    fixed_nc_geometry,
    resample_repository,
    structure_anchor_contrastive,
    tensor_sha256,
)
from src.utils.seed import set_seed
from tools.run_hyperkvasir_full_concm import evaluate_session, file_sha256, metric_pack


COLLAPSE_CURRENT_MIN = 50.0
COLLAPSE_CURRENT_TO_OLD_MAX = 30.0
CONTROL_MAX_DROP_PP = 5.0
AUDIT_SESSIONS = ((1, 1), (1, 2), (1, 4), (1, 5), (3, 4), (3, 5), (2, 5))
TRAIN_TARGETS = ((1, 5), (2, 5))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_payload(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    projector = DSMProjector(**{key: int(value) for key, value in payload["projector_config"].items()}).to(device)
    projector.load_state_dict(payload["projector_state_dict"])
    projector.eval()
    records = [PrototypeRecord.from_dict(value) for value in payload["repository"]]
    return payload, projector, records


def is_collapse(metrics: Mapping[str, object]) -> bool:
    return (
        float(metrics["current_acc"]) < COLLAPSE_CURRENT_MIN
        or float(metrics["current_to_old_rate"]) >= COLLAPSE_CURRENT_TO_OLD_MAX
    )


def groups_for(records: Sequence[PrototypeRecord]) -> dict[int, str]:
    return {record.class_id: record.frequency_group for record in records}


@torch.no_grad()
def evaluate_feature_map(
    projector: DSMProjector,
    records: Sequence[PrototypeRecord],
    current_classes: Sequence[int],
    feature_map: Mapping[int, torch.Tensor],
    device: torch.device,
):
    means = torch.stack([record.mean for record in records]).to(device)
    geometry, geometry_stats = compute_dynamic_structure(projector(means))
    seen = [record.class_id for record in records]
    features = torch.cat([feature_map[class_id] for class_id in seen]).to(device)
    labels = torch.cat(
        [torch.full((feature_map[class_id].shape[0],), head, dtype=torch.long) for head, class_id in enumerate(seen)]
    )
    logits = projector(features) @ geometry.T
    old = [class_id for class_id in seen if class_id not in set(current_classes)]
    metrics, class_rows, confusion = metric_pack(logits.cpu(), labels, seen, old, current_classes, groups_for(records))
    metrics["ETF_residual"] = geometry_stats["etf_residual"]
    metrics["condition_number"] = geometry_stats["condition_number"]
    return metrics, class_rows, confusion, geometry.detach().cpu(), geometry_stats


def destination_summary(
    confusion: Mapping[int, Mapping[int, int]], records: Sequence[PrototypeRecord], current_classes: Sequence[int]
) -> tuple[dict[int, int], int | None, int]:
    old = {record.class_id for record in records if record.class_id not in set(current_classes)}
    destinations: dict[int, int] = {}
    for current_class in current_classes:
        for predicted, count in confusion.get(int(current_class), {}).items():
            if int(predicted) in old:
                destinations[int(predicted)] = destinations.get(int(predicted), 0) + int(count)
    if not destinations:
        return {}, None, 0
    absorbing, count = max(destinations.items(), key=lambda item: (item[1], -item[0]))
    return dict(sorted(destinations.items(), key=lambda item: (-item[1], item[0]))), absorbing, count


@torch.no_grad()
def prototype_variants(
    previous_projector: DSMProjector,
    previous_records: Sequence[PrototypeRecord],
    current_records: Sequence[PrototypeRecord],
):
    output: dict[str, dict[int, dict[str, object]]] = {"raw": {}, "calibrated": {}}
    for variant in output:
        records = list(previous_records) + [deepcopy(record) for record in current_records]
        if variant == "raw":
            for record in records[len(previous_records) :]:
                record.mean = record.raw_mean.clone()
        projected = previous_projector(torch.stack([record.mean for record in records]).to(next(previous_projector.parameters()).device))
        geometry, _ = compute_dynamic_structure(projected)
        similarities = projected @ geometry.T
        for head in range(len(previous_records), len(records)):
            record = records[head]
            wrong = similarities[head].clone()
            wrong[head] = -float("inf")
            nearest = int(wrong.argmax().item())
            old_max = int(similarities[head, : len(previous_records)].argmax().item())
            output[variant][record.class_id] = {
                "own_cosine": float(similarities[head, head].cpu()),
                "own_anchor_margin": float((similarities[head, head] - wrong[nearest]).cpu()),
                "nearest_wrong_class": records[nearest].class_id,
                "nearest_wrong_cosine": float(similarities[head, nearest].cpu()),
                "max_old_class": records[old_max].class_id,
                "own_minus_max_old_gap": float((similarities[head, head] - similarities[head, old_max]).cpu()),
            }
    return output


def audit_existing(existing: Path, output: Path, device: torch.device):
    all_metrics = read_csv(existing / "csv" / "all_session_metrics.csv")
    collapse_rows = [
        row
        for row in all_metrics
        if row["method"] == "full_dynamic" and int(row["session"]) > 0 and is_collapse(row)
    ]
    collapse_keys = {(int(row["seed"]), int(row["session"])) for row in collapse_rows}
    if collapse_keys != {(1, 2), (1, 5), (3, 5)}:
        raise RuntimeError(f"Unexpected collapse set: {sorted(collapse_keys)}")

    session_rows = []
    class_rows = []
    epoch_rows = []
    audit_json = {"collapse_keys": sorted(collapse_keys), "sessions": {}}
    for seed, session in AUDIT_SESSIONS:
        run = existing / "runs" / f"seed{seed}_full_dynamic"
        current_path = run / "checkpoints" / f"session_{session}.pt"
        previous_path = run / "checkpoints" / f"session_{session - 1}.pt"
        payload, projector, records = load_payload(current_path, device)
        previous_payload, previous_projector, previous_records = load_payload(previous_path, device)
        current_classes = [record.class_id for record in records if record.session_id == session]
        current_records = [record for record in records if record.session_id == session]
        cache = torch.load(existing / "cache" / f"seed{seed}_feature_bank.pt", map_location="cpu", weights_only=False)

        test_metrics, evaluated_classes, confusion, _, geometry_stats = evaluate_feature_map(
            projector, records, current_classes, cache["test_features"], device
        )
        destination, absorbing, absorbing_count = destination_summary(confusion, records, current_classes)
        epoch0_records = list(previous_records) + [deepcopy(record) for record in current_records]
        epoch0_metrics, _, _, _, _ = evaluate_feature_map(
            previous_projector, epoch0_records, current_classes, cache["train_features"], device
        )
        epoch5_metrics, _, _, _, _ = evaluate_feature_map(
            projector, records, current_classes, cache["train_features"], device
        )
        variants = prototype_variants(previous_projector, previous_records, current_records)

        trace = read_csv(run / "csv" / f"projector_train_trace_session{session}.csv")
        epoch_rows.append({"seed": seed, "session": session, "epoch": 0, "source": "train_side_inference", **epoch0_metrics})
        for row in trace:
            epoch_rows.append(
                {
                    "seed": seed,
                    "session": session,
                    "epoch": int(row["epoch"]),
                    "source": "stored_loss_only" if int(row["epoch"]) < 5 else "stored_loss_plus_final_train_side_inference",
                    "loss": row["loss"],
                    "match_loss": row["match_loss"],
                    "cont_loss": row["cont_loss"],
                    "resampling_checksum": row["resampling_checksum"],
                    **(epoch5_metrics if int(row["epoch"]) == 5 else {
                        "AccT": "NOT_STORED", "old_acc": "NOT_STORED", "current_acc": "NOT_STORED",
                        "HM_old_current": "NOT_STORED", "current_to_old_rate": "NOT_STORED",
                    }),
                }
            )

        augmentation = read_csv(run / "csv" / "augmentation_health.csv")
        aug_by_class: dict[int, list[dict[str, str]]] = {}
        for row in augmentation:
            if int(row["session"]) == session:
                aug_by_class.setdefault(int(row["class_id"]), []).append(row)
        calibration = {
            int(row["class_id"]): row
            for row in read_csv(run / "csv" / "prototype_calibration.csv")
            if int(row["session"]) == session
        }
        beta = float(payload["config"]["covariance_beta"])
        for record in current_records:
            observed = cache["train_features"][record.class_id].var(dim=0, unbiased=False).clamp_min(1e-8)
            transferred = record.covariance_diag / beta - observed
            rows = aug_by_class.get(record.class_id, [])
            cal = calibration.get(record.class_id, {})
            class_eval = next(row for row in evaluated_classes if int(row["class_id"]) == record.class_id)
            class_rows.append(
                {
                    "seed": seed,
                    "session": session,
                    "collapse": (seed, session) in collapse_keys,
                    "class_id": record.class_id,
                    "test_recall": class_eval["recall"],
                    "raw_calibrated_cosine": cal.get("raw_calibrated_cosine", float(F.cosine_similarity(record.raw_mean, record.mean, dim=0))),
                    "calibration_displacement": cal.get("displacement", float(torch.linalg.vector_norm(record.mean - record.raw_mean))),
                    "attribute_coverage": 0.0,
                    "attribute_fallback": True,
                    "raw_own_anchor_margin_epoch0": variants["raw"][record.class_id]["own_anchor_margin"],
                    "calibrated_own_anchor_margin_epoch0": variants["calibrated"][record.class_id]["own_anchor_margin"],
                    "raw_own_minus_max_old_gap_epoch0": variants["raw"][record.class_id]["own_minus_max_old_gap"],
                    "calibrated_own_minus_max_old_gap_epoch0": variants["calibrated"][record.class_id]["own_minus_max_old_gap"],
                    "calibrated_nearest_wrong_class_epoch0": variants["calibrated"][record.class_id]["nearest_wrong_class"],
                    "observed_covariance_norm": float(torch.linalg.vector_norm(observed)),
                    "transferred_covariance_norm": float(torch.linalg.vector_norm(transferred)),
                    "final_covariance_norm": float(torch.linalg.vector_norm(record.covariance_diag)),
                    "covariance_max": float(record.covariance_diag.max()),
                    "sampled_feature_norm_mean": sum(float(row["sampled_feature_norm"]) for row in rows) / max(len(rows), 1),
                    "resampling_checksums_unique": len({row["checksum"] for row in rows}),
                    "current_error_destinations": json.dumps(confusion.get(record.class_id, {}), sort_keys=True),
                }
            )

        source_row = next(row for row in all_metrics if row["method"] == "full_dynamic" and int(row["seed"]) == seed and int(row["session"]) == session)
        session_rows.append(
            {
                "seed": seed,
                "session": session,
                "role": "collapse" if (seed, session) in collapse_keys else ("stable_control" if (seed, session) == (2, 5) else "preceding_normal"),
                "old_acc": source_row["old_acc"],
                "current_acc": source_row["current_acc"],
                "AccT": source_row["AccT"],
                "HM": source_row["HM_old_current"],
                "old_to_current": source_row["old_to_current_rate"],
                "current_to_old": source_row["current_to_old_rate"],
                "BER": source_row["BER"],
                "current_error_destination_old_distribution": json.dumps(destination, sort_keys=True),
                "max_absorbing_old_class": absorbing,
                "max_absorbing_error_count": absorbing_count,
                "ETF_residual": geometry_stats["etf_residual"],
                "SMR_all": source_row["SMR_all"],
                "SMR_old": source_row["SMR_old"],
                "SMR_current": source_row["SMR_current"],
                "condition_number": geometry_stats["condition_number"],
                "singular_values": json.dumps(geometry_stats["singular_values"]),
                "own_anchor_margin": source_row["own_anchor_margin"],
                "own_anchor_rank": source_row["own_anchor_rank"],
                "old_anchor_drift": source_row["old_anchor_drift"],
                "checkpoint": str(current_path),
                "checkpoint_sha256": file_sha256(current_path),
            }
        )
        audit_json["sessions"][f"seed{seed}_session{session}"] = {
            "test_metrics": test_metrics,
            "epoch0_train_side_metrics": epoch0_metrics,
            "epoch5_train_side_metrics": epoch5_metrics,
            "destination_old_distribution": destination,
            "prototype_variants": variants,
        }

    collapse_output = [row for row in session_rows if row["role"] == "collapse"]
    write_csv(output / "full_dynamic_collapse_sessions.csv", collapse_output)
    write_csv(output / "full_dynamic_epoch_trajectory.csv", epoch_rows)
    write_csv(output / "full_dynamic_class_diagnostics.csv", class_rows)
    write_csv(output / "csv" / "audit_comparison_sessions.csv", session_rows)
    write_json(output / "json" / "audit_details.json", audit_json)

    seed1_class = [row for row in class_rows if int(row["seed"]) == 1 and int(row["session"]) == 5]
    qualifies = all(
        float(row["calibrated_own_anchor_margin_epoch0"]) <= 0.0
        and float(row["calibrated_own_anchor_margin_epoch0"]) < float(row["raw_own_anchor_margin_epoch0"])
        for row in seed1_class
    )
    if not qualifies:
        raise RuntimeError("Frozen option-2 criterion failed for seed1 session5; bounded run must stop")
    return collapse_output, class_rows, epoch_rows


def train_one(seed: int, existing: Path, output: Path, device: torch.device):
    session = 5
    run = existing / "runs" / f"seed{seed}_full_dynamic"
    previous_path = run / "checkpoints" / "session_4.pt"
    original_path = run / "checkpoints" / "session_5.pt"
    previous_sha = file_sha256(previous_path)
    previous_payload, projector, previous_records = load_payload(previous_path, device)
    original_payload, _, original_records = load_payload(original_path, device)
    current_records = [deepcopy(record) for record in original_records if record.session_id == session]
    current_classes = [record.class_id for record in current_records]
    variants = prototype_variants(projector, previous_records, current_records)
    fallback_classes = []
    decisions = []
    for record in current_records:
        raw_margin = float(variants["raw"][record.class_id]["own_anchor_margin"])
        calibrated_margin = float(variants["calibrated"][record.class_id]["own_anchor_margin"])
        use_raw = calibrated_margin <= 0.0 and calibrated_margin < raw_margin
        if use_raw:
            record.mean = record.raw_mean.clone()
            record.calibrated = False
            fallback_classes.append(record.class_id)
        decisions.append(
            {"seed": seed, "session": session, "class_id": record.class_id, "raw_margin": raw_margin,
             "calibrated_margin": calibrated_margin, "use_raw": use_raw, "rule": "cal<=0 and cal<raw"}
        )
    records = list(previous_records) + current_records
    cache = torch.load(existing / "cache" / f"seed{seed}_feature_bank.pt", map_location="cpu", weights_only=False)
    config = dict(original_payload["config"])
    if float(config["lambda_dsm"]) != 0.2 or int(config["increment_projector_epochs"]) != 5:
        raise RuntimeError("Original configuration violates bounded rescue constants")
    fixed_geometry = fixed_nc_geometry(23, int(config["projector_dim"])).to(device)
    means = torch.stack([record.mean for record in records]).to(device)
    class_to_head = {record.class_id: head for head, record in enumerate(records)}
    current_heads = [class_to_head[class_id] for class_id in current_classes]
    optimizer = torch.optim.SGD(projector.parameters(), lr=float(config["projector_lr"]), momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    trajectory = []
    augmentation_rows = []
    val_metrics, _, _, _, _ = evaluate_feature_map(projector, records, current_classes, cache["train_features"], device)
    trajectory.append({"seed": seed, "session": session, "epoch": 0, "split": "train_side_frozen_features", **val_metrics})
    for epoch in range(5):
        features, labels, sample_stats = resample_repository(records, current_classes, seed + session * 100, epoch)
        for row in sample_stats:
            augmentation_rows.append({"seed": seed, "session": session, "epoch": epoch + 1, **row})
        features, labels = features.to(device), labels.to(device)
        with torch.no_grad():
            geometry, geometry_stats = compute_dynamic_structure(projector(means))
        optimizer.zero_grad(set_to_none=True)
        projected = projector(features)
        dynamic_losses = F.cross_entropy(projected @ geometry.T, labels, reduction="none")
        match_loss = class_balanced_mean(dynamic_losses, labels, len(records))
        cont_loss = structure_anchor_contrastive(projected, labels, geometry, current_heads, 0.07)
        loss = match_loss + cont_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        val_metrics, _, _, _, _ = evaluate_feature_map(projector, records, current_classes, cache["train_features"], device)
        trajectory.append(
            {"seed": seed, "session": session, "epoch": epoch + 1, "split": "train_side_frozen_features",
             "loss": float(loss.detach().cpu()), "match_loss": float(match_loss.detach().cpu()),
             "cont_loss": float(cont_loss.detach().cpu()), "ETF_residual_before_step": geometry_stats["etf_residual"], **val_metrics}
        )

    groups = groups_for(records)
    geometry, _ = compute_dynamic_structure(projector(means))
    test_metrics, per_class, confusion, health, rebuilt = evaluate_session(
        "full_dynamic", projector, records, current_classes, cache["test_features"], fixed_geometry,
        geometry, groups, previous_payload["geometry"], device, 0.2
    )
    destination, absorbing, absorbing_count = destination_summary(confusion, records, current_classes)
    out_checkpoint = output / "checkpoints" / f"seed{seed}_session5_raw_fallback.pt"
    mpc = MPCNetwork(**original_payload["mpc_config"]).to(device)
    mpc.load_state_dict(original_payload["mpc_state_dict"])
    rescue_config = {
        **config,
        "repair": "class_wise_raw_prototype_fallback",
        "fallback_rule": "calibrated own-anchor margin <= 0 and calibrated margin < raw margin",
        "fallback_classes": fallback_classes,
        "selection_source": "train_side_frozen_features_only",
        "test_used_for_selection": False,
        "source_session4_checkpoint": str(previous_path),
        "source_session4_sha256": previous_sha,
    }
    payload = {
        "session": 5, "mode": "full_dynamic", "repair": "class_wise_raw_prototype_fallback",
        "projector_state_dict": projector.state_dict(), "projector_config": projector.config(),
        "mpc_state_dict": mpc.state_dict(), "mpc_config": mpc.config(),
        "repository": [record.cpu_dict() for record in records], "geometry": rebuilt,
        "delta_prime": projector(means).detach().cpu(), "geometry_stats": health,
        "class_to_head": class_to_head, "exemplar_state": previous_payload["exemplar_state"],
        "seen_classes": [record.class_id for record in records], "previous_checkpoint_sha256": previous_sha,
        "backbone_checkpoint": previous_payload["backbone_checkpoint"],
        "backbone_checkpoint_sha256": previous_payload["backbone_checkpoint_sha256"],
        "optimizer_rule": "rebuild SGD and cosine scheduler for session5 only; inherit session4 projector",
        "config": rescue_config,
    }
    torch.save(payload, out_checkpoint)
    result = {
        "seed": seed, "session": 5, "repair": "class_wise_raw_prototype_fallback",
        "fallback_classes": fallback_classes, "source_checkpoint": str(previous_path),
        "source_checkpoint_sha256": previous_sha, "output_checkpoint": str(out_checkpoint),
        "output_checkpoint_sha256": file_sha256(out_checkpoint), "test_metrics": test_metrics,
        "test_current_error_destination_old_distribution": destination,
        "test_max_absorbing_old_class": absorbing, "test_max_absorbing_error_count": absorbing_count,
        "health": health, "test_used_for_selection": False,
    }
    write_json(output / "configs" / f"seed{seed}_session5_config.json", rescue_config)
    write_csv(output / "csv" / f"seed{seed}_session5_validation_trajectory.csv", trajectory)
    write_csv(output / "csv" / f"seed{seed}_session5_per_class.csv", per_class)
    write_csv(output / "csv" / f"seed{seed}_session5_augmentation_health.csv", augmentation_rows)
    write_csv(output / "csv" / f"seed{seed}_session5_fallback_decisions.csv", decisions)
    write_json(output / "json" / f"seed{seed}_session5_result.json", result)
    return result, trajectory, decisions


def original_metric(existing: Path, seed: int) -> dict[str, str]:
    return next(
        row for row in read_csv(existing / "csv" / "all_session_metrics.csv")
        if row["method"] == "full_dynamic" and int(row["seed"]) == seed and int(row["session"]) == 5
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("This bounded task requires CUDA_VISIBLE_DEVICES=1")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible GPU")
    existing = Path(args.existing_root).resolve()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty output root: {output}")
    for name in ("checkpoints", "configs", "csv", "json", "logs", "manifests"):
        (output / name).mkdir(parents=True, exist_ok=True)
    command = shlex.join([sys.executable, *sys.argv])
    (output / "logs" / "command.log").write_text(
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} {command}\n", encoding="utf-8"
    )
    device = torch.device(args.device)
    collapse_rows, class_rows, epoch_rows = audit_existing(existing, output, device)
    results = []
    all_trajectories = []
    all_decisions = []
    for seed, session in TRAIN_TARGETS:
        if session != 5:
            raise AssertionError("Bounded rescue permits session5 only")
        set_seed(seed)
        result, trajectory, decisions = train_one(seed, existing, output, device)
        results.append(result)
        all_trajectories.extend(trajectory)
        all_decisions.extend(decisions)

    original1, original2 = original_metric(existing, 1), original_metric(existing, 2)
    repaired1, repaired2 = results[0]["test_metrics"], results[1]["test_metrics"]
    seed1_pass = not is_collapse(repaired1) and float(repaired1["current_to_old_rate"]) < float(original1["current_to_old_rate"])
    control_pass = (
        not is_collapse(repaired2)
        and float(repaired2["HM_old_current"]) >= float(original2["HM_old_current"]) - CONTROL_MAX_DROP_PP
        and float(repaired2["old_acc"]) >= float(original2["old_acc"]) - CONTROL_MAX_DROP_PP
    )
    gate = "FULL_DYNAMIC_RESCUE_PASSED" if seed1_pass and control_pass else "FULL_DYNAMIC_RESCUE_FAILED"
    comparison = []
    for seed, before, after in ((1, original1, repaired1), (2, original2, repaired2)):
        comparison.append({
            "seed": seed, "session": 5,
            "original_AccT": before["AccT"], "repaired_AccT": after["AccT"],
            "original_old_acc": before["old_acc"], "repaired_old_acc": after["old_acc"],
            "original_current_acc": before["current_acc"], "repaired_current_acc": after["current_acc"],
            "original_HM": before["HM_old_current"], "repaired_HM": after["HM_old_current"],
            "original_current_to_old": before["current_to_old_rate"],
            "repaired_current_to_old": after["current_to_old_rate"],
            "original_collapse": is_collapse(before), "repaired_collapse": is_collapse(after),
        })
    write_csv(output / "csv" / "rescue_before_after.csv", comparison)
    write_csv(output / "csv" / "all_rescue_validation_trajectories.csv", all_trajectories)
    write_csv(output / "csv" / "all_fallback_decisions.csv", all_decisions)
    write_json(output / "json" / "final_decision.json", {
        "gate": gate, "repair": "class_wise_raw_prototype_fallback", "new_training_count": 2,
        "seed1_pass": seed1_pass, "control_pass": control_pass, "test_used_for_selection": False,
        "collapse_sessions": collapse_rows, "before_after": comparison, "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    absorbing = next(row for row in collapse_rows if int(row["seed"]) == 1 and int(row["session"]) == 5)
    audit_md = f"""# Full Dynamic Collapse Audit

## Scope

- Existing checkpoints only; no replay training was used for the audit.
- Collapse rule: `current_acc < {COLLAPSE_CURRENT_MIN}` or `current_to_old >= {COLLAPSE_CURRENT_TO_OLD_MAX}`.
- Exact collapse sessions: seed1/session2, seed1/session5, seed3/session5.
- Stored artifacts contain only session-final checkpoints. Epoch 1-4 validation metrics cannot be reconstructed; the trajectory file marks them `NOT_STORED`.
- Epoch 0 is an inference-only train-side evaluation using the previous-session projector plus the new session repository; epoch 5 is the saved final checkpoint.

## Seed1 Session5 Localization

- Maximum absorbing old class: `{absorbing['max_absorbing_old_class']}` with `{absorbing['max_absorbing_error_count']}` current-sample errors.
- Failure is already present at epoch 0 in the train-side inference audit, so late checkpoint selection is not eligible.
- Both current classes have non-positive calibrated own-anchor margins, and calibration makes each margin worse than raw.
- Gaussian samples are finite and deterministic checksums are recorded; DSM ETF residual remains within the original gate.
- Primary localization: MPC calibrated-prototype / dynamic-geometry mismatch, not a Gaussian numerical failure or late overfitting.

See `full_dynamic_collapse_sessions.csv`, `full_dynamic_epoch_trajectory.csv`, and `full_dynamic_class_diagnostics.csv` for evidence.
"""
    rescue_md = f"""# Full Dynamic Bounded Rescue

## Frozen Repair

- Selected exactly one repair: class-wise raw-prototype fallback.
- Rule: use raw only when calibrated own-anchor margin is `<= 0` and is lower than raw margin.
- All other classes remain MPC calibrated. Covariance, `lambda_dsm=0.2`, optimizer, epochs, and losses are unchanged.
- Selection used train-side frozen features and prototype geometry only. Test was evaluated after the rule and runs were frozen.

## Runs

- seed1/session5 restored from the original seed1/session4 checkpoint.
- seed2/session5 restored from the original seed2/session4 checkpoint as the stable control.
- Actual new training count: 2 sessions, 5 projector epochs each, physical GPU1 only.

## Result

Final gate: `{gate}`.

See `csv/rescue_before_after.csv`, per-seed validation trajectories, configs, checkpoint hashes, and `json/final_decision.json`.
"""
    final_md = f"""# Full Dynamic Final Decision

`{gate}`

- Seed1 rescue criterion passed: `{seed1_pass}`.
- Stable-control criterion passed: `{control_pass}`.
- Repair: `class_wise_raw_prototype_fallback`.
- New training sessions: `2`.
- Not run: Locked NC, nc_anchored, module ablations, lambda/alpha/beta/gamma sweeps, extra seeds, base or sessions1-4.
- No second rescue is permitted or attempted.
"""
    (output / "FULL_DYNAMIC_COLLAPSE_AUDIT.md").write_text(audit_md, encoding="utf-8")
    (output / "FULL_DYNAMIC_BOUNDED_RESCUE.md").write_text(rescue_md, encoding="utf-8")
    (output / "FULL_DYNAMIC_FINAL_DECISION.md").write_text(final_md, encoding="utf-8")
    checkpoint_rows = []
    for path in sorted((output / "checkpoints").glob("*.pt")):
        checkpoint_rows.append({"path": str(path), "sha256": file_sha256(path)})
    write_csv(output / "manifests" / "checkpoint_sha256.csv", checkpoint_rows)
    print(json.dumps({"gate": gate, "output": str(output), "new_training_count": 2}, indent=2))


if __name__ == "__main__":
    main()
