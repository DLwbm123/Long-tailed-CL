#!/usr/bin/env python3
"""Zero-training geometry forensics for HyperKvasir Full Dynamic checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
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
from src.methods.full_concm import PrototypeRecord, fixed_nc_geometry
from tools.run_hyperkvasir_full_concm import file_sha256


CURRENT_ACCURACY_MIN = 50.0
CURRENT_TO_OLD_MAX = 30.0
ANALYSES = (
    *({"seed": 1, "session": 5, "epoch": epoch, "role": "persistent_target"} for epoch in range(6)),
    {"seed": 1, "session": 2, "epoch": 1, "role": "rescued_target_reference"},
    {"seed": 3, "session": 5, "epoch": 5, "role": "rescued_target_reference"},
    {"seed": 2, "session": 5, "epoch": 5, "role": "stable_control"},
)
COUNTERFACTUALS = (
    "CF0_original_full_dynamic",
    "CF1_unit_norm_shared_scale",
    "CF2_raw_prototype_cosine_head",
    "CF3_calibrated_prototype_cosine_head",
    "CF4_nc_anchored_head",
    "CF5_nc_old_plus_dynamic_current",
    "CF6_dynamic_old_plus_nc_current",
    "CF7_class9_reference_replacement",
    "CF8_same_space_dynamic_head",
)


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


def quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.float().cpu(), q)) if values.numel() else float("nan")


def class_name(cache: Mapping[str, object], class_id: int) -> str:
    return Path(cache["train_paths"][int(class_id)][0]).parent.name


def validation_features(cache, manifest, seed: int, session: int):
    selected: dict[int, set[str]] = defaultdict(set)
    for row in manifest:
        if int(row["seed"]) == seed and int(row["session"]) == session:
            selected[int(row["class_id"])].add(row["basename"])
    output = {}
    for class_id, names in selected.items():
        indices = [index for index, path in enumerate(cache["train_paths"][class_id]) if Path(path).name in names]
        if len(indices) != len(names):
            raise RuntimeError(f"Validation manifest mismatch for seed{seed}/session{session}/class{class_id}")
        output[class_id] = cache["train_features"][class_id][indices].float()
    return output


def load_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    projector = DSMProjector(**payload["projector_config"]).to(device)
    projector.load_state_dict(payload["projector_state_dict"])
    projector.eval()
    records = [PrototypeRecord.from_dict(value) for value in payload["repository"]]
    return payload, projector, records


def collapse(micro_accuracy: float, current_to_old: float) -> bool:
    return micro_accuracy < CURRENT_ACCURACY_MIN or current_to_old >= CURRENT_TO_OLD_MAX


def metric_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    seen: Sequence[int],
    current_classes: Sequence[int],
):
    predictions = logits.argmax(dim=1)
    class_to_head = {class_id: head for head, class_id in enumerate(seen)}
    current_heads = [class_to_head[class_id] for class_id in current_classes]
    current_head_set = set(current_heads)
    old_heads = [head for head in range(len(seen)) if head not in current_head_set]
    restricted_predictions = logits[:, current_heads].argmax(dim=1)
    restricted_labels = torch.tensor([current_heads.index(int(value)) for value in labels.tolist()], device=labels.device)
    micro = float(predictions.eq(labels).float().mean().cpu()) * 100.0
    restricted = float(restricted_predictions.eq(restricted_labels).float().mean().cpu()) * 100.0
    per_class = {}
    for class_id, head in zip(current_classes, current_heads):
        mask = labels == head
        per_class[class_id] = float(predictions[mask].eq(labels[mask]).float().mean().cpu()) * 100.0
    macro = sum(per_class.values()) / len(per_class)
    pred_old = torch.tensor([int(value) in set(old_heads) for value in predictions.tolist()], device=labels.device)
    current_to_old = float(pred_old.float().mean().cpu()) * 100.0
    current_to_current_error = float((~pred_old & predictions.ne(labels)).float().mean().cpu()) * 100.0
    correct_logits = logits[torch.arange(labels.numel(), device=labels.device), labels]
    max_old = logits[:, old_heads].max(dim=1).values
    max_other_current_rows = []
    for index, label in enumerate(labels.tolist()):
        alternatives = [head for head in current_heads if head != int(label)]
        max_other_current_rows.append(logits[index, alternatives].max() if alternatives else logits.new_tensor(float("-inf")))
    max_other_current = torch.stack(max_other_current_rows)
    absorption = {int(seen[head]): int(predictions[pred_old].eq(head).sum().cpu()) for head in old_heads}
    absorbing_class, absorbing_count = max(absorption.items(), key=lambda item: (item[1], -item[0]))
    max_absorption_ratio = 100.0 * absorbing_count / max(int(labels.numel()), 1)
    return {
        "current_micro_accuracy": micro,
        "current_macro_accuracy": macro,
        "restricted_current_accuracy": restricted,
        "current_to_old": current_to_old,
        "current_to_current_error_rate": current_to_current_error,
        "max_old_absorption_ratio": max_absorption_ratio,
        "absorbing_old_class_id": absorbing_class,
        "per_class_accuracy": json.dumps(per_class, sort_keys=True),
        "mean_correct_current_vs_max_old_margin": float((correct_logits - max_old).mean().cpu()),
        "p10_correct_current_vs_max_old_margin": quantile(correct_logits - max_old, 0.1),
        "mean_correct_current_vs_max_other_current_margin": float((correct_logits - max_other_current).mean().cpu()),
        "p10_correct_current_vs_max_other_current_margin": quantile(correct_logits - max_other_current, 0.1),
        "collapse": collapse(micro, current_to_old),
        "eligible": not collapse(micro, current_to_old),
        "predictions": predictions,
        "restricted_predictions": restricted_predictions,
        "max_old": max_old,
        "max_other_current": max_other_current,
        "correct_logits": correct_logits,
        "absorption": absorption,
        "old_heads": old_heads,
        "current_heads": current_heads,
    }


def logits_for_counterfactual(
    name: str,
    backbone_features: torch.Tensor,
    projected_features: torch.Tensor,
    projected_raw: torch.Tensor,
    projected_calibrated: torch.Tensor,
    raw_prototypes: torch.Tensor,
    calibrated_prototypes: torch.Tensor,
    dynamic: torch.Tensor,
    recomputed_dynamic: torch.Tensor,
    nc: torch.Tensor,
    previous_dynamic: torch.Tensor,
    class9_head: int | None,
    old_heads: Sequence[int],
    current_heads: Sequence[int],
):
    if name in {"CF0_original_full_dynamic", "CF1_unit_norm_shared_scale"}:
        return F.normalize(projected_features, dim=1) @ F.normalize(dynamic, dim=1).T, "SUPPORTED"
    if name == "CF2_raw_prototype_cosine_head":
        return F.normalize(backbone_features, dim=1) @ F.normalize(raw_prototypes, dim=1).T, "SUPPORTED"
    if name == "CF3_calibrated_prototype_cosine_head":
        return F.normalize(backbone_features, dim=1) @ F.normalize(calibrated_prototypes, dim=1).T, "SUPPORTED"
    if name == "CF4_nc_anchored_head":
        return F.normalize(projected_features, dim=1) @ F.normalize(nc, dim=1).T, "SUPPORTED"
    if name == "CF5_nc_old_plus_dynamic_current":
        anchors = dynamic.clone()
        anchors[old_heads] = nc[old_heads]
        return F.normalize(projected_features, dim=1) @ F.normalize(anchors, dim=1).T, "SUPPORTED"
    if name == "CF6_dynamic_old_plus_nc_current":
        anchors = dynamic.clone()
        anchors[current_heads] = nc[current_heads]
        return F.normalize(projected_features, dim=1) @ F.normalize(anchors, dim=1).T, "SUPPORTED"
    if name == "CF7_class9_reference_replacement":
        if class9_head is None or class9_head >= previous_dynamic.shape[0]:
            return None, "UNSUPPORTED_COUNTERFACTUAL"
        anchors = dynamic.clone()
        anchors[class9_head] = previous_dynamic[class9_head]
        return F.normalize(projected_features, dim=1) @ F.normalize(anchors, dim=1).T, "SUPPORTED"
    if name == "CF8_same_space_dynamic_head":
        return F.normalize(projected_features, dim=1) @ F.normalize(recomputed_dynamic, dim=1).T, "SUPPORTED"
    raise ValueError(name)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", required=True)
    parser.add_argument("--existing-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    v2 = Path(args.v2_root).resolve()
    existing = Path(args.existing_root).resolve()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    validation_manifest = read_csv(v2 / "validation_manifest.csv")
    protocol = json.loads((v2 / "protocol_frozen_v2.json").read_text())
    if protocol["selected_official_fold"] != 0 or protocol["validation_test_overlap"] != 0:
        raise RuntimeError("Frozen fold0 protocol mismatch")

    tensor_metadata = []
    per_sample_rows = []
    epoch_summary_rows = []
    absorption_rows = []
    assignment_rows = []
    displacement_rows = []
    class9_rows = []
    counterfactual_rows = []
    raw_cosines = {}
    calibrated_cosines = {}
    checkpoint_hashes = []

    for analysis in ANALYSES:
        seed, session, epoch = analysis["seed"], analysis["session"], analysis["epoch"]
        key = f"seed{seed}_session{session}_epoch{epoch}"
        session_dir = v2 / ("seed2/session5_control" if (seed, session) == (2, 5) else f"seed{seed}/session{session}")
        checkpoint_path = session_dir / "checkpoints" / f"epoch_{epoch}.pt"
        payload, projector, records = load_checkpoint(checkpoint_path, device)
        checkpoint_hashes.append({"key": key, "path": str(checkpoint_path), "sha256": file_sha256(checkpoint_path)})
        cache = torch.load(existing / "cache" / f"seed{seed}_feature_bank.pt", map_location="cpu", weights_only=False)
        validation = validation_features(cache, validation_manifest, seed, session)
        seen = [record.class_id for record in records]
        current_classes = [record.class_id for record in records if record.session_id == session]
        class_to_head = {class_id: head for head, class_id in enumerate(seen)}
        current_heads = [class_to_head[class_id] for class_id in current_classes]
        old_heads = [head for head in range(len(seen)) if head not in set(current_heads)]
        labels = torch.cat([torch.full((validation[class_id].shape[0],), class_to_head[class_id], dtype=torch.long)
                            for class_id in current_classes]).to(device)
        backbone_features = torch.cat([validation[class_id] for class_id in current_classes]).to(device)
        projected_features = projector(backbone_features)
        raw_prototypes = torch.stack([record.raw_mean for record in records]).to(device)
        calibrated_prototypes = torch.stack([record.mean for record in records]).to(device)
        projected_raw = projector(raw_prototypes)
        projected_calibrated = projector(calibrated_prototypes)
        dynamic = payload["dynamic_anchors"].to(device)
        recomputed_dynamic, _ = compute_dynamic_structure(projected_calibrated)
        nc = fixed_nc_geometry(23, projected_features.shape[1]).to(device)[: len(seen)]
        source_checkpoint = Path(payload["source_checkpoint"])
        source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        previous_dynamic = source_payload["geometry"].to(device)
        class9_head = class_to_head.get(9)
        class9_name = class_name(cache, 9) if 9 in cache["train_paths"] else None

        semantic_rows = (
            ("raw_prototype", "frozen_backbone_feature_space", raw_prototypes, False, "fit-subset arithmetic mean"),
            ("calibrated_prototype", "frozen_backbone_feature_space", calibrated_prototypes, False, "calibrate_prototype"),
            ("projected_feature", "DSM_classifier_space", projected_features, True, "DSMProjector.forward"),
            ("dynamic_anchor", "DSM_classifier_space", dynamic, True, "compute_dynamic_structure"),
            ("nc_anchor", "DSM_classifier_space", nc, True, "fixed_nc_geometry"),
        )
        for role, space, tensor, normalized, source in semantic_rows:
            tensor_metadata.append({
                "checkpoint": key, "semantic_role": role, "coordinate_space": space, "normalized": normalized,
                "source_function": source, "source_checkpoint": str(checkpoint_path), "source_session": session,
                "class_ordering": json.dumps(seen), "dtype": str(tensor.dtype), "shape": list(tensor.shape),
            })

        raw_cosines[key] = (F.normalize(projected_raw, dim=1) @ F.normalize(dynamic, dim=1).T).cpu().numpy()
        calibrated_cosines[key] = (F.normalize(projected_calibrated, dim=1) @ F.normalize(dynamic, dim=1).T).cpu().numpy()
        raw_assign = raw_cosines[key].argmax(axis=1)
        cal_assign = calibrated_cosines[key].argmax(axis=1)
        for head, class_id in enumerate(seen):
            raw_sorted = np.sort(raw_cosines[key][head])[::-1]
            cal_sorted = np.sort(calibrated_cosines[key][head])[::-1]
            assignment_rows.append({
                "checkpoint": key, "class_id": class_id,
                "mpc_assignment_status": "NOT_IMPLEMENTED_IN_MODEL",
                "mpc_cost_matrix_status": "NOT_IMPLEMENTED_IN_MODEL",
                "raw_cosine_argmax_class": seen[int(raw_assign[head])],
                "calibrated_cosine_argmax_class": seen[int(cal_assign[head])],
                "raw_calibrated_cosine_argmax_consistent": bool(raw_assign[head] == cal_assign[head]),
                "raw_cosine_first_second_margin": float(raw_sorted[0] - raw_sorted[1]),
                "calibrated_cosine_first_second_margin": float(cal_sorted[0] - cal_sorted[1]),
            })
        dynamic_nc_cosine = (F.normalize(dynamic, dim=1) * F.normalize(nc, dim=1)).sum(dim=1)
        dynamic_nc_l2 = torch.linalg.vector_norm(F.normalize(dynamic, dim=1) - F.normalize(nc, dim=1), dim=1)
        for head, class_id in enumerate(seen):
            displacement_rows.append({
                "checkpoint": key, "class_id": class_id,
                "dynamic_vs_nc_angle_degrees": float(torch.rad2deg(torch.acos(dynamic_nc_cosine[head].clamp(-1, 1))).cpu()),
                "dynamic_vs_nc_l2": float(dynamic_nc_l2[head].cpu()),
                "is_old": head in old_heads, "is_current": head in current_heads,
            })
        for current_head in current_heads:
            similarities = calibrated_cosines[key][current_head]
            order = np.argsort(similarities[old_heads])[::-1]
            top_old_heads = [old_heads[int(index)] for index in order[:2]]
            class9_rows.append({
                "checkpoint": key, "current_class_id": seen[current_head], "class9_id": 9,
                "class9_name": class9_name, "class9_head": class9_head,
                "class9_similarity_to_calibrated_current": (
                    float(similarities[class9_head]) if class9_head is not None else None
                ),
                "current_top1_old_class": seen[top_old_heads[0]], "current_top1_old_similarity": float(similarities[top_old_heads[0]]),
                "current_top2_old_class": seen[top_old_heads[1]], "current_top2_old_similarity": float(similarities[top_old_heads[1]]),
                "own_similarity": float(similarities[current_head]),
                "own_minus_top_old_margin": float(similarities[current_head] - similarities[top_old_heads[0]]),
                "class9_dynamic_vs_nc_displacement": (
                    float(dynamic_nc_l2[class9_head].cpu()) if class9_head is not None else None
                ),
            })

        original_logits = projected_features @ dynamic.T
        original = metric_from_logits(original_logits, labels, seen, current_classes)
        class9_absorption = 0
        if class9_head is not None:
            class9_absorption = int(original["predictions"].eq(class9_head).sum().cpu())
        old_logits = original_logits[:, old_heads]
        correct_logits = original["correct_logits"]
        epoch_summary_rows.append({
            "checkpoint": key, "seed": seed, "session": session, "epoch": epoch, "role": analysis["role"],
            "global_current_accuracy": original["current_micro_accuracy"],
            "restricted_current_accuracy": original["restricted_current_accuracy"],
            "current_to_old": original["current_to_old"],
            "current_to_current_error_rate": original["current_to_current_error_rate"],
            "mean_current_vs_old_margin": original["mean_correct_current_vs_max_old_margin"],
            "p10_current_vs_old_margin": original["p10_correct_current_vs_max_old_margin"],
            "mean_current_vs_other_current_margin": original["mean_correct_current_vs_max_other_current_margin"],
            "p10_current_vs_other_current_margin": original["p10_correct_current_vs_max_other_current_margin"],
            "class9_absorption_count": class9_absorption,
            "class9_absorption_ratio": 100.0 * class9_absorption / max(labels.numel(), 1),
            "old_block_top_logit_mean": float(old_logits.max(dim=1).values.mean().cpu()),
            "old_block_top_logit_std": float(old_logits.max(dim=1).values.std(unbiased=False).cpu()),
            "current_block_correct_logit_mean": float(correct_logits.mean().cpu()),
            "current_block_correct_logit_std": float(correct_logits.std(unbiased=False).cpu()),
            "old_current_boundary_failure": (
                original["restricted_current_accuracy"] >= CURRENT_ACCURACY_MIN and original["current_micro_accuracy"] < CURRENT_ACCURACY_MIN
            ),
        })
        for old_class, count in original["absorption"].items():
            absorption_rows.append({
                "checkpoint": key, "seed": seed, "session": session, "epoch": epoch,
                "old_class_id": old_class, "absorption_count": count,
                "absorption_ratio": 100.0 * count / max(labels.numel(), 1), "is_class9": old_class == 9,
            })
        sample_index = 0
        for class_id in current_classes:
            for local_index in range(validation[class_id].shape[0]):
                true_head = class_to_head[class_id]
                predicted_head = int(original["predictions"][sample_index])
                old_relative = int(original_logits[sample_index, old_heads].argmax())
                absorbing_head = old_heads[old_relative]
                per_sample_rows.append({
                    "checkpoint": key, "seed": seed, "session": session, "epoch": epoch,
                    "sample_index": local_index, "true_current_class": class_id,
                    "predicted_class": seen[predicted_head],
                    "predicted_block": "old" if predicted_head in old_heads else "current",
                    "correct_current_logit": float(original["correct_logits"][sample_index].cpu()),
                    "max_other_current_logit": float(original["max_other_current"][sample_index].cpu()),
                    "max_old_logit": float(original["max_old"][sample_index].cpu()),
                    "correct_minus_other_current_margin": float((original["correct_logits"][sample_index] - original["max_other_current"][sample_index]).cpu()),
                    "correct_minus_max_old_margin": float((original["correct_logits"][sample_index] - original["max_old"][sample_index]).cpu()),
                    "backbone_feature_norm": float(torch.linalg.vector_norm(backbone_features[sample_index]).cpu()),
                    "classifier_feature_norm": float(torch.linalg.vector_norm(projected_features[sample_index]).cpu()),
                    "correct_anchor_norm": float(torch.linalg.vector_norm(dynamic[true_head]).cpu()),
                    "absorbing_old_class_id": seen[absorbing_head],
                    "absorbing_old_class_logit": float(original_logits[sample_index, absorbing_head].cpu()),
                    "class9_logit": float(original_logits[sample_index, class9_head].cpu()) if class9_head is not None else None,
                    "temperature": None, "shared_scale": 1.0, "bias_contribution": 0.0,
                    "task_block_calibration": False, "old_current_block_scale_difference": False,
                })
                sample_index += 1

        for counterfactual in COUNTERFACTUALS:
            logits, status = logits_for_counterfactual(
                counterfactual, backbone_features, projected_features, projected_raw, projected_calibrated,
                raw_prototypes, calibrated_prototypes, dynamic, recomputed_dynamic, nc, previous_dynamic,
                class9_head, old_heads, current_heads,
            )
            if logits is None:
                counterfactual_rows.append({"checkpoint": key, "seed": seed, "session": session, "epoch": epoch,
                                            "counterfactual": counterfactual, "status": status, "eligible": False})
                continue
            result = metric_from_logits(logits, labels, seen, current_classes)
            counterfactual_rows.append({
                "checkpoint": key, "seed": seed, "session": session, "epoch": epoch,
                "counterfactual": counterfactual, "status": status,
                "current_micro_accuracy": result["current_micro_accuracy"],
                "current_macro_accuracy": result["current_macro_accuracy"],
                "restricted_current_accuracy": result["restricted_current_accuracy"],
                "current_to_old": result["current_to_old"],
                "max_old_absorption_ratio": result["max_old_absorption_ratio"],
                "absorbing_old_class_id": result["absorbing_old_class_id"],
                "per_class_accuracy": result["per_class_accuracy"],
                "mean_correct_current_vs_max_old_margin": result["mean_correct_current_vs_max_old_margin"],
                "p10_correct_current_vs_max_old_margin": result["p10_correct_current_vs_max_old_margin"],
                "collapse": result["collapse"], "eligible": result["eligible"],
            })

    np.save(output / "raw_to_anchor_cosine.npy", raw_cosines, allow_pickle=True)
    np.save(output / "calibrated_to_anchor_cosine.npy", calibrated_cosines, allow_pickle=True)
    np.savez(output / "mpc_cost_matrices.npz", status=np.array(["NOT_IMPLEMENTED_IN_MODEL"]),
             reason=np.array(["MPC is attention-based prototype completion; no assignment cost matrix exists"]))
    write_json(output / "tensor_provenance.json", {"tensors": tensor_metadata, "checkpoints": checkpoint_hashes})
    write_json(output / "space_compatibility.json", {
        "raw_vs_dynamic_anchor": "SPACE_MISMATCH until raw prototype passes through DSMProjector",
        "calibrated_vs_dynamic_anchor": "SPACE_MISMATCH until calibrated prototype passes through DSMProjector",
        "projected_raw_vs_dynamic_anchor": "COMPATIBLE DSM_classifier_space",
        "projected_calibrated_vs_dynamic_anchor": "COMPATIBLE DSM_classifier_space",
        "projected_feature_vs_nc_anchor": "COMPATIBLE DSM_classifier_space",
        "backbone_feature_vs_raw_or_calibrated_prototype": "COMPATIBLE frozen_backbone_feature_space",
    })
    classifier_path = """# Classifier Path

`image -> frozen backbone cached feature [64] -> DSMProjector(normalize -> Linear/ReLU/Linear -> normalize) [128] -> dot(dynamic_anchor.T) -> argmax`

- Raw prototype: arithmetic mean of fit-subset frozen-backbone features, shape `[64]`, not normalized on storage.
- Calibrated prototype: `calibrate_prototype` output in frozen-backbone feature space, shape `[64]`, not normalized on storage.
- Dynamic anchor: `compute_dynamic_structure(projector(calibrated_prototypes))`, centered-SVD, shape `[seen_classes,128]`, normalized.
- NC reference anchor: `fixed_nc_geometry(23,128)[:seen_classes]`, normalized.
- MPC implementation is attention-based completion. There is no matching cost matrix or discrete assignment in this code path.
- Classifier has no temperature, learned/shared scale, bias, task-block calibration, or different old/current block scale. The implicit shared scale is 1.
- Class 9 uses local/global dataset class id 9; head index follows checkpoint `class_mapping` and class order.
"""
    (output / "classifier_path.md").write_text(classifier_path, encoding="utf-8")
    write_csv(output / "epoch_logit_summary.csv", epoch_summary_rows)
    write_csv(output / "absorption_by_epoch_and_class.csv", absorption_rows)
    write_csv(output / "assignment_comparison.csv", assignment_rows)
    write_csv(output / "anchor_displacement.csv", displacement_rows)
    write_csv(output / "class9_geometry.csv", class9_rows)
    write_csv(output / "counterfactual_epoch_metrics.csv", counterfactual_rows)

    try:
        import pandas as pd
        pd.DataFrame(per_sample_rows).to_parquet(output / "per_sample_logit_decomposition.parquet", index=False)
        parquet_status = "OK"
    except Exception as error:
        write_csv(output / "per_sample_logit_decomposition.csv", per_sample_rows)
        parquet_status = f"PARQUET_UNAVAILABLE: {type(error).__name__}: {error}"

    grouped = defaultdict(list)
    for row in counterfactual_rows:
        grouped[(row["seed"], row["session"], row["counterfactual"])].append(row)
    counterfactual_summary = []
    for (seed, session, name), rows in grouped.items():
        supported = [row for row in rows if row["status"] == "SUPPORTED"]
        eligible_epochs = [int(row["epoch"]) for row in supported if bool(row["eligible"])]
        best = max(supported, key=lambda row: (float(row["current_macro_accuracy"]), -float(row["current_to_old"]), -int(row["epoch"]))) if supported else None
        counterfactual_summary.append({
            "seed": seed, "session": session, "counterfactual": name,
            "status": supported[0]["status"] if supported else rows[0]["status"],
            "eligible_epochs": json.dumps(eligible_epochs), "any_eligible": bool(eligible_epochs),
            "diagnostic_best_epoch": int(best["epoch"]) if best else None,
            "diagnostic_best_current_macro_accuracy": best["current_macro_accuracy"] if best else None,
            "diagnostic_best_current_to_old": best["current_to_old"] if best else None,
        })
    write_csv(output / "counterfactual_summary.csv", counterfactual_summary)

    seed1s5 = [row for row in counterfactual_rows if row["seed"] == 1 and row["session"] == 5 and row["status"] == "SUPPORTED"]
    effective = {
        name: sorted(int(row["epoch"]) for row in seed1s5 if row["counterfactual"] == name and bool(row["eligible"]))
        for name in COUNTERFACTUALS
    }
    verdict = "INCONCLUSIVE_HEAD_GEOMETRY"
    if effective["CF1_unit_norm_shared_scale"]:
        verdict = "LOGIT_NORM_OR_SCALE_MISMATCH"
    elif effective["CF5_nc_old_plus_dynamic_current"] and not effective["CF6_dynamic_old_plus_nc_current"]:
        verdict = "OLD_ANCHOR_DRIFT"
    elif effective["CF6_dynamic_old_plus_nc_current"] and not effective["CF5_nc_old_plus_dynamic_current"]:
        verdict = "CURRENT_ANCHOR_PLACEMENT_FAILURE"
    elif effective["CF7_class9_reference_replacement"] and not any(
        effective[name] for name in ("CF4_nc_anchored_head", "CF5_nc_old_plus_dynamic_current", "CF6_dynamic_old_plus_nc_current")
    ):
        verdict = "LOCAL_CLASS9_ABSORPTION"
    elif effective["CF4_nc_anchored_head"] and not any(
        effective[name] for name in ("CF5_nc_old_plus_dynamic_current", "CF6_dynamic_old_plus_nc_current", "CF7_class9_reference_replacement")
    ):
        verdict = "GLOBAL_DYNAMIC_ANCHOR_GEOMETRY_FAILURE"
    elif not any(effective.values()):
        restricted = [row["restricted_current_accuracy"] for row in epoch_summary_rows if row["seed"] == 1 and row["session"] == 5]
        if all(float(value) < CURRENT_ACCURACY_MIN for value in restricted):
            verdict = "FEATURE_OR_OPTIMIZATION_FAILURE"
    flags = []
    rescuing_epochs = sorted({epoch for epochs in effective.values() for epoch in epochs})
    if 0 in rescuing_epochs:
        flags.append("INITIALIZATION_HEAD_FAILURE")
    if rescuing_epochs and 0 not in rescuing_epochs:
        flags.append("MIXED_GEOMETRY_AND_TRAJECTORY_FAILURE")
    anchor_verdicts = {
        "OLD_ANCHOR_DRIFT", "CURRENT_ANCHOR_PLACEMENT_FAILURE", "LOCAL_CLASS9_ABSORPTION",
        "CALIBRATION_MATCHING_MISMATCH", "GLOBAL_DYNAMIC_ANCHOR_GEOMETRY_FAILURE",
    }
    stage_b = verdict in anchor_verdicts or verdict == "LOGIT_NORM_OR_SCALE_MISMATCH"
    recommended = {
        "verdict": verdict,
        "flags": flags,
        "stage_b_justified": stage_b,
        "recommended_single_fix": (
            "unit_normalized_shared_scale_classifier" if verdict == "LOGIT_NORM_OR_SCALE_MISMATCH"
            else "nc_safe_dynamic_v1" if verdict in anchor_verdicts
            else "none"
        ),
        "effective_counterfactual_epochs": effective,
    }
    write_json(output / "causal_verdict.json", {**recommended, "parquet_status": parquet_status})
    write_json(output / "recommended_single_fix.json", recommended)
    write_json(output / "source_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "training_steps": 0, "test_accessed": False,
        "files": [
            {"path": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__).resolve())},
            {"path": str(REPO_ROOT / "src/methods/full_concm.py"), "sha256": file_sha256(REPO_ROOT / "src/methods/full_concm.py")},
            {"path": str(REPO_ROOT / "src/methods/concm_dsm.py"), "sha256": file_sha256(REPO_ROOT / "src/methods/concm_dsm.py")},
        ],
        "checkpoint_inputs": checkpoint_hashes,
    })
    seed1_summary = [row for row in epoch_summary_rows if row["seed"] == 1 and row["session"] == 5]
    report = f"""# Full Dynamic Geometry Forensics

## Scope

- Zero training, zero optimizer steps, no test access, official fold0 current-only validation only.
- Analyzed seed1/session5 epoch0-5 plus seed1/session2 epoch1, seed3/session5 epoch5, and control epoch5.
- Per-sample parquet status: `{parquet_status}`.

## Seed1 Session5 Boundary

- Restricted-current accuracy by epoch: {[round(float(row['restricted_current_accuracy']), 4) for row in seed1_summary]}.
- Global current accuracy by epoch: {[round(float(row['global_current_accuracy']), 4) for row in seed1_summary]}.
- Current-to-old by epoch: {[round(float(row['current_to_old']), 4) for row in seed1_summary]}.
- Old/current boundary failure markers: {[bool(row['old_current_boundary_failure']) for row in seed1_summary]}.
- Errors are primarily old-block absorption when current-to-old dominates; detailed per-class absorption is in `absorption_by_epoch_and_class.csv`.

## Provenance Findings

- Raw/calibrated prototypes are 64-D backbone-feature-space tensors; dynamic/NC anchors are 128-D DSM-classifier-space tensors.
- Prototype-to-anchor comparisons are performed only after the prototype passes through the checkpoint projector.
- No MPC assignment or cost matrix exists in the actual model. Raw/calibrated projected cosine argmax consistency is reported only as a geometry diagnostic.
- Original logits already use unit-normalized features and anchors with shared implicit scale 1, no bias, no temperature, and no block-specific scaling.

## Counterfactual Verdict

- Effective seed1/session5 counterfactual epochs: `{json.dumps(effective, sort_keys=True)}`.
- Causal verdict: `{verdict}`.
- Additional flags: `{flags}`.
- Stage B justified: `{stage_b}`.
- Recommended single fix: `{recommended['recommended_single_fix']}`.
"""
    (output / "geometry_forensics_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(recommended, indent=2))


if __name__ == "__main__":
    main()
