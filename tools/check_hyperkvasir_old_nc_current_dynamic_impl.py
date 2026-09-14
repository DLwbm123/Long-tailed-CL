#!/usr/bin/env python3
"""Reproduce Stage-A CF5 with the independent OLCD anchor assembler."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.concm_dsm import DSMProjector
from src.methods.full_concm import PrototypeRecord, fixed_nc_geometry
from src.methods.old_nc_current_dynamic import assemble_old_nc_current_dynamic_anchors, tensor_sha256


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_csv(path: Path, rows) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validation_features(cache, manifest, seed: int, session: int):
    selected = {}
    for row in manifest:
        if int(row["seed"]) == seed and int(row["session"]) == session:
            selected.setdefault(int(row["class_id"]), set()).add(row["basename"])
    output = {}
    for class_id, names in selected.items():
        indices = [index for index, path in enumerate(cache["train_paths"][class_id]) if Path(path).name in names]
        output[class_id] = cache["train_features"][class_id][indices].float()
    return output


def metric(logits, labels, old_heads):
    predictions = logits.argmax(dim=1)
    old = torch.tensor([int(value) in set(old_heads) for value in predictions.tolist()])
    accuracy = float(predictions.eq(labels).float().mean()) * 100.0
    current_to_old = float(old.float().mean()) * 100.0
    return {"current_micro_accuracy": accuracy, "current_to_old": current_to_old}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", required=True)
    parser.add_argument("--existing-root", required=True)
    parser.add_argument("--stage-a-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    v2 = Path(args.v2_root).resolve()
    existing = Path(args.existing_root).resolve()
    stage_a = Path(args.stage_a_root).resolve()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    stage_a_metrics = read_csv(stage_a / "counterfactual_epoch_metrics.csv")
    manifest = read_csv(v2 / "validation_manifest.csv")
    cache = torch.load(existing / "cache/seed1_feature_bank.pt", map_location="cpu", weights_only=False)
    validation = validation_features(cache, manifest, 1, 5)
    sample_rows = []
    class_rows = []
    provenance = []
    epoch_results = []
    checks = []
    for epoch in (2, 3):
        checkpoint = v2 / f"seed1/session5/checkpoints/epoch_{epoch}.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        projector = DSMProjector(**payload["projector_config"])
        projector.load_state_dict(payload["projector_state_dict"])
        projector.eval()
        records = [PrototypeRecord.from_dict(row) for row in payload["repository"]]
        class_order = [record.class_id for record in records]
        current_classes = [record.class_id for record in records if record.session_id == 5]
        current_heads = [class_order.index(class_id) for class_id in current_classes]
        old_heads = [head for head in range(len(class_order)) if head not in current_heads]
        features = torch.cat([validation[class_id] for class_id in current_classes])
        labels = torch.cat([
            torch.full((validation[class_id].shape[0],), class_order.index(class_id), dtype=torch.long)
            for class_id in current_classes
        ])
        projected = projector(features)
        dynamic = payload["dynamic_anchors"].float()
        nc = fixed_nc_geometry(23, projected.shape[1])[: len(class_order)]
        reference = dynamic.clone()
        reference[old_heads] = nc[old_heads]
        reference = F.normalize(reference, dim=1)
        assembled, metadata = assemble_old_nc_current_dynamic_anchors(nc, dynamic, class_order, current_classes)
        reference_logits = F.normalize(projected, dim=1) @ reference.T
        implementation_logits = F.normalize(projected, dim=1) @ assembled.T
        reference_predictions = reference_logits.argmax(dim=1)
        implementation_predictions = implementation_logits.argmax(dim=1)
        reference_metric = metric(reference_logits, labels, old_heads)
        implementation_metric = metric(implementation_logits, labels, old_heads)
        stage_row = next(
            row for row in stage_a_metrics
            if int(row["seed"]) == 1 and int(row["session"]) == 5 and int(row["epoch"]) == epoch
            and row["counterfactual"] == "CF5_nc_old_plus_dynamic_current"
        )
        max_logit_diff = float((reference_logits - implementation_logits).abs().max())
        prediction_mismatch = int(reference_predictions.ne(implementation_predictions).sum())
        metric_diff = max(
            abs(reference_metric["current_micro_accuracy"] - implementation_metric["current_micro_accuracy"]),
            abs(reference_metric["current_to_old"] - implementation_metric["current_to_old"]),
            abs(float(stage_row["current_micro_accuracy"]) - implementation_metric["current_micro_accuracy"]),
            abs(float(stage_row["current_to_old"]) - implementation_metric["current_to_old"]),
        )
        old_hash_match = tensor_sha256(reference[old_heads]) == tensor_sha256(assembled[old_heads])
        current_hash_match = tensor_sha256(reference[current_heads]) == tensor_sha256(assembled[current_heads])
        class_order_match = metadata["class_order"] == class_order
        epoch_check = {
            "epoch": epoch,
            "max_abs_logit_diff": max_logit_diff,
            "prediction_mismatch_count": prediction_mismatch,
            "metric_diff": metric_diff,
            "class_order_match": class_order_match,
            "old_anchor_hash_match": old_hash_match,
            "current_anchor_hash_match": current_hash_match,
            "passed": max_logit_diff <= 1e-6 and prediction_mismatch == 0 and metric_diff <= 1e-8
            and class_order_match and old_hash_match and current_hash_match,
        }
        checks.append(epoch_check)
        epoch_results.append({**epoch_check, **implementation_metric})
        for index in range(labels.numel()):
            sample_rows.append({
                "epoch": epoch,
                "sample_index": index,
                "true_class": class_order[int(labels[index])],
                "reference_prediction": class_order[int(reference_predictions[index])],
                "implementation_prediction": class_order[int(implementation_predictions[index])],
                "max_class_logit_abs_diff": float((reference_logits[index] - implementation_logits[index]).abs().max()),
            })
        for head, class_id in enumerate(class_order):
            class_rows.append({
                "epoch": epoch,
                "row": head,
                "class_id": class_id,
                "block": "current" if head in current_heads else "old",
                "reference_source": "original_dynamic" if head in current_heads else "canonical_nc",
                "implementation_source": "original_dynamic" if head in current_heads else "canonical_nc",
                "row_hash_match": tensor_sha256(reference[head]) == tensor_sha256(assembled[head]),
            })
        provenance.append({
            "epoch": epoch,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "anchor_metadata": metadata,
            "canonical_nc": {"source": "fixed_nc_geometry(23,128)", "shape": list(nc.shape),
                             "normalized": True, "requires_grad": False, "sha256": tensor_sha256(nc)},
            "original_dynamic": {"source": "saved Full Dynamic checkpoint dynamic_anchors", "shape": list(dynamic.shape),
                                 "normalized": True, "requires_grad": False, "sha256": tensor_sha256(dynamic)},
        })
    passed = all(row["passed"] for row in checks)
    status = "CF5_IMPLEMENTATION_EQUIVALENT" if passed else "CF5_IMPLEMENTATION_MISMATCH"
    result = {
        "status": status,
        "training_started": False,
        "checks": checks,
        "thresholds": {"max_abs_logit_diff": 1e-6, "prediction_mismatch_count": 0, "metric_diff": 1e-8,
                       "class_order_match": True, "old_anchor_hash_match": True, "current_anchor_hash_match": True},
    }
    write_json(output / "implementation_equivalence.json", result)
    write_csv(output / "per_sample_logit_diff.csv", sample_rows)
    write_json(output / "anchor_provenance.json", provenance)
    write_csv(output / "class_order_audit.csv", class_rows)
    write_json(output / "source_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "training_started": False, "test_accessed": False,
        "files": [
            {"path": str(REPO_ROOT / "src/methods/old_nc_current_dynamic.py"),
             "sha256": file_sha256(REPO_ROOT / "src/methods/old_nc_current_dynamic.py")},
            {"path": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__).resolve())},
        ],
    })
    report = f"""# OLCD Stage 0 Implementation Check

- Status: `{status}`.
- Training started: false.
- Project test accessed: false.
- Epoch2/3 checks: `{json.dumps(checks, sort_keys=True)}`.
- The independent implementation assembles one normalized matrix in checkpoint class order, with canonical NC rows for old classes and untouched original Dynamic rows for current-session classes.
"""
    (output / "implcheck_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit("TRAINING_NOT_STARTED")


if __name__ == "__main__":
    main()
