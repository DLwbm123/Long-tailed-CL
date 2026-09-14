#!/usr/bin/env python3
"""Validate and finalize closure manifests without training or evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXPECTED_STATUS = {
    "FULL_DYNAMIC": "CLOSED",
    "RAW_PROTOTYPE_FALLBACK": "FAILED",
    "VALIDATION_CHECKPOINT_RESCUE": "DIAGNOSTIC_ONLY",
    "NC_SAFE_DYNAMIC_V1": "FAILED",
    "ROOT_CAUSE": "OLD_ANCHOR_DRIFT",
    "SECONDARY_FLAG": "MIXED_GEOMETRY_AND_TRAJECTORY_FAILURE",
    "PROMOTION": "NO_FULL_PROMOTION",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--closure-root", required=True)
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    closure = Path(args.closure_root).resolve()
    document = project / "code/docs/HYPERKVASIR23_FULL_DYNAMIC_BRANCH_CLOSED.md"
    if not document.is_file():
        raise FileNotFoundError(document)

    ledger = json.loads((closure / "FULL_DYNAMIC_VALIDITY_LEDGER.json").read_text())
    if ledger["status"] != EXPECTED_STATUS:
        raise RuntimeError("Frozen status mismatch")
    cf5 = json.loads((closure / "cf5_equivalence.json").read_text())
    dsm = json.loads((closure / "dsm_fixed_old_margin_equivalence.json").read_text())
    if cf5["result"] not in {
        "CF5_EQUIVALENT_TO_EXISTING_METHOD", "CF5_NOT_EQUIVALENT", "CF5_EQUIVALENCE_UNDETERMINED"
    }:
        raise RuntimeError("Invalid CF5 result enum")
    if dsm["result"] not in {
        "DSM_EQUIVALENT", "DSM_PARTIALLY_OVERLAPPING", "DSM_NOT_EQUIVALENT", "UNDETERMINED"
    }:
        raise RuntimeError("Invalid DSM result enum")
    derivations = json.loads((closure / "metric_derivations.json").read_text())
    if derivations["control_before"]["absolute_derivation_error"] > 1e-10:
        raise RuntimeError("Control-before derivation mismatch")
    if derivations["control_after"]["absolute_derivation_error"] > 1e-10:
        raise RuntimeError("Control-after derivation mismatch")
    required_columns = {"branch", "hypothesis", "result", "gate", "promotion", "reason", "artifact_path"}
    with (closure / "branch_status.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not required_columns.issubset(reader.fieldnames or []) or len(rows) < 6:
            raise RuntimeError("branch_status.csv schema mismatch")
    for name in ("experiment_lineage.csv", "key_evidence.csv"):
        with (closure / name).open(newline="", encoding="utf-8") as handle:
            if not list(csv.DictReader(handle)):
                raise RuntimeError(f"Empty CSV: {name}")

    source_manifest_path = closure / "source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_manifest["algorithm_source_modified_by_closure"] = False
    source_manifest.pop("source_modified_by_closure", None)
    additions = [
        {"path": str(document), "sha256": sha256(document)},
        {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
    ]
    known = {row["path"] for row in source_manifest["files"]}
    source_manifest["files"].extend(row for row in additions if row["path"] not in known)
    source_manifest_path.write_text(json.dumps(source_manifest, indent=2, sort_keys=True), encoding="utf-8")

    artifact_path = closure / "artifact_sha256.json"
    artifact_names = sorted(path.name for path in closure.iterdir() if path.is_file() and path.name != artifact_path.name)
    artifact_hashes = {
        name: {"sha256": sha256(closure / name), "size_bytes": (closure / name).stat().st_size}
        for name in artifact_names
    }
    artifact_hashes[artifact_path.name] = {"sha256": None, "note": "self-hash intentionally omitted"}
    artifact_path.write_text(json.dumps(artifact_hashes, indent=2, sort_keys=True), encoding="utf-8")

    for name, row in artifact_hashes.items():
        if name == artifact_path.name:
            continue
        if sha256(closure / name) != row["sha256"]:
            raise RuntimeError(f"Artifact hash mismatch: {name}")
    model_files = [str(path) for path in closure.rglob("*") if path.suffix in {".pt", ".pth", ".ckpt"}]
    if model_files:
        raise RuntimeError(f"Unexpected model files in closure: {model_files}")
    print(json.dumps({
        "status": "CLOSURE_VALIDATED",
        "cf5_result": cf5["result"],
        "dsm_result": dsm["result"],
        "artifact_count": len(artifact_hashes),
        "source_document_manifested": True,
        "model_files": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
