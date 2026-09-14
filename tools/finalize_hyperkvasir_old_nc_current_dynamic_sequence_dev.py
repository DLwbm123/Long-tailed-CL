#!/usr/bin/env python3
"""Finalize the stopped seed1 OLCD sequence gate without training or test access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


REQUIRED = (
    "method_frozen.json",
    "implementation_equivalence.json",
    "common_base_manifest.csv",
    "matched_run_manifest.csv",
    "per_session_metrics.csv",
    "per_seed_summary.csv",
    "transition_audit.csv",
    "old_anchor_hash_audit.csv",
    "checkpoint_restore_audit.csv",
    "three_seed_comparison.csv",
    "gate.json",
    "report.md",
    "source_manifest.json",
    "artifact_sha256.json",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash_audit(rows):
    references = {}
    for row in rows:
        if row["block"] == "old":
            references.setdefault((row["method"], int(row["class_id"])), row["anchor_sha256"])
    for row in rows:
        key = (row["method"], int(row["class_id"]))
        if row["block"] == "old":
            reference = references[key]
            row["bitwise_frozen_reference_sha256"] = reference
            row["bitwise_frozen_match"] = str(row["anchor_sha256"] == reference)
        else:
            row["bitwise_frozen_reference_sha256"] = ""
            row["bitwise_frozen_match"] = ""
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--command-log", required=True)
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    root_hash_rows = stable_hash_audit(read_csv(root / "old_anchor_hash_audit.csv"))
    write_csv(root / "old_anchor_hash_audit.csv", root_hash_rows)
    for method in ("locked_nc_matched", "old_nc_current_dynamic_v1"):
        path = root / f"seed1/{method}/old_anchor_hash_audit.csv"
        write_csv(path, stable_hash_audit(read_csv(path)))

    olcd_old = [
        row for row in root_hash_rows
        if row["method"] == "old_nc_current_dynamic_v1" and row["block"] == "old"
    ]
    old_frozen = bool(olcd_old) and all(
        row["source"] == "canonical_nc"
        and row["requires_grad"] == "False"
        and row["in_optimizer"] == "False"
        and row["bitwise_frozen_match"] == "True"
        for row in olcd_old
    )
    gate = read_json(root / "gate.json")
    seed_gate = gate["seed_gates"]["1"]
    seed_gate["checks"]["old_anchor_hash_frozen"] = old_frozen
    seed_gate["passed"] = all(seed_gate["checks"].values())
    seed_gate["status"] = "OLD_NC_CURRENT_DYNAMIC_SEED1_FAILED"
    gate["status"] = "OLD_NC_CURRENT_DYNAMIC_SEED1_FAILED / CLOSE_OLCD_BRANCH / NO_SEED2_OR_SEED3"
    gate["hash_audit_correction"] = {
        "reason": "original expected hash used scalar instead of matrix-row normalization on GPU",
        "original_fields_preserved": True,
        "gate_uses": "per-class hash stability across every old epoch/session",
        "old_anchor_bitwise_frozen": old_frozen,
    }
    write_json(root / "gate.json", gate)
    write_json(root / "seed1/seed_gate.json", seed_gate)

    metrics = read_csv(root / "per_session_metrics.csv")
    locked = [row for row in metrics if row["method"] == "locked_nc_matched"]
    olcd = [row for row in metrics if row["method"] == "old_nc_current_dynamic_v1"]
    olcd_s5 = next(row for row in olcd if int(row["session"]) == 5)
    transitions = [
        float(row["accuracy_delta"])
        for row in read_csv(root / "transition_audit.csv")
        if row.get("accuracy_delta") not in (None, "")
    ]
    common = read_csv(root / "common_base_manifest.csv")[0]
    coverage_gaps = common.get("validation_coverage_gaps", "")
    checkpoint_rows = read_csv(root / "checkpoint_restore_audit.csv")
    restore_pass = sum(row["passed"] == "True" for row in checkpoint_rows)
    report = f"""# Old NC / Current Dynamic Sequence Development

## Final Gate

`OLD_NC_CURRENT_DYNAMIC_SEED1_FAILED / CLOSE_OLCD_BRANCH / NO_SEED2_OR_SEED3`

- Stage 0 exact CF5 reproduction: `CF5_IMPLEMENTATION_EQUIVALENT` with zero logit, prediction, metric, class-order, and anchor-hash mismatch at seed1/session5 epochs 2 and 3.
- Seed1 OLCD selected checkpoints: `{len(olcd)}/5`; current-side collapse count: `{sum(row['collapse'] == 'True' for row in olcd)}/5`.
- Matched Locked NC selected checkpoints: `{len(locked)}/5`; session2 had no eligible epoch, so sessions3-5 and a complete five-session paired comparison are unavailable.
- Seed2/3: not run by the frozen seed-order gate.
- Project test: not accessed. No full-data refit and no fixed-old hard-margin.

## Findings

1. The new implementation exactly reproduces Stage A CF5.
2. Old anchors are bitwise frozen: `{old_frozen}`. Every old row is non-grad, absent from the optimizer, sourced from canonical NC, and has one stable per-class SHA256 across all saved epochs/sessions.
3. The current dynamic path is active: `{seed_gate['checks']['current_dynamic_path_active']}`; saved current rows differ from canonical NC.
4. The current-to-old anchor transition has a severe cliff: mean available-class delta `{sum(transitions) / len(transitions):.6f}pp`, worst delta `{min(transitions):.6f}pp`.
5. Seed1/session5 does not trigger the current-side collapse predicate at selected epoch `{olcd_s5['selected_epoch']}`: current `{float(olcd_s5['current_accuracy']):.6f}`, current-to-old `{float(olcd_s5['current_to_old']):.6f}%`. It nevertheless has old accuracy `{float(olcd_s5['old_accuracy']):.6f}` and HM `{float(olcd_s5['HM']):.6f}`, which is severe old-class degradation.
6. OLCD does not establish protection of old accuracy/HM. Its selected old accuracies are `{', '.join(f"{float(row['old_accuracy']):.3f}" for row in olcd)}` and HMs are `{', '.join(f"{float(row['HM']):.3f}" for row in olcd)}`.
7. Current-side accuracy is often high only before/without effective session training; sessions2-5 selected epoch0, so there is no stable trained current-side gain.
8. A 2/3-seed benefit cannot be established because the frozen seed1 gate failed.
9. The method must not enter confirmatory evaluation.
10. Failure is localized primarily to anchor transition and old-class degradation. The matched Locked baseline also became selector-ineligible at session2, so no valid full paired protection claim is possible.

## Audit Completeness

- Fresh restore: `{restore_pass}/{len(checkpoint_rows)}` checkpoints passed.
- Saved full-state checkpoints: `{len(checkpoint_rows)}` (`12` Locked through failed session2; `30` OLCD through session5).
- Common base SHA256: `{common['sha256']}`.
- Official fold0 validation coverage gaps: `{coverage_gaps}`; absent classes remain unavailable and were never replaced with project test samples.
- GPU: physical GPU0 only, shared-mode free-VRAM gate passed; no OOM and no process termination commands.
"""
    (root / "report.md").write_text(report, encoding="utf-8")

    command_log = Path(args.command_log)
    logs = root / "command_logs"
    logs.mkdir(exist_ok=True)
    (logs / "sequence_dev.log").write_bytes(command_log.read_bytes())

    source = read_json(root / "source_manifest.json")
    source["finalized_utc"] = datetime.now(timezone.utc).isoformat()
    source["finalizer"] = {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())}
    source["hash_audit_correction_is_report_only"] = True
    source["checkpoints_modified"] = False
    source["metrics_modified"] = False
    write_json(root / "source_manifest.json", source)

    validation = {
        "required_paths": {}, "json_valid": {}, "csv_valid": {},
        "checkpoint_sha256_valid": True, "checkpoint_count": len(checkpoint_rows),
        "test_accessed": False, "training_started_during_finalization": False,
    }
    for name in REQUIRED:
        validation["required_paths"][name] = (root / name).is_file()
    for path in root.rglob("*.json"):
        read_json(path)
        validation["json_valid"][str(path.relative_to(root))] = True
    for path in root.rglob("*.csv"):
        read_csv(path)
        validation["csv_valid"][str(path.relative_to(root))] = True
    for row in checkpoint_rows:
        path = (
            root / f"seed{row['seed']}" / row["method"] / f"session{row['session']}"
            / "checkpoints" / f"epoch_{row['epoch']}.pt"
        )
        if not path.is_file() or sha256(path) != row["checkpoint_sha256"]:
            validation["checkpoint_sha256_valid"] = False
    if not all(validation["required_paths"].values()) or not validation["checkpoint_sha256_valid"]:
        raise RuntimeError("Final artifact validation failed")
    write_json(root / "finalization_validation.json", validation)

    artifacts = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_sha256.json":
            artifacts.append({"path": str(path.relative_to(root)), "sha256": sha256(path), "size": path.stat().st_size})
    write_json(root / "artifact_sha256.json", artifacts)
    print(json.dumps({"status": gate["status"], "old_anchor_bitwise_frozen": old_frozen,
                      "checkpoint_restore": f"{restore_pass}/{len(checkpoint_rows)}"}, indent=2))


if __name__ == "__main__":
    main()
