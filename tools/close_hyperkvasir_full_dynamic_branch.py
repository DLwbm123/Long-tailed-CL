#!/usr/bin/env python3
"""Create the read-only HyperKvasir Full Dynamic branch closure ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


FINAL_STATE = {
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


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
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


def old_from_hm_current(hm: float, current: float) -> float:
    denominator = 2.0 * current - hm
    if denominator <= 0:
        raise ValueError("Cannot derive old accuracy from HM/current")
    return hm * current / denominator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty closure output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    original = project / "outputs/hyperkvasir23_full_concm_20260710_gpu1"
    rescue = project / "outputs/hyperkvasir23_full_dynamic_rescue_20260710"
    valckpt = project / "outputs/hyperkvasir23_full_dynamic_valckpt_official5fold_v2_20260726"
    forensics = project / "outputs/hyperkvasir23_full_dynamic_geometry_forensics_20260728"
    nc_safe = project / "outputs/hyperkvasir23_nc_safe_dynamic_microcheck_20260728"
    inputs = [original, rescue, valckpt, forensics, nc_safe]
    missing = [str(path) for path in inputs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing input artifact directories: {missing}")

    original_metrics = read_csv(original / "csv/all_session_metrics.csv")
    original_control = next(
        row for row in original_metrics
        if row["method"] == "full_dynamic" and int(row["seed"]) == 2 and int(row["session"]) == 5
    )
    rescue_gate = json.loads((rescue / "json/final_decision.json").read_text())
    val_gate = json.loads((valckpt / "gate.json").read_text())
    causal = json.loads((forensics / "causal_verdict.json").read_text())
    nc_gate = json.loads((nc_safe / "gate.json").read_text())
    nc_control = json.loads((nc_safe / "seed2/session5_control/fresh_test_metrics.json").read_text())
    nc_target = json.loads((nc_safe / "seed1/session5/fresh_test_metrics.json").read_text())
    epoch_rows = [
        row for row in read_csv(forensics / "epoch_logit_summary.csv")
        if int(row["seed"]) == 1 and int(row["session"]) == 5
    ]
    restricted = [float(row["restricted_current_accuracy"]) for row in epoch_rows]
    control_old_before = old_from_hm_current(float(original_control["HM_old_current"]), float(original_control["current_acc"]))
    control_old_after = old_from_hm_current(float(nc_control["test_HM"]), float(nc_control["test_current_accuracy"]))

    metric_derivations = {
        "harmonic_mean_definition": "HM = 2 * old * current / (old + current)",
        "solved_old_accuracy": "old = HM * current / (2 * current - HM)",
        "control_before": {
            "HM": float(original_control["HM_old_current"]),
            "current": float(original_control["current_acc"]),
            "derived_old": control_old_before,
            "saved_old": float(original_control["old_acc"]),
            "absolute_derivation_error": abs(control_old_before - float(original_control["old_acc"])),
        },
        "control_after": {
            "HM": float(nc_control["test_HM"]),
            "current": float(nc_control["test_current_accuracy"]),
            "derived_old": control_old_after,
            "saved_old": float(nc_control["test_old_accuracy"]),
            "absolute_derivation_error": abs(control_old_after - float(nc_control["test_old_accuracy"])),
        },
        "percentage_point_changes": {
            "AccT": float(nc_control["test_AccT"]) - float(original_control["AccT"]),
            "current": float(nc_control["test_current_accuracy"]) - float(original_control["current_acc"]),
            "HM": float(nc_control["test_HM"]) - float(original_control["HM_old_current"]),
            "old": control_old_after - control_old_before,
        },
    }
    write_json(output / "metric_derivations.json", metric_derivations)

    ledger = {
        "status": FINAL_STATE,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "development_policy": {
            "test_further_method_development": False,
            "threshold_sweeps": False,
            "alpha_sweeps": False,
            "class9_sweeps": False,
            "full_15_session_run": False,
            "recommended_mainline": "Locked NC-ConCM",
        },
        "gates": {
            "raw_prototype_rescue": rescue_gate["gate"],
            "validation_checkpoint": val_gate["gate"],
            "geometry_forensics": causal["verdict"],
            "nc_safe_dynamic": nc_gate["gate"],
        },
        "integrity": {
            "stage_b_all_12_checkpoints_verified": nc_gate["all_12_checkpoints_verified"],
            "oom": False,
            "four_session_confirmation_started": nc_gate["confirmation_sessions_started"],
            "full_15_session_started": nc_gate["full_15_session_started"],
        },
    }
    write_json(output / "FULL_DYNAMIC_VALIDITY_LEDGER.json", ledger)

    lineage = [
        {"order": 1, "stage": "original full_dynamic", "input": "original Full ConCM session checkpoints",
         "output": "3/15 incremental collapse sessions", "gate": "FULL_CONCM_PROMISING_BUT_UNSTABLE",
         "artifact_path": str(original)},
        {"order": 2, "stage": "raw-prototype fallback", "input": "seed1/session5 and stable control",
         "output": "target improved but remained collapse", "gate": rescue_gate["gate"], "artifact_path": str(rescue)},
        {"order": 3, "stage": "official-fold validation checkpoint", "input": "official fold0 current-only validation",
         "output": "2/3 targets rescued; seed1/session5 all epochs collapsed", "gate": val_gate["gate"],
         "artifact_path": str(valckpt)},
        {"order": 4, "stage": "geometry forensics", "input": "zero-training head counterfactuals",
         "output": "CF5 only; OLD_ANCHOR_DRIFT", "gate": causal["verdict"], "artifact_path": str(forensics)},
        {"order": 5, "stage": "nc_safe_dynamic_v1", "input": "two-session bounded microcheck",
         "output": "target test collapse and control regression", "gate": nc_gate["gate"], "artifact_path": str(nc_safe)},
        {"order": 6, "stage": "branch closure", "input": "all saved reports and source audit",
         "output": "Full Dynamic closed", "gate": "FULL_DYNAMIC_CLOSED", "artifact_path": str(output)},
    ]
    write_csv(output / "experiment_lineage.csv", lineage)

    branches = [
        {"branch": "full_dynamic", "hypothesis": "all-class dynamic centered-SVD anchors improve incremental balance",
         "result": "unstable with 3/15 collapses", "gate": "CLOSED", "promotion": "NO_FULL_PROMOTION",
         "reason": "persistent old-block absorption", "artifact_path": str(original)},
        {"branch": "raw_prototype_fallback", "hypothesis": "calibration mismatch is the dominant failure",
         "result": "FAILED", "gate": rescue_gate["gate"], "promotion": "NO_FULL_PROMOTION",
         "reason": "seed1/session5 still collapsed", "artifact_path": str(rescue)},
        {"branch": "validation_checkpoint_selector", "hypothesis": "collapse is avoidable by validation-only epoch selection",
         "result": "DIAGNOSTIC_ONLY", "gate": val_gate["gate"], "promotion": "NO_FULL_PROMOTION",
         "reason": "seed1/session5 had no eligible epoch", "artifact_path": str(valckpt)},
        {"branch": "cf5_head_counterfactual", "hypothesis": "fixed NC old anchors restore the old/current boundary",
         "result": "SUPPORTED_DIAGNOSTIC", "gate": causal["verdict"], "promotion": "BOUNDED_MICROCHECK_ONLY",
         "reason": "CF5 alone rescued validation epochs 2 and 3", "artifact_path": str(forensics)},
        {"branch": "nc_safe_dynamic_v1", "hypothesis": "fit-geometry safety gate approximates CF5 without harming control",
         "result": "FAILED", "gate": nc_gate["gate"], "promotion": "NO_FULL_PROMOTION",
         "reason": "target current-to-old 31.71 and control AccT/HM regression", "artifact_path": str(nc_safe)},
        {"branch": "full_dynamic_closure", "hypothesis": "all bounded repair evidence is sufficient for final decision",
         "result": "CLOSED", "gate": "FULL_DYNAMIC_CLOSED", "promotion": "NO_FULL_PROMOTION",
         "reason": "repair branch exhausted under frozen gates", "artifact_path": str(output)},
    ]
    write_csv(output / "branch_status.csv", branches)

    evidence = [
        {"evidence_id": "E01", "claim": "seed1/session5 restricted-current accuracy stays high",
         "value": f"min={min(restricted):.4f}; max={max(restricted):.4f}", "source": str(forensics / "epoch_logit_summary.csv")},
        {"evidence_id": "E02", "claim": "classifier normalization and scale", "value": "unit normalized; shared scale=1; no bias/temperature/block scale",
         "source": str(forensics / "classifier_path.md")},
        {"evidence_id": "E03", "claim": "CF1 outcome", "value": "no eligible epoch", "source": str(forensics / "counterfactual_summary.csv")},
        {"evidence_id": "E04", "claim": "only effective head counterfactual", "value": "CF5 eligible at epochs 2,3",
         "source": str(forensics / "causal_verdict.json")},
        {"evidence_id": "E05", "claim": "class9 absorption after repair", "value": f"83 -> {nc_target['class9_absorption_count']}",
         "source": str(nc_safe / "seed1/session5/fresh_test_metrics.json")},
        {"evidence_id": "E06", "claim": "seed1/session5 remains test collapse", "value": f"current_to_old={nc_target['test_current_to_old']:.6f}",
         "source": str(nc_safe / "seed1/session5/fresh_test_metrics.json")},
        {"evidence_id": "E07", "claim": "control regression", "value":
         f"AccT {float(original_control['AccT']):.4f}->{nc_control['test_AccT']:.4f}; current {float(original_control['current_acc']):.4f}->{nc_control['test_current_accuracy']:.4f}; HM {float(original_control['HM_old_current']):.4f}->{nc_control['test_HM']:.4f}",
         "source": str(nc_safe / "seed2/session5_control/fresh_test_metrics.json")},
        {"evidence_id": "E08", "claim": "control old accuracy", "value": f"{control_old_before:.4f}->{control_old_after:.4f}",
         "source": str(output / "metric_derivations.json")},
        {"evidence_id": "E09", "claim": "Stage B checkpoint integrity", "value": "12/12 SHA256 and restore passed",
         "source": str(nc_safe / "gate.json")},
        {"evidence_id": "E10", "claim": "bounded execution", "value": "no OOM; no four-session confirmation; no full 15-session",
         "source": str(nc_safe / "gate.json")},
    ]
    write_csv(output / "key_evidence.csv", evidence)

    cf5_json = {
        "result": "CF5_NOT_EQUIVALENT",
        "cf5_definition": "old classes use fixed NC/reference anchors; current classes use original dynamic anchors",
        "audited_methods": {
            "Locked NC-ConCM": {
                "equivalent": False,
                "old_anchor_source": "fixed_nc_geometry",
                "current_anchor_source": "fixed_nc_geometry",
                "classifier": "projected feature dot fixed NC anchors",
                "difference": "CF5 current block is dynamic, Locked NC current block is NC",
                "code": ["code/src/methods/full_concm.py:333", "code/src/methods/full_concm.py:385-390",
                         "code/tools/run_hyperkvasir_full_concm.py:287-293"],
            },
            "nc_anchored": {
                "equivalent": False,
                "old_anchor_source": "both fixed NC and rebuilt dynamic geometry",
                "current_anchor_source": "both fixed NC and rebuilt dynamic geometry",
                "classifier": "normalize(fixed_logits) + lambda_dsm * normalize(dynamic_logits)",
                "difference": "two-logit residual for every class, not one asymmetric anchor matrix",
                "code": ["code/src/methods/full_concm.py:381-394", "code/tools/run_hyperkvasir_full_concm.py:287-295"],
            },
            "full_dynamic": {
                "equivalent": False,
                "old_anchor_source": "centered-SVD geometry rebuilt from all seen projected prototypes",
                "current_anchor_source": "same rebuilt dynamic geometry",
                "old_anchor_update_path": "recomputed each session/epoch using current-session records",
                "difference": "CF5 freezes old NC anchors",
                "code": ["code/src/methods/full_concm.py:375-392", "code/src/methods/concm_dsm.py:45-74"],
            },
            "ltconcm_tdsm": {
                "equivalent": False,
                "old_anchor_source": "previous/prototype anchor regularization target",
                "current_anchor_source": "optimized calibrated-prototype anchors",
                "difference": "not fixed NC old anchors and not original Full Dynamic current anchors",
                "code": ["code/src/methods/gpa.py:243-270", "code/src/methods/ltconcm.py:227-285"],
            },
        },
        "normalization": "all compared heads normalize anchors/features before cosine logits",
        "gradient_path": "CF5 diagnostic is head-only; existing train modes optimize projector under different losses",
        "session_transition": "no existing mode preserves an NC old block while rebuilding only the current dynamic block",
        "minimal_difference_to_implement": "assemble one classifier anchor tensor with immutable NC rows for old heads and original dynamic rows for current heads",
        "implementation_allowed": False,
    }
    write_json(output / "cf5_equivalence.json", cf5_json)
    cf5_md = """# CF5 Equivalence Audit

`CF5_NOT_EQUIVALENT`

CF5 uses one asymmetric classifier matrix: old rows are immutable NC/reference anchors, while current rows are the original centered-SVD dynamic anchors. No existing Full ConCM mode has that source/update split.

| Method | Old source | Current source | Classifier | Why not equivalent |
|---|---|---|---|---|
| Locked NC-ConCM | fixed NC | fixed NC | one NC dot-product head | Current rows are not dynamic. |
| full_dynamic | rebuilt dynamic | rebuilt dynamic | one dynamic dot-product head | Old rows drift with current-session data. |
| nc_anchored | NC and dynamic | NC and dynamic | normalized NC logits plus lambda times normalized dynamic logits | It is a two-logit residual over every class, not an asymmetric row assembly. |
| LTConCM/TDSM | retained/prototype regularization target | optimized calibrated-prototype anchors | optimized anchor head | Old anchors are not fixed NC and current anchors are not original Full Dynamic rows. |

Source trace:

- Fixed NC construction: `code/src/methods/full_concm.py:333-340`.
- Full Dynamic geometry and losses: `code/src/methods/full_concm.py:375-395`.
- Inference assembly for all three modes: `code/tools/run_hyperkvasir_full_concm.py:287-295`.
- Centered-SVD dynamic construction: `code/src/methods/concm_dsm.py:45-74`.
- TDSM old-anchor regularization: `code/src/methods/ltconcm.py:227-285` and `code/src/methods/gpa.py:243-270`.

The minimum mathematical difference would be a single anchor tensor whose old rows never leave fixed NC coordinates and whose current rows come directly from the current Full Dynamic geometry. This audit does not implement it.
"""
    (output / "cf5_equivalence_audit.md").write_text(cf5_md, encoding="utf-8")

    dsm_json = {
        "result": "DSM_PARTIALLY_OVERLAPPING",
        "candidate": "relu(margin - max_similarity(current_feature,fixed_old_anchors) - similarity(current_feature,correct_current_anchor))",
        "existing_dsm": {
            "positive_operand": "same-label projected features plus own geometry anchor in supervised contrastive positives",
            "negative_candidate_set": "all resampled features and all old/current geometry anchors through a log-sum-exp denominator",
            "uses_old_anchors": True,
            "old_anchors_frozen": "detached within each epoch but dynamic geometry is rebuilt from all seen prototypes",
            "hardest_or_topk_old": False,
            "gradient_to_old_anchors": False,
            "gradient_to_current_anchors": False,
            "gradient_to_backbone": False,
            "gradient_to_projector": True,
            "margin": "no explicit hinge margin; temperature=0.07 contrastive softmax",
            "reduction": "mean positives per anchor, then mean per current class; LMatch uses class-balanced cross entropy",
            "session_applicability": "current feature anchors only in incremental sessions; all-class LMatch",
        },
        "overlap": ["current projected features", "correct-current anchor attraction", "old anchors participate as negatives"],
        "non_equivalence": ["softmax/log-sum-exp versus ReLU hinge", "all negatives versus max fixed-old negative",
                            "dynamic detached old anchors versus permanently fixed old anchors", "no explicit margin parameter",
                            "same-label sample positives are included", "different reduction"],
        "source": ["code/src/methods/full_concm.py:299-330", "code/src/methods/full_concm.py:375-395",
                   "code/src/methods/concm_dsm.py:18-35", "code/src/methods/concm_dsm.py:296-403"],
    }
    write_json(output / "dsm_fixed_old_margin_equivalence.json", dsm_json)
    dsm_md = """# DSM vs Fixed-Old Hard-Margin Audit

`DSM_PARTIALLY_OVERLAPPING`

The losses share a high-level interaction: a current projected feature is attracted to its correct anchor while old anchors can act as negatives. They are not mathematically equivalent.

| Property | Existing Full Dynamic DSM | Proposed fixed-old hard margin |
|---|---|---|
| Positive | Same-label samples and own anchor | Correct current anchor only |
| Negatives | All samples and all old/current anchors | Hardest fixed old anchor |
| Form | Temperature-scaled softmax/log-sum-exp plus cross entropy | ReLU hinge |
| Explicit margin | None | Required |
| Old anchors | Detached per epoch, but rebuilt dynamic geometry | Permanently fixed |
| Anchor gradients | None, geometry detached | Presumably none for fixed old; current behavior unspecified |
| Backbone gradient | None; cached frozen features | Candidate does not specify |
| Trainable path | DSM projector | Not enough to imply equivalence |
| Reduction | Positive mean, then current-class mean; class-balanced LMatch | Unspecified hinge reduction |

Relevant source: `code/src/methods/full_concm.py:299-330,375-395`, `code/src/methods/concm_dsm.py:18-35,296-403`.
"""
    (output / "dsm_fixed_old_margin_equivalence.md").write_text(dsm_md, encoding="utf-8")

    closure_report = f"""# Full Dynamic Branch Closure Report

## Frozen Status

- `FULL_DYNAMIC = CLOSED`
- `RAW_PROTOTYPE_FALLBACK = FAILED`
- `VALIDATION_CHECKPOINT_RESCUE = DIAGNOSTIC_ONLY`
- `NC_SAFE_DYNAMIC_V1 = FAILED`
- `ROOT_CAUSE = OLD_ANCHOR_DRIFT`
- `SECONDARY_FLAG = MIXED_GEOMETRY_AND_TRAJECTORY_FAILURE`
- `PROMOTION = NO_FULL_PROMOTION`

## Evidence

- seed1/session5 restricted-current accuracy spans `{min(restricted):.1f}%` to `{max(restricted):.1f}%`.
- The original classifier is unit-normalized with shared scale 1 and no bias, temperature, or block-specific scale; CF1 is ineffective.
- CF5 (`NC old + dynamic current`) is the only effective head-only counterfactual.
- Removing class9 absorption (`83 -> {nc_target['class9_absorption_count']}`) does not clear the target: fresh-test current-to-old is `{nc_target['test_current_to_old']:.2f}%`.
- Control AccT `{float(original_control['AccT']):.2f} -> {nc_control['test_AccT']:.2f}`, current `{float(original_control['current_acc']):.2f} -> {nc_control['test_current_accuracy']:.2f}`, HM `{float(original_control['HM_old_current']):.2f} -> {nc_control['test_HM']:.2f}`, old `{control_old_before:.2f} -> {control_old_after:.2f}`.
- Stage B checkpoint integrity is 12/12. No OOM occurred. Four-session confirmation and full 15-session training were not run.

## Equivalence Decisions

- CF5: `CF5_NOT_EQUIVALENT` to Locked NC-ConCM, full_dynamic, nc_anchored, or existing LTConCM/TDSM modes.
- DSM vs fixed-old hard-margin: `DSM_PARTIALLY_OVERLAPPING`, not mathematically equivalent.

## Mainline

The recommended project mainline is Locked NC-ConCM. Validation checkpoint selection remains diagnostic only. Local prototype/anchor safety does not guarantee global sample safety because prototype gates constrain a small set of class centers, while sample logits depend on the full feature distribution and all competing old-class rows. No further threshold, alpha, or class9 sweep is justified, and test is closed to further method development.
"""
    (output / "FULL_DYNAMIC_BRANCH_CLOSURE_REPORT.md").write_text(closure_report, encoding="utf-8")

    source_files = [
        project / "code/src/methods/full_concm.py",
        project / "code/src/methods/concm_dsm.py",
        project / "code/src/methods/ltconcm.py",
        project / "code/src/methods/gpa.py",
        project / "code/tools/run_hyperkvasir_full_concm.py",
        project / "code/tools/analyze_hyperkvasir_full_dynamic_geometry_forensics.py",
        project / "code/tools/run_hyperkvasir_nc_safe_dynamic_v1.py",
        Path(__file__).resolve(),
    ]
    source_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_started": False,
        "test_evaluation_started": False,
        "source_modified_by_closure": False,
        "files": [{"path": str(path), "sha256": sha256(path)} for path in source_files],
        "input_artifacts": [
            {"path": str(rescue / "json/final_decision.json"), "sha256": sha256(rescue / "json/final_decision.json")},
            {"path": str(valckpt / "gate.json"), "sha256": sha256(valckpt / "gate.json")},
            {"path": str(forensics / "causal_verdict.json"), "sha256": sha256(forensics / "causal_verdict.json")},
            {"path": str(nc_safe / "gate.json"), "sha256": sha256(nc_safe / "gate.json")},
        ],
    }
    write_json(output / "source_manifest.json", source_manifest)

    expected = [
        "FULL_DYNAMIC_BRANCH_CLOSURE_REPORT.md", "FULL_DYNAMIC_VALIDITY_LEDGER.json", "experiment_lineage.csv",
        "branch_status.csv", "key_evidence.csv", "metric_derivations.json", "source_manifest.json",
        "cf5_equivalence_audit.md", "cf5_equivalence.json", "dsm_fixed_old_margin_equivalence.md",
        "dsm_fixed_old_margin_equivalence.json",
    ]
    missing_outputs = [name for name in expected if not (output / name).is_file()]
    if missing_outputs:
        raise RuntimeError(f"Closure output missing: {missing_outputs}")
    for name in ("FULL_DYNAMIC_VALIDITY_LEDGER.json", "metric_derivations.json", "source_manifest.json",
                 "cf5_equivalence.json", "dsm_fixed_old_margin_equivalence.json"):
        json.loads((output / name).read_text())
    for name in ("experiment_lineage.csv", "branch_status.csv", "key_evidence.csv"):
        rows = read_csv(output / name)
        if not rows:
            raise RuntimeError(f"CSV has no rows: {name}")
    artifact_hashes = {
        name: {"sha256": sha256(output / name), "size_bytes": (output / name).stat().st_size}
        for name in expected
    }
    artifact_hashes["artifact_sha256.json"] = {"sha256": None, "note": "self-hash intentionally omitted"}
    write_json(output / "artifact_sha256.json", artifact_hashes)
    print(json.dumps({"status": FINAL_STATE, "output": str(output), "artifacts": len(artifact_hashes)}, indent=2))


if __name__ == "__main__":
    main()
