#!/usr/bin/env python3
"""Finalize the failed NC-safe Dynamic v1 microcheck without new evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    stage_a = Path(args.stage_a_root).resolve()
    output = Path(args.output_root).resolve()
    method = json.loads((output / "method_frozen.json").read_text())
    method_hash = json.loads((output / "method_source_hash.json").read_text())
    micro = json.loads((output / "microcheck_gate.json").read_text())
    control = json.loads((output / "seed2/session5_control/fresh_test_metrics.json").read_text())
    target = json.loads((output / "seed1/session5/fresh_test_metrics.json").read_text())
    control_selection = json.loads((output / "seed2/session5_control/selection.json").read_text())
    target_selection = json.loads((output / "seed1/session5/selection.json").read_text())
    verdict = json.loads((stage_a / "causal_verdict.json").read_text())
    checkpoint_verification = []
    for relative in ("seed2/session5_control", "seed1/session5"):
        manifest = json.loads((output / relative / "checkpoint_manifest.json").read_text())["checkpoints"]
        for row in manifest:
            path = Path(row["path"])
            checkpoint_verification.append(
                {
                    "session": relative,
                    "epoch": row["epoch"],
                    "sha256_matches": sha256(path) == row["sha256"],
                    "restore_ok": bool(row["restore_ok"]),
                }
            )
    all_verified = len(checkpoint_verification) == 12 and all(
        row["sha256_matches"] and row["restore_ok"] for row in checkpoint_verification
    )
    gate = {
        "gate": "NC_SAFE_DYNAMIC_MICROCHECK_FAILED",
        "branch_action": "CLOSE_FULL_DYNAMIC_REPAIR_BRANCH",
        "promotion": "NO_FULL_PROMOTION",
        "confirmation_sessions_started": False,
        "full_15_session_started": False,
        "target_validation_checkpoint_exists": target_selection["selected_epoch"] is not None,
        "target_fresh_test_non_collapse": not bool(target["test_collapse"]),
        "target_class9_significant_reduction": int(target["class9_absorption_count"]) <= 41,
        "control_preserved": bool(micro["control_success"]),
        "all_12_checkpoints_verified": all_verified,
    }
    (output / "gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")
    report = f"""# NC-safe Dynamic v1 Microcheck

## Stage A Basis

- Causal verdict: `{verdict['verdict']}`.
- Additional flag: `{verdict['flags']}`.
- Only CF5 (`NC old + dynamic current`) rescued seed1/session5 validation epochs 2 and 3.
- Restricted-current accuracy was high while global accuracy was suppressed by old-block absorption, supporting an old-anchor boundary failure.

## Frozen Method

- Branch: `nc_safe_dynamic_v1`.
- Alpha candidates: `{method['alpha_candidates']}`, descending; epsilon `{method['epsilon']}`.
- Alpha selection used fit-prototype geometry only, without validation or test.
- Frozen method source SHA256: `{method_hash['sha256']}`.

## Persistent Target: seed1/session5

- First legal validation checkpoint: yes.
- Eligible epochs: `{target_selection['eligible_epochs']}`.
- Selected epoch: `{target_selection['selected_epoch']}`.
- Fresh test: current `{target['test_current_accuracy']:.4f}`, HM `{target['test_HM']:.4f}`, current-to-old `{target['test_current_to_old']:.4f}`.
- Fresh-test collapse: `{target['test_collapse']}` because current-to-old remains above 30.
- Class 9 absorption: original 83 -> `{target['class9_absorption_count']}`.
- Class 9 alpha: `{target['class9_alpha']}`.
- Dynamic acceptance ratio: `{target['dynamic_acceptance_ratio']:.4f}`.
- Selected alpha by class: `{json.dumps(target['selected_alpha_by_class'], sort_keys=True)}`.

## Stable Control: seed2/session5

- Eligible epochs: `{control_selection['eligible_epochs']}`.
- Selected epoch: `{control_selection['selected_epoch']}`.
- Fresh test: AccT `{control['test_AccT']:.4f}`, current `{control['test_current_accuracy']:.4f}`, HM `{control['test_HM']:.4f}`, current-to-old `{control['test_current_to_old']:.4f}`.
- Original control: AccT `50.9099`, current `57.1429`, HM `53.7408`.
- Control preserved within 1 percentage point: `{micro['control_success']}`.
- Class 9 alpha: `{control['class9_alpha']}`.
- Dynamic acceptance ratio: `{control['dynamic_acceptance_ratio']:.4f}`.
- Selected alpha by class: `{json.dumps(control['selected_alpha_by_class'], sort_keys=True)}`.

## Decision

- Gate: `NC_SAFE_DYNAMIC_MICROCHECK_FAILED`.
- Action: `CLOSE_FULL_DYNAMIC_REPAIR_BRANCH` and `NO_FULL_PROMOTION`.
- seed1/session2 and seed3/session5 confirmation sessions were not started.
- Full 15-session training was not started.
- The anchor gate removes class9 absorption and creates a legal validation checkpoint, but it does not generalize past the original current-to-old test threshold and substantially damages the control old/current balance.
- All 12 saved epoch checkpoints passed SHA256 and restore verification: `{all_verified}`.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    source_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_method_source": method_hash,
        "finalizer": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "stage_a_source_manifest": str(stage_a / "source_manifest.json"),
        "checkpoint_verification": checkpoint_verification,
        "new_training_sessions": 2,
        "confirmation_training_sessions": 0,
    }
    (output / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
