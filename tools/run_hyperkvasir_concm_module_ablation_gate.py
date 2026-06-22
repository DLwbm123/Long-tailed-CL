#!/usr/bin/env python3
"""HyperKvasir23 ConCM two-module ablation gate.

The new branches are feature-level only: the backbone checkpoints are loaded,
frozen, and used for feature extraction.  MPC-lite here is a bounded visual
memory prototype calibration proxy, not original semantic MPC.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets import build_protocol
from src.methods.concm_dsm import (
    DSMProjector,
    compute_dynamic_structure,
    evaluate_dsm,
    extract_features_by_class,
    train_dsm_projector,
)
from src.methods.concm_mpc import calibrate_current_stds, compute_mpc_lite_calibration, load_semantic_prior
from src.models.resnet_cifar import resnet32
from src.utils.seed import set_seed

from run_hyperkvasir_concm_min_gate import (
    _collect_logits_features,
    _confusion_rows as _class_confusion_rows,
    _correction_vector,
    _evaluate_pack,
    _per_class_metrics_rows as _class_per_class_metrics_rows,
    _risk_and_reliability,
)
from run_hyperkvasir_pure_dsm_concm_gate import (
    ExemplarPathDataset,
    _evaluate_model_classifier,
    _freeze_all,
    _json_default,
    _load_exemplar_rows,
    _load_result_row,
    _make_loader,
    _make_protocol_args,
    _markdown_table,
    _pairwise_rows,
    _per_class_rows,
    _stack_stats,
    _summary_row,
    _write_csv,
    _write_json,
)


METRIC_COLUMNS = [
    "AccT",
    "balanced_acc",
    "macro_f1",
    "old_acc",
    "current_acc",
    "old_to_current_rate",
    "current_to_old_rate",
]

EXPECTED_C = {
    "AccT": 77.5273,
    "old_acc": 75.5594,
    "current_acc": 88.2629,
    "old_to_current_rate": 14.0275,
    "current_to_old_rate": 9.8592,
}
EXPECTED_D = {
    "AccT": 76.5818,
    "old_acc": 74.4406,
    "current_acc": 88.2629,
    "old_to_current_rate": 9.3804,
    "current_to_old_rate": 9.8592,
}


def _read_json(path: str | Path) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _requested(groups: str) -> set[str]:
    parsed = {item.strip().upper() for item in groups.split(",") if item.strip()}
    valid = {"A", "B", "C", "D", "E", "M", "N", "O"}
    unknown = parsed - valid
    if unknown:
        raise ValueError(f"Unknown groups requested: {sorted(unknown)}")
    return parsed


def _as_float(value) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _assert_reference(label: str, row: Mapping[str, object], expected: Mapping[str, float], tolerance: float = 0.02) -> None:
    errors = []
    for key, expected_value in expected.items():
        observed = _as_float(row.get(key))
        if observed is None or abs(observed - float(expected_value)) > float(tolerance):
            errors.append(f"{label}.{key}: observed={observed} expected={expected_value}")
    if errors:
        raise RuntimeError("Reference reproduction check failed before new branches: " + "; ".join(errors))


def _load_dsm_row(path: str | Path, group: str, method: str) -> Dict[str, object]:
    payload = _read_json(path)
    row = dict(payload.get("dsm_row") or {})
    if not row:
        raise RuntimeError(f"DSM row not found in {path}")
    row["group"] = group
    row["method"] = method
    row["source_json"] = str(path)
    return row


def _infer_no_cont_json(dsm_json: str | Path) -> Path:
    path = Path(dsm_json)
    parent = path.parent
    sibling = parent.with_name(parent.name + "_no_cont")
    return sibling / "final_results.json"


def _class_confusion_rows_from_tensor(
    confusion: torch.Tensor,
    head_to_class: Mapping[int, int],
    seen_classes: Sequence[int],
) -> List[Dict[str, object]]:
    rows = []
    class_to_head = {int(class_id): int(head) for head, class_id in head_to_class.items()}
    for true_class in seen_classes:
        true_head = class_to_head[int(true_class)]
        row = {"true_class": int(true_class), "total": int(confusion[true_head].sum().item())}
        for pred_class in seen_classes:
            pred_head = class_to_head[int(pred_class)]
            row[f"pred_{int(pred_class)}"] = int(confusion[true_head, pred_head].item())
        rows.append(row)
    return rows


@torch.no_grad()
def _evaluate_prototype_classifier(
    features: torch.Tensor,
    head_labels: torch.Tensor,
    prototypes: torch.Tensor,
    old_head_indices: Sequence[int],
    current_head_indices: Sequence[int],
) -> Dict[str, object]:
    x = F.normalize(torch.nan_to_num(features.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0), p=2, dim=1)
    p = F.normalize(torch.nan_to_num(prototypes.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0), p=2, dim=1)
    labels = head_labels.detach().cpu().long()
    logits = x @ p.transpose(0, 1)
    preds = logits.argmax(dim=1)
    matches = preds.eq(labels)
    old_set = {int(item) for item in old_head_indices}
    current_set = {int(item) for item in current_head_indices}
    old_mask = torch.tensor([int(item) in old_set for item in labels.tolist()], dtype=torch.bool)
    current_mask = torch.tensor([int(item) in current_set for item in labels.tolist()], dtype=torch.bool)
    pred_old_mask = torch.tensor([int(item) in old_set for item in preds.tolist()], dtype=torch.bool)
    pred_current_mask = torch.tensor([int(item) in current_set for item in preds.tolist()], dtype=torch.bool)
    confusion = torch.zeros((p.shape[0], p.shape[0]), dtype=torch.long)
    for label, pred in zip(labels.tolist(), preds.tolist()):
        confusion[int(label), int(pred)] += 1

    recalls: List[float] = []
    f1_values: List[float] = []
    per_class: List[Dict[str, object]] = []
    for head in range(p.shape[0]):
        total = int(confusion[head].sum().item())
        correct = int(confusion[head, head].item())
        predicted = int(confusion[:, head].sum().item())
        precision = 100.0 * correct / predicted if predicted else 0.0
        recall = 100.0 * correct / total if total else None
        f1 = 0.0
        if recall is not None and precision + recall > 0.0:
            f1 = 2.0 * precision * recall / (precision + recall)
        if recall is not None:
            recalls.append(recall)
            f1_values.append(f1)
        per_class.append(
            {
                "head_idx": int(head),
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "correct": correct,
                "total": total,
                "predicted": predicted,
            }
        )

    def pct(numerator: int, denominator: int) -> float | None:
        return 100.0 * float(numerator) / float(denominator) if denominator > 0 else None

    return {
        "AccT": pct(int(matches.sum().item()), int(labels.numel())),
        "balanced_acc": sum(recalls) / len(recalls) if recalls else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "old_acc": pct(int(matches[old_mask].sum().item()), int(old_mask.sum().item())),
        "current_acc": pct(int(matches[current_mask].sum().item()), int(current_mask.sum().item())),
        "old_to_current_rate": pct(int((old_mask & pred_current_mask).sum().item()), int(old_mask.sum().item())),
        "current_to_old_rate": pct(int((current_mask & pred_old_mask).sum().item()), int(current_mask.sum().item())),
        "total": int(labels.numel()),
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


def _load_dsm_projector(path: str | Path, device: torch.device) -> tuple[DSMProjector, torch.Tensor, Dict[str, object]]:
    payload = torch.load(path, map_location=device)
    config = payload["projector_config"]
    projector = DSMProjector(**config).to(device)
    projector.load_state_dict(payload["projector_state_dict"])
    projector.eval()
    geometry = payload["geometry_vectors"].detach().to(device)
    return projector, geometry, dict(payload)


def _add_group(rows: Sequence[Mapping[str, object]], group: str, method: str) -> List[Dict[str, object]]:
    out = []
    for row in rows:
        item = dict(row)
        item["group"] = group
        item["method"] = method
        out.append(item)
    return out


def _delta_rows(rows: Sequence[Mapping[str, object]], reference: Mapping[str, object], reference_group: str) -> List[Dict[str, object]]:
    output = []
    for row in rows:
        item = {
            "group": row.get("group"),
            "method": row.get("method"),
            "reference_group": reference_group,
        }
        for metric in METRIC_COLUMNS:
            value = _as_float(row.get(metric))
            ref_value = _as_float(reference.get(metric))
            item[f"delta_{metric}"] = value - ref_value if value is not None and ref_value is not None else None
        output.append(item)
    return output


def _branch_acceptance(rows_by_group: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    d = rows_by_group.get("D_pure_DSM_ConCM")
    c = rows_by_group.get("C_locked_NCConCM")
    m = rows_by_group.get("M_MPC_lite_only")
    n = rows_by_group.get("N_MPC_lite_DSM")
    if d is None or n is None:
        return {"status": "not_evaluated", "reason": "D or N row missing"}

    def val(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
        parsed = _as_float(row.get(key))
        return default if parsed is None else parsed

    checks = {
        "N_current_ge_85": val(n, "current_acc") >= 85.0,
        "N_current_to_old_lt_15": val(n, "current_to_old_rate", 100.0) < 15.0,
        "N_old_to_current_not_worse_than_D_by_gt_2": val(n, "old_to_current_rate", 100.0) <= val(d, "old_to_current_rate", 100.0) + 2.0,
        "N_AccT_ge_D": val(n, "AccT") >= val(d, "AccT"),
        "N_old_acc_ge_D": val(n, "old_acc") >= val(d, "old_acc"),
        "N_balanced_ge_D": val(n, "balanced_acc") >= val(d, "balanced_acc"),
        "N_macro_ge_D": val(n, "macro_f1") >= val(d, "macro_f1"),
    }
    if m is not None:
        checks["M_current_to_old_lt_15"] = val(m, "current_to_old_rate", 100.0) < 15.0
    strong = all(bool(checks[key]) for key in ["N_current_ge_85", "N_current_to_old_lt_15", "N_AccT_ge_D", "N_old_acc_ge_D", "N_balanced_ge_D", "N_macro_ge_D"])
    beats_d = val(n, "AccT") > val(d, "AccT") or val(n, "old_acc") > val(d, "old_acc") or val(n, "balanced_acc") > val(d, "balanced_acc")
    approaches_c = c is not None and val(n, "AccT") >= val(c, "AccT") - 2.0
    return {
        "status": "strong_success" if strong else ("partial_success" if beats_d else "failed_vs_D"),
        "checks": checks,
        "N_minus_D": {metric: val(n, metric) - val(d, metric) for metric in METRIC_COLUMNS},
        "N_minus_C": ({metric: val(n, metric) - val(c, metric) for metric in METRIC_COLUMNS} if c is not None else None),
        "approaches_locked_C": bool(approaches_c),
    }


def _format_float(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}" if not math.isnan(value) else "nan"
    return str(value)


def _build_report(payload: Mapping[str, object]) -> str:
    rows = payload["summary_rows"]
    acceptance = payload["acceptance"]
    columns = ["group", "method", *METRIC_COLUMNS]
    lines = [
        "# HyperKvasir23 ConCM Module Ablation Results",
        "",
        "## Purpose",
        "",
        "Test ConCM's two separable modules under the same seed0 HyperKvasir23 medical CIL setting: DSM structure matching and MPC-lite visual-memory prototype calibration.",
        "",
        "MPC-lite is a medical-safe visual-memory proxy. It uses no external semantic resources and should not be read as faithful original semantic MPC.",
        "",
        "## Exact Command",
        "",
        "```bash",
        str(payload["command"]),
        "```",
        "",
        "## Module Branches",
        "",
        "- `A`: raw frozen phase1 classifier reference.",
        "- `B`: task-block calibration reference using balanced_hmean_exemplar.",
        "- `C`: locked NC-ConCM reference.",
        "- `D`: reused Pure DSM-ConCM reference.",
        "- `E`: reused or optional Pure DSM without contrastive loss.",
        "- `M`: MPC-lite prototype classifier in frozen feature space.",
        "- `N`: MPC-lite calibrated prototypes/stds feeding DSM projector training.",
        "- `O`: optional MPC-lite DSM without contrastive loss.",
        "",
        "## Results",
        "",
        _markdown_table(rows, columns),
        "",
        "## Acceptance",
        "",
        "```json",
        json.dumps(acceptance, indent=2, sort_keys=True),
        "```",
        "",
        "## Interpretation",
        "",
    ]
    status = str(acceptance.get("status"))
    if status == "strong_success":
        lines.append("MPC-lite helps DSM under the bounded seed0 gate and satisfies the current/old confusion guardrails.")
    elif status == "partial_success":
        lines.append("MPC-lite gives some benefit over DSM-only, but not across every main metric. Treat it as a useful proxy diagnostic rather than a settled replacement for locked NC-ConCM.")
    elif status == "failed_vs_D":
        lines.append("MPC-lite does not improve over DSM-only in this gate. DSM-only remains the cleaner branch unless a stronger medical prior is added.")
    else:
        lines.append("The main MPC-lite DSM branch was not evaluated, so only the reference and diagnostic rows should be used.")
    lines.extend(
        [
            "",
            "## Artifact Paths",
            "",
        ]
    )
    for name, path in payload["artifact_paths"].items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            "",
            "## Safety Checks",
            "",
            "- Backbone parameters were frozen before feature extraction.",
            "- No backbone checkpoint is saved by this runner.",
            "- Only `DSMProjector` parameters are optimized for DSM branches.",
            "- Semantic prior inputs are optional local JSON/CSV files; the runner has no network fetch path.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--phase0-ckpt", required=True)
    parser.add_argument("--phase1-ckpt", required=True)
    parser.add_argument("--exemplar-csv", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--diagnostics-json", required=True)
    parser.add_argument("--locked-json", required=True)
    parser.add_argument("--dsm-json", required=True)
    parser.add_argument("--dsm-projector", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--projector-hidden", type=int, default=2048)
    parser.add_argument("--projector-dim", type=int, default=128)
    parser.add_argument("--base-projector-epochs", type=int, default=5)
    parser.add_argument("--increment-projector-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--sample-num-old", type=int, default=100)
    parser.add_argument("--sample-num-current", type=int, default=50)
    parser.add_argument("--cont-weight", type=float, default=1.0)
    parser.add_argument("--mpc-lite-alpha", type=float, default=0.6)
    parser.add_argument("--mpc-lite-topk", type=int, default=5)
    parser.add_argument("--mpc-lite-tau", type=float, default=16.0)
    parser.add_argument("--mpc-lite-gamma", type=float, default=0.6)
    parser.add_argument("--groups", default="A,B,C,D,E,M,N")
    parser.add_argument("--reuse-existing-dsm", action="store_true")
    parser.add_argument("--run-mpc-lite", action="store_true")
    parser.add_argument("--run-mpc-lite-dsm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--semantic-prior-json", default=None)
    parser.add_argument("--semantic-prior-csv", default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--run-no-cont", action="store_true")
    args = parser.parse_args()

    set_seed(int(args.seed))
    requested = _requested(args.groups)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    protocol = build_protocol(_make_protocol_args(args))

    phase0_payload = torch.load(args.phase0_ckpt, map_location="cpu")
    phase1_payload = torch.load(args.phase1_ckpt, map_location=device)
    old_classes = [int(class_id) for class_id in phase0_payload["old_classes"]]
    current_classes = [int(class_id) for class_id in phase0_payload["current_classes"]]
    seen_classes = old_classes + current_classes
    if [int(class_id) for class_id in phase1_payload["seen_classes"]] != seen_classes:
        raise RuntimeError("phase1 checkpoint seen_classes do not match phase0 metadata")

    phase0_class_to_head = {int(class_id): idx for idx, class_id in enumerate(old_classes)}
    phase1_class_to_head = {int(class_id): idx for idx, class_id in enumerate(seen_classes)}
    head_to_class = {idx: int(class_id) for class_id, idx in phase1_class_to_head.items()}
    old_heads = list(range(len(old_classes)))
    current_heads = list(range(len(old_classes), len(seen_classes)))

    phase0_model = resnet32().to(device)
    phase0_model.expand_classifier(len(old_classes))
    phase0_model.load_state_dict(phase0_payload["model_state"])
    phase1_model = resnet32().to(device)
    phase1_model.expand_classifier(len(seen_classes))
    phase1_model.load_state_dict(phase1_payload["model_state"])
    phase0_freeze = _freeze_all(phase0_model)
    phase1_freeze = _freeze_all(phase1_model)

    phase0_proto_loader = _make_loader(
        protocol.prototype_dataset_for_classes(old_classes),
        args.batch_size,
        args.num_workers,
        shuffle=False,
        seed=int(args.seed),
    )
    current_proto_loader = _make_loader(
        protocol.prototype_dataset_for_classes(current_classes),
        args.batch_size,
        args.num_workers,
        shuffle=False,
        seed=int(args.seed) + 1,
    )
    current_train_loader_for_c = _make_loader(
        protocol.train_dataset_for_classes(current_classes),
        args.batch_size,
        args.num_workers,
        shuffle=False,
        seed=int(args.seed) + 11,
    )
    seen_test_loader = _make_loader(
        protocol.test_dataset_for_classes(seen_classes),
        args.batch_size,
        args.num_workers,
        shuffle=False,
        seed=int(args.seed) + 2,
    )
    exemplar_rows = _load_exemplar_rows(Path(args.exemplar_csv))
    exemplar_loader = _make_loader(
        ExemplarPathDataset(exemplar_rows, protocol.eval_transform),
        max(1, min(int(args.batch_size), len(exemplar_rows))),
        args.num_workers,
        shuffle=False,
        seed=int(args.seed) + 3,
    )

    base_stats = extract_features_by_class(phase0_model, phase0_proto_loader, phase0_class_to_head, device)
    current_stats = extract_features_by_class(phase1_model, current_proto_loader, phase1_class_to_head, device)
    exemplar_stats = extract_features_by_class(phase1_model, exemplar_loader, phase1_class_to_head, device)
    test_stats = extract_features_by_class(phase1_model, seen_test_loader, phase1_class_to_head, device)

    base_means = _stack_stats(base_stats.means_by_head, old_heads)
    base_stds = _stack_stats(base_stats.stds_by_head, old_heads)
    current_means = _stack_stats(current_stats.means_by_head, current_heads)
    current_stds = _stack_stats(current_stats.stds_by_head, current_heads)
    real_features = torch.cat([exemplar_stats.features, current_stats.features], dim=0)
    real_labels = torch.cat([exemplar_stats.head_labels, current_stats.head_labels], dim=0)
    input_dim = int(base_means.shape[1])

    raw_row = _load_result_row(args.diagnostics_json, "diagnostics")
    if raw_row is None:
        raw_row = _evaluate_model_classifier(phase1_model, seen_test_loader, phase1_class_to_head, old_classes, current_classes, device)
    raw_summary = _summary_row("A_raw_freeze", "raw_freeze", raw_row)
    calibration_row = _load_result_row(args.calibration_json, "calibration")
    locked_row = _load_result_row(args.locked_json, "locked")
    if calibration_row is None:
        raise RuntimeError("B task-block row not found in calibration JSON")
    if locked_row is None:
        raise RuntimeError("C locked NC-ConCM row not found in locked JSON")
    c_row = _summary_row(
        "C_locked_NCConCM",
        "locked_NCConCM",
        locked_row,
        alpha=locked_row.get("alpha"),
        lambda_value=locked_row.get("lambda_value"),
    )
    d_row = _load_dsm_row(args.dsm_json, "D_pure_DSM_ConCM", "pure_DSM_ConCM_reused")
    _assert_reference("C", c_row, EXPECTED_C)
    _assert_reference("D", d_row, EXPECTED_D)

    d_projector, d_geometry, d_projector_payload = _load_dsm_projector(args.dsm_projector, device)
    d_metrics = evaluate_dsm(test_stats.features, test_stats.head_labels, d_projector, d_geometry, old_heads, current_heads, device=device)
    d_confusion_rows = _class_confusion_rows_from_tensor(d_metrics["confusion_matrix"], head_to_class, seen_classes)
    d_per_class_rows = _add_group(_per_class_rows(d_metrics, head_to_class), "D_pure_DSM_ConCM", "pure_DSM_ConCM_recomputed_confusion")

    calibration_payload = _read_json(args.calibration_json)
    diagnostics_payload = _read_json(args.diagnostics_json)
    selected_by_rule = {row["rule"]: row for row in calibration_payload.get("selected_test_rows", [])}
    balanced = selected_by_rule.get("balanced_hmean_exemplar")
    if balanced is None:
        raise RuntimeError("balanced_hmean_exemplar missing from calibration JSON")
    c_alpha = float(balanced["alpha"])
    c_lambda = float(c_row.get("lambda_value") or 0.30)
    seen_pack = _collect_logits_features(phase1_model, seen_test_loader, device)
    current_train_pack = _collect_logits_features(phase1_model, current_train_loader_for_c, device)
    exemplar_pack = _collect_logits_features(phase1_model, exemplar_loader, device)
    old_risk, current_absorb, _, reliability, _ = _risk_and_reliability(
        exemplar_pack,
        current_train_pack,
        diagnostics_payload,
        phase0_payload,
        phase1_model,
        seen_classes,
        old_classes,
        current_classes,
        c_alpha,
    )
    c_correction = _correction_vector(
        seen_classes,
        old_classes,
        current_classes,
        old_risk,
        current_absorb,
        reliability,
        c_lambda,
        use_reliability=False,
    )
    c_recomputed_metrics, c_confusion = _evaluate_pack(seen_pack, seen_classes, old_classes, current_classes, alpha=c_alpha, correction=c_correction)
    c_confusion_rows = _class_confusion_rows(c_confusion, seen_classes)
    c_per_class_rows = _add_group(
        _class_per_class_metrics_rows(c_confusion, seen_classes),
        "C_locked_NCConCM",
        "locked_NCConCM_recomputed_confusion",
    )

    semantic_prior = load_semantic_prior(
        json_path=args.semantic_prior_json,
        csv_path=args.semantic_prior_csv,
        current_class_ids=current_classes,
        old_class_ids=old_classes,
    )
    calibrated_current_means, mpc_state, mpc_stats_rows, mpc_topk_rows = compute_mpc_lite_calibration(
        current_means,
        base_means,
        current_class_ids=current_classes,
        old_class_ids=old_classes,
        alpha=float(args.mpc_lite_alpha),
        topk=int(args.mpc_lite_topk),
        tau=float(args.mpc_lite_tau),
        semantic_prior=semantic_prior,
    )
    calibrated_current_stds, std_rows = calibrate_current_stds(
        current_stds,
        base_stds,
        top_indices=mpc_state["top_indices"],
        weights=mpc_state["weights"],
        gamma=float(args.mpc_lite_gamma),
    )
    for idx, row in enumerate(std_rows):
        row["current_class"] = int(current_classes[idx])
        mpc_stats_rows[idx].update({k: v for k, v in row.items() if k != "current_row"})

    mpc_means = torch.cat([base_means, calibrated_current_means], dim=0)
    mpc_stds = torch.cat([base_stds, calibrated_current_stds], dim=0)
    summary_rows: List[Dict[str, object]] = []
    per_class_rows: List[Dict[str, object]] = []
    train_traces: Dict[str, List[Dict[str, object]]] = {}
    pairwise_rows_by_group: Dict[str, List[Dict[str, object]]] = {}
    geometry_stats_by_group: Dict[str, Dict[str, object]] = {
        "D_pure_DSM_ConCM": dict(d_projector_payload.get("geometry_stats") or {}),
    }
    confusion_by_group: Dict[str, List[Dict[str, object]]] = {
        "C_locked_NCConCM": c_confusion_rows,
        "D_pure_DSM_ConCM": d_confusion_rows,
    }
    rows_by_group: Dict[str, Dict[str, object]] = {}

    if "A" in requested:
        summary_rows.append(raw_summary)
    if "B" in requested:
        summary_rows.append(_summary_row("B_task_block_only", "task_block_only", calibration_row, alpha=calibration_row.get("alpha")))
    if "C" in requested:
        summary_rows.append(c_row)
        per_class_rows.extend(c_per_class_rows)
    if "D" in requested:
        summary_rows.append(d_row)
        per_class_rows.extend(d_per_class_rows)
        pairwise_rows_by_group["D_pure_DSM_ConCM"] = _pairwise_rows(d_geometry.detach().cpu(), head_to_class)
    if "E" in requested:
        no_cont_json = _infer_no_cont_json(args.dsm_json)
        if no_cont_json.exists():
            summary_rows.append(_load_dsm_row(no_cont_json, "E_pure_DSM_ConCM_no_cont", "pure_DSM_ConCM_no_cont_reused"))
        else:
            summary_rows.append(
                {
                    "group": "E_pure_DSM_ConCM_no_cont",
                    "method": "pure_DSM_ConCM_no_cont_missing",
                    "status": f"not_found: {no_cont_json}",
                }
            )

    if args.dry_run:
        dry_payload = {
            "status": "dry_run_ok",
            "command": shlex.join([sys.executable, *sys.argv]),
            "output_dir": str(output_dir),
            "requested_groups": sorted(requested),
            "old_classes": old_classes,
            "current_classes": current_classes,
            "feature_counts_by_head": {
                "base": base_stats.counts_by_head,
                "current": current_stats.counts_by_head,
                "old_exemplars": exemplar_stats.counts_by_head,
                "test": test_stats.counts_by_head,
            },
            "mpc_lite_stats": mpc_stats_rows,
            "c_recomputed_metrics": c_recomputed_metrics,
            "freeze_checks": {"phase0": phase0_freeze, "phase1": phase1_freeze},
        }
        _write_json(output_dir / "module_ablation_config.json", dry_payload)
        _write_csv(output_dir / "mpc_lite_calibration_stats.csv", mpc_stats_rows)
        _write_csv(output_dir / "mpc_lite_topk_memory.csv", mpc_topk_rows)
        (output_dir / "run_summary.md").write_text(
            "# HyperKvasir23 ConCM Module Ablation Dry Run\n\nReference checks, feature extraction, and MPC-lite calibration completed. No projector training was launched.\n",
            encoding="utf-8",
        )
        print(json.dumps(dry_payload, indent=2, sort_keys=True, default=_json_default))
        return

    run_m = bool(args.run_mpc_lite or "M" in requested or "N" in requested or "O" in requested)
    if run_m and "M" in requested:
        m_metrics = _evaluate_prototype_classifier(test_stats.features, test_stats.head_labels, mpc_means, old_heads, current_heads)
        m_row = _summary_row(
            "M_MPC_lite_only",
            "MPC_lite_prototype_classifier",
            m_metrics,
            evaluation_mode="cosine_NCM_on_frozen_features",
        )
        summary_rows.append(m_row)
        per_class_rows.extend(_add_group(_per_class_rows(m_metrics, head_to_class), "M_MPC_lite_only", "MPC_lite_prototype_classifier"))
        confusion_by_group["M_MPC_lite_only"] = _class_confusion_rows_from_tensor(m_metrics["confusion_matrix"], head_to_class, seen_classes)

    def train_mpc_dsm_branch(
        group: str,
        method: str,
        no_cont: bool,
    ) -> tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]], Dict[str, object], torch.Tensor, DSMProjector, List[Dict[str, object]]]:
        projector = DSMProjector(input_dim=input_dim, hidden_dim=int(args.projector_hidden), output_dim=int(args.projector_dim)).to(device)
        trace_rows: List[Dict[str, object]] = []
        if int(args.base_projector_epochs) > 0:
            projector, _, base_trace, _ = train_dsm_projector(
                means=base_means,
                stds=base_stds,
                projector=projector,
                epochs=int(args.base_projector_epochs),
                lr=float(args.lr),
                batch_size=int(args.batch_size),
                sample_num_old=int(args.sample_num_old),
                sample_num_current=int(args.sample_num_current),
                num_old_classes=len(old_classes),
                cont_weight=float(args.cont_weight),
                no_cont_loss=bool(no_cont),
                device=device,
                max_train_batches=args.max_train_batches,
                seed=int(args.seed) + 40,
            )
            for trace in base_trace:
                item = dict(trace)
                item["group"] = group
                item["stage"] = "base_projector_alignment"
                trace_rows.append(item)
        projector, geometry, inc_trace, geometry_stats = train_dsm_projector(
            means=mpc_means,
            stds=mpc_stds,
            projector=projector,
            epochs=int(args.increment_projector_epochs),
            lr=float(args.lr),
            batch_size=int(args.batch_size),
            sample_num_old=int(args.sample_num_old),
            sample_num_current=int(args.sample_num_current),
            num_old_classes=len(old_classes),
            cont_weight=float(args.cont_weight),
            no_cont_loss=bool(no_cont),
            real_features=real_features,
            real_labels=real_labels,
            device=device,
            max_train_batches=args.max_train_batches,
            seed=int(args.seed) + (50 if not no_cont else 60),
        )
        for trace in inc_trace:
            item = dict(trace)
            item["group"] = group
            item["stage"] = "incremental_mpc_lite_dsm"
            trace_rows.append(item)
        metrics = evaluate_dsm(test_stats.features, test_stats.head_labels, projector, geometry, old_heads, current_heads, device=device)
        row = _summary_row(
            group,
            method,
            metrics,
            cont_loss_enabled=not bool(no_cont),
            mpc_lite_alpha=float(args.mpc_lite_alpha),
            mpc_lite_topk=int(args.mpc_lite_topk),
            mpc_lite_tau=float(args.mpc_lite_tau),
            mpc_lite_gamma=float(args.mpc_lite_gamma),
        )
        return (
            row,
            _per_class_rows(metrics, head_to_class),
            _class_confusion_rows_from_tensor(metrics["confusion_matrix"], head_to_class, seen_classes),
            geometry_stats,
            geometry,
            projector,
            trace_rows,
        )

    n_projector = None
    n_geometry = None
    if (args.run_mpc_lite_dsm or "N" in requested) and "N" in requested:
        n_row, n_per_class, n_confusion, n_geometry_stats, n_geometry, n_projector, n_trace = train_mpc_dsm_branch(
            "N_MPC_lite_DSM",
            "MPC_lite_DSM",
            no_cont=False,
        )
        summary_rows.append(n_row)
        per_class_rows.extend(_add_group(n_per_class, "N_MPC_lite_DSM", "MPC_lite_DSM"))
        train_traces["N_MPC_lite_DSM"] = [dict(row, method="MPC_lite_DSM") for row in n_trace]
        confusion_by_group["N_MPC_lite_DSM"] = n_confusion
        geometry_stats_by_group["N_MPC_lite_DSM"] = n_geometry_stats
        pairwise_rows_by_group["N_MPC_lite_DSM"] = _pairwise_rows(n_geometry.detach().cpu(), head_to_class)

    if "O" in requested:
        if bool(args.run_no_cont):
            o_row, o_per_class, o_confusion, o_geometry_stats, o_geometry, _, o_trace = train_mpc_dsm_branch(
                "O_MPC_lite_DSM_no_cont",
                "MPC_lite_DSM_no_cont",
                no_cont=True,
            )
            summary_rows.append(o_row)
            per_class_rows.extend(_add_group(o_per_class, "O_MPC_lite_DSM_no_cont", "MPC_lite_DSM_no_cont"))
            train_traces["O_MPC_lite_DSM_no_cont"] = [dict(row, method="MPC_lite_DSM_no_cont") for row in o_trace]
            confusion_by_group["O_MPC_lite_DSM_no_cont"] = o_confusion
            geometry_stats_by_group["O_MPC_lite_DSM_no_cont"] = o_geometry_stats
            pairwise_rows_by_group["O_MPC_lite_DSM_no_cont"] = _pairwise_rows(o_geometry.detach().cpu(), head_to_class)
        else:
            summary_rows.append(
                {
                    "group": "O_MPC_lite_DSM_no_cont",
                    "method": "MPC_lite_DSM_no_cont_skipped",
                    "status": "requires --run-no-cont",
                }
            )

    rows_by_group = {str(row.get("group")): dict(row) for row in summary_rows}
    acceptance = _branch_acceptance(rows_by_group)

    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_csv(output_dir / "delta_vs_C.csv", _delta_rows(summary_rows, c_row, "C_locked_NCConCM"))
    _write_csv(output_dir / "delta_vs_D.csv", _delta_rows(summary_rows, d_row, "D_pure_DSM_ConCM"))
    _write_csv(output_dir / "mpc_lite_calibration_stats.csv", mpc_stats_rows)
    _write_csv(output_dir / "mpc_lite_topk_memory.csv", mpc_topk_rows)
    _write_csv(output_dir / "per_class_metrics.csv", per_class_rows)
    if "N_MPC_lite_DSM" in train_traces:
        _write_csv(output_dir / "train_trace_N.csv", train_traces["N_MPC_lite_DSM"])
    if "O_MPC_lite_DSM_no_cont" in train_traces:
        _write_csv(output_dir / "train_trace_O.csv", train_traces["O_MPC_lite_DSM_no_cont"])
    _write_json(output_dir / "geometry_stats_D.json", geometry_stats_by_group.get("D_pure_DSM_ConCM", {}))
    if "N_MPC_lite_DSM" in geometry_stats_by_group:
        _write_json(output_dir / "geometry_stats_N.json", geometry_stats_by_group["N_MPC_lite_DSM"])
    for group, rows in confusion_by_group.items():
        suffix = group.split("_", 1)[0]
        _write_csv(output_dir / f"confusion_matrix_{suffix}.csv", rows)
    if "D_pure_DSM_ConCM" in pairwise_rows_by_group:
        _write_csv(output_dir / "pairwise_geometry_dot_D.csv", pairwise_rows_by_group["D_pure_DSM_ConCM"])
    if "N_MPC_lite_DSM" in pairwise_rows_by_group:
        _write_csv(output_dir / "pairwise_geometry_dot_N.csv", pairwise_rows_by_group["N_MPC_lite_DSM"])
    if n_projector is not None and n_geometry is not None:
        torch.save(
            {
                "projector_state_dict": n_projector.state_dict(),
                "projector_config": n_projector.config(),
                "geometry_vectors": n_geometry.detach().cpu(),
                "seen_classes": seen_classes,
                "class_to_head": phase1_class_to_head,
                "head_to_class": head_to_class,
                "geometry_stats": geometry_stats_by_group.get("N_MPC_lite_DSM", {}),
                "mpc_lite": {
                    "alpha": float(args.mpc_lite_alpha),
                    "topk": int(args.mpc_lite_topk),
                    "tau": float(args.mpc_lite_tau),
                    "gamma": float(args.mpc_lite_gamma),
                },
            },
            output_dir / "projector_mpc_lite_dsm.pt",
        )

    config = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "output_dir": str(output_dir),
        "requested_groups": sorted(requested),
        "seed": int(args.seed),
        "device": str(device),
        "paths": {
            "phase0_ckpt": args.phase0_ckpt,
            "phase1_ckpt": args.phase1_ckpt,
            "exemplar_csv": args.exemplar_csv,
            "calibration_json": args.calibration_json,
            "diagnostics_json": args.diagnostics_json,
            "locked_json": args.locked_json,
            "dsm_json": args.dsm_json,
            "dsm_projector": args.dsm_projector,
        },
        "split": {
            "old_classes": old_classes,
            "current_classes": current_classes,
            "seen_classes": seen_classes,
            "class_to_head": phase1_class_to_head,
        },
        "mpc_lite": {
            "alpha": float(args.mpc_lite_alpha),
            "topk": int(args.mpc_lite_topk),
            "tau": float(args.mpc_lite_tau),
            "gamma": float(args.mpc_lite_gamma),
            "semantic_prior_json": args.semantic_prior_json,
            "semantic_prior_csv": args.semantic_prior_csv,
        },
        "projector": {
            "input_dim": input_dim,
            "hidden": int(args.projector_hidden),
            "dim": int(args.projector_dim),
            "base_epochs": int(args.base_projector_epochs),
            "increment_epochs": int(args.increment_projector_epochs),
            "lr": float(args.lr),
            "sample_num_old": int(args.sample_num_old),
            "sample_num_current": int(args.sample_num_current),
            "cont_weight": float(args.cont_weight),
            "max_train_batches": args.max_train_batches,
        },
        "feature_counts_by_head": {
            "base": base_stats.counts_by_head,
            "current": current_stats.counts_by_head,
            "old_exemplars": exemplar_stats.counts_by_head,
            "test": test_stats.counts_by_head,
        },
        "freeze_checks": {
            "phase0": phase0_freeze,
            "phase1": phase1_freeze,
            "optimizer_scope": "DSMProjector.parameters_only",
            "backbone_checkpoint_saved": False,
        },
        "reference_checks": {
            "C_expected": EXPECTED_C,
            "D_expected": EXPECTED_D,
            "tolerance": 0.02,
            "passed": True,
        },
        "c_recomputed_metrics": c_recomputed_metrics,
    }
    _write_json(output_dir / "module_ablation_config.json", config)

    artifact_paths = {
        "output_dir": str(output_dir),
        "final_results": str(output_dir / "final_results.json"),
        "summary": str(output_dir / "summary.csv"),
        "delta_vs_C": str(output_dir / "delta_vs_C.csv"),
        "delta_vs_D": str(output_dir / "delta_vs_D.csv"),
        "mpc_lite_calibration_stats": str(output_dir / "mpc_lite_calibration_stats.csv"),
        "mpc_lite_topk_memory": str(output_dir / "mpc_lite_topk_memory.csv"),
        "per_class_metrics": str(output_dir / "per_class_metrics.csv"),
        "module_ablation_config": str(output_dir / "module_ablation_config.json"),
        "run_summary": str(output_dir / "run_summary.md"),
        "projector_mpc_lite_dsm": str(output_dir / "projector_mpc_lite_dsm.pt"),
    }
    payload = {
        **config,
        "summary_rows": summary_rows,
        "acceptance": acceptance,
        "geometry_stats": geometry_stats_by_group,
        "artifact_paths": artifact_paths,
    }
    _write_json(output_dir / "final_results.json", payload)
    report = _build_report(payload)
    (output_dir / "run_summary.md").write_text(report, encoding="utf-8")
    docs_path = REPO_ROOT / "docs" / "HYPERKVASIR23_CONCM_MODULE_ABLATION_RESULTS.md"
    docs_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
