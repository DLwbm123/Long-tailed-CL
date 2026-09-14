#!/usr/bin/env python3
"""DSM-assisted locked NC-ConCM gate for HyperKvasir23 phase1 checkpoints.

This is intentionally evaluation-only for the NC-ConCM branch: the locked
task-block + NC correction remains the main classifier, while a DSM projector is
trained on frozen feature tensors and used as an auxiliary residual at
evaluation time.
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
from torch import nn
from torch.utils.data import DataLoader

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
from src.models.resnet_cifar import resnet32
from src.utils.seed import set_seed
from train import apply_method_aliases, build_parser

from diagnose_hyperkvasir_task_block_calibration import (
    ExemplarPathDataset,
    _load_exemplar_rows,
    _write_csv,
    _write_json,
)
from run_hyperkvasir_concm_min_gate import (
    _block_confusion_rows,
    _collect_logits_features,
    _confusion_rows,
    _correction_vector,
    _evaluate_pack,
    _per_class_metrics_rows,
    _risk_and_reliability,
)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def _make_protocol_args(args: argparse.Namespace) -> argparse.Namespace:
    parser = build_parser()
    parsed = parser.parse_args([])
    parsed.dataset = "hyper_kvasir23"
    parsed.data_root = args.data_root
    parsed.method = "finetune"
    parsed.order = "shuffled"
    parsed.seed = int(args.seed)
    parsed.base_classes = 13
    parsed.incremental_steps = 5
    parsed.max_phases = 2
    parsed.epochs = 5
    parsed.batch_size = int(args.batch_size)
    parsed.num_workers = int(args.num_workers)
    parsed.lr = 0.01
    parsed.scheduler = "cosine"
    parsed.device = args.device
    parsed.download = False
    return apply_method_aliases(parsed)


def _freeze_all(model: nn.Module) -> Dict[str, object]:
    trainable_before = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    trainable_after = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    return {
        "trainable_params_before_freeze": trainable_before,
        "trainable_params_after_freeze": trainable_after,
        "backbone_requires_grad_false": len(trainable_after) == 0,
    }


def _make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool = False, seed: int = 0) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def _stack_stats(stats_by_head: Mapping[int, torch.Tensor], heads: Sequence[int]) -> torch.Tensor:
    missing = [int(head) for head in heads if int(head) not in stats_by_head]
    if missing:
        raise RuntimeError(f"Missing feature stats for head indices: {missing}")
    return torch.stack([stats_by_head[int(head)] for head in heads], dim=0)


def _parse_float_grid(value: str) -> List[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Float grid cannot be empty")
    return parsed


def _class_maps(class_ids: Sequence[int]) -> tuple[Dict[int, int], Dict[int, int]]:
    class_to_head = {int(class_id): idx for idx, class_id in enumerate(class_ids)}
    head_to_class = {idx: int(class_id) for class_id, idx in class_to_head.items()}
    return class_to_head, head_to_class


def _safe_percent(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return 100.0 * float(numerator) / float(denominator)


def _harmonic_mean(left, right) -> float | None:
    if left is None or right is None:
        return None
    left_f = float(left)
    right_f = float(right)
    if left_f + right_f <= 0.0:
        return 0.0
    return 2.0 * left_f * right_f / (left_f + right_f)


def _with_common_aliases(row: Mapping[str, object]) -> Dict[str, object]:
    out = dict(row)
    out["HM_old_current"] = _harmonic_mean(out.get("old_acc"), out.get("current_acc"))
    out["old_to_current"] = out.get("old_to_current_rate")
    out["current_to_old"] = out.get("current_to_old_rate")
    return out


def _selected_alpha(calibration: Mapping[str, object], rule: str) -> float:
    for row in calibration.get("selected_test_rows", []):
        if row.get("rule") == rule:
            return float(row["alpha"])
    raise ValueError(f"Calibration JSON does not contain selected_test_rows rule={rule!r}")


def _locked_reference_row(locked: Mapping[str, object] | None, rule: str) -> Dict[str, object] | None:
    if not locked:
        return None
    for row in locked.get("summary_rows", []):
        if row.get("group") in {"C_locked_NCConCM", "C_task_block_nc_concm"} and row.get("alpha_rule") == rule:
            return _with_common_aliases(dict(row))
    return None


def _locked_logits_and_labels(
    pack: Mapping[str, torch.Tensor],
    seen_classes: Sequence[int],
    current_classes: Sequence[int],
    alpha: float,
    correction: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    class_to_head, _ = _class_maps(seen_classes)
    logits = pack["logits"].clone().float()
    for class_id in current_classes:
        logits[:, class_to_head[int(class_id)]] *= float(alpha)
    if correction is not None:
        logits += correction.view(1, -1).cpu().float()
    labels_global = pack["labels_global"].long()
    labels_head = torch.tensor([class_to_head[int(label)] for label in labels_global.tolist()], dtype=torch.long)
    return logits, labels_head


def _metrics_from_logits(
    logits: torch.Tensor,
    labels_head: torch.Tensor,
    seen_classes: Sequence[int],
    old_classes: Sequence[int],
    current_classes: Sequence[int],
) -> tuple[Dict[str, object], Dict[int, Dict[int, int]]]:
    class_to_head, head_to_class = _class_maps(seen_classes)
    old_head_set = {class_to_head[int(class_id)] for class_id in old_classes}
    current_head_set = {class_to_head[int(class_id)] for class_id in current_classes}
    preds = logits.argmax(dim=1).cpu()
    labels_head = labels_head.cpu()
    matches = preds.eq(labels_head)
    confusion: Dict[int, Dict[int, int]] = {}
    per_class: Dict[int, Dict[str, int]] = {}
    old_total = old_correct = current_total = current_correct = 0
    old_pred_current = current_pred_old = 0
    for idx, label_head in enumerate(labels_head.tolist()):
        pred_head = int(preds[idx].item())
        true_class = int(head_to_class[int(label_head)])
        pred_class = int(head_to_class[pred_head])
        confusion.setdefault(true_class, {})
        confusion[true_class][pred_class] = confusion[true_class].get(pred_class, 0) + 1
        true_item = per_class.setdefault(true_class, {"correct": 0, "total": 0, "predicted": 0})
        true_item["total"] += 1
        if bool(matches[idx].item()):
            true_item["correct"] += 1
        pred_item = per_class.setdefault(pred_class, {"correct": 0, "total": 0, "predicted": 0})
        pred_item["predicted"] += 1
        if int(label_head) in old_head_set:
            old_total += 1
            old_correct += int(matches[idx].item())
            old_pred_current += int(pred_head in current_head_set)
        if int(label_head) in current_head_set:
            current_total += 1
            current_correct += int(matches[idx].item())
            current_pred_old += int(pred_head in old_head_set)

    recalls: List[float] = []
    f1_values: List[float] = []
    for stats in per_class.values():
        total = int(stats.get("total", 0))
        predicted = int(stats.get("predicted", 0))
        correct = int(stats.get("correct", 0))
        if total <= 0:
            continue
        recall = correct / total
        precision = correct / predicted if predicted else 0.0
        recalls.append(recall)
        f1_values.append(2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0)
    total = int(labels_head.numel())
    correct = int(matches.sum().item())
    row = {
        "AccT": _safe_percent(correct, total),
        "balanced_acc": 100.0 * sum(recalls) / len(recalls) if recalls else 0.0,
        "macro_f1": 100.0 * sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "old_acc": _safe_percent(old_correct, old_total),
        "current_acc": _safe_percent(current_correct, current_total),
        "old_to_current_rate": _safe_percent(old_pred_current, old_total),
        "current_to_old_rate": _safe_percent(current_pred_old, current_total),
        "total": total,
        "old_total": old_total,
        "current_total": current_total,
    }
    return row, confusion


@torch.no_grad()
def _dsm_logits(
    features: torch.Tensor,
    projector: DSMProjector,
    geometry: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    projector.eval()
    features = features.to(device=device, dtype=torch.float32)
    geometry = F.normalize(geometry.to(device=device, dtype=torch.float32), p=2, dim=1)
    return (projector(features) @ geometry.transpose(0, 1)).detach().cpu()


def _summary_row(
    group: str,
    method: str,
    metrics: Mapping[str, object],
    *,
    seed: int,
    lambda_dsm: float | None = None,
    alpha: float | None = None,
    alpha_rule: str | None = None,
    nc_lambda: float | None = None,
    note: str | None = None,
) -> Dict[str, object]:
    row = {
        "group": group,
        "method": method,
        "seed": int(seed),
        "lambda_dsm": lambda_dsm,
        "alpha_rule": alpha_rule,
        "alpha": alpha,
        "nc_lambda": nc_lambda,
        "AccT": metrics.get("AccT"),
        "balanced_acc": metrics.get("balanced_acc"),
        "macro_f1": metrics.get("macro_f1"),
        "old_acc": metrics.get("old_acc"),
        "current_acc": metrics.get("current_acc"),
        "old_to_current_rate": metrics.get("old_to_current_rate"),
        "current_to_old_rate": metrics.get("current_to_old_rate"),
        "total": metrics.get("total"),
    }
    if note:
        row["note"] = note
    return _with_common_aliases(row)


def _format_float(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def _markdown_table(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format_float(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def _build_report(payload: Mapping[str, object]) -> str:
    columns = [
        "method",
        "seed",
        "lambda_dsm",
        "AccT",
        "old_acc",
        "current_acc",
        "HM_old_current",
        "old_to_current",
        "current_to_old",
    ]
    lambda0 = payload.get("lambda0_reproduction", {})
    lines = [
        "# HyperKvasir23 Locked NC-DSM Auxiliary Gate",
        "",
        "## Scope",
        "",
        "Evaluation-only locked NC-ConCM branch plus a trained DSM auxiliary projector. No MPC-lite, no Medical-MPC, no phase2/full phases.",
        "",
        "The main prediction branch is the locked task-block + NC-ConCM correction. DSM is trained on frozen features with `LMatch + lambda_cont * LCont`; `lambda_dsm` controls the auxiliary normalized residual at evaluation time.",
        "",
        "## Exact Command",
        "",
        "```bash",
        str(payload["command"]),
        "```",
        "",
        "## Summary",
        "",
        _markdown_table(payload["summary_rows"], columns),
        "",
        "## Lambda 0 Sanity",
        "",
        "```json",
        json.dumps(lambda0, indent=2, sort_keys=True),
        "```",
        "",
        "## Best Rows",
        "",
        _markdown_table(payload["best_rows"], columns),
        "",
        "## Artifacts",
        "",
        f"- Output dir: `{payload['output_dir']}`",
        "- `final_results.json`",
        "- `summary.csv`",
        "- `lambda_sweep.csv`",
        "- `train_trace_dsm_aux.csv`",
        "- `geometry_stats_dsm_aux.json`",
        "- `confusion_matrix_lambda_*.csv`",
        "- `projector_dsm_aux.pt`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--phase0-ckpt", required=True)
    parser.add_argument("--phase1-ckpt", required=True)
    parser.add_argument("--exemplar-csv", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--diagnostics-json", required=True)
    parser.add_argument("--locked-json", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--alpha-rule", default="balanced_hmean_exemplar")
    parser.add_argument("--locked-nc-lambda", type=float, default=0.30)
    parser.add_argument("--lambda-dsm-grid", default="0.0,0.05,0.1,0.2,0.5")
    parser.add_argument("--projector-hidden", type=int, default=2048)
    parser.add_argument("--projector-dim", type=int, default=128)
    parser.add_argument("--base-projector-epochs", type=int, default=5)
    parser.add_argument("--increment-projector-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--sample-num-old", type=int, default=100)
    parser.add_argument("--sample-num-current", type=int, default=50)
    parser.add_argument("--cont-weight", type=float, default=1.0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(args.seed))
    device = torch.device(args.device)
    lambda_grid = _parse_float_grid(args.lambda_dsm_grid)
    if 0.0 not in lambda_grid:
        lambda_grid = [0.0, *lambda_grid]

    protocol = build_protocol(_make_protocol_args(args))
    phase0_payload = torch.load(args.phase0_ckpt, map_location="cpu")
    phase1_payload = torch.load(args.phase1_ckpt, map_location=device)
    calibration = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))
    diagnostics = json.loads(Path(args.diagnostics_json).read_text(encoding="utf-8"))
    locked_payload = json.loads(Path(args.locked_json).read_text(encoding="utf-8")) if args.locked_json else None

    old_classes = [int(class_id) for class_id in phase0_payload["old_classes"]]
    current_classes = [int(class_id) for class_id in phase0_payload["current_classes"]]
    seen_classes = old_classes + current_classes
    if [int(class_id) for class_id in phase1_payload["seen_classes"]] != seen_classes:
        raise RuntimeError("phase1 checkpoint seen_classes do not match phase0 metadata")
    if old_classes != [int(class_id) for class_id in protocol.tasks[0]]:
        raise RuntimeError("phase0 old classes do not match HyperKvasir23 protocol task0")
    if current_classes != [int(class_id) for class_id in protocol.tasks[1]]:
        raise RuntimeError("phase1 current classes do not match HyperKvasir23 protocol task1")

    class_to_head, head_to_class = _class_maps(seen_classes)
    phase0_class_to_head = {int(class_id): idx for idx, class_id in enumerate(old_classes)}
    alpha = _selected_alpha(calibration, args.alpha_rule)

    phase0_model = resnet32().to(device)
    phase0_model.expand_classifier(len(old_classes))
    phase0_model.load_state_dict(phase0_payload["model_state"])
    phase1_model = resnet32().to(device)
    phase1_model.expand_classifier(len(seen_classes))
    phase1_model.load_state_dict(phase1_payload["model_state"])
    phase0_freeze = _freeze_all(phase0_model)
    phase1_freeze = _freeze_all(phase1_model)

    phase0_proto_loader = _make_loader(protocol.prototype_dataset_for_classes(old_classes), args.batch_size, args.num_workers, shuffle=False, seed=int(args.seed))
    current_proto_loader = _make_loader(protocol.prototype_dataset_for_classes(current_classes), args.batch_size, args.num_workers, shuffle=False, seed=int(args.seed) + 1)
    current_train_loader = _make_loader(protocol.train_dataset_for_classes(current_classes), args.batch_size, args.num_workers, shuffle=False, seed=int(args.seed) + 2)
    seen_test_loader = _make_loader(protocol.test_dataset_for_classes(seen_classes), args.batch_size, args.num_workers, shuffle=False, seed=int(args.seed) + 3)
    exemplar_rows = _load_exemplar_rows(Path(args.exemplar_csv))
    exemplar_loader = _make_loader(
        ExemplarPathDataset(exemplar_rows, protocol.eval_transform),
        max(1, min(int(args.batch_size), len(exemplar_rows))),
        args.num_workers,
        shuffle=False,
        seed=int(args.seed) + 4,
    )

    seen_pack = _collect_logits_features(phase1_model, seen_test_loader, device)
    current_train_pack = _collect_logits_features(phase1_model, current_train_loader, device)
    exemplar_pack = _collect_logits_features(phase1_model, exemplar_loader, device)
    old_risk, current_absorb, current_to_old_risk, reliability, reliability_rows = _risk_and_reliability(
        exemplar_pack,
        current_train_pack,
        diagnostics,
        phase0_payload,
        phase1_model,
        seen_classes,
        old_classes,
        current_classes,
        alpha,
    )
    locked_correction = _correction_vector(
        seen_classes,
        old_classes,
        current_classes,
        old_risk,
        current_absorb,
        reliability,
        float(args.locked_nc_lambda),
        use_reliability=False,
    )
    locked_metrics, locked_confusion = _evaluate_pack(seen_pack, seen_classes, old_classes, current_classes, alpha=alpha, correction=locked_correction)
    locked_row = _summary_row(
        "C_locked_NCConCM_recomputed",
        "locked_NCConCM_recomputed",
        locked_metrics,
        seed=args.seed,
        lambda_dsm=0.0,
        alpha=alpha,
        alpha_rule=args.alpha_rule,
        nc_lambda=float(args.locked_nc_lambda),
        note="main_prediction_branch",
    )
    locked_reference = _locked_reference_row(locked_payload, args.alpha_rule)

    base_stats = extract_features_by_class(phase0_model, phase0_proto_loader, phase0_class_to_head, device)
    current_stats = extract_features_by_class(phase1_model, current_proto_loader, class_to_head, device)
    exemplar_stats = extract_features_by_class(phase1_model, exemplar_loader, class_to_head, device)
    old_heads = list(range(len(old_classes)))
    current_heads = list(range(len(old_classes), len(seen_classes)))
    base_means = _stack_stats(base_stats.means_by_head, old_heads)
    base_stds = _stack_stats(base_stats.stds_by_head, old_heads)
    current_means = _stack_stats(current_stats.means_by_head, current_heads)
    current_stds = _stack_stats(current_stats.stds_by_head, current_heads)
    combined_means = torch.cat([base_means, current_means], dim=0)
    combined_stds = torch.cat([base_stds, current_stds], dim=0)
    real_features = torch.cat([exemplar_stats.features, current_stats.features], dim=0)
    real_labels = torch.cat([exemplar_stats.head_labels, current_stats.head_labels], dim=0)
    input_dim = int(combined_means.shape[1])

    config = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "method": "locked_nc_dsm_aux",
        "alias": "dsm_assisted_nc_concm",
        "output_dir": str(output_dir),
        "data_root": args.data_root,
        "phase0_ckpt": args.phase0_ckpt,
        "phase1_ckpt": args.phase1_ckpt,
        "exemplar_csv": args.exemplar_csv,
        "calibration_json": args.calibration_json,
        "diagnostics_json": args.diagnostics_json,
        "locked_json": args.locked_json,
        "seed": int(args.seed),
        "device": str(device),
        "alpha_rule": args.alpha_rule,
        "alpha": alpha,
        "locked_nc_lambda": float(args.locked_nc_lambda),
        "lambda_dsm_grid": lambda_grid,
        "projector_hidden": int(args.projector_hidden),
        "projector_dim": int(args.projector_dim),
        "base_projector_epochs": int(args.base_projector_epochs),
        "increment_projector_epochs": int(args.increment_projector_epochs),
        "lr": float(args.lr),
        "sample_num_old": int(args.sample_num_old),
        "sample_num_current": int(args.sample_num_current),
        "cont_weight": float(args.cont_weight),
        "max_train_batches": args.max_train_batches,
        "loss_contract": "L_nc is the frozen locked NC-ConCM branch; DSM projector optimizes L_match_dsm + lambda_cont * L_cont_dsm, and lambda_dsm controls the auxiliary residual at evaluation time.",
        "split": {
            "old_classes": old_classes,
            "current_classes": current_classes,
            "seen_classes": seen_classes,
            "class_order": [int(class_id) for class_id in protocol.class_order],
            "class_to_head": class_to_head,
        },
        "freeze_checks": {
            "phase0": phase0_freeze,
            "phase1": phase1_freeze,
            "optimizer_scope": "DSMProjector.parameters_only",
        },
    }
    _write_json(output_dir / "dsm_aux_config.json", config)

    if args.dry_run:
        payload = {
            **config,
            "status": "dry_run_ok",
            "locked_row": locked_row,
            "locked_reference_row": locked_reference,
            "reliability_rows": reliability_rows,
        }
        _write_json(output_dir / "dry_run.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
        return

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
            no_cont_loss=False,
            device=device,
            max_train_batches=args.max_train_batches,
            seed=int(args.seed) + 10,
        )
        for row in base_trace:
            item = dict(row)
            item["stage"] = "base_projector_alignment"
            trace_rows.append(item)

    projector, geometry, inc_trace, geometry_stats = train_dsm_projector(
        means=combined_means,
        stds=combined_stds,
        projector=projector,
        epochs=int(args.increment_projector_epochs),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        sample_num_old=int(args.sample_num_old),
        sample_num_current=int(args.sample_num_current),
        num_old_classes=len(old_classes),
        cont_weight=float(args.cont_weight),
        no_cont_loss=False,
        real_features=real_features,
        real_labels=real_labels,
        device=device,
        max_train_batches=args.max_train_batches,
        seed=int(args.seed) + 20,
    )
    for row in inc_trace:
        item = dict(row)
        item["stage"] = "incremental_dsm_adaptation"
        trace_rows.append(item)

    dsm_metrics = evaluate_dsm(
        seen_pack["features"],
        torch.tensor([class_to_head[int(label)] for label in seen_pack["labels_global"].long().tolist()], dtype=torch.long),
        projector,
        geometry,
        old_head_indices=old_heads,
        current_head_indices=current_heads,
        device=device,
    )
    dsm_row = _summary_row(
        "D_pure_DSM_aux_projector",
        "pure_DSM_aux_projector",
        dsm_metrics,
        seed=args.seed,
        lambda_dsm=None,
        alpha=None,
        alpha_rule=None,
        nc_lambda=None,
        note="standalone_DSM_projector_check",
    )

    locked_logits, labels_head = _locked_logits_and_labels(seen_pack, seen_classes, current_classes, alpha, locked_correction)
    dsm_residual = _dsm_logits(seen_pack["features"], projector, geometry, device)
    locked_norm = F.normalize(locked_logits, p=2, dim=1)
    dsm_norm = F.normalize(dsm_residual, p=2, dim=1)
    lambda_rows: List[Dict[str, object]] = []
    confusion_by_lambda: Dict[str, Mapping[int, Mapping[int, int]]] = {}
    for lambda_dsm in lambda_grid:
        aux_logits = locked_norm + float(lambda_dsm) * dsm_norm
        metrics, confusion = _metrics_from_logits(aux_logits, labels_head, seen_classes, old_classes, current_classes)
        row = _summary_row(
            "H_locked_NC_DSM_aux",
            "locked_nc_dsm_aux",
            metrics,
            seed=args.seed,
            lambda_dsm=float(lambda_dsm),
            alpha=alpha,
            alpha_rule=args.alpha_rule,
            nc_lambda=float(args.locked_nc_lambda),
            note="normalized_locked_logits_plus_lambda_dsm_normalized_dsm_logits",
        )
        lambda_rows.append(row)
        key = f"{float(lambda_dsm):.4f}".replace(".", "p")
        confusion_by_lambda[key] = confusion
        _write_csv(output_dir / f"confusion_matrix_lambda_{key}.csv", _confusion_rows(confusion, seen_classes))
        _write_csv(output_dir / f"per_class_metrics_lambda_{key}.csv", _per_class_metrics_rows(confusion, seen_classes))
        _write_csv(output_dir / f"old_to_current_confusion_lambda_{key}.csv", _block_confusion_rows(confusion, old_classes, current_classes, "old_to_current"))
        _write_csv(output_dir / f"current_to_old_confusion_lambda_{key}.csv", _block_confusion_rows(confusion, current_classes, old_classes, "current_to_old"))

    lambda0_row = next(row for row in lambda_rows if float(row["lambda_dsm"]) == 0.0)
    lambda0_reproduction = {
        "matches_recomputed_locked": all(
            abs(float(lambda0_row.get(key) or 0.0) - float(locked_row.get(key) or 0.0)) < 1e-9
            for key in ["AccT", "old_acc", "current_acc", "old_to_current_rate", "current_to_old_rate"]
        ),
        "delta_vs_recomputed_locked": {
            key: float(lambda0_row.get(key) or 0.0) - float(locked_row.get(key) or 0.0)
            for key in ["AccT", "old_acc", "current_acc", "old_to_current_rate", "current_to_old_rate"]
        },
        "locked_reference_available": locked_reference is not None,
    }
    if locked_reference is not None:
        lambda0_reproduction["delta_vs_locked_json"] = {
            key: float(lambda0_row.get(key) or 0.0) - float(locked_reference.get(key) or 0.0)
            for key in ["AccT", "old_acc", "current_acc", "old_to_current_rate", "current_to_old_rate"]
        }

    best_by_acct = sorted(lambda_rows, key=lambda row: float(row.get("AccT") or -1.0), reverse=True)[:3]
    best_by_old_to_current = sorted(lambda_rows, key=lambda row: (float(row.get("old_to_current_rate") or 100.0), -float(row.get("AccT") or 0.0)))[:3]
    best_rows = []
    for row in [*best_by_acct, *best_by_old_to_current]:
        if not any(float(existing["lambda_dsm"]) == float(row["lambda_dsm"]) for existing in best_rows):
            best_rows.append(row)

    summary_rows: List[Dict[str, object]] = [locked_row]
    if locked_reference is not None:
        ref = dict(locked_reference)
        ref["method"] = "locked_NCConCM_reference_json"
        ref["lambda_dsm"] = 0.0
        ref["seed"] = int(args.seed)
        summary_rows.append(ref)
    summary_rows.append(dsm_row)
    summary_rows.extend(lambda_rows)

    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_csv(output_dir / "lambda_sweep.csv", lambda_rows)
    _write_csv(output_dir / "best_rows.csv", best_rows)
    _write_csv(output_dir / "train_trace_dsm_aux.csv", trace_rows)
    _write_json(output_dir / "geometry_stats_dsm_aux.json", geometry_stats)
    _write_csv(output_dir / "reliability_rows.csv", reliability_rows)
    _write_csv(output_dir / "locked_confusion_matrix.csv", _confusion_rows(locked_confusion, seen_classes))
    torch.save(
        {
            "projector_state_dict": projector.state_dict(),
            "projector_config": projector.config(),
            "geometry_vectors": geometry.detach().cpu(),
            "seen_classes": seen_classes,
            "class_to_head": class_to_head,
            "head_to_class": head_to_class,
            "geometry_stats": geometry_stats,
        },
        output_dir / "projector_dsm_aux.pt",
    )

    payload = {
        **config,
        "output_dir": str(output_dir),
        "locked_row": locked_row,
        "locked_reference_row": locked_reference,
        "dsm_row": dsm_row,
        "lambda_rows": lambda_rows,
        "summary_rows": summary_rows,
        "best_rows": best_rows,
        "lambda0_reproduction": lambda0_reproduction,
        "train_trace_rows": trace_rows,
        "geometry_stats": geometry_stats,
        "reliability_rows": reliability_rows,
        "artifact_paths": {
            "final_results": str(output_dir / "final_results.json"),
            "summary_csv": str(output_dir / "summary.csv"),
            "lambda_sweep_csv": str(output_dir / "lambda_sweep.csv"),
            "best_rows_csv": str(output_dir / "best_rows.csv"),
            "train_trace": str(output_dir / "train_trace_dsm_aux.csv"),
            "geometry_stats": str(output_dir / "geometry_stats_dsm_aux.json"),
            "run_summary": str(output_dir / "run_summary.md"),
            "projector": str(output_dir / "projector_dsm_aux.pt"),
        },
    }
    _write_json(output_dir / "final_results.json", payload)
    report = _build_report(payload)
    (output_dir / "run_summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
