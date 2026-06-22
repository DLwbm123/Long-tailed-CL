#!/usr/bin/env python3
"""Evaluation-only ConCM-min gate for HyperKvasir23 freeze checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets import build_protocol
from src.models.resnet_cifar import resnet32
from train import apply_method_aliases, build_parser

from diagnose_hyperkvasir_task_block_calibration import (
    ExemplarPathDataset,
    _load_exemplar_rows,
    _make_loader,
    _write_csv,
    _write_json,
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


def _safe_percent(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return 100.0 * float(numerator) / float(denominator)


def _balanced_f1(per_class: Mapping[int, Mapping[str, int]]) -> tuple[float, float]:
    recalls: List[float] = []
    f1_values: List[float] = []
    for stats in per_class.values():
        total = int(stats.get("total", 0))
        predicted = int(stats.get("predicted", 0))
        correct = int(stats.get("correct", 0))
        if total <= 0:
            continue
        recall = correct / total
        precision = correct / predicted if predicted > 0 else 0.0
        recalls.append(recall)
        f1_values.append(2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)
    return (
        100.0 * sum(recalls) / len(recalls) if recalls else 0.0,
        100.0 * sum(f1_values) / len(f1_values) if f1_values else 0.0,
    )


@torch.no_grad()
def _collect_logits_features(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, torch.Tensor]:
    model.eval()
    logits_list: List[torch.Tensor] = []
    features_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    for images, labels in loader:
        images = images.to(device)
        features = model.extract_features(images)
        logits = model.classifier(features)
        features_list.append(features.detach().cpu())
        logits_list.append(logits.detach().cpu())
        labels_list.append(labels.detach().cpu())
    return {
        "logits": torch.cat(logits_list, dim=0),
        "features": torch.cat(features_list, dim=0),
        "labels_global": torch.cat(labels_list, dim=0),
    }


def _class_maps(class_ids: Sequence[int]) -> tuple[Dict[int, int], Dict[int, int]]:
    class_to_head = {int(class_id): idx for idx, class_id in enumerate(class_ids)}
    head_to_class = {idx: int(class_id) for class_id, idx in class_to_head.items()}
    return class_to_head, head_to_class


def _evaluate_pack(
    pack: Mapping[str, torch.Tensor],
    seen_classes: Sequence[int],
    old_classes: Sequence[int],
    current_classes: Sequence[int],
    alpha: float,
    correction: torch.Tensor | None = None,
) -> tuple[Dict[str, object], Dict[int, Dict[int, int]]]:
    class_to_head, head_to_class = _class_maps(seen_classes)
    old_head_set = set(class_to_head[int(class_id)] for class_id in old_classes)
    current_head_set = set(class_to_head[int(class_id)] for class_id in current_classes)
    logits = pack["logits"].clone()
    for class_id in current_classes:
        logits[:, class_to_head[int(class_id)]] *= float(alpha)
    if correction is not None:
        logits += correction.view(1, -1).cpu()
    labels_global = pack["labels_global"].long()
    labels_head = torch.tensor([class_to_head[int(x)] for x in labels_global.tolist()], dtype=torch.long)
    preds = logits.argmax(dim=1)
    matches = preds.eq(labels_head)
    per_class: Dict[int, Dict[str, int]] = {}
    confusion: Dict[int, Dict[int, int]] = {}
    total = int(labels_head.numel())
    correct = int(matches.sum().item())
    old_total = old_correct = current_total = current_correct = 0
    old_pred_current = current_pred_old = 0
    for idx, label_head in enumerate(labels_head.tolist()):
        pred_head = int(preds[idx].item())
        true_class = int(head_to_class[int(label_head)])
        pred_class = int(head_to_class[pred_head])
        confusion.setdefault(true_class, {})
        confusion[true_class][pred_class] = confusion[true_class].get(pred_class, 0) + 1
        item = per_class.setdefault(true_class, {"correct": 0, "total": 0, "predicted": 0})
        item["total"] += 1
        if bool(matches[idx].item()):
            item["correct"] += 1
        pred_item = per_class.setdefault(pred_class, {"correct": 0, "total": 0, "predicted": 0})
        pred_item["predicted"] += 1
        if label_head in old_head_set:
            old_total += 1
            old_correct += int(matches[idx].item())
            old_pred_current += int(pred_head in current_head_set)
        if label_head in current_head_set:
            current_total += 1
            current_correct += int(matches[idx].item())
            current_pred_old += int(pred_head in old_head_set)
    balanced, macro = _balanced_f1(per_class)
    return (
        {
            "AccT": _safe_percent(correct, total),
            "balanced_acc": balanced,
            "macro_f1": macro,
            "old_acc": _safe_percent(old_correct, old_total),
            "current_acc": _safe_percent(current_correct, current_total),
            "old_to_current_rate": _safe_percent(old_pred_current, old_total),
            "current_to_old_rate": _safe_percent(current_pred_old, current_total),
            "total": total,
            "alpha": float(alpha),
        },
        confusion,
    )


def _confusion_rows(confusion: Mapping[int, Mapping[int, int]], seen_classes: Sequence[int]) -> List[Dict[str, object]]:
    rows = []
    for true_class in seen_classes:
        total = sum(int(v) for v in confusion.get(int(true_class), {}).values())
        row = {"true_class": int(true_class), "total": total}
        for pred_class in seen_classes:
            row[f"pred_{int(pred_class)}"] = int(confusion.get(int(true_class), {}).get(int(pred_class), 0))
        rows.append(row)
    return rows


def _block_confusion_rows(
    confusion: Mapping[int, Mapping[int, int]],
    source_classes: Sequence[int],
    target_classes: Sequence[int],
    block_name: str,
) -> List[Dict[str, object]]:
    rows = []
    for true_class in source_classes:
        total = sum(int(v) for v in confusion.get(int(true_class), {}).values())
        target_count = sum(int(confusion.get(int(true_class), {}).get(int(pred), 0)) for pred in target_classes)
        best_target = None
        best_count = 0
        for target in target_classes:
            count = int(confusion.get(int(true_class), {}).get(int(target), 0))
            if count > best_count:
                best_count = count
                best_target = int(target)
        rows.append(
            {
                "block": block_name,
                "true_class": int(true_class),
                "total": total,
                "target_block_count": target_count,
                "target_block_rate": _safe_percent(target_count, total),
                "top_target_class": best_target,
                "top_target_count": best_count,
            }
        )
    return rows


def _per_class_metrics_rows(confusion: Mapping[int, Mapping[int, int]], seen_classes: Sequence[int]) -> List[Dict[str, object]]:
    predicted_counts = {int(class_id): 0 for class_id in seen_classes}
    for pred_map in confusion.values():
        for pred_class, count in pred_map.items():
            predicted_counts[int(pred_class)] = predicted_counts.get(int(pred_class), 0) + int(count)
    rows = []
    for true_class in seen_classes:
        total = sum(int(v) for v in confusion.get(int(true_class), {}).values())
        correct = int(confusion.get(int(true_class), {}).get(int(true_class), 0))
        predicted = int(predicted_counts.get(int(true_class), 0))
        precision = 100.0 * correct / predicted if predicted else 0.0
        recall = 100.0 * correct / total if total else None
        f1 = 2.0 * precision * recall / (precision + recall) if recall is not None and precision + recall > 0 else 0.0
        rows.append(
            {
                "class_id": int(true_class),
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "correct": correct,
                "total": total,
                "predicted": predicted,
            }
        )
    return rows


def _proxy_predictions(pack: Mapping[str, torch.Tensor], seen_classes: Sequence[int], alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    class_to_head, _ = _class_maps(seen_classes)
    logits = pack["logits"].clone()
    current_heads = [idx for class_id, idx in class_to_head.items() if int(class_id) not in []]
    # Scaling is done by caller through explicit correction helper.
    return logits, pack["labels_global"].long()


def _risk_and_reliability(
    exemplar_pack: Mapping[str, torch.Tensor],
    current_train_pack: Mapping[str, torch.Tensor],
    diagnostics: Mapping[str, object],
    phase0_payload: Mapping[str, object],
    model: nn.Module,
    seen_classes: Sequence[int],
    old_classes: Sequence[int],
    current_classes: Sequence[int],
    alpha: float,
) -> tuple[Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float], List[Dict[str, object]]]:
    class_to_head, head_to_class = _class_maps(seen_classes)
    old_set = set(int(c) for c in old_classes)
    current_set = set(int(c) for c in current_classes)
    exemplar_metrics, exemplar_confusion = _evaluate_pack(exemplar_pack, seen_classes, old_classes, current_classes, alpha)
    current_metrics, current_confusion = _evaluate_pack(current_train_pack, seen_classes, old_classes, current_classes, alpha)
    old_risk: Dict[int, float] = {}
    current_absorb: Dict[int, float] = {int(c): 0.0 for c in current_classes}
    current_to_old_risk: Dict[int, float] = {}
    for old_class in old_classes:
        total = sum(exemplar_confusion.get(int(old_class), {}).values())
        old_to_current = sum(exemplar_confusion.get(int(old_class), {}).get(int(c), 0) for c in current_classes)
        old_risk[int(old_class)] = old_to_current / total if total else 0.0
        for current_class in current_classes:
            current_absorb[int(current_class)] += exemplar_confusion.get(int(old_class), {}).get(int(current_class), 0)
    denom = max(1, len(old_classes))
    current_absorb = {int(k): float(v) / denom for k, v in current_absorb.items()}
    for current_class in current_classes:
        total = sum(current_confusion.get(int(current_class), {}).values())
        current_to_old = sum(current_confusion.get(int(current_class), {}).get(int(c), 0) for c in old_classes)
        current_to_old_risk[int(current_class)] = current_to_old / total if total else 0.0

    phase0_per_class = {
        int(row["class_id"]): row
        for row in diagnostics.get("per_class_metrics", [])
        if row.get("split") == "phase0"
    }
    if not phase0_per_class and "phase0_quality" in diagnostics:
        # Fallback when only summary is available.
        phase0_per_class = {}
    anchors = {int(k): v.float() for k, v in phase0_payload["anchors"].items()}
    old_heads = torch.tensor([class_to_head[int(c)] for c in old_classes], dtype=torch.long)
    current_heads = torch.tensor([class_to_head[int(c)] for c in current_classes], dtype=torch.long)
    reliability: Dict[int, float] = {}
    rows = []
    for old_class in old_classes:
        anchor = anchors[int(old_class)].view(1, -1)
        logits = model.classifier(anchor.to(next(model.parameters()).device)).detach().cpu()
        anchor_ret = float((logits[:, old_heads].max(dim=1).values >= float(alpha) * logits[:, current_heads].max(dim=1).values).item())
        total = sum(exemplar_confusion.get(int(old_class), {}).values())
        correct = exemplar_confusion.get(int(old_class), {}).get(int(old_class), 0)
        exemplar_ret = correct / total if total else 0.0
        phase0_recall = None
        if int(old_class) in phase0_per_class and phase0_per_class[int(old_class)].get("recall") is not None:
            phase0_recall = float(phase0_per_class[int(old_class)]["recall"]) / 100.0
        else:
            phase0_recall = 0.5
        r = max(0.0, min(1.0, (anchor_ret + exemplar_ret + phase0_recall) / 3.0))
        reliability[int(old_class)] = r
        rows.append(
            {
                "class_id": int(old_class),
                "block": "old",
                "anchor_retention": anchor_ret,
                "exemplar_retention": exemplar_ret,
                "phase0_recall_proxy": phase0_recall,
                "old_to_current_risk": old_risk[int(old_class)],
                "reliability": r,
            }
        )
    for current_class in current_classes:
        total = sum(current_confusion.get(int(current_class), {}).values())
        correct = current_confusion.get(int(current_class), {}).get(int(current_class), 0)
        current_ret = correct / total if total else 0.0
        reliability[int(current_class)] = current_ret
        rows.append(
            {
                "class_id": int(current_class),
                "block": "current",
                "current_train_retention": current_ret,
                "current_to_old_risk": current_to_old_risk[int(current_class)],
                "current_absorb_old_exemplar_risk": current_absorb[int(current_class)],
                "reliability": current_ret,
            }
        )
    return old_risk, current_absorb, current_to_old_risk, reliability, rows


def _correction_vector(
    seen_classes: Sequence[int],
    old_classes: Sequence[int],
    current_classes: Sequence[int],
    old_risk: Mapping[int, float],
    current_absorb: Mapping[int, float],
    reliability: Mapping[int, float],
    lambda_value: float,
    use_reliability: bool,
    matching_strength: float = 0.0,
    drift_norm: Mapping[int, float] | None = None,
) -> torch.Tensor:
    class_to_head, _ = _class_maps(seen_classes)
    correction = torch.zeros(len(seen_classes), dtype=torch.float32)
    old_rel_values = [float(reliability.get(int(c), 1.0)) for c in old_classes]
    mean_old_reliability = sum(old_rel_values) / len(old_rel_values) if old_rel_values else 1.0
    for old_class in old_classes:
        gate = float(reliability.get(int(old_class), 1.0)) if use_reliability else 1.0
        delta = float(lambda_value) * float(old_risk.get(int(old_class), 0.0)) * gate
        if drift_norm is not None and matching_strength > 0.0:
            drift = float(drift_norm.get(int(old_class), 0.0))
            if gate >= 0.4 and drift >= 0.5:
                delta += float(matching_strength) * gate * drift
        correction[class_to_head[int(old_class)]] += delta
    for current_class in current_classes:
        gate = mean_old_reliability if use_reliability else 1.0
        delta = -float(lambda_value) * float(current_absorb.get(int(current_class), 0.0)) * gate
        correction[class_to_head[int(current_class)]] += delta
    return correction


def _select_diagnostic_row(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    candidates = [
        row
        for row in rows
        if row.get("current_acc") is not None
        and row.get("current_to_old_rate") is not None
        and float(row["current_acc"]) >= 85.0
        and float(row["current_to_old_rate"]) < 15.0
    ]
    if candidates:
        return max(candidates, key=lambda row: (float(row.get("old_acc") or 0.0), float(row.get("AccT") or 0.0)))
    return max(rows, key=lambda row: float(row.get("AccT") or -1.0))


def _prototype_relations(
    phase0_payload: Mapping[str, object],
    exemplar_pack: Mapping[str, torch.Tensor],
    old_classes: Sequence[int],
) -> tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    Dict[int, float],
]:
    anchors = {int(k): v.float() for k, v in phase0_payload["anchors"].items()}
    labels = exemplar_pack["labels_global"].long().tolist()
    features = exemplar_pack["features"].float()
    phase1_proto = {}
    for old_class in old_classes:
        idxs = [idx for idx, label in enumerate(labels) if int(label) == int(old_class)]
        if idxs:
            phase1_proto[int(old_class)] = features[idxs].mean(dim=0)
        else:
            phase1_proto[int(old_class)] = anchors[int(old_class)].float()
    phase0_mat = []
    phase1_mat = []
    drift_rows = []
    per_class_values = {int(c): [] for c in old_classes}
    for ci in old_classes:
        for cj in old_classes:
            s0 = float(F.cosine_similarity(anchors[int(ci)].view(1, -1), anchors[int(cj)].view(1, -1)).item())
            s1 = float(F.cosine_similarity(phase1_proto[int(ci)].view(1, -1), phase1_proto[int(cj)].view(1, -1)).item())
            drift = abs(s1 - s0)
            phase0_mat.append({"class_i": int(ci), "class_j": int(cj), "cosine": s0})
            phase1_mat.append({"class_i": int(ci), "class_j": int(cj), "cosine": s1})
            drift_rows.append({"class_i": int(ci), "class_j": int(cj), "drift_abs": drift, "phase0_cosine": s0, "phase1_cosine": s1})
            if int(ci) != int(cj):
                per_class_values[int(ci)].append(drift)
    per_class_rows = []
    raw = {}
    for class_id, values in per_class_values.items():
        mean = sum(values) / len(values) if values else 0.0
        raw[class_id] = mean
        per_class_rows.append(
            {
                "class_id": int(class_id),
                "relation_drift_mean": mean,
                "relation_drift_max": max(values) if values else 0.0,
            }
        )
    max_mean = max(raw.values()) if raw else 1.0
    drift_norm = {class_id: (value / max_mean if max_mean > 0 else 0.0) for class_id, value in raw.items()}
    for row in per_class_rows:
        row["relation_drift_norm"] = drift_norm[int(row["class_id"])]
    top_pairs = sorted([row for row in drift_rows if row["class_i"] != row["class_j"]], key=lambda row: row["drift_abs"], reverse=True)[:25]
    return phase0_mat, phase1_mat, drift_rows, top_pairs, per_class_rows, drift_norm


def _with_meta(row: Mapping[str, object], **meta) -> Dict[str, object]:
    output = dict(meta)
    output.update({k: row.get(k) for k in ["alpha", "AccT", "balanced_acc", "macro_f1", "old_acc", "current_acc", "old_to_current_rate", "current_to_old_rate", "total"]})
    return output


def _selector_stats(
    rule: str,
    alpha: float,
    old_risk: Mapping[int, float],
    current_to_old_risk: Mapping[int, float],
    reliability: Mapping[int, float],
    old_classes: Sequence[int],
    current_classes: Sequence[int],
) -> Dict[str, object]:
    old_rel = [float(reliability.get(int(c), 0.0)) for c in old_classes]
    current_rel = [float(reliability.get(int(c), 0.0)) for c in current_classes]
    old_risk_values = [float(old_risk.get(int(c), 0.0)) for c in old_classes]
    current_risk_values = [float(current_to_old_risk.get(int(c), 0.0)) for c in current_classes]
    return {
        "rule": rule,
        "alpha": float(alpha),
        "mean_old_reliability": sum(old_rel) / len(old_rel) if old_rel else 0.0,
        "mean_current_reliability": sum(current_rel) / len(current_rel) if current_rel else 0.0,
        "mean_old_to_current_proxy": sum(old_risk_values) / len(old_risk_values) if old_risk_values else 0.0,
        "mean_current_to_old_proxy": sum(current_risk_values) / len(current_risk_values) if current_risk_values else 0.0,
    }


def _quality_gated_selector(selector_stats: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    stats_by_rule = {str(row["rule"]): row for row in selector_stats}
    balanced = stats_by_rule.get("balanced_hmean_exemplar")
    constrained = stats_by_rule.get("constrained_anchor") or stats_by_rule.get("constrained_exemplar")
    if balanced is None:
        if constrained is None:
            raise ValueError("quality-gated selector requires at least one candidate rule")
        selected = constrained
        reason = "balanced_hmean_exemplar unavailable; selected constrained rule"
    elif constrained is None:
        selected = balanced
        reason = "constrained rules unavailable; selected balanced_hmean_exemplar"
    else:
        old_rel = float(balanced["mean_old_reliability"])
        old_to_current_proxy = float(balanced["mean_old_to_current_proxy"])
        current_rel = float(balanced["mean_current_reliability"])
        if old_rel >= 0.40 and old_to_current_proxy <= 0.45 and current_rel >= 0.80:
            selected = balanced
            reason = (
                "balanced_hmean_exemplar selected: old reliability/current retention proxies are sufficient "
                f"(old_rel={old_rel:.3f}, old_to_current_proxy={old_to_current_proxy:.3f}, current_rel={current_rel:.3f})"
            )
        else:
            selected = constrained
            reason = (
                f"{selected['rule']} selected: balanced_hmean_exemplar proxy is weak "
                f"(old_rel={old_rel:.3f}, old_to_current_proxy={old_to_current_proxy:.3f}, current_rel={current_rel:.3f})"
            )
    output = dict(selected)
    output["selected_rule"] = selected["rule"]
    output["selection_reason"] = reason
    return output


def _copy_locked_metrics(row: Mapping[str, object], seed: int) -> Dict[str, object]:
    copied = _with_meta(
        row,
        seed=seed,
        group="C_locked_NCConCM",
        alpha_rule=row.get("alpha_rule", "balanced_hmean_exemplar"),
        lambda_value=row.get("lambda_value", 0.30),
        matching_strength=0.0,
        reliability="off",
        selector="fixed_balanced",
        selection_reason="loaded_from_locked_rule_lock_json",
    )
    copied["selected"] = True
    return copied


def _markdown_table(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> str:
    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
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
    parser.add_argument("--alpha-rule", default="balanced_hmean_exemplar,constrained_anchor")
    parser.add_argument("--lambda-grid", default="0.00,0.05,0.10,0.15,0.20,0.30")
    parser.add_argument("--fixed-lambda", type=float, default=None)
    parser.add_argument("--disable-lambda-grid", action="store_true")
    parser.add_argument("--groups", default="A,B,C,D,E")
    parser.add_argument("--matching-grid", default="0.00,0.05,0.10,0.15")
    parser.add_argument("--enable-reliability-gate", action="store_true")
    parser.add_argument("--disable-reliability-gate", dest="enable_reliability_gate", action="store_false")
    parser.add_argument("--enable-matching-diagnostic", action="store_true")
    parser.add_argument("--disable-matching-diagnostic", dest="enable_matching_diagnostic", action="store_false")
    parser.add_argument("--enable-matching-correction", action="store_true")
    parser.add_argument("--disable-matching-correction", dest="enable_matching_correction", action="store_false")
    parser.add_argument("--enable-quality-gated-selector", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    protocol = build_protocol(_make_protocol_args(args))
    phase0_payload = torch.load(args.phase0_ckpt, map_location="cpu")
    phase1_payload = torch.load(args.phase1_ckpt, map_location=device)
    calibration = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))
    diagnostics = json.loads(Path(args.diagnostics_json).read_text(encoding="utf-8"))
    locked = json.loads(Path(args.locked_json).read_text(encoding="utf-8")) if args.locked_json else None
    old_classes = [int(c) for c in phase0_payload["old_classes"]]
    current_classes = [int(c) for c in phase0_payload["current_classes"]]
    seen_classes = old_classes + current_classes
    selected_by_rule = {row["rule"]: row for row in calibration["selected_test_rows"]}
    requested_rules = [item.strip() for item in args.alpha_rule.split(",") if item.strip()]
    if args.fixed_lambda is not None:
        lambda_grid = [float(args.fixed_lambda)]
    else:
        lambda_grid = [float(item) for item in args.lambda_grid.split(",") if item.strip()]
    if args.disable_lambda_grid and len(lambda_grid) != 1:
        raise ValueError("--disable-lambda-grid requires --fixed-lambda or a single-value --lambda-grid")
    matching_grid = [float(item) for item in args.matching_grid.split(",") if item.strip()]
    requested_groups = {item.strip().upper() for item in args.groups.split(",") if item.strip()}
    valid_groups = {"A", "B", "C", "D", "E", "F", "G"}
    unknown_groups = requested_groups - valid_groups
    if unknown_groups:
        raise ValueError(f"Unknown groups requested: {sorted(unknown_groups)}")
    lambda_locked = bool(args.disable_lambda_grid or args.fixed_lambda is not None or len(lambda_grid) == 1)
    full_diag_groups = {"F", "G"} & requested_groups

    model = resnet32().to(device)
    model.expand_classifier(len(seen_classes))
    model.load_state_dict(phase1_payload["model_state"])
    model.eval()
    seen_loader = _make_loader(protocol.test_dataset_for_classes(seen_classes), args.batch_size, args.num_workers)
    current_train_loader = _make_loader(protocol.train_dataset_for_classes(current_classes), args.batch_size, args.num_workers)
    exemplar_rows = _load_exemplar_rows(Path(args.exemplar_csv))
    exemplar_loader = _make_loader(
        ExemplarPathDataset(exemplar_rows, protocol.eval_transform),
        max(1, min(args.batch_size, len(exemplar_rows))),
        args.num_workers,
    )
    seen_pack = _collect_logits_features(model, seen_loader, device)
    current_train_pack = _collect_logits_features(model, current_train_loader, device)
    exemplar_pack = _collect_logits_features(model, exemplar_loader, device)

    raw_metrics, raw_confusion = _evaluate_pack(seen_pack, seen_classes, old_classes, current_classes, alpha=1.0)
    if args.enable_matching_diagnostic or args.enable_matching_correction:
        phase0_rel, phase1_rel, drift_rows, top_drift_pairs, per_class_drift_rows, drift_norm = _prototype_relations(phase0_payload, exemplar_pack, old_classes)
    else:
        phase0_rel, phase1_rel, drift_rows, top_drift_pairs, per_class_drift_rows, drift_norm = [], [], [], [], [], {}

    landscape_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []
    reliability_rows_all: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    selector_stats_rows: List[Dict[str, object]] = []
    selector_decisions: List[Dict[str, object]] = []
    rule_context: Dict[str, Dict[str, object]] = {}

    raw_row = _with_meta(
        raw_metrics,
        seed=args.seed,
        group="A_raw_freeze",
        alpha_rule="none",
        lambda_value=0.0,
        matching_strength=0.0,
        reliability="off",
        selector="none",
        selection_reason="raw_phase1_freeze",
        selected=True,
    )
    if "A" in requested_groups:
        selected_rows.append(raw_row)
        summary_rows.append(raw_row)

    locked_c_row = None
    if locked is not None:
        for row in locked.get("summary_rows", []):
            if row.get("group") in {"C_locked_NCConCM", "C_task_block_nc_concm"} and row.get("alpha_rule") == "balanced_hmean_exemplar":
                locked_c_row = _copy_locked_metrics(row, args.seed)
                break
    if locked_c_row is not None and "C" in requested_groups:
        selected_rows.append(locked_c_row)
        summary_rows.append(locked_c_row)

    for rule in requested_rules:
        if rule not in selected_by_rule:
            raise ValueError(f"Requested alpha rule not found in calibration JSON: {rule}")
        alpha = float(selected_by_rule[rule]["alpha"])
        old_risk, current_absorb, current_to_old_risk, reliability, reliability_rows = _risk_and_reliability(
            exemplar_pack,
            current_train_pack,
            diagnostics,
            phase0_payload,
            model,
            seen_classes,
            old_classes,
            current_classes,
            alpha,
        )
        rule_context[rule] = {
            "alpha": alpha,
            "old_risk": old_risk,
            "current_absorb": current_absorb,
            "current_to_old_risk": current_to_old_risk,
            "reliability": reliability,
        }
        selector_stats = _selector_stats(rule, alpha, old_risk, current_to_old_risk, reliability, old_classes, current_classes)
        selector_stats["seed"] = args.seed
        selector_stats_rows.append(selector_stats)
        for row in reliability_rows:
            row = dict(row)
            row["seed"] = args.seed
            row["alpha_rule"] = rule
            row["alpha"] = alpha
            reliability_rows_all.append(row)
        base_metrics, _ = _evaluate_pack(seen_pack, seen_classes, old_classes, current_classes, alpha=alpha)
        base_row = _with_meta(
            base_metrics,
            seed=args.seed,
            group="B_task_block_only",
            alpha_rule=rule,
            lambda_value=0.0,
            matching_strength=0.0,
            reliability="off",
            selector="none",
            selection_reason="task_block_only",
            selected=True,
        )
        if "B" in requested_groups:
            landscape_rows.append(base_row)
            selected_rows.append(base_row)
            summary_rows.append(base_row)

        c_rows = []
        d_rows = []
        e_rows = []
        f_rows = []
        if {"C", "D", "E", "F"} & requested_groups:
            for lamb in lambda_grid:
                if "C" in requested_groups and locked_c_row is None:
                    correction = _correction_vector(seen_classes, old_classes, current_classes, old_risk, current_absorb, reliability, lamb, use_reliability=False)
                    metrics, _ = _evaluate_pack(seen_pack, seen_classes, old_classes, current_classes, alpha=alpha, correction=correction)
                    row = _with_meta(
                        metrics,
                        seed=args.seed,
                        group="C_task_block_nc_concm",
                        alpha_rule=rule,
                        lambda_value=lamb,
                        matching_strength=0.0,
                        reliability="off",
                        selector="none",
                        selection_reason="fixed_or_diagnostic_ncconcm_without_reliability_or_matching",
                    )
                    row["selection_note"] = "fixed_lambda_locked" if lambda_locked else "diagnostic_grid"
                    c_rows.append(row)
                    landscape_rows.append(row)
                if args.enable_reliability_gate and "D" in requested_groups and rule == "balanced_hmean_exemplar":
                    gated = _correction_vector(seen_classes, old_classes, current_classes, old_risk, current_absorb, reliability, lamb, use_reliability=True)
                    metrics, _ = _evaluate_pack(seen_pack, seen_classes, old_classes, current_classes, alpha=alpha, correction=gated)
                    row = _with_meta(
                        metrics,
                        seed=args.seed,
                        group="D_reliability_locked",
                        alpha_rule=rule,
                        lambda_value=lamb,
                        matching_strength=0.0,
                        reliability="on",
                        selector="fixed_balanced",
                        selection_reason="balanced_hmean_exemplar_with_reliability_only",
                    )
                    row["selection_note"] = "fixed_lambda_locked" if lambda_locked else "diagnostic_grid"
                    d_rows.append(row)
                    landscape_rows.append(row)
                if args.enable_matching_correction and "E" in requested_groups and rule == "balanced_hmean_exemplar":
                    for m in matching_grid:
                        matched = _correction_vector(
                            seen_classes,
                            old_classes,
                            current_classes,
                            old_risk,
                            current_absorb,
                            reliability,
                            lamb,
                            use_reliability=False,
                            matching_strength=m,
                            drift_norm=drift_norm,
                        )
                        metrics, _ = _evaluate_pack(seen_pack, seen_classes, old_classes, current_classes, alpha=alpha, correction=matched)
                        row = _with_meta(
                            metrics,
                            seed=args.seed,
                            group="E_matching_locked",
                            alpha_rule=rule,
                            lambda_value=lamb,
                            matching_strength=m,
                            reliability="off",
                            selector="fixed_balanced",
                            selection_reason="balanced_hmean_exemplar_with_matching_only",
                        )
                        row["selection_note"] = "fixed_lambda_locked" if lambda_locked and len(matching_grid) == 1 else "diagnostic_grid"
                        e_rows.append(row)
                        landscape_rows.append(row)
                if args.enable_reliability_gate and args.enable_matching_correction and "F" in requested_groups and rule == "balanced_hmean_exemplar":
                    for m in matching_grid:
                        matched = _correction_vector(
                            seen_classes,
                            old_classes,
                            current_classes,
                            old_risk,
                            current_absorb,
                            reliability,
                            lamb,
                            use_reliability=True,
                            matching_strength=m,
                            drift_norm=drift_norm,
                        )
                        metrics, _ = _evaluate_pack(seen_pack, seen_classes, old_classes, current_classes, alpha=alpha, correction=matched)
                        row = _with_meta(
                            metrics,
                            seed=args.seed,
                            group="F_full_all_fixed_balanced",
                            alpha_rule=rule,
                            lambda_value=lamb,
                            matching_strength=m,
                            reliability="on",
                            selector="fixed_balanced",
                            selection_reason="diagnostic_grid_over_lambda_and_matching_for_fixed_balanced_rule",
                        )
                        row["selection_note"] = "fixed_lambda_locked" if lambda_locked and len(matching_grid) == 1 else "diagnostic_grid"
                        f_rows.append(row)
                        landscape_rows.append(row)
        for group_rows in [c_rows, d_rows, e_rows, f_rows]:
            if group_rows:
                if lambda_locked and len(group_rows) == 1:
                    selected = dict(group_rows[0])
                    selected["selection_note"] = "fixed_lambda_locked"
                else:
                    selected = dict(_select_diagnostic_row(group_rows))
                    selected["selection_note"] = "diagnostic_best_guardrail_else_best_AccT"
                selected["selected"] = True
                selected_rows.append(selected)
                summary_rows.append(selected)

    if args.enable_quality_gated_selector and "G" in requested_groups:
        selector_decision = _quality_gated_selector(selector_stats_rows)
        selector_decision["seed"] = args.seed
        selector_decisions.append(selector_decision)
        selected_rule = str(selector_decision["selected_rule"])
        if selected_rule not in rule_context:
            raise ValueError(f"Selector chose unavailable rule: {selected_rule}")
        ctx = rule_context[selected_rule]
        g_rows = []
        for lamb in lambda_grid:
            for m in matching_grid:
                correction = _correction_vector(
                    seen_classes,
                    old_classes,
                    current_classes,
                    ctx["old_risk"],
                    ctx["current_absorb"],
                    ctx["reliability"],
                    lamb,
                    use_reliability=True,
                    matching_strength=m,
                    drift_norm=drift_norm,
                )
                metrics, _ = _evaluate_pack(seen_pack, seen_classes, old_classes, current_classes, alpha=float(ctx["alpha"]), correction=correction)
                row = _with_meta(
                    metrics,
                    seed=args.seed,
                    group="G_full_all_selector",
                    alpha_rule=selected_rule,
                    lambda_value=lamb,
                    matching_strength=m,
                    reliability="on",
                    selector="quality_gated",
                    selection_reason=selector_decision["selection_reason"],
                )
                row["selection_note"] = "diagnostic_grid_after_non_oracle_rule_selection"
                g_rows.append(row)
                landscape_rows.append(row)
        if g_rows:
            selected = dict(_select_diagnostic_row(g_rows))
            selected["selected"] = True
            selected["selection_note"] = "diagnostic_best_guardrail_else_best_AccT_after_non_oracle_rule_selection"
            selected_rows.append(selected)
            summary_rows.append(selected)

    confusion_rows = _confusion_rows(raw_confusion, seen_classes)
    old_to_current_rows = _block_confusion_rows(raw_confusion, old_classes, current_classes, "old_to_current")
    current_to_old_rows = _block_confusion_rows(raw_confusion, current_classes, old_classes, "current_to_old")
    per_class_rows = _per_class_metrics_rows(raw_confusion, seen_classes)
    selected_per_class_rows: List[Dict[str, object]] = []
    selected_old_to_current_rows: List[Dict[str, object]] = []
    selected_current_to_old_rows: List[Dict[str, object]] = []
    for selected in summary_rows:
        group = str(selected.get("group"))
        rule = str(selected.get("alpha_rule"))
        alpha = float(selected.get("alpha") or 1.0)
        lamb = float(selected.get("lambda_value") or 0.0)
        match = float(selected.get("matching_strength") or 0.0)
        correction = None
        if group != "A_raw_freeze" and lamb != 0.0:
            ctx = rule_context.get(rule)
            if ctx is not None:
                use_reliability = str(selected.get("reliability")) == "on"
                correction = _correction_vector(
                    seen_classes,
                    old_classes,
                    current_classes,
                    ctx["old_risk"],
                    ctx["current_absorb"],
                    ctx["reliability"],
                    lamb,
                    use_reliability=use_reliability,
                    matching_strength=match,
                    drift_norm=drift_norm,
                )
        _, selected_confusion = _evaluate_pack(seen_pack, seen_classes, old_classes, current_classes, alpha=alpha, correction=correction)
        for row in _per_class_metrics_rows(selected_confusion, seen_classes):
            row = dict(row)
            row.update(
                {
                    "seed": args.seed,
                    "group": group,
                    "alpha_rule": rule,
                    "alpha": alpha,
                    "lambda_value": lamb,
                    "matching_strength": match,
                    "reliability": selected.get("reliability"),
                    "selector": selected.get("selector"),
                }
            )
            selected_per_class_rows.append(row)
        for row in _block_confusion_rows(selected_confusion, old_classes, current_classes, "old_to_current"):
            row = dict(row)
            row.update({"seed": args.seed, "group": group, "alpha_rule": rule, "lambda_value": lamb, "matching_strength": match})
            selected_old_to_current_rows.append(row)
        for row in _block_confusion_rows(selected_confusion, current_classes, old_classes, "current_to_old"):
            row = dict(row)
            row.update({"seed": args.seed, "group": group, "alpha_rule": rule, "lambda_value": lamb, "matching_strength": match})
            selected_current_to_old_rows.append(row)

    result = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "seed": args.seed,
        "paths": {
            "phase0_ckpt": args.phase0_ckpt,
            "phase1_ckpt": args.phase1_ckpt,
            "exemplar_csv": args.exemplar_csv,
            "calibration_json": args.calibration_json,
            "diagnostics_json": args.diagnostics_json,
            "locked_json": args.locked_json,
            "output_dir": str(output_dir),
        },
        "old_classes": old_classes,
        "current_classes": current_classes,
        "raw_freeze": raw_row,
        "selected_rows": selected_rows,
        "summary_rows": summary_rows,
        "alpha_lambda_landscape_rows": landscape_rows,
        "selector_stats_rows": selector_stats_rows,
        "selector_decisions": selector_decisions,
        "matching_diagnostic": {
            "enabled": bool(args.enable_matching_diagnostic),
            "correction_enabled": bool(args.enable_matching_correction),
            "top_drift_pairs": top_drift_pairs,
            "per_class_drift_norm": drift_norm,
        },
    }
    _write_json(output_dir / "final_results.json", result)
    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_csv(output_dir / "selected_rows.csv", selected_rows)
    _write_csv(output_dir / "alpha_lambda_landscape.csv", landscape_rows)
    _write_csv(output_dir / "selector_stats.csv", selector_stats_rows)
    _write_csv(output_dir / "selector_decisions.csv", selector_decisions)
    _write_csv(output_dir / "confusion_matrix.csv", confusion_rows)
    _write_csv(output_dir / "old_to_current_confusion.csv", old_to_current_rows)
    _write_csv(output_dir / "current_to_old_confusion.csv", current_to_old_rows)
    _write_csv(output_dir / "per_class_confusion_strength.csv", old_to_current_rows + current_to_old_rows)
    _write_csv(output_dir / "per_class_reliability.csv", reliability_rows_all)
    _write_csv(output_dir / "selected_per_class_metrics.csv", selected_per_class_rows)
    _write_csv(output_dir / "selected_old_to_current_confusion.csv", selected_old_to_current_rows)
    _write_csv(output_dir / "selected_current_to_old_confusion.csv", selected_current_to_old_rows)
    if args.enable_matching_diagnostic or args.enable_matching_correction:
        _write_csv(output_dir / "prototype_relation_phase0.csv", phase0_rel)
        _write_csv(output_dir / "prototype_relation_phase1.csv", phase1_rel)
        _write_csv(output_dir / "prototype_relation_drift.csv", drift_rows)
        _write_csv(output_dir / "top_drift_pairs.csv", top_drift_pairs)
        _write_csv(output_dir / "per_class_relation_drift.csv", per_class_drift_rows)
    _write_csv(output_dir / "per_class_metrics.csv", per_class_rows)
    run_summary = "\n".join(
        [
            "# HyperKvasir23 ConCM-min Gate",
            "",
            "Evaluation-only. No training, no phase2, no full phases, no seed sweep, no FDM v2.",
            "",
            _markdown_table(
                summary_rows,
                ["seed", "group", "alpha_rule", "alpha", "lambda_value", "matching_strength", "AccT", "old_acc", "current_acc", "old_to_current_rate", "current_to_old_rate"],
            ),
            "",
        ]
    )
    (output_dir / "run_summary.md").write_text(run_summary, encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
