#!/usr/bin/env python3
"""Isolated Pure-DSM-ConCM gate for HyperKvasir23 phase1 checkpoints."""

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
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

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


class ExemplarPathDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping[str, object]], transform) -> None:
        self.rows = list(rows)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(str(row["path"])) as image:
            image = image.convert("RGB")
        return self.transform(image), int(row["class_id"])


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _load_exemplar_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "class_id": int(row["class_id"]),
                    "class_name": row.get("class_name", str(row["class_id"])),
                    "train_index": int(row["train_index"]),
                    "path": row["path"],
                }
            )
    return rows


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


def _load_result_row(path: str | None, kind: str) -> Dict[str, object] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if kind == "diagnostics":
        for key in ["raw_freeze_metrics", "raw_freeze", "raw_freeze_summary"]:
            value = payload.get(key)
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("group", "A_raw_freeze")
                row.setdefault("method", "raw_freeze")
                return row
        if isinstance(payload.get("raw_freeze_metrics"), list) and payload["raw_freeze_metrics"]:
            row = dict(payload["raw_freeze_metrics"][0])
            row.setdefault("group", "A_raw_freeze")
            row.setdefault("method", "raw_freeze")
            return row
    if kind == "calibration":
        for row in payload.get("selected_test_rows", []):
            if row.get("rule") == "balanced_hmean_exemplar":
                out = dict(row)
                out.setdefault("group", "B_task_block_only")
                out.setdefault("method", "task_block_only")
                return out
    if kind == "locked":
        for row in payload.get("summary_rows", []):
            if row.get("group") in {"C_locked_NCConCM", "C_task_block_nc_concm"} and row.get("alpha_rule") == "balanced_hmean_exemplar":
                out = dict(row)
                out["group"] = "C_locked_NCConCM"
                out.setdefault("method", "locked_NCConCM")
                return out
    return None


@torch.no_grad()
def _evaluate_model_classifier(
    model: nn.Module,
    loader: DataLoader,
    class_to_head: Mapping[int, int],
    old_classes: Sequence[int],
    current_classes: Sequence[int],
    device: torch.device,
) -> Dict[str, object]:
    if model.classifier is None:
        raise RuntimeError("Cannot evaluate raw freeze: classifier is not initialized")
    model.eval()
    features = extract_features_by_class(model, loader, class_to_head, device)
    logits = model.classifier(features.features.to(device))
    preds = logits.argmax(dim=1).cpu()
    labels = features.head_labels.cpu()
    num_classes = len(class_to_head)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for label, pred in zip(labels.tolist(), preds.tolist()):
        confusion[int(label), int(pred)] += 1
    old_heads = {int(class_to_head[int(c)]) for c in old_classes}
    current_heads = {int(class_to_head[int(c)]) for c in current_classes}
    matches = preds.eq(labels)

    def pct(n: int, d: int) -> float | None:
        return 100.0 * float(n) / float(d) if d > 0 else None

    recalls = []
    f1_values = []
    for head in range(num_classes):
        total = int(confusion[head].sum().item())
        correct = int(confusion[head, head].item())
        predicted = int(confusion[:, head].sum().item())
        if total <= 0:
            continue
        recall = correct / total
        precision = correct / predicted if predicted else 0.0
        recalls.append(recall)
        f1_values.append(2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)
    old_mask = torch.tensor([int(x) in old_heads for x in labels.tolist()], dtype=torch.bool)
    current_mask = torch.tensor([int(x) in current_heads for x in labels.tolist()], dtype=torch.bool)
    pred_old = torch.tensor([int(x) in old_heads for x in preds.tolist()], dtype=torch.bool)
    pred_current = torch.tensor([int(x) in current_heads for x in preds.tolist()], dtype=torch.bool)
    return {
        "group": "A_raw_freeze",
        "method": "raw_freeze_recomputed",
        "AccT": pct(int(matches.sum().item()), int(labels.numel())),
        "balanced_acc": 100.0 * sum(recalls) / len(recalls) if recalls else 0.0,
        "macro_f1": 100.0 * sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "old_acc": pct(int(matches[old_mask].sum().item()), int(old_mask.sum().item())),
        "current_acc": pct(int(matches[current_mask].sum().item()), int(current_mask.sum().item())),
        "old_to_current_rate": pct(int((old_mask & pred_current).sum().item()), int(old_mask.sum().item())),
        "current_to_old_rate": pct(int((current_mask & pred_old).sum().item()), int(current_mask.sum().item())),
        "total": int(labels.numel()),
    }


def _stack_stats(stats_by_head: Mapping[int, torch.Tensor], heads: Sequence[int]) -> torch.Tensor:
    missing = [int(head) for head in heads if int(head) not in stats_by_head]
    if missing:
        raise RuntimeError(f"Missing feature stats for head indices: {missing}")
    return torch.stack([stats_by_head[int(head)] for head in heads], dim=0)


def _summary_row(group: str, method: str, metrics: Mapping[str, object], **extra) -> Dict[str, object]:
    row = {
        "group": group,
        "method": method,
        "AccT": metrics.get("AccT"),
        "balanced_acc": metrics.get("balanced_acc"),
        "macro_f1": metrics.get("macro_f1"),
        "old_acc": metrics.get("old_acc"),
        "current_acc": metrics.get("current_acc"),
        "old_to_current_rate": metrics.get("old_to_current_rate"),
        "current_to_old_rate": metrics.get("current_to_old_rate"),
        "total": metrics.get("total"),
    }
    row.update(extra)
    return row


def _pairwise_rows(geometry: torch.Tensor, head_to_class: Mapping[int, int]) -> List[Dict[str, object]]:
    dots = geometry.detach().cpu() @ geometry.detach().cpu().transpose(0, 1)
    rows = []
    for i in range(dots.shape[0]):
        for j in range(dots.shape[1]):
            rows.append(
                {
                    "head_i": int(i),
                    "class_i": int(head_to_class[int(i)]),
                    "head_j": int(j),
                    "class_j": int(head_to_class[int(j)]),
                    "dot": float(dots[i, j].item()),
                }
            )
    return rows


def _confusion_rows(confusion: torch.Tensor, head_to_class: Mapping[int, int]) -> List[Dict[str, object]]:
    rows = []
    for true_head in range(confusion.shape[0]):
        row = {
            "true_head": int(true_head),
            "true_class": int(head_to_class[int(true_head)]),
            "total": int(confusion[true_head].sum().item()),
        }
        for pred_head in range(confusion.shape[1]):
            row[f"pred_head_{pred_head}"] = int(confusion[true_head, pred_head].item())
        rows.append(row)
    return rows


def _per_class_rows(metrics: Mapping[str, object], head_to_class: Mapping[int, int]) -> List[Dict[str, object]]:
    rows = []
    for row in metrics["per_class"]:
        out = dict(row)
        out["class_id"] = int(head_to_class[int(out["head_idx"])])
        rows.append(out)
    return rows


def _acceptance(raw_row: Mapping[str, object] | None, locked_row: Mapping[str, object] | None, dsm_row: Mapping[str, object]) -> Dict[str, object]:
    raw_old = float(raw_row.get("old_acc", 12.39)) if raw_row and raw_row.get("old_acc") is not None else 12.39
    raw_old_to_current = (
        float(raw_row.get("old_to_current_rate", 87.61))
        if raw_row and raw_row.get("old_to_current_rate") is not None
        else 87.61
    )
    locked_acct = float(locked_row.get("AccT", 77.53)) if locked_row and locked_row.get("AccT") is not None else 77.53
    old_acc = float(dsm_row.get("old_acc") or 0.0)
    current_acc = float(dsm_row.get("current_acc") or 0.0)
    current_to_old = float(dsm_row.get("current_to_old_rate") or 0.0)
    old_to_current = float(dsm_row.get("old_to_current_rate") or 100.0)
    acct = float(dsm_row.get("AccT") or 0.0)
    checks = {
        "old_acc_above_raw": old_acc > raw_old,
        "current_acc_ge_85": current_acc >= 85.0,
        "current_to_old_lt_15": current_to_old < 15.0,
        "old_to_current_below_raw": old_to_current < raw_old_to_current,
        "AccT_close_or_above_locked": acct >= locked_acct - 2.0,
    }
    passed = all(bool(v) for v in checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "raw_old_acc_reference": raw_old,
        "raw_old_to_current_reference": raw_old_to_current,
        "locked_AccT_reference": locked_acct,
        "AccT_gap_vs_locked": acct - locked_acct,
    }


def _build_report(payload: Mapping[str, object]) -> str:
    rows = payload["summary_rows"]
    dsm_row = payload["dsm_row"]
    locked = payload.get("locked_row")
    gap = None
    if locked and locked.get("AccT") is not None and dsm_row.get("AccT") is not None:
        gap = float(dsm_row["AccT"]) - float(locked["AccT"])
    lines = [
        "# HyperKvasir23 Pure DSM-ConCM Gate Results",
        "",
        "## Purpose",
        "",
        "Validate whether original ConCM Dynamic Structure Matching, without MPC or task-block calibration, transfers to the HyperKvasir23 medical CIL/FSCIL phase1 setting.",
        "",
        "This is not the previous evaluation-time matching diagnostic. This gate trains a two-layer projector on frozen-backbone feature tensors, then evaluates a geometric classifier built by SVD/Procrustes-style dynamic structure matching.",
        "",
        "## Scope Guardrails",
        "",
        "- Phase2/full phases: not run.",
        "- Seed sweep: not run; this tool defaults to seed0.",
        "- FDM v2: not run.",
        "- MPC/WordNet/GloVe/large-model priors: not implemented or downloaded.",
        "- Backbone: frozen; only `DSMProjector` parameters are optimized.",
        "",
        "## Exact Command",
        "",
        "```bash",
        str(payload["command"]),
        "```",
        "",
        "## Implementation Details",
        "",
        "- `DSMProjector`: normalize input feature -> Linear -> ReLU -> Linear -> normalize projected feature.",
        "- Gaussian feature augmentation: class feature mean/std diagonal with explicit `head_idx < num_old_classes` old/current split.",
        "- Dynamic structure: `torch.linalg.svd(projected_prototypes.T @ M)` with ETF-style centered geometry.",
        "- Loss: `LMatch` cross entropy against geometry vectors plus optional anchor-augmented supervised contrastive loss.",
        "- Evaluation: project all-seen frozen features and classify with `projected @ geometry.T`.",
        "",
        "## Files Changed",
        "",
        "- `src/methods/concm_dsm.py`",
        "- `tools/run_hyperkvasir_pure_dsm_concm_gate.py`",
        "- `docs/HYPERKVASIR23_PURE_DSM_CONCM_GATE_RESULTS.md`",
        "",
        "## A/B/C/D/E Comparison",
        "",
        _markdown_table(
            rows,
            ["group", "method", "AccT", "balanced_acc", "macro_f1", "old_acc", "current_acc", "old_to_current_rate", "current_to_old_rate"],
        ),
        "",
        "## Pure DSM vs Locked NC-ConCM",
        "",
        f"- AccT gap D/E minus C: `{gap:.4f}`" if gap is not None else "- Locked NC-ConCM row unavailable.",
        "",
        "## Geometry Sanity",
        "",
        "```json",
        json.dumps(payload["geometry_stats"], indent=2, sort_keys=True),
        "```",
        "",
        "## Acceptance Gate",
        "",
        "```json",
        json.dumps(payload["acceptance_gate"], indent=2, sort_keys=True),
        "```",
        "",
        "## Interpretation",
        "",
    ]
    acc = payload["acceptance_gate"]
    if acc["passed"]:
        lines.append("Pure DSM-ConCM passes the bounded seed0 gate. Original ConCM structure matching is a credible medical-CIL component for the next bounded seed1 check.")
    elif (float(dsm_row.get("AccT") or 0.0) > float((payload.get("raw_row") or {}).get("AccT") or 0.0)):
        lines.append("Pure DSM-ConCM improves over raw freeze but does not pass all locked-baseline guardrails. DSM has transfer value, but it likely needs task-block calibration or NC-ConCM-style bias control.")
    else:
        lines.append("Pure DSM-ConCM does not improve the bounded raw-freeze reference enough. The bottleneck may still be old/current block dominance rather than projector geometry alone.")
    lines.extend(
        [
            "",
            "## Next Steps",
            "",
            "- If D/E is below locked C but above raw freeze, test DSM combined with task-block calibration before adding MPC.",
            "- Add seed1 only after seed0 shows a clear positive signal.",
            "- Keep MPC disabled until DSM-only has a stable benefit.",
            "- Stop this branch if current accuracy drops below 85 or current->old exceeds 15.",
            "",
            "## Artifact Paths",
            "",
            f"- Output dir: `{payload['output_dir']}`",
            "- `final_results.json`",
            "- `summary.csv`",
            "- `train_trace.csv`",
            "- `dsm_config.json`",
            "- `geometry_stats.json`",
            "- `pairwise_geometry_dot.csv`",
            "- `confusion_matrix.csv`",
            "- `per_class_metrics.csv`",
            "- `projector.pt`",
            "",
            "## GPU/Process Cleanup Status",
            "",
            "The tool does not launch child training processes. It saves only the projector checkpoint and exits after evaluation.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--phase0-ckpt", required=True)
    parser.add_argument("--phase1-ckpt", required=True)
    parser.add_argument("--exemplar-csv", required=True)
    parser.add_argument("--calibration-json", default=None)
    parser.add_argument("--diagnostics-json", default=None)
    parser.add_argument("--locked-json", default=None)
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
    parser.add_argument("--no-cont-loss", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(args.seed))
    device = torch.device(args.device)
    protocol_args = _make_protocol_args(args)
    protocol = build_protocol(protocol_args)

    phase0_payload = torch.load(args.phase0_ckpt, map_location="cpu")
    phase1_payload = torch.load(args.phase1_ckpt, map_location=device)
    old_classes = [int(class_id) for class_id in phase0_payload["old_classes"]]
    current_classes = [int(class_id) for class_id in phase0_payload["current_classes"]]
    seen_classes = old_classes + current_classes
    if [int(class_id) for class_id in phase1_payload["seen_classes"]] != seen_classes:
        raise RuntimeError("phase1 checkpoint seen_classes do not match phase0 metadata")
    if old_classes != [int(class_id) for class_id in protocol.tasks[0]]:
        raise RuntimeError("phase0 old classes do not match HyperKvasir23 protocol task0")
    if current_classes != [int(class_id) for class_id in protocol.tasks[1]]:
        raise RuntimeError("phase1 current classes do not match HyperKvasir23 protocol task1")

    phase0_class_to_head = {int(class_id): idx for idx, class_id in enumerate(old_classes)}
    phase1_class_to_head = {int(class_id): idx for idx, class_id in enumerate(seen_classes)}
    head_to_class = {idx: int(class_id) for class_id, idx in phase1_class_to_head.items()}

    phase0_model = resnet32().to(device)
    phase0_model.expand_classifier(len(old_classes))
    phase0_model.load_state_dict(phase0_payload["model_state"])
    phase1_model = resnet32().to(device)
    phase1_model.expand_classifier(len(seen_classes))
    phase1_model.load_state_dict(phase1_payload["model_state"])
    phase0_freeze = _freeze_all(phase0_model)
    phase1_freeze = _freeze_all(phase1_model)
    if not hasattr(phase1_model, "extract_features"):
        raise RuntimeError("phase1 checkpoint model does not expose extract_features(images); check src/models/resnet_cifar.py.")

    phase0_proto_loader = _make_loader(
        protocol.prototype_dataset_for_classes(old_classes),
        args.batch_size,
        args.num_workers,
        shuffle=False,
        seed=int(args.seed),
    )
    current_train_loader = _make_loader(
        protocol.prototype_dataset_for_classes(current_classes),
        args.batch_size,
        args.num_workers,
        shuffle=False,
        seed=int(args.seed) + 1,
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
    current_stats = extract_features_by_class(phase1_model, current_train_loader, phase1_class_to_head, device)
    exemplar_stats = extract_features_by_class(phase1_model, exemplar_loader, phase1_class_to_head, device)
    test_stats = extract_features_by_class(phase1_model, seen_test_loader, phase1_class_to_head, device)

    old_heads = list(range(len(old_classes)))
    current_heads = list(range(len(old_classes), len(seen_classes)))
    seen_heads = list(range(len(seen_classes)))
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
        "data_root": args.data_root,
        "phase0_ckpt": args.phase0_ckpt,
        "phase1_ckpt": args.phase1_ckpt,
        "exemplar_csv": args.exemplar_csv,
        "calibration_json": args.calibration_json,
        "diagnostics_json": args.diagnostics_json,
        "locked_json": args.locked_json,
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "projector_hidden": int(args.projector_hidden),
        "projector_dim": int(args.projector_dim),
        "base_projector_epochs": int(args.base_projector_epochs),
        "increment_projector_epochs": int(args.increment_projector_epochs),
        "lr": float(args.lr),
        "sample_num_old": int(args.sample_num_old),
        "sample_num_current": int(args.sample_num_current),
        "cont_weight": float(args.cont_weight),
        "no_cont_loss": bool(args.no_cont_loss),
        "max_train_batches": args.max_train_batches,
        "input_dim": input_dim,
        "dry_run": bool(args.dry_run),
        "split": {
            "old_classes": old_classes,
            "current_classes": current_classes,
            "seen_classes": seen_classes,
            "class_order": [int(class_id) for class_id in protocol.class_order],
            "class_to_head": phase1_class_to_head,
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
            "optimizer_scope": "projector.parameters_only",
            "backbone_checkpoint_saved": False,
        },
    }
    _write_json(output_dir / "dsm_config.json", config)

    if args.dry_run:
        projected = torch.nn.functional.normalize(torch.randn(len(seen_classes), int(args.projector_dim)), dim=1)
        _, geometry_stats = compute_dynamic_structure(projected)
        payload = {
            **config,
            "status": "dry_run_ok",
            "geometry_sanity_on_random_projected_prototypes": geometry_stats,
        }
        _write_json(output_dir / "dry_run.json", payload)
        (output_dir / "run_summary.md").write_text(
            "# HyperKvasir23 Pure DSM-ConCM Dry Run\n\nSetup, checkpoint loading, feature extraction, and geometry sanity completed. No projector training or phase2/full run was launched.\n",
            encoding="utf-8",
        )
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
            no_cont_loss=bool(args.no_cont_loss),
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
        no_cont_loss=bool(args.no_cont_loss),
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
        test_stats.features,
        test_stats.head_labels,
        projector,
        geometry,
        old_head_indices=old_heads,
        current_head_indices=current_heads,
        device=device,
    )
    group = "E_pure_DSM_ConCM_no_cont" if args.no_cont_loss else "D_pure_DSM_ConCM"
    method = "pure_DSM_ConCM_no_cont" if args.no_cont_loss else "pure_DSM_ConCM"
    dsm_row = _summary_row(group, method, dsm_metrics, cont_loss_enabled=not bool(args.no_cont_loss))

    raw_row = _load_result_row(args.diagnostics_json, "diagnostics")
    if raw_row is None:
        raw_row = _evaluate_model_classifier(phase1_model, seen_test_loader, phase1_class_to_head, old_classes, current_classes, device)
    raw_summary = _summary_row("A_raw_freeze", "raw_freeze", raw_row)
    calibration_row = _load_result_row(args.calibration_json, "calibration")
    locked_row = _load_result_row(args.locked_json, "locked")
    summary_rows = [raw_summary]
    if calibration_row is not None:
        summary_rows.append(_summary_row("B_task_block_only", "task_block_only", calibration_row, alpha=calibration_row.get("alpha")))
    if locked_row is not None:
        summary_rows.append(
            _summary_row(
                "C_locked_NCConCM",
                "locked_NCConCM",
                locked_row,
                alpha=locked_row.get("alpha"),
                lambda_value=locked_row.get("lambda_value"),
            )
        )
    summary_rows.append(dsm_row)
    acceptance = _acceptance(raw_summary, locked_row, dsm_row)

    pairwise_rows = _pairwise_rows(geometry, head_to_class)
    confusion_rows = _confusion_rows(dsm_metrics["confusion_matrix"], head_to_class)
    per_class_rows = _per_class_rows(dsm_metrics, head_to_class)
    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_csv(output_dir / "train_trace.csv", trace_rows)
    _write_json(output_dir / "geometry_stats.json", geometry_stats)
    _write_csv(output_dir / "pairwise_geometry_dot.csv", pairwise_rows)
    _write_csv(output_dir / "confusion_matrix.csv", confusion_rows)
    _write_csv(output_dir / "per_class_metrics.csv", per_class_rows)
    torch.save(
        {
            "projector_state_dict": projector.state_dict(),
            "projector_config": projector.config(),
            "geometry_vectors": geometry.detach().cpu(),
            "seen_classes": seen_classes,
            "class_to_head": phase1_class_to_head,
            "head_to_class": head_to_class,
            "geometry_stats": geometry_stats,
        },
        output_dir / "projector.pt",
    )

    payload = {
        **config,
        "raw_row": raw_summary,
        "calibration_row": calibration_row,
        "locked_row": locked_row,
        "dsm_row": dsm_row,
        "summary_rows": summary_rows,
        "train_trace_rows": trace_rows,
        "geometry_stats": geometry_stats,
        "acceptance_gate": acceptance,
        "artifact_paths": {
            "final_results": str(output_dir / "final_results.json"),
            "summary_csv": str(output_dir / "summary.csv"),
            "train_trace_csv": str(output_dir / "train_trace.csv"),
            "dsm_config": str(output_dir / "dsm_config.json"),
            "geometry_stats": str(output_dir / "geometry_stats.json"),
            "pairwise_geometry_dot": str(output_dir / "pairwise_geometry_dot.csv"),
            "confusion_matrix": str(output_dir / "confusion_matrix.csv"),
            "per_class_metrics": str(output_dir / "per_class_metrics.csv"),
            "run_summary": str(output_dir / "run_summary.md"),
            "projector": str(output_dir / "projector.pt"),
        },
    }
    _write_json(output_dir / "final_results.json", payload)
    report = _build_report(payload)
    (output_dir / "run_summary.md").write_text(report, encoding="utf-8")
    docs_path = REPO_ROOT / "docs" / "HYPERKVASIR23_PURE_DSM_CONCM_GATE_RESULTS.md"
    docs_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
