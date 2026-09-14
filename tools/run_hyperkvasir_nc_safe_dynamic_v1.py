#!/usr/bin/env python3
"""NC-safe Dynamic v1 bounded microcheck and four-session confirmation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.concm_dsm import DSMProjector, compute_dynamic_structure
from src.methods.full_concm import (
    MPCNetwork,
    PrototypeRecord,
    blend_current_covariance,
    build_attribute_mapping,
    calibrate_prototype,
    class_balanced_mean,
    class_semantic_embedding,
    fixed_nc_geometry,
    resample_repository,
    structure_anchor_contrastive,
)
from src.utils.seed import set_seed
from tools.run_hyperkvasir_full_concm import file_sha256, metric_pack
from tools.run_hyperkvasir_full_dynamic_valckpt_v2 import (
    cuda_snapshot,
    current_only_features,
    fit_features,
    git_commit,
    read_csv,
    rng_state,
    write_csv,
    write_json,
)


ALPHAS = (1.0, 0.75, 0.5, 0.25, 0.0)
EPSILON = 0.0
EPOCHS = 5
CURRENT_ACCURACY_MIN = 50.0
CURRENT_TO_OLD_MAX = 30.0
TARGETS = {
    (2, 5): {"role": "stable_control", "output": "seed2/session5_control", "source_session": 4},
    (1, 5): {"role": "persistent_target", "output": "seed1/session5", "source_session": 4},
    (1, 2): {"role": "confirmation_target", "output": "seed1/session2", "source_session": 1},
    (3, 5): {"role": "confirmation_target", "output": "seed3/session5", "source_session": 4},
}
MICROCHECK = ((2, 5), (1, 5))
CONFIRMATION = ((1, 2), (3, 5))


def load_source(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    projector = DSMProjector(**payload["projector_config"]).to(device)
    projector.load_state_dict(payload["projector_state_dict"])
    mpc = MPCNetwork(**payload["mpc_config"]).to(device)
    mpc.load_state_dict(payload["mpc_state_dict"])
    mpc.eval()
    records = [PrototypeRecord.from_dict(value) for value in payload["repository"]]
    return payload, projector, mpc, records


@torch.no_grad()
def geometry_stats(
    projected_calibrated: torch.Tensor,
    projected_raw: torch.Tensor,
    anchors: torch.Tensor,
    old_heads: Sequence[int],
    current_heads: Sequence[int],
):
    anchors = F.normalize(anchors, dim=1)
    cal_sim = F.normalize(projected_calibrated, dim=1) @ anchors.T
    raw_sim = F.normalize(projected_raw, dim=1) @ anchors.T
    cal_pred = cal_sim.argmax(dim=1)
    raw_pred = raw_sim.argmax(dim=1)
    own = torch.arange(anchors.shape[0], device=anchors.device)
    old_tensor = torch.tensor(old_heads, device=anchors.device)
    old_self_accuracy = float(cal_pred[old_tensor].eq(old_tensor).float().mean().cpu()) * 100.0
    old_margins = []
    for head in old_heads:
        wrong = cal_sim[head].clone()
        wrong[head] = -float("inf")
        old_margins.append(cal_sim[head, head] - wrong.max())
    old_margins_tensor = torch.stack(old_margins)
    current_old_margins = {}
    for head in current_heads:
        current_old_margins[head] = float((cal_sim[head, head] - cal_sim[head, old_tensor].max()).cpu())
    return {
        "cal_sim": cal_sim,
        "raw_sim": raw_sim,
        "cal_pred": cal_pred,
        "raw_pred": raw_pred,
        "old_self_accuracy": old_self_accuracy,
        "old_min_margin": float(old_margins_tensor.min().cpu()),
        "old_mean_margin": float(old_margins_tensor.mean().cpu()),
        "current_old_margins": current_old_margins,
        "all_calibrated_self_accuracy": float(cal_pred.eq(own).float().mean().cpu()) * 100.0,
        "raw_calibrated_assignment_consistency": float(raw_pred.eq(cal_pred).float().mean().cpu()) * 100.0,
    }


@torch.no_grad()
def build_nc_safe_anchors(
    projector: DSMProjector,
    records: Sequence[PrototypeRecord],
    current_classes: Sequence[int],
    total_classes: int = 23,
):
    device = next(projector.parameters()).device
    calibrated = torch.stack([record.mean for record in records]).to(device)
    raw = torch.stack([record.raw_mean for record in records]).to(device)
    projected_calibrated = projector(calibrated)
    projected_raw = projector(raw)
    dynamic, dynamic_info = compute_dynamic_structure(projected_calibrated)
    nc = fixed_nc_geometry(total_classes, projected_calibrated.shape[1]).to(device)[: len(records)]
    seen = [record.class_id for record in records]
    current_heads = [seen.index(class_id) for class_id in current_classes]
    old_heads = [head for head in range(len(seen)) if head not in set(current_heads)]
    reference = geometry_stats(projected_calibrated, projected_raw, nc, old_heads, current_heads)
    safe = nc.clone()
    accepted_heads: list[int] = []
    decisions = []
    for head, record in enumerate(records):
        accepted_alpha = 0.0
        accepted_reason = "NO_ALPHA_GT_ZERO_PASSED_FALLBACK_NC"
        before = geometry_stats(projected_calibrated, projected_raw, safe, old_heads, current_heads)
        for alpha in ALPHAS[:-1]:
            candidate = safe.clone()
            candidate[head] = F.normalize((1.0 - alpha) * nc[head] + alpha * dynamic[head], dim=0)
            stats = geometry_stats(projected_calibrated, projected_raw, candidate, old_heads, current_heads)
            checks = {
                "calibrated_argmax_self": int(stats["cal_pred"][head]) == head,
                "current_margin_not_below_nc": all(
                    stats["current_old_margins"][current_head] >= reference["current_old_margins"][current_head] + EPSILON
                    for current_head in current_heads
                ),
                "old_self_accuracy_not_below_nc": stats["old_self_accuracy"] >= reference["old_self_accuracy"] + EPSILON,
                "old_min_margin_not_below_nc": stats["old_min_margin"] >= reference["old_min_margin"] + EPSILON,
                "raw_calibrated_assignment_consistent": int(stats["raw_pred"][head]) == int(stats["cal_pred"][head]),
                "accepted_prefix_still_self": all(int(stats["cal_pred"][accepted]) == accepted for accepted in accepted_heads),
                "accepted_prefix_assignment_consistent": all(
                    int(stats["raw_pred"][accepted]) == int(stats["cal_pred"][accepted]) for accepted in accepted_heads
                ),
            }
            if all(checks.values()):
                safe = candidate
                accepted_alpha = alpha
                accepted_reason = "ALL_SAFETY_CHECKS_PASSED"
                accepted_heads.append(head)
                break
            accepted_reason = "REJECTED:" + ",".join(name for name, passed in checks.items() if not passed)
        if accepted_alpha == 0.0:
            accepted_reason = "NO_ALPHA_GT_ZERO_PASSED_FALLBACK_NC"
        after = geometry_stats(projected_calibrated, projected_raw, safe, old_heads, current_heads)
        decisions.append({
            "class_id": record.class_id,
            "head_idx": head,
            "block": "current" if head in current_heads else "old",
            "alpha": accepted_alpha,
            "dynamic_accepted": accepted_alpha > 0.0,
            "reason": accepted_reason,
            "nc_own_cosine": float(reference["cal_sim"][head, head].cpu()),
            "safe_own_cosine": float(after["cal_sim"][head, head].cpu()),
            "nc_current_old_margin": reference["current_old_margins"].get(head),
            "safe_current_old_margin": after["current_old_margins"].get(head),
            "old_self_accuracy_before": before["old_self_accuracy"],
            "old_self_accuracy_after": after["old_self_accuracy"],
            "old_min_margin_before": before["old_min_margin"],
            "old_min_margin_after": after["old_min_margin"],
            "raw_assignment_class": seen[int(after["raw_pred"][head])],
            "calibrated_assignment_class": seen[int(after["cal_pred"][head])],
        })
    final = geometry_stats(projected_calibrated, projected_raw, safe, old_heads, current_heads)
    global_checks = {
        "current_margin_not_below_nc": all(
            final["current_old_margins"][head] >= reference["current_old_margins"][head] + EPSILON
            for head in current_heads
        ),
        "old_self_accuracy_not_below_nc": final["old_self_accuracy"] >= reference["old_self_accuracy"] + EPSILON,
        "old_min_margin_not_below_nc": final["old_min_margin"] >= reference["old_min_margin"] + EPSILON,
    }
    if not all(global_checks.values()):
        raise RuntimeError(f"Global NC-safe check failed: {global_checks}")
    metadata = {
        "alphas": {record.class_id: decisions[head]["alpha"] for head, record in enumerate(records)},
        "dynamic_acceptance_ratio": sum(row["dynamic_accepted"] for row in decisions) / len(decisions),
        "class9_alpha": decisions[seen.index(9)]["alpha"] if 9 in seen else None,
        "reference_old_self_accuracy": reference["old_self_accuracy"],
        "final_old_self_accuracy": final["old_self_accuracy"],
        "reference_old_min_margin": reference["old_min_margin"],
        "final_old_min_margin": final["old_min_margin"],
        "global_checks": global_checks,
        "dynamic_etf_residual": dynamic_info["etf_residual"],
    }
    return safe.detach(), dynamic.detach(), nc.detach(), decisions, metadata


@torch.no_grad()
def validate_current(projector, records, current_classes, validation, safe_anchors, nc_anchors, device):
    projector.eval()
    seen = [record.class_id for record in records]
    class_to_head = {class_id: head for head, class_id in enumerate(seen)}
    current_heads = [class_to_head[class_id] for class_id in current_classes]
    old_heads = [head for head in range(len(seen)) if head not in set(current_heads)]
    features = torch.cat([validation[class_id] for class_id in current_classes]).to(device)
    labels = torch.cat([torch.full((validation[class_id].shape[0],), class_to_head[class_id], dtype=torch.long)
                        for class_id in current_classes]).to(device)
    logits = projector(features) @ F.normalize(safe_anchors.to(device), dim=1).T
    predictions = logits.argmax(dim=1)
    per_class = {}
    for class_id, head in zip(current_classes, current_heads):
        mask = labels == head
        per_class[class_id] = float(predictions[mask].eq(labels[mask]).float().mean().cpu()) * 100.0
    micro = float(predictions.eq(labels).float().mean().cpu()) * 100.0
    pred_old = torch.tensor([int(value) in set(old_heads) for value in predictions.tolist()], device=device)
    current_to_old = float(pred_old.float().mean().cpu()) * 100.0
    absorption = {seen[head]: int(predictions[pred_old].eq(head).sum().cpu()) for head in old_heads}
    absorbing_class, absorbing_count = max(absorption.items(), key=lambda item: (item[1], -item[0]))
    projected_cal = projector(torch.stack([record.mean for record in records]).to(device))
    projected_raw = projector(torch.stack([record.raw_mean for record in records]).to(device))
    stats = geometry_stats(projected_cal, projected_raw, safe_anchors.to(device), old_heads, current_heads)
    current_margins = torch.tensor([stats["current_old_margins"][head] for head in current_heads])
    metrics = {
        "val_loss": float(F.cross_entropy(logits, labels).cpu()),
        "val_current_micro_accuracy": micro,
        "val_current_macro_accuracy": sum(per_class.values()) / len(per_class),
        "val_per_current_class_accuracy": json.dumps(per_class, sort_keys=True),
        "val_current_to_old": current_to_old,
        "val_max_old_absorption_count": absorbing_count,
        "val_max_old_absorption_ratio": 100.0 * absorbing_count / max(labels.numel(), 1),
        "val_absorbing_old_class_id": absorbing_class,
    }
    diagnostics = {
        "current_prototype_self_accuracy": float(stats["cal_pred"][current_heads].eq(torch.tensor(current_heads, device=device)).float().mean().cpu()) * 100.0,
        "old_prototype_self_accuracy": stats["old_self_accuracy"],
        "mean_current_vs_old_margin": float(current_margins.mean()),
        "p10_current_vs_old_margin": float(torch.quantile(current_margins, 0.1)),
        "prototype_anchor_cosine_mean": float(torch.diagonal(stats["cal_sim"])[current_heads].mean().cpu()),
        "prototype_anchor_cosine_min": float(torch.diagonal(stats["cal_sim"])[current_heads].min().cpu()),
        "old_min_margin": stats["old_min_margin"],
    }
    return metrics, diagnostics, absorption


def is_eligible(metrics, diagnostics, restore_ok):
    values = [value for value in list(metrics.values()) + list(diagnostics.values()) if isinstance(value, (float, int))]
    return (
        all(math.isfinite(float(value)) for value in values)
        and restore_ok
        and float(metrics["val_current_micro_accuracy"]) >= CURRENT_ACCURACY_MIN
        and float(metrics["val_current_to_old"]) < CURRENT_TO_OLD_MAX
    )


def selection_key(row):
    return (
        float(row["val_current_macro_accuracy"]), -float(row["val_current_to_old"]),
        float(row["val_current_micro_accuracy"]), -float(row["val_max_old_absorption_ratio"]),
        float(row["p10_current_vs_old_margin"]), float(row["old_prototype_self_accuracy"]), -int(row["epoch"]),
    )


def save_checkpoint(path, epoch, projector, optimizer, scheduler, mpc, records, safe, dynamic, nc, decisions,
                    metadata, source_path, source_sha, config):
    payload = {
        "epoch": epoch, "method": "nc_safe_dynamic_v1",
        "model_state_dict": projector.state_dict(), "projector_state_dict": projector.state_dict(),
        "projector_config": projector.config(), "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(), "mpc_state_dict": mpc.state_dict(), "mpc_config": mpc.config(),
        "repository": [record.cpu_dict() for record in records], "safe_anchors": safe.cpu(),
        "dynamic_proposal_anchors": dynamic.cpu(), "nc_reference_anchors": nc.cpu(),
        "alpha_decisions": decisions, "safety_metadata": metadata,
        "class_mapping": {record.class_id: head for head, record in enumerate(records)},
        "rng_state": rng_state(), "config": config, "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_sha, "git_commit": git_commit(),
    }
    torch.save(payload, path)
    return file_sha256(path)


def restore_check(path, sha):
    if file_sha256(path) != sha:
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "repository", "safe_anchors",
                "dynamic_proposal_anchors", "nc_reference_anchors", "alpha_decisions", "rng_state"}
    if not required.issubset(payload):
        return False
    model = DSMProjector(**payload["projector_config"])
    model.load_state_dict(payload["projector_state_dict"])
    return bool(torch.isfinite(payload["safe_anchors"]).all())


def freeze(args):
    stage_a = Path(args.stage_a_root).resolve()
    verdict = json.loads((stage_a / "causal_verdict.json").read_text())
    if verdict["verdict"] != "OLD_ANCHOR_DRIFT" or not verdict["stage_b_justified"]:
        raise RuntimeError(f"Stage B is not justified: {verdict}")
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    source_sha = file_sha256(Path(__file__).resolve())
    method = {
        "method": "nc_safe_dynamic_v1", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "stage_a_verdict": verdict["verdict"], "stage_a_flags": verdict["flags"],
        "alpha_candidates": list(ALPHAS), "alpha_order": "descending", "epsilon": EPSILON,
        "selection_data": "fit-train projected raw/calibrated prototype geometry only",
        "validation_used_for_alpha": False, "test_used_for_alpha": False,
        "per_class_rules": ["calibrated argmax remains self", "all current-vs-old margins not below NC",
                            "old prototype self accuracy not below NC", "old minimum margin not below NC",
                            "raw/calibrated assignment consistent", "accepted-prefix global safety"],
        "fallback": "alpha=0 NC reference when no alpha>0 passes", "targets": list(TARGETS),
        "class9_significant_reduction_gate": "fresh-test class9 absorption count <= 41 (at least 50 percent below frozen original count 83)",
        "source_sha256": source_sha,
    }
    write_json(output / "method_frozen.json", method)
    write_json(output / "method_source_hash.json", {"path": str(Path(__file__).resolve()), "sha256": source_sha})
    write_json(output / "gpu_shared_mode_policy.json", {
        "physical_gpu": 0, "shared_mode": True, "initial_min_free_mib": 12000,
        "subsequent_min_free_mib": 2186, "kill_allowed": False, "auto_retry": False,
    })
    print(json.dumps(method, indent=2))


def train_select(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or torch.cuda.device_count() != 1:
        raise RuntimeError("Require physical GPU0 only")
    key = (args.seed, args.session)
    target = TARGETS[key]
    output_root = Path(args.output_root).resolve()
    frozen = json.loads((output_root / "method_frozen.json").read_text())
    if file_sha256(Path(__file__).resolve()) != frozen["source_sha256"]:
        raise RuntimeError("Frozen method source hash mismatch")
    session_output = output_root / target["output"]
    if session_output.exists() and any(session_output.iterdir()):
        raise FileExistsError(session_output)
    (session_output / "checkpoints").mkdir(parents=True, exist_ok=True)
    snapshots = cuda_snapshot("before_train")
    if int(snapshots[0]["free_mib"]) < args.min_free_mib:
        write_csv(session_output / "gpu_snapshots.csv", snapshots)
        write_json(session_output / "selection.json", {"status": "GPU_INSUFFICIENT_FREE_MEMORY", "selected_epoch": None})
        raise SystemExit("GPU_INSUFFICIENT_FREE_MEMORY")
    set_seed(args.seed)
    device = torch.device("cuda")
    existing = Path(args.existing_root).resolve()
    v2 = Path(args.v2_root).resolve()
    source_path = existing / "runs" / f"seed{args.seed}_full_dynamic/checkpoints/session_{target['source_session']}.pt"
    source_sha = file_sha256(source_path)
    source_payload, projector, mpc, previous_records = load_source(source_path, device)
    config = dict(source_payload["config"])
    cache = torch.load(existing / "cache" / f"seed{args.seed}_feature_bank.pt", map_location="cpu", weights_only=False)
    validation = current_only_features(cache, read_csv(v2 / "validation_manifest.csv"), args.seed, args.session)
    fit = fit_features(cache, read_csv(v2 / "fit_manifest.csv"), args.seed, args.session)
    current_classes = [int(value) for value in cache["tasks"][args.session]]
    pool = torch.load(existing / f"runs/seed{args.seed}_full_dynamic/checkpoints/attribute_pool.pt", map_location="cpu", weights_only=False)
    attribute_visual, attribute_semantic = pool["visual_prototypes"], pool["semantic_embeddings"]
    names = {class_id: Path(cache["train_paths"][class_id][0]).parent.name for class_id in cache["class_order"]}
    mapping, _ = build_attribute_mapping(names)
    semantics = {class_id: class_semantic_embedding(mapping[class_id]) for class_id in cache["class_order"]}
    base_records = [record for record in previous_records if record.session_id == 0]
    records = list(previous_records)
    calibration = []
    for class_id in current_classes:
        values = fit[class_id]
        raw_mean = values.mean(dim=0)
        observed = values.var(dim=0, unbiased=False).clamp_min(1e-8)
        calibrated, calibration_stats = calibrate_prototype(
            mpc, raw_mean, semantics[class_id], attribute_visual, attribute_semantic, float(config["mpc_alpha"]), device
        )
        covariance, weights = blend_current_covariance(
            observed, calibrated, base_records, float(config["covariance_gamma"]), float(config["covariance_beta"])
        )
        records.append(PrototypeRecord(class_id, args.session, calibrated, covariance, raw_mean, len(values), "current", True))
        calibration.append({"class_id": class_id, **calibration_stats, "base_covariance_weight_max": float(weights.max())})
    optimizer = torch.optim.SGD(projector.parameters(), lr=float(config["projector_lr"]), momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    means = torch.stack([record.mean for record in records]).to(device)
    current_heads = list(range(len(previous_records), len(records)))
    run_config = {**config, "method": "nc_safe_dynamic_v1", "alpha_candidates": list(ALPHAS),
                  "epsilon": EPSILON, "validation_used_for_alpha": False, "test_used_in_training": False}
    epoch_rows, geometry_rows, checkpoint_rows, alpha_rows, decision_rows, logs = [], [], [], [], [], []
    try:
        for epoch in range(EPOCHS + 1):
            if epoch > 0:
                projector.train()
                safe_before, _, _, _, _ = build_nc_safe_anchors(projector, records, current_classes)
                features, labels, sample_stats = resample_repository(records, current_classes, args.seed + args.session * 100, epoch - 1)
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                projected = projector(features)
                match = class_balanced_mean(F.cross_entropy(projected @ safe_before.T, labels, reduction="none"), labels, len(records))
                cont = structure_anchor_contrastive(projected, labels, safe_before, current_heads, 0.07)
                loss = match + cont
                loss.backward()
                optimizer.step()
                scheduler.step()
                train = {"train_loss": float(loss.detach().cpu()), "train_match_loss": float(match.detach().cpu()),
                         "train_cont_loss": float(cont.detach().cpu()),
                         "resampling_checksum": hashlib.sha256("".join(row["checksum"] for row in sample_stats).encode()).hexdigest()}
            else:
                train = {"train_loss": None, "train_match_loss": None, "train_cont_loss": None, "resampling_checksum": None}
            safe, dynamic, nc, decisions, safety = build_nc_safe_anchors(projector, records, current_classes)
            metrics, diagnostics, _ = validate_current(projector, records, current_classes, validation, safe, nc, device)
            checkpoint = session_output / f"checkpoints/epoch_{epoch}.pt"
            sha = save_checkpoint(checkpoint, epoch, projector, optimizer, scheduler, mpc, records, safe, dynamic, nc,
                                  decisions, safety, source_path, source_sha, run_config)
            restored = restore_check(checkpoint, sha)
            eligible = is_eligible(metrics, diagnostics, restored)
            epoch_rows.append({"seed": args.seed, "session": args.session, "epoch": epoch, **train, **metrics, **diagnostics,
                               "dynamic_acceptance_ratio": safety["dynamic_acceptance_ratio"], "class9_alpha": safety["class9_alpha"],
                               "restore_ok": restored, "eligible": eligible, "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha})
            geometry_rows.append({"epoch": epoch, **diagnostics, **{k: v for k, v in safety.items() if k != "alphas" and k != "global_checks"}})
            checkpoint_rows.append({"epoch": epoch, "path": str(checkpoint), "sha256": sha, "restore_ok": restored, "eligible": eligible})
            for class_id, alpha in safety["alphas"].items():
                alpha_rows.append({"epoch": epoch, "class_id": class_id, "alpha": alpha, "class9": int(class_id) == 9})
            decision_rows.extend({"epoch": epoch, **row} for row in decisions)
            logs.append(f"epoch={epoch} micro={metrics['val_current_micro_accuracy']:.8f} c2o={metrics['val_current_to_old']:.8f} "
                        f"accept={safety['dynamic_acceptance_ratio']:.8f} eligible={eligible}")
    except torch.cuda.OutOfMemoryError:
        snapshots.extend(cuda_snapshot("oom"))
        write_csv(session_output / "gpu_snapshots.csv", snapshots)
        write_json(session_output / "selection.json", {"status": "GPU_SHARED_MODE_OOM", "selected_epoch": None})
        (session_output / "run.log").write_text("\n".join(logs + ["GPU_SHARED_MODE_OOM"]) + "\n")
        raise SystemExit("GPU_SHARED_MODE_OOM")
    eligible_rows = [row for row in epoch_rows if row["eligible"]]
    selected = max(eligible_rows, key=selection_key) if eligible_rows else None
    selection = {
        "status": "SELECTED" if selected else "ALL_EPOCHS_COLLAPSED", "seed": args.seed, "session": args.session,
        "selected_epoch": int(selected["epoch"]) if selected else None,
        "selected_checkpoint": selected["checkpoint_path"] if selected else None,
        "selected_checkpoint_sha256": selected["checkpoint_sha256"] if selected else None,
        "eligible_epochs": [int(row["epoch"]) for row in eligible_rows],
        "diagnostic_least_bad_epoch": int(max(epoch_rows, key=selection_key)["epoch"]),
        "selector_accessed_test": False, "official_rescue": selected is not None,
    }
    write_csv(session_output / "epoch_metrics.csv", epoch_rows)
    write_csv(session_output / "geometry_diagnostics.csv", geometry_rows)
    write_csv(session_output / "alpha_by_epoch.csv", alpha_rows)
    write_csv(session_output / "safety_decisions.csv", decision_rows)
    write_json(session_output / "checkpoint_manifest.json", {"checkpoints": checkpoint_rows})
    write_json(session_output / "selection.json", selection)
    snapshots.extend(cuda_snapshot("after_train"))
    write_csv(session_output / "gpu_snapshots.csv", snapshots)
    write_json(session_output / "memory_peak.json", {"max_memory_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
                                                       "max_memory_reserved_mib": torch.cuda.max_memory_reserved()/1024**2})
    (session_output / "run.log").write_text("\n".join(logs + [f"status={selection['status']} selected={selection['selected_epoch']}"]) + "\n")
    print(json.dumps(selection, indent=2))


@torch.no_grad()
def fresh_test(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Require GPU0")
    target = TARGETS[(args.seed, args.session)]
    output = Path(args.output_root).resolve() / target["output"]
    selection = json.loads((output / "selection.json").read_text())
    if selection["selected_epoch"] is None:
        write_json(output / "fresh_test_metrics.json", {"status": "NOT_RUN", "reason": selection["status"]})
        write_csv(output / "confusion_matrix.csv", [], ("true_class", "predicted_class", "count"))
        write_csv(output / "absorption_by_old_class.csv", [], ("old_class_id", "absorption_count", "absorption_ratio"))
        return
    checkpoint = Path(selection["selected_checkpoint"])
    sha = file_sha256(checkpoint)
    if sha != selection["selected_checkpoint_sha256"]:
        raise RuntimeError("SHA mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda")
    projector = DSMProjector(**payload["projector_config"]).to(device)
    projector.load_state_dict(payload["projector_state_dict"])
    projector.eval()
    records = [PrototypeRecord.from_dict(row) for row in payload["repository"]]
    seen = [record.class_id for record in records]
    current_classes = [record.class_id for record in records if record.session_id == args.session]
    current_heads = {seen.index(class_id) for class_id in current_classes}
    old_heads = set(range(len(seen))) - current_heads
    cache = torch.load(Path(args.existing_root).resolve() / f"cache/seed{args.seed}_feature_bank.pt", map_location="cpu", weights_only=False)
    features = torch.cat([cache["test_features"][class_id] for class_id in seen]).to(device)
    labels = torch.cat([torch.full((cache["test_features"][class_id].shape[0],), head, dtype=torch.long)
                        for head, class_id in enumerate(seen)])
    logits = projector(features) @ F.normalize(payload["safe_anchors"].to(device), dim=1).T
    groups = {record.class_id: record.frequency_group for record in records}
    metrics, class_rows, confusion = metric_pack(logits.cpu(), labels, seen, [seen[h] for h in old_heads], current_classes, groups)
    predictions = logits.cpu().argmax(dim=1)
    current_mask = torch.tensor([int(value) in current_heads for value in labels.tolist()])
    absorption = {seen[head]: int(predictions[current_mask].eq(head).sum()) for head in old_heads}
    absorbing_class, absorbing_count = max(absorption.items(), key=lambda item: (item[1], -item[0]))
    class9_count = absorption.get(9, 0)
    fresh = {
        "status": "OK", "seed": args.seed, "session": args.session, "selected_epoch": selection["selected_epoch"],
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha, "restore_equality": restore_check(checkpoint, sha),
        "test_AccT": metrics["AccT"], "test_old_accuracy": metrics["old_acc"],
        "test_current_accuracy": metrics["current_acc"], "test_HM": metrics["HM_old_current"],
        "test_old_to_current": metrics["old_to_current_rate"], "test_current_to_old": metrics["current_to_old_rate"],
        "test_BER": metrics["BER"], "absorbing_old_class_id": absorbing_class,
        "max_old_absorption_count": absorbing_count,
        "max_old_absorption_ratio": 100.0 * absorbing_count / max(int(current_mask.sum()), 1),
        "class9_absorption_count": class9_count,
        "class9_absorption_ratio": 100.0 * class9_count / max(int(current_mask.sum()), 1),
        "test_collapse": metrics["current_acc"] < CURRENT_ACCURACY_MIN or metrics["current_to_old_rate"] >= CURRENT_TO_OLD_MAX,
        "selected_alpha_by_class": payload["safety_metadata"]["alphas"],
        "dynamic_acceptance_ratio": payload["safety_metadata"]["dynamic_acceptance_ratio"],
        "class9_alpha": payload["safety_metadata"]["class9_alpha"],
    }
    write_json(output / "fresh_test_metrics.json", fresh)
    write_csv(output / "confusion_matrix.csv", [
        {"true_class": true, "predicted_class": pred, "count": count}
        for true in sorted(confusion) for pred, count in sorted(confusion[true].items())
    ])
    write_csv(output / "absorption_by_old_class.csv", [
        {"old_class_id": class_id, "absorption_count": count,
         "absorption_ratio": 100.0 * count / max(int(current_mask.sum()), 1)} for class_id, count in sorted(absorption.items())
    ])
    write_csv(output / "fresh_test_per_class.csv", class_rows)
    print(json.dumps(fresh, indent=2))


def summarize(args):
    output = Path(args.output_root).resolve()
    existing = Path(args.existing_root).resolve()
    keys = MICROCHECK if args.phase == "microcheck" else tuple(TARGETS)
    original_rows = read_csv(existing / "csv/all_session_metrics.csv")
    rows = []
    for seed, session in keys:
        target = TARGETS[(seed, session)]
        directory = output / target["output"]
        selection = json.loads((directory / "selection.json").read_text())
        fresh = json.loads((directory / "fresh_test_metrics.json").read_text())
        original = next(row for row in original_rows if row["method"] == "full_dynamic" and int(row["seed"]) == seed and int(row["session"]) == session)
        rows.append({
            "seed": seed, "session": session, "role": target["role"], "selection_status": selection["status"],
            "eligible_epochs": json.dumps(selection["eligible_epochs"]), "selected_epoch": selection["selected_epoch"],
            "test_status": fresh.get("status"), "test_AccT": fresh.get("test_AccT"),
            "test_current_accuracy": fresh.get("test_current_accuracy"), "test_HM": fresh.get("test_HM"),
            "test_current_to_old": fresh.get("test_current_to_old"), "test_collapse": fresh.get("test_collapse"),
            "class9_absorption_count": fresh.get("class9_absorption_count"), "class9_alpha": fresh.get("class9_alpha"),
            "dynamic_acceptance_ratio": fresh.get("dynamic_acceptance_ratio"), "restore_equality": fresh.get("restore_equality"),
            "original_AccT": original["AccT"], "original_current_accuracy": original["current_acc"],
            "original_HM": original["HM_old_current"], "original_current_to_old": original["current_to_old_rate"],
        })
    write_csv(output / ("microcheck_summary.csv" if args.phase == "microcheck" else "four_session_summary.csv"), rows)
    if args.phase == "microcheck":
        control = next(row for row in rows if row["role"] == "stable_control")
        target = next(row for row in rows if row["role"] == "persistent_target")
        target_ok = (
            target["selected_epoch"] is not None and target["test_status"] == "OK" and not target["test_collapse"]
            and float(target["test_current_to_old"]) < CURRENT_TO_OLD_MAX
            and int(target["class9_absorption_count"]) <= 41 and bool(target["restore_equality"])
        )
        control_ok = (
            control["test_status"] == "OK" and not control["test_collapse"] and bool(control["restore_equality"])
            and float(control["test_AccT"]) >= float(control["original_AccT"]) - 1.0
            and float(control["test_HM"]) >= float(control["original_HM"]) - 1.0
            and float(control["test_current_accuracy"]) >= float(control["original_current_accuracy"]) - 1.0
            and float(json.loads((output / "seed2/session5_control/fresh_test_metrics.json").read_text())["max_old_absorption_ratio"]) < 30.0
        )
        gate = "NC_SAFE_DYNAMIC_MICROCHECK_PASSED" if target_ok and control_ok else "NC_SAFE_DYNAMIC_MICROCHECK_FAILED"
        promotion = "PROMOTE_TO_FOUR_SESSION_CONFIRMATION" if target_ok and control_ok else "CLOSE_FULL_DYNAMIC_REPAIR_BRANCH"
        write_json(output / "microcheck_gate.json", {"gate": gate, "promotion": promotion,
                                                       "target_success": target_ok, "control_success": control_ok,
                                                       "next": "RUN_CONFIRMATION" if target_ok and control_ok else "NO_FULL_PROMOTION"})
        print(json.dumps({"gate": gate, "promotion": promotion}, indent=2))
        return
    micro = json.loads((output / "microcheck_gate.json").read_text())
    confirmation = [row for row in rows if row["role"] == "confirmation_target"]
    confirmation_ok = all(
        row["selected_epoch"] is not None and row["test_status"] == "OK" and not row["test_collapse"] and bool(row["restore_equality"])
        for row in confirmation
    )
    gate = "NC_SAFE_DYNAMIC_FOUR_SESSION_CONFIRMATION_PASSED" if micro["gate"] == "NC_SAFE_DYNAMIC_MICROCHECK_PASSED" and confirmation_ok else "NC_SAFE_DYNAMIC_FOUR_SESSION_CONFIRMATION_FAILED"
    write_json(output / "gate.json", {"gate": gate, "microcheck_gate": micro["gate"],
                                       "confirmation_success": confirmation_ok, "promotion": "STOP_NO_FULL_15_SESSION"})
    lines = "\n".join(
        f"- seed{row['seed']}/session{row['session']} {row['role']}: eligible={row['eligible_epochs']}, selected={row['selected_epoch']}, "
        f"current={row['test_current_accuracy']}, HM={row['test_HM']}, c2o={row['test_current_to_old']}, "
        f"class9_abs={row['class9_absorption_count']}, class9_alpha={row['class9_alpha']}, accept={row['dynamic_acceptance_ratio']}"
        for row in rows
    )
    report = f"""# NC-safe Dynamic v1 Bounded Confirmation

## Frozen Method

- Stage A verdict: `OLD_ANCHOR_DRIFT` with `MIXED_GEOMETRY_AND_TRAJECTORY_FAILURE`.
- Alpha candidates: `{list(ALPHAS)}`, descending, epsilon 0, selected from fit-prototype geometry only.
- Validation/test never selects alpha. Raw-prototype rescue is disabled.

## Sessions

{lines}

## Decision

- Microcheck: `{micro['gate']}`.
- Four-session confirmation: `{gate}`.
- Full 15-session run was not started.
- Final action: `STOP_NO_FULL_15_SESSION`.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    write_json(output / "source_manifest.json", {"git_commit": git_commit(), "files": [
        {"path": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__).resolve())},
        {"path": str(REPO_ROOT / "src/methods/full_concm.py"), "sha256": file_sha256(REPO_ROOT / "src/methods/full_concm.py")},
        {"path": str(REPO_ROOT / "src/methods/concm_dsm.py"), "sha256": file_sha256(REPO_ROOT / "src/methods/concm_dsm.py")},
    ]})
    print(json.dumps({"gate": gate, "promotion": "STOP_NO_FULL_15_SESSION"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subs.add_parser("freeze")
    freeze_parser.add_argument("--stage-a-root", required=True)
    freeze_parser.add_argument("--output-root", required=True)
    for command in ("train-select", "fresh-test"):
        sub = subs.add_parser(command)
        sub.add_argument("--seed", type=int, required=True)
        sub.add_argument("--session", type=int, required=True)
        sub.add_argument("--existing-root", required=True)
        sub.add_argument("--v2-root", required=True)
        sub.add_argument("--output-root", required=True)
        if command == "train-select":
            sub.add_argument("--min-free-mib", type=int, required=True)
    summary = subs.add_parser("summarize")
    summary.add_argument("--phase", choices=("microcheck", "four"), required=True)
    summary.add_argument("--existing-root", required=True)
    summary.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if args.command == "freeze": freeze(args)
    elif args.command == "train-select": train_select(args)
    elif args.command == "fresh-test": fresh_test(args)
    else: summarize(args)


if __name__ == "__main__":
    main()
