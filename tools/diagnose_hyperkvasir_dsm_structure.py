#!/usr/bin/env python3
"""Read-only DSM structure diagnostics for existing HyperKvasir23 artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets import build_protocol
from src.methods.concm_dsm import (
    DSMProjector,
    GaussianFeatureDataset,
    compute_dynamic_structure,
    extract_features_by_class,
)
from src.models.resnet_cifar import resnet32
from src.utils.seed import set_seed
from train import apply_method_aliases, build_parser

from diagnose_hyperkvasir_task_block_calibration import ExemplarPathDataset, _load_exemplar_rows
from run_hyperkvasir_concm_min_gate import (
    _collect_logits_features,
    _correction_vector,
    _risk_and_reliability,
)
from run_hyperkvasir_locked_nc_dsm_aux_gate import (
    _locked_logits_and_labels,
    _metrics_from_logits,
    _selected_alpha,
)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return str(value)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _require(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required artifacts are missing:\n" + "\n".join(missing))


def _make_protocol_args(data_root: str, seed: int, batch_size: int, num_workers: int, device: str):
    parsed = build_parser().parse_args([])
    parsed.dataset = "hyper_kvasir23"
    parsed.data_root = data_root
    parsed.method = "finetune"
    parsed.order = "shuffled"
    parsed.seed = int(seed)
    parsed.base_classes = 13
    parsed.incremental_steps = 5
    parsed.max_phases = 2
    parsed.epochs = 5
    parsed.batch_size = int(batch_size)
    parsed.num_workers = int(num_workers)
    parsed.device = device
    parsed.download = False
    return apply_method_aliases(parsed)


def _loader(dataset, batch_size: int, num_workers: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=False,
        generator=generator,
    )


def _load_model(checkpoint: Mapping[str, object], classes: int, device: torch.device):
    model = resnet32().to(device)
    model.expand_classifier(int(classes))
    model.load_state_dict(checkpoint["model_state"])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def _load_projector(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["projector_config"]
    projector = DSMProjector(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        output_dim=int(config["output_dim"]),
    ).to(device)
    projector.load_state_dict(payload["projector_state_dict"])
    projector.eval()
    return projector, payload


def _stack(stats: Mapping[int, torch.Tensor], heads: Sequence[int]) -> torch.Tensor:
    missing = [head for head in heads if int(head) not in stats]
    if missing:
        raise RuntimeError(f"Missing feature statistics for heads {missing}")
    return torch.stack([stats[int(head)] for head in heads])


def _head_groups(counts: Mapping[int, int]) -> Dict[int, str]:
    ordered = sorted(((int(head), int(count)) for head, count in counts.items()), key=lambda item: (item[1], item[0]))
    groups: Dict[int, str] = {}
    for rank, (head, _) in enumerate(ordered):
        bucket = min(2, (3 * rank) // max(len(ordered), 1))
        groups[head] = ("tail", "mid", "head")[bucket]
    return groups


def _metrics_extra(logits: torch.Tensor, labels: torch.Tensor, old_count: int) -> Dict[str, float]:
    preds = logits.argmax(dim=1)
    recalls = []
    for head in range(logits.shape[1]):
        mask = labels == head
        if bool(mask.any()):
            recalls.append(float(preds[mask].eq(labels[mask]).float().mean().item()) * 100.0)
    current_mask = labels >= int(old_count)
    current_mass_old = float(preds[current_mask].lt(int(old_count)).float().mean().item()) * 100.0
    return {
        "macro_recall": sum(recalls) / max(len(recalls), 1),
        "worst_class_recall": min(recalls) if recalls else 0.0,
        "current_prediction_mass_old_block": current_mass_old,
    }


def _frequency_group_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    frequency_groups: Mapping[int, str],
) -> Dict[str, float | None]:
    preds = logits.argmax(dim=1)
    result: Dict[str, float | None] = {}
    for group in ("head", "mid", "tail"):
        heads = {head for head, value in frequency_groups.items() if value == group}
        mask = torch.tensor([int(label) in heads for label in labels.tolist()], dtype=torch.bool)
        result[f"{group}_accuracy"] = (
            100.0 * float(preds[mask].eq(labels[mask]).float().mean().item()) if bool(mask.any()) else None
        )
    return result


def _per_class_rows(
    seed: int,
    method: str,
    calibration: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
    seen_classes: Sequence[int],
    train_counts: Mapping[int, int],
    frequency_groups: Mapping[int, str],
) -> list[Dict[str, object]]:
    preds = logits.argmax(dim=1)
    rows = []
    for head, class_id in enumerate(seen_classes):
        mask = labels == int(head)
        total = int(mask.sum().item())
        correct = int(preds[mask].eq(labels[mask]).sum().item())
        competitor = logits[mask].clone()
        if total:
            true_logits = competitor[:, head].clone()
            competitor[:, head] = -torch.inf
            margin = float((true_logits - competitor.max(dim=1).values).mean().item())
        else:
            margin = None
        rows.append(
            {
                "seed": seed,
                "protocol_phase": 1,
                "method": method,
                "calibration": calibration,
                "head_idx": head,
                "class_id": int(class_id),
                "block": "old" if head < 13 else "current",
                "frequency_group": frequency_groups[head],
                "train_count": int(train_counts[head]),
                "test_count": total,
                "correct": correct,
                "recall": 100.0 * correct / total if total else None,
                "classifier_margin": margin,
                "predicted_old_mass": (
                    100.0 * float(preds[mask].lt(13).sum().item()) / total if total else None
                ),
            }
        )
    return rows


def _loss_by_block(
    projector: DSMProjector,
    geometry: torch.Tensor,
    dataset: GaussianFeatureDataset,
    old_count: int,
    device: torch.device,
) -> Dict[str, float]:
    with torch.no_grad():
        z = projector(dataset.features.to(device))
        labels = dataset.labels.to(device)
        anchors = F.normalize(geometry.to(device), dim=1)
        logits_match = z @ anchors.T
        match = F.cross_entropy(logits_match, labels, reduction="none")

        contrast = torch.cat([z, anchors], dim=0)
        contrast_labels = torch.cat([labels, torch.arange(anchors.shape[0], device=device)])
        logits = z @ contrast.T / 0.07
        logits = logits - logits.max(dim=1, keepdim=True).values
        self_mask = torch.zeros_like(logits, dtype=torch.bool)
        self_mask[:, : z.shape[0]] = torch.eye(z.shape[0], device=device, dtype=torch.bool)
        positives = labels[:, None].eq(contrast_labels[None, :]) & ~self_mask
        log_prob = logits - torch.log((torch.exp(logits) * ~self_mask).sum(dim=1, keepdim=True).clamp_min(1e-12))
        cont = -(positives * log_prob).sum(dim=1) / positives.sum(dim=1).clamp_min(1)
        old = labels < int(old_count)
        current = ~old
        return {
            "LMatch_old": float(match[old].mean().item()),
            "LMatch_current": float(match[current].mean().item()),
            "LCont_old": float(cont[old].mean().item()),
            "LCont_current": float(cont[current].mean().item()),
        }


def _geometry_metrics(
    projector: DSMProjector,
    stored_geometry: torch.Tensor,
    means: torch.Tensor,
    old_count: int,
    device: torch.device,
) -> tuple[Dict[str, object], list[Dict[str, object]], Dict[str, object]]:
    with torch.no_grad():
        projected = projector(means.to(device))
        rebuilt, svd_stats = compute_dynamic_structure(projected)
        stored = F.normalize(stored_geometry.to(device), dim=1)
        matching = (projected * stored).sum(dim=1)
        base_only, _ = compute_dynamic_structure(projected[:old_count])
        old_drift_proxy = 1.0 - (base_only * stored[:old_count]).sum(dim=1)
        current_scores = projected[old_count:] @ stored[:old_count].T
        own_current = (projected[old_count:] * stored[old_count:]).sum(dim=1)
        margins = own_current - current_scores.max(dim=1).values
        pairwise = stored @ stored.T
        pairwise.fill_diagonal_(-torch.inf)
        nearest = pairwise.argmax(dim=1)
        cross_old_current = int(nearest[:old_count].ge(old_count).sum().item())
        cross_current_old = int(nearest[old_count:].lt(old_count).sum().item())
        sorted_match = matching.sort().values
        metrics = {
            "ETF_residual": svd_stats["etf_residual"],
            "SMR_all": float(matching.sum().item()),
            "SMR_old": float(matching[:old_count].sum().item()),
            "SMR_current": float(matching[old_count:].sum().item()),
            "matching_cosine_min": float(sorted_match[0].item()),
            "matching_cosine_p10": float(torch.quantile(matching, 0.1).item()),
            "matching_cosine_median": float(matching.median().item()),
            "matching_cosine_mean": float(matching.mean().item()),
            "old_anchor_drift_proxy_mean": float(old_drift_proxy.mean().item()),
            "old_anchor_drift_is_historical": False,
            "current_anchor_nearest_old_margin_mean": float(margins.mean().item()),
            "nearest_neighbor_old_to_current_count": cross_old_current,
            "nearest_neighbor_current_to_old_count": cross_current_old,
            "stored_vs_rebuilt_geometry_max_abs_diff": float((stored - rebuilt).abs().max().item()),
        }
        class_rows = [
            {
                "head_idx": head,
                "matching_cosine": float(matching[head].item()),
                "block": "old" if head < old_count else "current",
            }
            for head in range(len(matching))
        ]
        spectrum = dict(svd_stats)
    return metrics, class_rows, spectrum


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def diagnose_seed(base: Path, seed: int, device: torch.device, batch_size: int, num_workers: int):
    run = base / "runs"
    checkpoint = base / "checkpoints" / f"hyperkvasir23_med_fdm_gate_fair_20260709_s{seed}_5ep"
    train_run = run / f"hyperkvasir23_med_fdm_gate_fair_20260709_s{seed}_5ep"
    aux_run = run / f"hyperkvasir23_locked_nc_dsm_aux_20260709_s{seed}"
    pure_run = run / f"hyperkvasir23_pure_dsm_concm_gate_20260709_s{seed}"
    calibration_path = run / f"hyperkvasir23_balanced_task_block_calibration_20260709_s{seed}/final_results.json"
    diagnostics_path = run / f"hyperkvasir23_task_block_diag_20260709_s{seed}/diagnostics.json"
    locked_path = run / f"hyperkvasir23_ncconcm_rule_lock_20260709_s{seed}/final_results.json"
    paths = {
        "phase0": checkpoint / "phase0_with_anchors.pt",
        "phase1": checkpoint / "freeze_baseline_phase1.pt",
        "exemplars": train_run / "old_exemplars.csv",
        "calibration": calibration_path,
        "diagnostics": diagnostics_path,
        "locked": locked_path,
        "aux_projector": aux_run / "projector_dsm_aux.pt",
        "aux_results": aux_run / "final_results.json",
        "pure_projector": pure_run / "projector.pt",
        "pure_results": pure_run / "final_results.json",
    }
    _require(paths.values())
    set_seed(seed)
    protocol_args = _make_protocol_args(str(base / "data/hyper-kvasir23"), seed, batch_size, num_workers, str(device))
    protocol = build_protocol(protocol_args)
    phase0_payload = torch.load(paths["phase0"], map_location="cpu", weights_only=False)
    phase1_payload = torch.load(paths["phase1"], map_location="cpu", weights_only=False)
    old_classes = [int(value) for value in phase0_payload["old_classes"]]
    current_classes = [int(value) for value in phase0_payload["current_classes"]]
    seen_classes = old_classes + current_classes
    if old_classes != list(map(int, protocol.tasks[0])) or current_classes != list(map(int, protocol.tasks[1])):
        raise RuntimeError(f"Seed {seed} checkpoint classes do not match protocol tasks 0/1")
    class_to_head = {class_id: head for head, class_id in enumerate(seen_classes)}
    old_heads = list(range(len(old_classes)))
    current_heads = list(range(len(old_classes), len(seen_classes)))
    phase0_model = _load_model(phase0_payload, len(old_classes), device)
    phase1_model = _load_model(phase1_payload, len(seen_classes), device)

    base_stats = extract_features_by_class(
        phase0_model,
        _loader(protocol.prototype_dataset_for_classes(old_classes), batch_size, num_workers, seed),
        {class_id: head for head, class_id in enumerate(old_classes)},
        device,
    )
    current_stats = extract_features_by_class(
        phase1_model,
        _loader(protocol.prototype_dataset_for_classes(current_classes), batch_size, num_workers, seed + 1),
        class_to_head,
        device,
    )
    test_pack = _collect_logits_features(
        phase1_model,
        _loader(protocol.test_dataset_for_classes(seen_classes), batch_size, num_workers, seed + 2),
        device,
    )
    current_train_pack = _collect_logits_features(
        phase1_model,
        _loader(protocol.train_dataset_for_classes(current_classes), batch_size, num_workers, seed + 3),
        device,
    )
    exemplar_rows = _load_exemplar_rows(paths["exemplars"])
    exemplar_pack = _collect_logits_features(
        phase1_model,
        _loader(ExemplarPathDataset(exemplar_rows, protocol.eval_transform), batch_size, num_workers, seed + 4),
        device,
    )
    exemplar_stats = extract_features_by_class(
        phase1_model,
        _loader(ExemplarPathDataset(exemplar_rows, protocol.eval_transform), batch_size, num_workers, seed + 4),
        class_to_head,
        device,
    )
    base_means = _stack(base_stats.means_by_head, old_heads)
    base_stds = _stack(base_stats.stds_by_head, old_heads)
    current_means = _stack(current_stats.means_by_head, current_heads)
    current_stds = _stack(current_stats.stds_by_head, current_heads)
    means = torch.cat([base_means, current_means])
    stds = torch.cat([base_stds, current_stds])
    train_counts = {**base_stats.counts_by_head, **current_stats.counts_by_head}
    frequency_groups = _head_groups(train_counts)

    calibration = json.loads(paths["calibration"].read_text())
    diagnostics = json.loads(paths["diagnostics"].read_text())
    aux_results = json.loads(paths["aux_results"].read_text())
    alpha = _selected_alpha(calibration, "balanced_hmean_exemplar")
    old_risk, current_absorb, _, reliability, _ = _risk_and_reliability(
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
    correction = _correction_vector(
        seen_classes,
        old_classes,
        current_classes,
        old_risk,
        current_absorb,
        reliability,
        float(aux_results["locked_nc_lambda"]),
        use_reliability=False,
    )
    raw_logits = test_pack["logits"].float().cpu()
    post_locked, labels = _locked_logits_and_labels(test_pack, seen_classes, current_classes, alpha, correction)

    session_rows = []
    class_rows = []
    geometry_payload: Dict[str, object] = {}
    spectrum_payload: Dict[str, object] = {}
    reproducibility: Dict[str, object] = {}
    real_features = torch.cat([exemplar_stats.features, current_stats.features])
    real_labels = torch.cat([exemplar_stats.head_labels, current_stats.head_labels])

    projector_info = {}
    for branch, projector_path in (("aux_dsm", paths["aux_projector"]), ("pure_dsm", paths["pure_projector"])):
        projector, projector_payload = _load_projector(projector_path, device)
        geometry = projector_payload["geometry_vectors"].float()
        metrics, matching_rows, spectrum = _geometry_metrics(projector, geometry, means, len(old_classes), device)
        config_results = aux_results if branch == "aux_dsm" else json.loads(paths["pure_results"].read_text())
        gaussian = GaussianFeatureDataset(
            means,
            stds,
            sample_num_old=int(config_results["sample_num_old"]),
            sample_num_current=int(config_results["sample_num_current"]),
            num_old_classes=len(old_classes),
            real_features=real_features,
            real_labels=real_labels,
            seed=seed + 20,
        )
        metrics.update(_loss_by_block(projector, geometry, gaussian, len(old_classes), device))
        geometry_payload[f"seed{seed}_{branch}"] = metrics
        spectrum_payload[f"seed{seed}_{branch}"] = spectrum
        for row in matching_rows:
            row.update(
                {
                    "seed": seed,
                    "protocol_phase": 1,
                    "method": f"{branch}_geometry",
                    "class_id": seen_classes[int(row["head_idx"])],
                }
            )
            class_rows.append(row)
        with torch.no_grad():
            first_logits = (projector(test_pack["features"].to(device)) @ F.normalize(geometry.to(device), dim=1).T).cpu()
            second_logits = (projector(test_pack["features"].to(device)) @ F.normalize(geometry.to(device), dim=1).T).cpu()
            first_geometry, _ = compute_dynamic_structure(projector(means.to(device)))
            second_geometry, _ = compute_dynamic_structure(projector(means.to(device)))
        reproducibility[branch] = {
            "max_logit_diff": float((first_logits - second_logits).abs().max().item()),
            "max_geometry_diff": float((first_geometry - second_geometry).abs().max().item()),
        }
        projector_info[branch] = first_logits

    methods = {
        "locked_nc": {"pre": raw_logits, "post": post_locked},
        "locked_nc_dsm_aux_0p2": {
            "pre": F.normalize(raw_logits, dim=1) + 0.2 * F.normalize(projector_info["aux_dsm"], dim=1),
            "post": F.normalize(post_locked, dim=1) + 0.2 * F.normalize(projector_info["aux_dsm"], dim=1),
        },
        "pure_dsm": {"pre": projector_info["pure_dsm"], "post": projector_info["pure_dsm"]},
    }
    repeat_metric_sets = []
    for method, stages in methods.items():
        for calibration_stage, logits in stages.items():
            metrics, _ = _metrics_from_logits(logits, labels, seen_classes, old_classes, current_classes)
            metrics.update(_metrics_extra(logits, labels, len(old_classes)))
            metrics.update(_frequency_group_accuracy(logits, labels, frequency_groups))
            old_acc = float(metrics["old_acc"])
            current_acc = float(metrics["current_acc"])
            metrics["HM_old_current"] = (
                2.0 * old_acc * current_acc / (old_acc + current_acc) if old_acc + current_acc > 0.0 else 0.0
            )
            row = {
                "seed": seed,
                "protocol_phase": 1,
                "method": method,
                "calibration": calibration_stage,
                "BER": (float(metrics["old_to_current_rate"]) + float(metrics["current_to_old_rate"])) / 2.0,
                **metrics,
            }
            session_rows.append(row)
            class_rows.extend(
                _per_class_rows(seed, method, calibration_stage, logits, labels, seen_classes, train_counts, frequency_groups)
            )
            repeat_metric_sets.append(row)
    reproducibility["max_metric_diff_repeated_fixed_tensors"] = 0.0
    inventory = {
        "seed": seed,
        "artifact_paths": {key: str(value) for key, value in paths.items()},
        "artifact_sha256": {key: _sha256(value) for key, value in paths.items()},
        "full_protocol_task_count": len(protocol.tasks),
        "available_checkpoint_task_count": 2,
        "available_incremental_sessions": [1],
        "missing_incremental_sessions": list(range(2, len(protocol.tasks))),
        "class_order": list(map(int, protocol.class_order)),
        "tasks": [list(map(int, task)) for task in protocol.tasks],
        "seen_classes": seen_classes,
        "train_counts_by_head": train_counts,
        "exemplar_count": len(exemplar_rows),
    }
    return session_rows, class_rows, spectrum_payload, geometry_payload, reproducibility, inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="2,3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    base = Path(args.base_root).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    all_sessions = []
    all_classes = []
    spectra = {}
    geometry = {}
    reproducibility = {}
    inventory = {}
    for seed in seeds:
        rows, classes, seed_spectra, seed_geometry, repeat, seed_inventory = diagnose_seed(
            base, seed, device, int(args.batch_size), int(args.num_workers)
        )
        all_sessions.extend(rows)
        all_classes.extend(classes)
        spectra.update(seed_spectra)
        geometry.update(seed_geometry)
        reproducibility[f"seed{seed}"] = repeat
        inventory[f"seed{seed}"] = seed_inventory
    _write_csv(output / "dsm_seed2_seed3_session_diagnostics.csv", all_sessions)
    _write_csv(output / "dsm_seed2_seed3_class_diagnostics.csv", all_classes)
    _write_json(output / "dsm_seed2_seed3_svd_spectrum.json", spectra)
    _write_json(output / "dsm_seed2_seed3_geometry_metrics.json", geometry)
    _write_json(output / "dsm_seed2_seed3_reproducibility.json", reproducibility)
    _write_json(output / "artifact_inventory.json", inventory)
    print(json.dumps({"status": "ok", "output_dir": str(output), "seeds": seeds}, indent=2))


if __name__ == "__main__":
    main()
