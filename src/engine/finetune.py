"""Exemplar-free Finetune baseline for CIFAR-100-LT."""

from __future__ import annotations

import logging
import platform
import shlex
import socket
import subprocess
import sys
import time
import csv
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader

from src.datasets import build_protocol
from src.methods.gpa import GPAPlugin, compute_class_prototypes
from src.models.resnet_cifar import resnet32
from src.utils.io import append_jsonl, ensure_dir, write_acc_matrix_csv, write_json, write_summary_csv, write_yaml
from src.utils.metrics import (
    build_label_map,
    evaluate,
    forgetting,
    frequency_groups,
    group_accuracy,
    normalize_histogram,
    prediction_task_histogram,
)
from src.utils.seed import set_seed
from src.utils.tracking import ExperimentTracker


def configure_logging(output_dir: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("ltcil")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if output_dir is not None:
        file_handler = logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def _args_to_config(args) -> Dict[str, object]:
    return {key: value for key, value in sorted(vars(args).items())}


def _resolve_output_dir(args) -> Path:
    if args.output:
        return ensure_dir(args.output)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return ensure_dir(Path(args.output_root) / f"{args.method}_{args.order}_seed{args.seed}_{timestamp}")


def _resolve_optional_dir(path: str | None) -> Path | None:
    if not path:
        return None
    return ensure_dir(path)


def _resolve_checkpoint_dir(args, output_dir: Path) -> Path:
    if args.ckpt_dir:
        return ensure_dir(args.ckpt_dir)
    return ensure_dir(output_dir / "checkpoints")


def _method_metadata(args) -> Dict[str, object]:
    return {
        "ablation_stage": getattr(args, "ablation_stage", args.method),
        "uses_gpa": bool(getattr(args, "uses_gpa", args.method != "finetune")),
        "base_only": args.base_only,
        "base_classes": args.base_classes,
        "incremental_steps": args.incremental_steps,
        "debug_num_classes": args.debug_num_classes,
        "max_train_per_class": args.max_train_per_class,
        "count_assignment": args.count_assignment,
        "class_order_source": args.class_order_source,
        "class_order_file": args.class_order_file,
        "cifar_normalization": args.cifar_normalization,
        "medical_fold": getattr(args, "medical_fold", None),
        "medical_imb_factor": getattr(args, "medical_imb_factor", None),
        "medical_split_root": getattr(args, "medical_split_root", None),
        "medical_label_column": getattr(args, "medical_label_column", None),
        "medical_eval_split": getattr(args, "medical_eval_split", None),
        "medical_image_size": getattr(args, "medical_image_size", None),
        "medical_normalization": getattr(args, "medical_normalization", None),
        "max_phases": args.max_phases,
        "scheduler": args.scheduler,
        "milestones": args.milestones,
        "gamma": args.gamma,
        "base_epochs": args.base_epochs,
        "inc_epochs": args.inc_epochs,
        "num_exemplars_per_class": args.num_exemplars_per_class,
        "exemplar_selection": args.exemplar_selection,
        "freeze_backbone_after_base": args.freeze_backbone_after_base,
        "freeze_bn_after_base": args.freeze_bn_after_base,
        "train_new_head_only_after_base": args.train_new_head_only_after_base,
        "lambda_gpa": args.lambda_gpa,
        "gpa_anchor_start_phase": args.gpa_anchor_start_phase,
        "gpa_init_bias": args.gpa_init_bias,
        "gpa_freeze_old_head": args.gpa_freeze_old_head,
        "gpa_restore_old_head_after_step": args.gpa_restore_old_head_after_step,
        "gpa_anchor_reduction": args.gpa_anchor_reduction,
        "gpa_bias_mode": args.gpa_bias_mode,
        "diagnose_task_hist": args.diagnose_task_hist,
        "use_ltconcm": args.use_ltconcm,
        "ltconcm_calibrate_prototypes": args.ltconcm_calibrate_prototypes,
        "ltconcm_anchor_target": args.ltconcm_anchor_target,
        "ltconcm_tail_anchor": args.ltconcm_tail_anchor,
        "ltconcm_anchor_gamma": args.ltconcm_anchor_gamma,
        "ltconcm_anchor_max_weight": args.ltconcm_anchor_max_weight,
        "ltconcm_memory_topk": args.ltconcm_memory_topk,
        "ltconcm_alpha_a": args.ltconcm_alpha_a,
        "ltconcm_alpha_b": args.ltconcm_alpha_b,
        "ltconcm_alpha_min": args.ltconcm_alpha_min,
        "ltconcm_alpha_max": args.ltconcm_alpha_max,
        "ltconcm_use_tdsm": args.ltconcm_use_tdsm,
        "ltconcm_use_match_loss": args.ltconcm_use_match_loss,
        "ltconcm_match_lambda": args.ltconcm_match_lambda,
    }


def _task_frequency_group_counts(protocol) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for task_idx, task_classes in enumerate(protocol.tasks):
        groups = _frequency_groups(protocol, task_classes)
        summaries.append(
            {
                "task": task_idx,
                "many": len(groups["many"]),
                "medium": len(groups["medium"]),
                "few": len(groups["few"]),
                "classes": {
                    "many": groups["many"],
                    "medium": groups["medium"],
                    "few": groups["few"],
                },
            }
        )
    return summaries


def _frequency_groups(protocol, class_ids: Sequence[int]) -> Dict[str, List[int]]:
    return frequency_groups(
        protocol.class_counts,
        class_ids,
        many_threshold=int(getattr(protocol, "frequency_many_threshold", 100)),
        few_threshold=int(getattr(protocol, "frequency_few_threshold", 20)),
    )


def _per_task_train_samples(protocol) -> List[Dict[str, int]]:
    return [
        {
            "task": task_idx,
            "train_samples": int(sum(protocol.class_counts[int(class_id)] for class_id in task_classes)),
        }
        for task_idx, task_classes in enumerate(protocol.tasks)
    ]


def _phase_epochs(args, phase: int) -> int:
    if phase == 0 and args.base_epochs is not None:
        return int(args.base_epochs)
    if phase > 0 and args.inc_epochs is not None:
        return int(args.inc_epochs)
    return int(args.epochs)


def _collect_phase_exemplars(
    protocol,
    task_classes: Sequence[int],
    num_exemplars_per_class: int,
    seed: int,
    phase: int,
) -> Dict[int, List[int]]:
    if num_exemplars_per_class <= 0:
        return {}
    selected: Dict[int, List[int]] = {}
    for class_id in task_classes:
        class_id = int(class_id)
        indices = list(protocol.train_indices_by_class[class_id])
        generator = torch.Generator()
        generator.manual_seed(int(seed) + 1000003 * int(phase) + int(class_id))
        permutation = torch.randperm(len(indices), generator=generator).tolist()
        keep = min(int(num_exemplars_per_class), len(indices))
        selected[class_id] = [int(indices[item]) for item in permutation[:keep]]
    return selected


def _exemplar_task_summary(
    tasks: Sequence[Sequence[int]],
    exemplar_indices_by_class: Mapping[int, Sequence[int]],
) -> List[Dict[str, int]]:
    summary: List[Dict[str, int]] = []
    for task_idx, task_classes in enumerate(tasks):
        count = sum(len(exemplar_indices_by_class.get(int(class_id), [])) for class_id in task_classes)
        classes = sum(1 for class_id in task_classes if exemplar_indices_by_class.get(int(class_id)))
        summary.append({"task": int(task_idx), "classes_with_exemplars": int(classes), "exemplars": int(count)})
    return summary


def _git_metadata(root: Path) -> Dict[str, object]:
    def run_git(command: List[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *command],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except FileNotFoundError:
            return subprocess.CompletedProcess(command, returncode=127, stdout="", stderr="")

    commit = run_git(["rev-parse", "--short", "HEAD"])
    if commit.returncode != 0:
        return {
            "git_commit_hash": None,
            "git_is_dirty": None,
            "git_status_short": None,
        }

    status = run_git(["status", "--short"])
    status_text = status.stdout.strip() if status.returncode == 0 else None
    return {
        "git_commit_hash": commit.stdout.strip(),
        "git_is_dirty": bool(status_text) if status_text is not None else None,
        "git_status_short": status_text,
    }


def _task_split_metadata(args, protocol) -> Dict[str, object]:
    task_sizes = [len(task) for task in protocol.tasks]
    return {
        "base_classes": args.base_classes,
        "incremental_steps": args.incremental_steps,
        "debug_num_classes": args.debug_num_classes,
        "num_selected_classes": len(protocol.selected_classes),
        "num_phases": len(protocol.tasks),
        "task_sizes": task_sizes,
        "incremental_classes_per_step": task_sizes[1:],
        "tasks": protocol.tasks,
        "task_classes": protocol.tasks,
        "task_frequency_group_counts": _task_frequency_group_counts(protocol),
        "per_task_many_medium_few_count": _task_frequency_group_counts(protocol),
        "per_task_train_samples": _per_task_train_samples(protocol),
        "max_train_per_class": args.max_train_per_class,
        "max_train_per_class_effective": protocol.max_train_per_class,
        "protocol_source": getattr(protocol, "protocol_source", None),
        "class_names": getattr(protocol, "class_names", None),
        "split_metadata": getattr(protocol, "split_metadata", None),
        "frequency_many_threshold": getattr(protocol, "frequency_many_threshold", 100),
        "frequency_few_threshold": getattr(protocol, "frequency_few_threshold", 20),
    }


def _runtime_metadata(args, output_dir: Path, checkpoints_dir: Path, cache_dir: Path | None) -> Dict[str, object]:
    gpu_names: List[str] = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_gpu_names": gpu_names,
        "resolved_cache_dir": str(cache_dir) if cache_dir is not None else None,
        "resolved_output_dir": str(output_dir),
        "resolved_checkpoint_dir": str(checkpoints_dir),
        "swanlab_enabled": bool(args.use_swanlab),
        "swanlab_project": args.swanlab_project,
        "swanlab_run_name": args.swanlab_run_name,
        "swanlab_mode": args.swanlab_mode,
    }


def _run_metadata(args, protocol, output_dir: Path, checkpoints_dir: Path, cache_dir: Path | None) -> Dict[str, object]:
    command_argv = [sys.executable, *sys.argv]
    return {
        "dataset": args.dataset,
        "method_name": args.method,
        "ablation_stage": getattr(args, "ablation_stage", args.method),
        "seed": args.seed,
        "rho": args.rho,
        "imbalance_ratio": args.rho,
        "order": args.order,
        "count_assignment": args.count_assignment,
        "class_order_source": args.class_order_source,
        "class_order_file": args.class_order_file,
        "cifar_normalization": args.cifar_normalization,
        "medical_fold": getattr(args, "medical_fold", None),
        "medical_imb_factor": getattr(args, "medical_imb_factor", None),
        "medical_split_root": getattr(args, "medical_split_root", None),
        "medical_label_column": getattr(args, "medical_label_column", None),
        "medical_eval_split": getattr(args, "medical_eval_split", None),
        "medical_image_size": getattr(args, "medical_image_size", None),
        "medical_normalization": getattr(args, "medical_normalization", None),
        "protocol_source": getattr(protocol, "protocol_source", None),
        "normalization_mean": protocol.normalization_mean,
        "normalization_std": protocol.normalization_std,
        "scheduler": args.scheduler,
        "milestones": args.milestones,
        "gamma": args.gamma,
        "base_epochs": args.base_epochs,
        "inc_epochs": args.inc_epochs,
        "max_phases": args.max_phases,
        "base_only": args.base_only,
        "num_exemplars_per_class": args.num_exemplars_per_class,
        "exemplar_selection": args.exemplar_selection,
        "task_split": _task_split_metadata(args, protocol),
        "gpa_flags": {
            "lambda_gpa": args.lambda_gpa,
            "gpa_anchor_start_phase": args.gpa_anchor_start_phase,
            "gpa_init_bias": args.gpa_init_bias,
            "gpa_freeze_old_head": args.gpa_freeze_old_head,
            "gpa_restore_old_head_after_step": args.gpa_restore_old_head_after_step,
            "gpa_anchor_reduction": args.gpa_anchor_reduction,
            "gpa_bias_mode": args.gpa_bias_mode,
        },
        "taconcm_flags": {
            "use_ltconcm": args.use_ltconcm,
            "ltconcm_calibrate_prototypes": args.ltconcm_calibrate_prototypes,
            "ltconcm_anchor_target": args.ltconcm_anchor_target,
            "ltconcm_tail_anchor": args.ltconcm_tail_anchor,
            "ltconcm_anchor_gamma": args.ltconcm_anchor_gamma,
            "ltconcm_anchor_max_weight": args.ltconcm_anchor_max_weight,
            "ltconcm_memory_topk": args.ltconcm_memory_topk,
            "ltconcm_alpha_a": args.ltconcm_alpha_a,
            "ltconcm_alpha_b": args.ltconcm_alpha_b,
            "ltconcm_alpha_min": args.ltconcm_alpha_min,
            "ltconcm_alpha_max": args.ltconcm_alpha_max,
            "ltconcm_use_tdsm": args.ltconcm_use_tdsm,
            "ltconcm_use_match_loss": args.ltconcm_use_match_loss,
            "ltconcm_match_lambda": args.ltconcm_match_lambda,
        },
        "reproduction_command": shlex.join(command_argv),
        "argv": sys.argv,
        "python_executable": sys.executable,
        "resolved_data_root": str(Path(args.data_root).resolve()),
        "resolved_output_dir": str(output_dir),
        "resolved_checkpoint_dir": str(checkpoints_dir),
        "resolved_cache_dir": str(cache_dir) if cache_dir is not None else None,
        **_runtime_metadata(args, output_dir, checkpoints_dir, cache_dir),
        **_git_metadata(Path.cwd()),
    }


def _make_loader(dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        torch.manual_seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        worker_init_fn=seed_worker if num_workers else None,
    )


def _max_abs_delta(current: torch.Tensor | None, reference: torch.Tensor | None) -> float | None:
    if current is None or reference is None or current.numel() == 0 or reference.numel() == 0:
        return None
    current_cpu = current.detach().cpu()
    reference_cpu = reference.detach().cpu()
    return float((current_cpu - reference_cpu).abs().max().item())


def _restore_old_head(
    model: nn.Module,
    old_dim: int,
    old_weight_snapshot: torch.Tensor | None,
    old_bias_snapshot: torch.Tensor | None,
) -> None:
    if old_dim <= 0 or model.classifier is None or old_weight_snapshot is None:
        return
    with torch.no_grad():
        model.classifier.weight[:old_dim].copy_(old_weight_snapshot.to(model.classifier.weight.device))
        if (
            model.classifier.bias is not None
            and old_bias_snapshot is not None
        ):
            model.classifier.bias[:old_dim].copy_(old_bias_snapshot.to(model.classifier.bias.device))


def _parse_milestones(value: str | Sequence[int] | None) -> List[int]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def _build_scheduler(args, optimizer: torch.optim.Optimizer, epochs: int):
    if args.scheduler == "constant":
        return None
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))
    if args.scheduler == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=_parse_milestones(args.milestones),
            gamma=float(args.gamma),
        )
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def _set_batchnorm_eval(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def _apply_phase_freeze_policy(model: nn.Module, args, phase: int) -> Dict[str, bool]:
    freeze_backbone = bool(phase > 0 and (args.freeze_backbone_after_base or args.train_new_head_only_after_base))
    freeze_bn = bool(phase > 0 and args.freeze_bn_after_base)
    for name, parameter in model.named_parameters():
        is_classifier = name.startswith("classifier.")
        parameter.requires_grad = not (freeze_backbone and not is_classifier)
    if freeze_bn:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                if module.weight is not None:
                    module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False
    return {
        "freeze_backbone_active": freeze_backbone,
        "freeze_bn_active": freeze_bn,
        "train_new_head_only_active": bool(phase > 0 and args.train_new_head_only_after_base),
    }


def _snapshot_backbone_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if not name.startswith("classifier.")
    }


def _snapshot_bn_buffers(model: nn.Module) -> Dict[str, torch.Tensor]:
    snapshots: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if module.running_mean is not None:
                snapshots[f"{name}.running_mean"] = module.running_mean.detach().cpu().clone()
            if module.running_var is not None:
                snapshots[f"{name}.running_var"] = module.running_var.detach().cpu().clone()
    return snapshots


def _max_named_delta(current: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]) -> float | None:
    values: List[float] = []
    for name, ref in reference.items():
        cur = current.get(name)
        if cur is None or cur.numel() == 0:
            continue
        values.append(float((cur.detach().cpu() - ref).abs().max().item()))
    return max(values) if values else None


def _bn_delta_diagnostics(model: nn.Module, reference: Mapping[str, torch.Tensor]) -> Dict[str, float | None]:
    current = _snapshot_bn_buffers(model)
    mean_reference = {name: value for name, value in reference.items() if name.endswith("running_mean")}
    var_reference = {name: value for name, value in reference.items() if name.endswith("running_var")}
    return {
        "bn_running_mean_max_delta_after_phase": _max_named_delta(current, mean_reference),
        "bn_running_var_max_delta_after_phase": _max_named_delta(current, var_reference),
    }


@torch.no_grad()
def _collect_features(model: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor | None:
    model.eval()
    chunks: List[torch.Tensor] = []
    for images, _ in loader:
        images = images.to(device)
        chunks.append(model.extract_features(images).detach().cpu())
    if not chunks:
        return None
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def _feature_drift_mean_cosine(
    model: nn.Module,
    loader: DataLoader | None,
    device: torch.device,
    before_features: torch.Tensor | None,
) -> float | None:
    if loader is None or before_features is None or before_features.numel() == 0:
        return None
    after_features = _collect_features(model, loader, device)
    if after_features is None or after_features.shape != before_features.shape:
        return None
    cosine = F.cosine_similarity(before_features, after_features, dim=1)
    return float(cosine.mean().item())


@torch.no_grad()
def _old_eval_logit_margin(
    model: nn.Module,
    loader: DataLoader | None,
    device: torch.device,
    old_dim: int,
    new_dim: int,
) -> float | None:
    if loader is None or old_dim <= 0 or new_dim <= old_dim or model.classifier is None:
        return None
    model.eval()
    values: List[torch.Tensor] = []
    for images, _ in loader:
        images = images.to(device)
        logits = model(images)
        max_old = logits[:, :old_dim].max(dim=1).values
        max_latest = logits[:, old_dim:new_dim].max(dim=1).values
        values.append((max_old - max_latest).detach().cpu())
    if not values:
        return None
    return float(torch.cat(values).mean().item())


def _tensor_stats(values: torch.Tensor | None, prefix: str) -> Dict[str, float | None]:
    if values is None or values.numel() == 0:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
        }
    values = values.detach().float().cpu()
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        f"{prefix}_min": float(values.min().item()),
        f"{prefix}_max": float(values.max().item()),
    }


def _prediction_task_hist_from_heads(
    pred_heads: Sequence[int],
    head_to_class: Mapping[int, int],
    tasks: Sequence[Sequence[int]],
) -> Dict[str, int]:
    class_to_task = {
        int(class_id): task_idx
        for task_idx, task_classes in enumerate(tasks)
        for class_id in task_classes
    }
    hist = {f"task_{task_idx}": 0 for task_idx in range(len(tasks))}
    for pred_head in pred_heads:
        pred_class = head_to_class.get(int(pred_head))
        if pred_class is None:
            continue
        task_idx = class_to_task.get(int(pred_class))
        if task_idx is None:
            continue
        hist[f"task_{task_idx}"] += 1
    return hist


@torch.no_grad()
def _classifier_logit_stats(
    model: nn.Module,
    old_dim: int,
    new_dim: int,
    class_to_head_index: Mapping[int, int],
    task_classes: Sequence[int],
    class_counts: Mapping[int, int],
    method_diagnostics: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    if model.classifier is None:
        return {}
    classifier = model.classifier
    weight = classifier.weight.detach()
    bias = classifier.bias.detach() if classifier.bias is not None else None
    old_weights = weight[:old_dim] if old_dim > 0 else None
    new_weights = weight[old_dim:new_dim] if new_dim > old_dim else None
    stats: Dict[str, object] = {}
    stats.update(
        _tensor_stats(
            torch.linalg.vector_norm(old_weights, dim=1) if old_weights is not None else None,
            "old_weight_norm",
        )
    )
    stats.update(
        _tensor_stats(
            torch.linalg.vector_norm(new_weights, dim=1) if new_weights is not None else None,
            "new_weight_norm",
        )
    )
    stats.update(_tensor_stats(bias[:old_dim] if bias is not None and old_dim > 0 else None, "old_bias"))
    stats.update(_tensor_stats(bias[old_dim:new_dim] if bias is not None and new_dim > old_dim else None, "new_bias"))
    stats.update(
        _tensor_stats(
            bias[old_dim:new_dim] if bias is not None and new_dim > old_dim else None,
            "latest_task_bias",
        )
    )
    if bias is not None:
        stats["new_bias_values_per_class"] = {
            int(class_id): float(bias[int(class_to_head_index[int(class_id)])].detach().cpu().item())
            for class_id in task_classes
            if int(class_id) in class_to_head_index
        }
    else:
        stats["new_bias_values_per_class"] = {}
    stats["new_bias_init_values_per_class"] = (
        dict(method_diagnostics.get("gpa_new_bias_init_values_per_class", {}))
        if method_diagnostics is not None
        else {}
    )
    stats["class_counts_for_new_classes"] = {
        int(class_id): int(class_counts[int(class_id)])
        for class_id in task_classes
    }
    stats["gpa_bias_n_ref"] = (
        method_diagnostics.get("gpa_bias_n_ref") if method_diagnostics is not None else None
    )
    stats["gpa_bias_formula"] = (
        method_diagnostics.get("gpa_bias_formula") if method_diagnostics is not None else None
    )
    return stats


@torch.no_grad()
def _logit_forensic_snapshot(
    model: nn.Module,
    loader: DataLoader | None,
    class_to_head_index: Mapping[int, int],
    tasks: Sequence[Sequence[int]],
    device: torch.device,
    old_dim: int,
    new_dim: int,
    phase: int,
    stage: str,
    task_classes: Sequence[int],
    class_counts: Mapping[int, int],
    method_diagnostics: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    if loader is None or model.classifier is None or old_dim <= 0:
        return {"phase": phase, "stage": stage}
    model.eval()
    label_map = build_label_map(class_to_head_index, device)
    head_to_class = {int(head_index): int(class_id) for class_id, head_index in class_to_head_index.items()}
    classifier = model.classifier
    weight = classifier.weight.detach()
    bias = classifier.bias.detach() if classifier.bias is not None else None

    max_old_logits: List[torch.Tensor] = []
    max_new_logits: List[torch.Tensor] = []
    max_latest_logits: List[torch.Tensor] = []
    old_new_margins: List[torch.Tensor] = []
    old_latest_margins: List[torch.Tensor] = []
    dot_old_max: List[torch.Tensor] = []
    dot_new_max: List[torch.Tensor] = []
    bias_old_argmax: List[torch.Tensor] = []
    bias_new_argmax: List[torch.Tensor] = []
    dot_margin_values: List[torch.Tensor] = []
    bias_margin_values: List[torch.Tensor] = []
    pred_heads_all: List[int] = []
    correct = 0
    total = 0

    has_new = new_dim > old_dim
    for images, labels_global in loader:
        images = images.to(device)
        labels_global = labels_global.to(device)
        labels_head = label_map[labels_global]
        if torch.any(labels_head < 0):
            bad = labels_global[labels_head < 0].detach().cpu().tolist()
            raise RuntimeError(f"Forensic eval found labels outside seen class map: {bad}")
        features = model.extract_features(images)
        logits = classifier(features)
        dots = features @ weight[:new_dim].T
        predictions = logits[:, :new_dim].argmax(dim=1)
        pred_heads_all.extend(int(item) for item in predictions.detach().cpu().tolist())
        correct += int(predictions.eq(labels_head).sum().item())
        total += int(labels_head.numel())

        old_logits = logits[:, :old_dim]
        old_dot = dots[:, :old_dim]
        old_arg = old_logits.argmax(dim=1)
        old_dot_arg = old_dot.argmax(dim=1)
        max_old_logit = old_logits.gather(1, old_arg[:, None]).squeeze(1)
        max_old_dot = old_dot.gather(1, old_dot_arg[:, None]).squeeze(1)
        max_old_logits.append(max_old_logit.detach().cpu())
        dot_old_max.append(max_old_dot.detach().cpu())
        if bias is not None:
            old_bias = bias[old_arg]
        else:
            old_bias = torch.zeros_like(max_old_logit)
        bias_old_argmax.append(old_bias.detach().cpu())

        if has_new:
            new_logits = logits[:, old_dim:new_dim]
            new_dot = dots[:, old_dim:new_dim]
            new_arg_rel = new_logits.argmax(dim=1)
            new_dot_arg_rel = new_dot.argmax(dim=1)
            max_new_logit = new_logits.gather(1, new_arg_rel[:, None]).squeeze(1)
            max_new_dot = new_dot.gather(1, new_dot_arg_rel[:, None]).squeeze(1)
            if bias is not None:
                new_bias = bias[old_dim + new_arg_rel]
            else:
                new_bias = torch.zeros_like(max_new_logit)
            max_new_logits.append(max_new_logit.detach().cpu())
            max_latest_logits.append(max_new_logit.detach().cpu())
            old_new_margins.append((max_old_logit - max_new_logit).detach().cpu())
            old_latest_margins.append((max_old_logit - max_new_logit).detach().cpu())
            dot_new_max.append(max_new_dot.detach().cpu())
            bias_new_argmax.append(new_bias.detach().cpu())
            dot_margin_values.append((max_old_dot - max_new_dot).detach().cpu())
            bias_margin_values.append((old_bias - new_bias).detach().cpu())

    def mean_or_none(values: List[torch.Tensor]) -> float | None:
        if not values:
            return None
        return float(torch.cat(values).mean().item())

    pred_hist = _prediction_task_hist_from_heads(pred_heads_all, head_to_class, tasks)
    result: Dict[str, object] = {
        "phase": phase,
        "stage": stage,
        "old_task_acc": 100.0 * correct / total if total else None,
        "max_old_logit_mean": mean_or_none(max_old_logits),
        "max_new_logit_mean": mean_or_none(max_new_logits),
        "max_latest_task_logit_mean": mean_or_none(max_latest_logits),
        "old_vs_new_margin_mean": mean_or_none(old_new_margins),
        "old_vs_latest_margin_mean": mean_or_none(old_latest_margins),
        "predicted_task_hist": pred_hist,
        "predicted_task_hist_frac": normalize_histogram(pred_hist, total),
        "dot_old_max_mean": mean_or_none(dot_old_max),
        "dot_new_max_mean": mean_or_none(dot_new_max),
        "bias_old_at_argmax_mean": mean_or_none(bias_old_argmax),
        "bias_new_at_argmax_mean": mean_or_none(bias_new_argmax),
        "old_vs_new_dot_margin_mean": mean_or_none(dot_margin_values),
        "old_vs_new_bias_margin_mean": mean_or_none(bias_margin_values),
        "old_eval_total": total,
    }
    result.update(
        _classifier_logit_stats(
            model=model,
            old_dim=old_dim,
            new_dim=new_dim,
            class_to_head_index=class_to_head_index,
            task_classes=task_classes,
            class_counts=class_counts,
            method_diagnostics=method_diagnostics,
        )
    )
    return result


@torch.no_grad()
def _eval_calibration_diagnostics(
    model: nn.Module,
    loader: DataLoader,
    class_to_head_index: Mapping[int, int],
    tasks: Sequence[Sequence[int]],
    device: torch.device,
) -> Dict[str, float | None]:
    if model.classifier is None:
        return {
            "raw_linear_acc": None,
            "no_bias_acc": None,
            "normalized_weight_acc": None,
            "cosine_eval_acc": None,
            "old_new_bias_corrected_acc": None,
        }
    model.eval()
    classifier = model.classifier
    weight = classifier.weight.detach()
    bias = classifier.bias.detach() if classifier.bias is not None else None
    label_map = build_label_map(class_to_head_index, device)
    head_to_task = {}
    for task_idx, task_classes in enumerate(tasks):
        for class_id in task_classes:
            class_id = int(class_id)
            if class_id in class_to_head_index:
                head_to_task[int(class_to_head_index[class_id])] = task_idx
    task_bias_means: Dict[int, float] = {}
    if bias is not None:
        for task_idx in range(len(tasks)):
            indices = [head for head, idx in head_to_task.items() if idx == task_idx]
            if indices:
                index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
                task_bias_means[task_idx] = float(bias.index_select(0, index_tensor).mean().detach().cpu().item())
    bias_correction = torch.zeros(weight.shape[0], dtype=weight.dtype, device=device)
    for head_idx, task_idx in head_to_task.items():
        bias_correction[head_idx] = float(task_bias_means.get(task_idx, 0.0))

    correct = {
        "raw_linear_acc": 0,
        "no_bias_acc": 0,
        "normalized_weight_acc": 0,
        "cosine_eval_acc": 0,
        "old_new_bias_corrected_acc": 0,
    }
    total = 0
    weight_norm = F.normalize(weight, dim=1)
    for images, labels_global in loader:
        images = images.to(device)
        labels_global = labels_global.to(device)
        labels_head = label_map[labels_global]
        features = model.extract_features(images)
        raw_logits = classifier(features)
        dot_logits = features @ weight.T
        normalized_weight_logits = features @ weight_norm.T
        cosine_logits = F.normalize(features, dim=1) @ weight_norm.T
        corrected_logits = raw_logits - bias_correction
        variants = {
            "raw_linear_acc": raw_logits,
            "no_bias_acc": dot_logits,
            "normalized_weight_acc": normalized_weight_logits,
            "cosine_eval_acc": cosine_logits,
            "old_new_bias_corrected_acc": corrected_logits,
        }
        for key, logits in variants.items():
            correct[key] += int(logits.argmax(dim=1).eq(labels_head).sum().item())
        total += int(labels_head.numel())
    return {key: 100.0 * value / total if total else None for key, value in correct.items()}


def _group_summary(class_counts: Mapping[int, int], groups: Mapping[str, Sequence[int]]) -> Dict[str, Dict[str, int | None]]:
    summary: Dict[str, Dict[str, int | None]] = {}
    for name, class_ids in groups.items():
        counts = [int(class_counts[int(class_id)]) for class_id in class_ids]
        summary[name] = {
            "num_classes": len(class_ids),
            "total_images": int(sum(counts)),
            "min_count": min(counts) if counts else None,
            "max_count": max(counts) if counts else None,
        }
    return summary


def _write_base_sanity_artifacts(
    output_dir: Path,
    per_class_stats: Mapping[int, Mapping[str, int]],
    per_class_acc: Mapping[int, float],
    groups: Mapping[str, Sequence[int]],
    class_counts: Mapping[int, int],
) -> None:
    group_for_class = {
        int(class_id): group_name
        for group_name, class_ids in groups.items()
        for class_id in class_ids
    }
    with (output_dir / "base_per_class_acc.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_id", "class_count", "group", "accuracy", "correct", "total", "predicted"],
        )
        writer.writeheader()
        for class_id in sorted(per_class_acc):
            stats = per_class_stats.get(int(class_id), {})
            writer.writerow(
                {
                    "class_id": int(class_id),
                    "class_count": int(class_counts[int(class_id)]),
                    "group": group_for_class.get(int(class_id), ""),
                    "accuracy": float(per_class_acc[int(class_id)]),
                    "correct": int(stats.get("correct", 0)),
                    "total": int(stats.get("total", 0)),
                    "predicted": int(stats.get("predicted", 0)),
                }
            )
    with (output_dir / "base_group_acc.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["group", "num_classes", "total_train_images", "accuracy"],
        )
        writer.writeheader()
        for group_name in ["many", "medium", "few"]:
            class_ids = list(groups.get(group_name, []))
            writer.writerow(
                {
                    "group": group_name,
                    "num_classes": len(class_ids),
                    "total_train_images": sum(int(class_counts[int(class_id)]) for class_id in class_ids),
                    "accuracy": group_accuracy(per_class_stats, class_ids),
                }
            )


def _classifier_task_stats(
    model: nn.Module,
    tasks: Sequence[Sequence[int]],
    class_to_head_index: Mapping[int, int],
    upto_phase: int,
) -> Dict[str, Dict[str, float | None]]:
    if model.classifier is None:
        return {"classifier_weight_norm_by_task": {}, "classifier_bias_mean_by_task": {}}
    weight_norm_by_task: Dict[str, float | None] = {}
    bias_mean_by_task: Dict[str, float | None] = {}
    with torch.no_grad():
        for task_idx in range(upto_phase + 1):
            indices = [
                int(class_to_head_index[int(class_id)])
                for class_id in tasks[task_idx]
                if int(class_id) in class_to_head_index
            ]
            key = f"task_{task_idx}"
            if not indices:
                weight_norm_by_task[key] = None
                bias_mean_by_task[key] = None
                continue
            index_tensor = torch.tensor(indices, dtype=torch.long, device=model.classifier.weight.device)
            weights = model.classifier.weight.index_select(0, index_tensor)
            weight_norm_by_task[key] = float(torch.linalg.vector_norm(weights, dim=1).mean().detach().cpu().item())
            if model.classifier.bias is None:
                bias_mean_by_task[key] = None
            else:
                biases = model.classifier.bias.index_select(0, index_tensor)
                bias_mean_by_task[key] = float(biases.mean().detach().cpu().item())
    return {
        "classifier_weight_norm_by_task": weight_norm_by_task,
        "classifier_bias_mean_by_task": bias_mean_by_task,
    }


def _new_weight_cosine_to_prototypes(
    model: nn.Module,
    task_classes: Sequence[int],
    class_to_head_index: Mapping[int, int],
    prototype_snapshot: Mapping[str, object],
) -> Dict[str, float | None]:
    if model.classifier is None or not prototype_snapshot:
        return {
            "new_weight_cos_to_frozen_proto_mean": None,
            "new_weight_cos_to_frozen_proto_min": None,
        }
    proto_init = prototype_snapshot.get("proto_init")
    if not isinstance(proto_init, torch.Tensor) or proto_init.numel() == 0:
        return {
            "new_weight_cos_to_frozen_proto_mean": None,
            "new_weight_cos_to_frozen_proto_min": None,
        }
    cosines: List[float] = []
    with torch.no_grad():
        proto_init = proto_init.to(device=model.classifier.weight.device, dtype=model.classifier.weight.dtype)
        for row, class_id in enumerate(task_classes):
            head_index = int(class_to_head_index[int(class_id)])
            weight = torch.nn.functional.normalize(model.classifier.weight[head_index], dim=0)
            proto = torch.nn.functional.normalize(proto_init[row], dim=0)
            cosines.append(float(torch.sum(weight * proto).detach().cpu().item()))
    if not cosines:
        return {
            "new_weight_cos_to_frozen_proto_mean": None,
            "new_weight_cos_to_frozen_proto_min": None,
        }
    return {
        "new_weight_cos_to_frozen_proto_mean": float(sum(cosines) / len(cosines)),
        "new_weight_cos_to_frozen_proto_min": float(min(cosines)),
    }


def _prototype_norm_diagnostics(estimate) -> Dict[str, float]:
    return {
        "prototype_norm_mean": float(estimate.raw_norms.mean().detach().cpu().item()),
        "prototype_norm_min": float(estimate.raw_norms.min().detach().cpu().item()),
        "prototype_norm_max": float(estimate.raw_norms.max().detach().cpu().item()),
    }


def _train_one_phase(
    model: nn.Module,
    loader: DataLoader,
    class_to_head_index: Mapping[int, int],
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int,
    logger: logging.Logger,
    method_plugin: GPAPlugin | None = None,
    old_dim: int = 0,
    old_weight_snapshot: torch.Tensor | None = None,
    old_bias_snapshot: torch.Tensor | None = None,
    freeze_old_head: bool = False,
    restore_old_head_after_step: bool = False,
    freeze_bn: bool = False,
) -> Dict[str, object]:
    criterion = nn.CrossEntropyLoss()
    label_map = build_label_map(class_to_head_index, device)
    model.train()
    if freeze_bn:
        _set_batchnorm_eval(model)
    totals: Dict[str, float] = {
        "train_loss": 0.0,
        "train_ce_loss": 0.0,
        "gpa_anchor_loss": 0.0,
        "ltconcm_match_loss": 0.0,
        "method_extra_loss": 0.0,
    }
    total_seen = 0
    feature_norm_sum = 0.0
    feature_norm_sq_sum = 0.0
    feature_norm_count = 0
    train_acc_last_epoch = 0.0
    train_loss_last_epoch = 0.0
    train_ce_loss_last_epoch = 0.0
    lr_curve: List[Dict[str, float]] = []

    for epoch in range(epochs):
        if freeze_bn:
            _set_batchnorm_eval(model)
        lr_curve.append({"epoch": float(epoch + 1), "lr": float(optimizer.param_groups[0]["lr"])})
        epoch_totals = {key: 0.0 for key in totals}
        epoch_seen = 0
        epoch_correct = 0
        for images, labels_global in loader:
            images = images.to(device)
            labels_global = labels_global.to(device)
            labels_head = label_map[labels_global]
            if torch.any(labels_head < 0):
                bad = labels_global[labels_head < 0].detach().cpu().tolist()
                raise RuntimeError(f"Current batch has labels outside the seen class map: {bad}")

            optimizer.zero_grad(set_to_none=True)
            if model.classifier is None:
                logits = model(images)
                features = None
            else:
                features = model.extract_features(images)
                logits = model.classifier(features)
                feature_norms = torch.linalg.vector_norm(features.detach(), dim=1)
                feature_norm_sum += float(feature_norms.sum().cpu().item())
                feature_norm_sq_sum += float(feature_norms.pow(2).sum().cpu().item())
                feature_norm_count += int(feature_norms.numel())
            if method_plugin is None:
                method_losses: Dict[str, torch.Tensor] = {}
            else:
                if features is None or model.classifier is None:
                    raise RuntimeError("Classifier has not been initialized")
                method_losses = method_plugin.loss(
                    classifier=model.classifier,
                    features=features,
                    labels_global=labels_global,
                    class_to_head_index=class_to_head_index,
                )

            ce_loss = criterion(logits, labels_head)
            loss = ce_loss + method_losses.get(
                "method_extra_loss",
                torch.zeros((), dtype=ce_loss.dtype, device=ce_loss.device),
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered: {float(loss.detach().cpu())}")
            loss.backward()
            if freeze_old_head and old_dim > 0 and model.classifier is not None:
                weight_grad = model.classifier.weight.grad
                if weight_grad is not None:
                    weight_grad[:old_dim].zero_()
                if model.classifier.bias is not None and model.classifier.bias.grad is not None:
                    model.classifier.bias.grad[:old_dim].zero_()
            optimizer.step()
            if restore_old_head_after_step:
                _restore_old_head(model, old_dim, old_weight_snapshot, old_bias_snapshot)

            batch_size = int(labels_head.numel())
            epoch_correct += int(logits.argmax(dim=1).eq(labels_head).sum().item())
            batch_values = {
                "train_loss": loss,
                "train_ce_loss": ce_loss,
                "gpa_anchor_loss": method_losses.get(
                    "gpa_anchor_loss",
                    torch.zeros((), dtype=loss.dtype, device=loss.device),
                ),
                "ltconcm_match_loss": method_losses.get(
                    "ltconcm_match_loss",
                    torch.zeros((), dtype=loss.dtype, device=loss.device),
                ),
                "method_extra_loss": method_losses.get(
                    "method_extra_loss",
                    torch.zeros((), dtype=loss.dtype, device=loss.device),
                ),
            }
            for key, value in batch_values.items():
                epoch_totals[key] += float(value.detach().cpu()) * batch_size
            epoch_seen += batch_size

        avg_epoch = {key: value / max(epoch_seen, 1) for key, value in epoch_totals.items()}
        train_acc_last_epoch = 100.0 * epoch_correct / max(epoch_seen, 1)
        train_loss_last_epoch = avg_epoch["train_loss"]
        train_ce_loss_last_epoch = avg_epoch["train_ce_loss"]
        if method_plugin is None:
            logger.info(
                "epoch=%d train_loss=%.6f train_acc=%.4f samples=%d",
                epoch + 1,
                avg_epoch["train_loss"],
                train_acc_last_epoch,
                epoch_seen,
            )
        else:
            logger.info(
                "epoch=%d train_loss=%.6f ce=%.6f anchor=%.6f match=%.6f train_acc=%.4f samples=%d",
                epoch + 1,
                avg_epoch["train_loss"],
                avg_epoch["train_ce_loss"],
                avg_epoch["gpa_anchor_loss"],
                avg_epoch["ltconcm_match_loss"],
                train_acc_last_epoch,
                epoch_seen,
            )
        for key, value in epoch_totals.items():
            totals[key] += value
        total_seen += epoch_seen
        if scheduler is not None:
            scheduler.step()

    result = {key: value / max(total_seen, 1) for key, value in totals.items()}
    feature_norm_mean = feature_norm_sum / max(feature_norm_count, 1)
    feature_norm_var = feature_norm_sq_sum / max(feature_norm_count, 1) - feature_norm_mean * feature_norm_mean
    result["feature_norm_mean"] = feature_norm_mean if feature_norm_count else None
    result["feature_norm_std"] = max(feature_norm_var, 0.0) ** 0.5 if feature_norm_count else None
    result["train_acc_last_epoch"] = train_acc_last_epoch
    result["train_loss_last_epoch"] = train_loss_last_epoch
    result["train_ce_loss_last_epoch"] = train_ce_loss_last_epoch
    result["lr_curve"] = lr_curve
    return result


def _write_static_artifacts(
    output_dir: Path,
    args,
    protocol,
    run_metadata: Dict[str, object],
    checkpoints_dir: Path,
    cache_dir: Path | None,
) -> None:
    config = _args_to_config(args)
    config["resolved_output_dir"] = str(output_dir)
    config["resolved_data_root"] = str(Path(args.data_root).resolve())
    config["resolved_checkpoint_dir"] = str(checkpoints_dir)
    config["resolved_cache_dir"] = str(cache_dir) if cache_dir is not None else None
    config["tasks"] = protocol.tasks
    config["task_classes"] = protocol.tasks
    config["task_frequency_group_counts"] = _task_frequency_group_counts(protocol)
    config["per_task_many_medium_few_count"] = _task_frequency_group_counts(protocol)
    config["per_task_train_samples"] = _per_task_train_samples(protocol)
    config["max_train_per_class_effective"] = protocol.max_train_per_class
    config["normalization_mean"] = protocol.normalization_mean
    config["normalization_std"] = protocol.normalization_std
    config["protocol_source"] = getattr(protocol, "protocol_source", None)
    config["class_names"] = getattr(protocol, "class_names", None)
    config["split_metadata"] = getattr(protocol, "split_metadata", None)
    config["frequency_many_threshold"] = getattr(protocol, "frequency_many_threshold", 100)
    config["frequency_few_threshold"] = getattr(protocol, "frequency_few_threshold", 20)
    config["run_metadata"] = run_metadata
    write_yaml(output_dir / "config.yaml", config)
    write_json(output_dir / "class_counts.json", protocol.class_counts)
    write_json(
        output_dir / "class_order.json",
        {
            "order": protocol.order,
            "count_assignment": protocol.count_assignment,
            "class_order_source": protocol.class_order_source,
            "class_order_file": protocol.class_order_file,
            "seed": protocol.seed,
            "class_order": protocol.class_order,
            "tasks": protocol.tasks,
            "task_frequency_group_counts": _task_frequency_group_counts(protocol),
        },
    )


def preview_dataset(args) -> None:
    protocol = build_protocol(args)
    logger = configure_logging()
    preview = protocol.preview()
    logger.info("dataset preview")
    for key, value in preview.items():
        if key in {"class_order", "class_counts"}:
            logger.info("%s=%s", key, value)
        else:
            logger.info("%s=%s", key, value)


def run_finetune(args) -> Path:
    set_seed(args.seed)
    output_dir = _resolve_output_dir(args)
    cache_dir = _resolve_optional_dir(args.cache_dir)
    checkpoints_dir = _resolve_checkpoint_dir(args, output_dir)
    logger = configure_logging(output_dir)
    device = torch.device(args.device)
    logger.info("output_dir=%s", output_dir)
    logger.info("checkpoint_dir=%s", checkpoints_dir)
    logger.info("cache_dir=%s", cache_dir)
    logger.info("device=%s", device)

    protocol = build_protocol(args)
    run_metadata = _run_metadata(args, protocol, output_dir, checkpoints_dir, cache_dir)
    _write_static_artifacts(output_dir, args, protocol, run_metadata, checkpoints_dir, cache_dir)
    swanlab_logdir = ensure_dir(cache_dir / "swanlab") if cache_dir is not None else ensure_dir(output_dir / ".swanlab")
    tracker = ExperimentTracker(
        enabled=bool(args.use_swanlab),
        project=args.swanlab_project,
        run_name=args.swanlab_run_name or f"{args.method}_seed{args.seed}_{output_dir.name}",
        mode=args.swanlab_mode,
        logdir=swanlab_logdir,
        config={**_args_to_config(args), **run_metadata},
        logger=logger,
    )

    model = resnet32().to(device)
    method_plugin = GPAPlugin.maybe_build(args)
    if method_plugin is not None:
        logger.info(
            "method_plugin=%s use_ltconcm=%s lambda_gpa=%.4f",
            args.method,
            method_plugin.use_ltconcm,
            method_plugin.lambda_gpa,
        )
    seen_classes: List[int] = []
    class_to_head_index: Dict[int, int] = {}
    if args.base_only:
        active_tasks = protocol.tasks[:1]
    elif args.max_phases is not None:
        max_phases = max(1, min(int(args.max_phases), len(protocol.tasks)))
        active_tasks = protocol.tasks[:max_phases]
    else:
        active_tasks = protocol.tasks
    task_count = len(active_tasks)
    acc_matrix: List[List[float | None]] = [[None for _ in range(task_count)] for _ in range(task_count)]
    phase_accs: List[float] = []
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    last_metrics: Dict[str, object] = {}
    base_train_acc_last_epoch: float | None = None
    base_train_loss_last_epoch: float | None = None
    base_test_acc: float | None = None
    phase1_logit_forensics: Dict[str, object] | None = None
    exemplar_indices_by_class: Dict[int, List[int]] = {}

    for phase, task_classes in enumerate(active_tasks):
        task_classes = [int(class_id) for class_id in task_classes]
        old_classes = seen_classes.copy()
        old_dim = len(seen_classes)
        logit_forensic_before_expansion: Dict[str, object] | None = None
        logit_forensic_after_init: Dict[str, object] | None = None
        logit_forensic_after_train: Dict[str, object] | None = None
        old_eval_loader_for_phase = None
        old_class_to_head_index = class_to_head_index.copy()
        if phase > 0 and old_classes:
            old_eval_loader_for_phase = _make_loader(
                protocol.test_dataset_for_classes(old_classes),
                batch_size=args.batch_size,
                shuffle=False,
                seed=args.seed,
                num_workers=args.num_workers,
            )
            logit_forensic_before_expansion = _logit_forensic_snapshot(
                model=model,
                loader=old_eval_loader_for_phase,
                class_to_head_index=old_class_to_head_index,
                tasks=protocol.tasks[:phase],
                device=device,
                old_dim=old_dim,
                new_dim=old_dim,
                phase=phase,
                stage="before_expansion",
                task_classes=task_classes,
                class_counts=protocol.class_counts,
                method_diagnostics=None,
            )
        old_weight_before_expansion = None
        old_bias_before_expansion = None
        if model.classifier is not None:
            old_weight_before_expansion = model.classifier.weight.detach().clone()
            old_bias_before_expansion = model.classifier.bias.detach().clone() if model.classifier.bias is not None else None
        seen_classes.extend(task_classes)
        for class_id in task_classes:
            class_to_head_index[int(class_id)] = len(class_to_head_index)

        previous_dim, new_dim = model.expand_classifier(len(seen_classes))
        if previous_dim != old_dim:
            raise RuntimeError(f"Classifier dimension mismatch: expected {old_dim}, got {previous_dim}")
        old_head_weight_delta_after_expansion = None
        old_head_bias_delta_after_expansion = None
        if old_weight_before_expansion is not None and model.classifier is not None:
            current_old_weight = model.classifier.weight[:old_dim].detach().cpu()
            old_head_weight_delta_after_expansion = _max_abs_delta(
                current_old_weight,
                old_weight_before_expansion,
            )
            if not torch.allclose(current_old_weight, old_weight_before_expansion.cpu()):
                raise RuntimeError("Classifier expansion changed old class weights")
            if old_bias_before_expansion is not None and model.classifier.bias is not None:
                current_old_bias = model.classifier.bias[:old_dim].detach().cpu()
                old_head_bias_delta_after_expansion = _max_abs_delta(
                    current_old_bias,
                    old_bias_before_expansion,
                )
                if not torch.allclose(current_old_bias, old_bias_before_expansion.cpu()):
                    raise RuntimeError("Classifier expansion changed old class bias")
        old_weight_snapshot = (
            model.classifier.weight[:old_dim].detach().clone()
            if model.classifier is not None and old_dim > 0
            else None
        )
        old_bias_snapshot = (
            model.classifier.bias[:old_dim].detach().clone()
            if model.classifier is not None and model.classifier.bias is not None and old_dim > 0
            else None
        )
        logger.info(
            "phase=%d new_classes=%s classifier_dim=%d->%d train_counts=%s",
            phase,
            task_classes,
            previous_dim,
            new_dim,
            {class_id: protocol.class_counts[class_id] for class_id in task_classes},
        )

        current_train_dataset = protocol.train_dataset_for_classes(task_classes)
        exemplar_train_indices: List[int] = []
        if phase > 0 and int(args.num_exemplars_per_class) > 0:
            for class_id in old_classes:
                exemplar_train_indices.extend(exemplar_indices_by_class.get(int(class_id), []))
        exemplar_train_samples_added = len(exemplar_train_indices)
        if exemplar_train_indices:
            exemplar_dataset = protocol.train_dataset_for_indices(exemplar_train_indices)
            train_dataset = ConcatDataset([current_train_dataset, exemplar_dataset])
        else:
            train_dataset = current_train_dataset
        train_loader = _make_loader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed + phase,
            num_workers=args.num_workers,
        )
        prototype_loader = _make_loader(
            protocol.prototype_dataset_for_classes(task_classes),
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed,
            num_workers=args.num_workers,
        )

        groups = _frequency_groups(protocol, seen_classes)
        method_diagnostics: Dict[str, object] = {}
        if method_plugin is not None and phase > 0:
            prototype_estimate = compute_class_prototypes(
                model=model,
                loader=prototype_loader,
                class_ids=task_classes,
                device=device,
            )
            method_diagnostics.update(
                method_plugin.prepare_incremental_phase(
                    model=model,
                    phase=phase,
                    task_classes=task_classes,
                    seen_classes=seen_classes,
                    class_to_head_index=class_to_head_index,
                    class_counts=protocol.class_counts,
                    groups=groups,
                    estimate=prototype_estimate,
                )
            )
            method_diagnostics.update(_prototype_norm_diagnostics(prototype_estimate))
            method_diagnostics.update(
                _new_weight_cosine_to_prototypes(
                    model=model,
                    task_classes=task_classes,
                    class_to_head_index=class_to_head_index,
                    prototype_snapshot=method_plugin.last_prototype_snapshot,
                )
            )
            logger.info(
                "phase=%d gpa_init prototypes=%d raw_norm_mean=%.6f raw_calib_cos=%.6f",
                phase,
                method_diagnostics.get("gpa_num_prototypes", 0),
                method_diagnostics.get("gpa_prototype_raw_norm_mean", 0.0),
                method_diagnostics.get("ltconcm_raw_calib_cosine_mean", 1.0),
            )

        if phase > 0 and old_eval_loader_for_phase is not None:
            logit_forensic_after_init = _logit_forensic_snapshot(
                model=model,
                loader=old_eval_loader_for_phase,
                class_to_head_index=class_to_head_index,
                tasks=protocol.tasks[: phase + 1],
                device=device,
                old_dim=old_dim,
                new_dim=new_dim,
                phase=phase,
                stage="after_gpa_init_before_training",
                task_classes=task_classes,
                class_counts=protocol.class_counts,
                method_diagnostics=method_diagnostics,
            )

        freeze_policy = _apply_phase_freeze_policy(model, args, phase)
        if freeze_policy["freeze_backbone_active"] or freeze_policy["freeze_bn_active"]:
            logger.info(
                "phase=%d freeze_backbone=%s freeze_bn=%s train_new_head_only=%s",
                phase,
                freeze_policy["freeze_backbone_active"],
                freeze_policy["freeze_bn_active"],
                freeze_policy["train_new_head_only_active"],
            )
        backbone_snapshot = _snapshot_backbone_parameters(model)
        bn_snapshot = _snapshot_bn_buffers(model)
        old_eval_loader_for_drift = None
        old_eval_features_before = None
        if phase > 0 and old_classes:
            old_eval_loader_for_drift = old_eval_loader_for_phase
            old_eval_features_before = _collect_features(model, old_eval_loader_for_drift, device)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        phase_epochs = _phase_epochs(args, phase)
        scheduler = _build_scheduler(args, optimizer, phase_epochs)
        gpa_anchor_active = method_plugin is not None and phase >= args.gpa_anchor_start_phase
        freeze_old_head_this_phase = bool(args.gpa_freeze_old_head and method_plugin is not None and phase > 0)
        restore_old_head_this_phase = bool(
            args.gpa_restore_old_head_after_step and method_plugin is not None and phase > 0
        )
        if freeze_old_head_this_phase:
            logger.info(
                "phase=%d gpa_freeze_old_head=True restore_old_head_after_step=%s old_dim=%d",
                phase,
                restore_old_head_this_phase,
                old_dim,
            )
        train_stats = _train_one_phase(
            model=model,
            loader=train_loader,
            class_to_head_index=class_to_head_index,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=phase_epochs,
            logger=logger,
            method_plugin=method_plugin if gpa_anchor_active else None,
            old_dim=old_dim,
            old_weight_snapshot=old_weight_snapshot,
            old_bias_snapshot=old_bias_snapshot,
            freeze_old_head=freeze_old_head_this_phase,
            restore_old_head_after_step=restore_old_head_this_phase,
            freeze_bn=freeze_policy["freeze_bn_active"],
        )
        if phase > 0 and old_eval_loader_for_phase is not None:
            logit_forensic_after_train = _logit_forensic_snapshot(
                model=model,
                loader=old_eval_loader_for_phase,
                class_to_head_index=class_to_head_index,
                tasks=protocol.tasks[: phase + 1],
                device=device,
                old_dim=old_dim,
                new_dim=new_dim,
                phase=phase,
                stage="after_phase_training",
                task_classes=task_classes,
                class_counts=protocol.class_counts,
                method_diagnostics=method_diagnostics,
            )
        drift_diagnostics = {
            "backbone_param_max_delta_after_phase": _max_named_delta(
                _snapshot_backbone_parameters(model),
                backbone_snapshot,
            ),
            **_bn_delta_diagnostics(model, bn_snapshot),
            "feature_drift_old_eval_mean_cos": _feature_drift_mean_cosine(
                model,
                old_eval_loader_for_drift,
                device,
                old_eval_features_before,
            ),
            "old_eval_logit_margin_old_vs_latest": _old_eval_logit_margin(
                model,
                old_eval_loader_for_drift,
                device,
                old_dim,
                new_dim,
            ),
        }
        old_head_weight_delta_after_phase = (
            _max_abs_delta(model.classifier.weight[:old_dim], old_weight_snapshot)
            if model.classifier is not None and old_dim > 0
            else None
        )
        old_head_bias_delta_after_phase = (
            _max_abs_delta(model.classifier.bias[:old_dim], old_bias_snapshot)
            if model.classifier is not None and model.classifier.bias is not None and old_dim > 0
            else None
        )

        if method_plugin is not None and phase == 0:
            prototype_estimate = compute_class_prototypes(
                model=model,
                loader=prototype_loader,
                class_ids=task_classes,
                device=device,
            )
            method_diagnostics.update(
                method_plugin.update_memory_after_phase(
                    phase=phase,
                    seen_classes=seen_classes,
                    class_counts=protocol.class_counts,
                    groups=groups,
                    estimate=prototype_estimate,
                )
            )
            method_diagnostics.update(_prototype_norm_diagnostics(prototype_estimate))
            logger.info(
                "phase=%d prototype_memory_classes=%s",
                phase,
                method_diagnostics.get("gpa_memory_classes", 0),
            )

        seen_eval_loader = _make_loader(
            protocol.test_dataset_for_classes(seen_classes),
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        seen_eval = evaluate(model, seen_eval_loader, class_to_head_index, device)
        calibration_eval = _eval_calibration_diagnostics(
            model=model,
            loader=seen_eval_loader,
            class_to_head_index=class_to_head_index,
            tasks=protocol.tasks[: phase + 1],
            device=device,
        )
        phase_accs.append(seen_eval.top1)

        predicted_task_hist_per_eval_task: Dict[str, Dict[str, int]] = {}
        predicted_task_hist_per_eval_task_frac: Dict[str, Dict[str, float]] = {}
        for task_idx in range(phase + 1):
            task_eval_loader = _make_loader(
                protocol.test_dataset_for_classes(protocol.tasks[task_idx]),
                batch_size=args.batch_size,
                shuffle=False,
                seed=args.seed,
                num_workers=args.num_workers,
            )
            task_eval = evaluate(model, task_eval_loader, class_to_head_index, device)
            acc_matrix[phase][task_idx] = task_eval.top1
            if args.diagnose_task_hist:
                task_hist = prediction_task_histogram(task_eval.per_class, protocol.tasks[: phase + 1])
                predicted_task_hist_per_eval_task[f"task_{task_idx}"] = task_hist
                task_hist_frac = normalize_histogram(task_hist, task_eval.total)
                predicted_task_hist_per_eval_task_frac[f"task_{task_idx}"] = task_hist_frac
                latest_task_frac = task_hist_frac.get(f"task_{phase}", 0.0)
                if task_idx < phase and latest_task_frac >= 0.9:
                    logger.warning(
                        "phase=%d eval_task=%d predicted %.2f%% of samples as latest task_%d",
                        phase,
                        task_idx,
                        100.0 * latest_task_frac,
                        phase,
                    )

        many_acc = group_accuracy(seen_eval.per_class, groups["many"])
        medium_acc = group_accuracy(seen_eval.per_class, groups["medium"])
        few_acc = group_accuracy(seen_eval.per_class, groups["few"])
        head_tail_gap = None
        if many_acc is not None and few_acc is not None:
            head_tail_gap = many_acc - few_acc
        per_class_acc = {
            int(class_id): 100.0 * int(stats["correct"]) / int(stats["total"])
            for class_id, stats in seen_eval.per_class.items()
            if int(stats["total"]) > 0
        }
        per_task_acc = {
            f"task_{task_idx}": acc_matrix[phase][task_idx]
            for task_idx in range(phase + 1)
        }
        avg_acc = sum(phase_accs) / len(phase_accs)
        paper_acc_exclude_base = sum(phase_accs[1:]) / len(phase_accs[1:]) if len(phase_accs) > 1 else None
        paper_accT = seen_eval.top1 if phase == task_count - 1 else None
        current_forgetting = forgetting(acc_matrix, phase)
        gpa_anchor_loss_raw = train_stats["gpa_anchor_loss"]
        gpa_anchor_loss_scaled = float(args.lambda_gpa) * float(gpa_anchor_loss_raw)
        ce_loss = train_stats["train_ce_loss"]
        anchor_to_ce_ratio = gpa_anchor_loss_scaled / ce_loss if ce_loss else None
        if gpa_anchor_active and anchor_to_ce_ratio is not None and anchor_to_ce_ratio < 0.01:
            logger.warning(
                "phase=%d anchor_to_ce_ratio=%.6f < 0.01; GPA regularization is likely too weak",
                phase,
                anchor_to_ce_ratio,
            )
        if phase == 0:
            base_train_acc_last_epoch = train_stats["train_acc_last_epoch"]
            base_train_loss_last_epoch = train_stats["train_loss_last_epoch"]
            base_test_acc = seen_eval.top1
        predicted_task_hist_over_all_seen = (
            prediction_task_histogram(seen_eval.per_class, protocol.tasks[: phase + 1])
            if args.diagnose_task_hist
            else None
        )
        predicted_task_hist_over_all_seen_frac = (
            normalize_histogram(predicted_task_hist_over_all_seen, seen_eval.total)
            if predicted_task_hist_over_all_seen is not None
            else None
        )
        classifier_stats = _classifier_task_stats(
            model=model,
            tasks=protocol.tasks,
            class_to_head_index=class_to_head_index,
            upto_phase=phase,
        )
        if int(args.num_exemplars_per_class) > 0:
            exemplar_indices_by_class.update(
                _collect_phase_exemplars(
                    protocol=protocol,
                    task_classes=task_classes,
                    num_exemplars_per_class=int(args.num_exemplars_per_class),
                    seed=int(args.seed),
                    phase=int(phase),
                )
            )
        exemplars_per_task = _exemplar_task_summary(protocol.tasks, exemplar_indices_by_class)
        if phase == 1:
            phase1_logit_forensics = {
                "before_expansion": logit_forensic_before_expansion,
                "after_init": logit_forensic_after_init,
                "after_train": logit_forensic_after_train,
            }

        metrics = {
            "phase": phase,
            "method": args.method,
            "seed": args.seed,
            "rho": args.rho,
            "order": args.order,
            "count_assignment": args.count_assignment,
            "class_order_source": args.class_order_source,
            "class_order_file": args.class_order_file,
            "cifar_normalization": args.cifar_normalization,
            "medical_fold": getattr(args, "medical_fold", None),
            "medical_imb_factor": getattr(args, "medical_imb_factor", None),
            "medical_image_size": getattr(args, "medical_image_size", None),
            "medical_normalization": getattr(args, "medical_normalization", None),
            "protocol_source": getattr(protocol, "protocol_source", None),
            "normalization_mean": protocol.normalization_mean,
            "normalization_std": protocol.normalization_std,
            "scheduler": args.scheduler,
            "milestones": args.milestones,
            "gamma": args.gamma,
            "epochs": args.epochs,
            "phase_epochs": phase_epochs,
            "base_epochs": args.base_epochs,
            "inc_epochs": args.inc_epochs,
            "max_phases": args.max_phases,
            "gpa_anchor_active": gpa_anchor_active,
            "new_classes": task_classes,
            "seen_classes": seen_classes.copy(),
            "num_seen_classes": len(seen_classes),
            "classifier_dim_before": previous_dim,
            "classifier_dim_after": new_dim,
            "train_samples": len(train_dataset),
            "current_train_samples": len(current_train_dataset),
            "exemplar_train_samples_added": exemplar_train_samples_added,
            "num_exemplars_per_class": args.num_exemplars_per_class,
            "exemplar_selection": args.exemplar_selection,
            "exemplars_per_task": exemplars_per_task,
            "train_class_counts": {class_id: protocol.class_counts[class_id] for class_id in task_classes},
            "train_loss": train_stats["train_loss"],
            "train_loss_last_epoch": train_stats["train_loss_last_epoch"],
            "train_ce_loss": train_stats["train_ce_loss"],
            "train_ce_loss_last_epoch": train_stats["train_ce_loss_last_epoch"],
            "ce_loss": ce_loss,
            "gpa_anchor_loss": train_stats["gpa_anchor_loss"],
            "gpa_anchor_loss_raw": gpa_anchor_loss_raw,
            "gpa_anchor_loss_scaled": gpa_anchor_loss_scaled,
            "anchor_to_ce_ratio": anchor_to_ce_ratio,
            "ltconcm_match_loss": train_stats["ltconcm_match_loss"],
            "method_extra_loss": train_stats["method_extra_loss"],
            "feature_norm_mean": train_stats["feature_norm_mean"],
            "feature_norm_std": train_stats["feature_norm_std"],
            "train_acc_last_epoch": train_stats["train_acc_last_epoch"],
            "base_train_acc_last_epoch": base_train_acc_last_epoch,
            "base_train_loss_last_epoch": base_train_loss_last_epoch,
            "base_test_acc": base_test_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "lr_curve": train_stats["lr_curve"],
            "top1_all_seen": seen_eval.top1,
            **calibration_eval,
            "balanced_acc": seen_eval.balanced_acc,
            "macro_f1": seen_eval.macro_f1,
            "avg_incremental_acc_so_far": avg_acc,
            "paper_acc_include_base": avg_acc,
            "paper_acc_exclude_base": paper_acc_exclude_base,
            "paper_accT": paper_accT,
            "forgetting_so_far": current_forgetting,
            "many_acc": many_acc,
            "medium_acc": medium_acc,
            "few_acc": few_acc,
            "head_tail_gap": head_tail_gap,
            "old_head_weight_max_delta_after_expansion": old_head_weight_delta_after_expansion,
            "old_head_bias_max_delta_after_expansion": old_head_bias_delta_after_expansion,
            "old_head_weight_max_delta_before_after_phase": old_head_weight_delta_after_phase,
            "old_head_bias_max_delta_before_after_phase": old_head_bias_delta_after_phase,
            "gpa_freeze_old_head_active": freeze_old_head_this_phase,
            "gpa_restore_old_head_after_step_active": restore_old_head_this_phase,
            "gpa_anchor_reduction": args.gpa_anchor_reduction,
            "gpa_bias_mode": args.gpa_bias_mode,
            **freeze_policy,
            **drift_diagnostics,
            "logit_forensic_before_expansion": logit_forensic_before_expansion,
            "logit_forensic_after_init": logit_forensic_after_init,
            "logit_forensic_after_train": logit_forensic_after_train,
            "phase1_task0_acc_after_init": (
                logit_forensic_after_init.get("old_task_acc")
                if phase == 1 and logit_forensic_after_init is not None
                else None
            ),
            "phase1_task0_acc_after_train": (
                logit_forensic_after_train.get("old_task_acc")
                if phase == 1 and logit_forensic_after_train is not None
                else None
            ),
            "phase1_old_vs_latest_margin_after_init": (
                logit_forensic_after_init.get("old_vs_latest_margin_mean")
                if phase == 1 and logit_forensic_after_init is not None
                else None
            ),
            "phase1_old_vs_latest_margin_after_train": (
                logit_forensic_after_train.get("old_vs_latest_margin_mean")
                if phase == 1 and logit_forensic_after_train is not None
                else None
            ),
            "frequency_groups": groups,
            "frequency_group_summary": _group_summary(protocol.class_counts, groups),
            "per_task_acc": per_task_acc,
            "per_class_acc": per_class_acc,
            "per_class_prediction_stats": seen_eval.per_class,
            "predicted_task_hist_over_all_seen": predicted_task_hist_over_all_seen,
            "predicted_task_hist_over_all_seen_frac": predicted_task_hist_over_all_seen_frac,
            "predicted_task_hist_per_eval_task": predicted_task_hist_per_eval_task if args.diagnose_task_hist else None,
            "predicted_task_hist_per_eval_task_frac": (
                predicted_task_hist_per_eval_task_frac if args.diagnose_task_hist else None
            ),
        }
        metrics.update(classifier_stats)
        metrics.update(
            {
                "train/loss": train_stats["train_loss"],
                "train/ce_loss": train_stats["train_ce_loss"],
                "train/gpa_loss": train_stats["gpa_anchor_loss"],
                "train/ltconcm_loss": train_stats["method_extra_loss"] if args.use_ltconcm else None,
                "train/tdsm_loss": method_diagnostics.get("tdsm_align_loss"),
                "train/match_loss": train_stats["ltconcm_match_loss"],
                "train/lr": optimizer.param_groups[0]["lr"],
                "eval/acc": seen_eval.top1,
                "eval/raw_linear_acc": calibration_eval.get("raw_linear_acc"),
                "eval/no_bias_acc": calibration_eval.get("no_bias_acc"),
                "eval/normalized_weight_acc": calibration_eval.get("normalized_weight_acc"),
                "eval/cosine_eval_acc": calibration_eval.get("cosine_eval_acc"),
                "eval/old_new_bias_corrected_acc": calibration_eval.get("old_new_bias_corrected_acc"),
                "eval/balanced_acc": seen_eval.balanced_acc,
                "eval/macro_f1": seen_eval.macro_f1,
                "eval/per_task_acc": per_task_acc,
                "eval/per_class_acc": per_class_acc,
                "eval/head_acc": many_acc,
                "eval/medium_acc": medium_acc,
                "eval/tail_acc": few_acc,
                "eval/many_shot_acc": many_acc,
                "eval/medium_shot_acc": medium_acc,
                "eval/few_shot_acc": few_acc,
                "eval/avg_acc": avg_acc,
                "eval/final_acc": seen_eval.top1 if phase == task_count - 1 else None,
                "eval/bwt": -current_forgetting,
                "eval/forgetting": current_forgetting,
            }
        )
        metrics.update(run_metadata)
        metrics.update(_method_metadata(args))
        metrics.update(method_diagnostics)
        last_metrics = metrics
        append_jsonl(metrics_path, metrics)
        tracker.log(metrics)
        write_acc_matrix_csv(output_dir / "acc_matrix.csv", acc_matrix)
        if args.base_only and phase == 0:
            _write_base_sanity_artifacts(
                output_dir=output_dir,
                per_class_stats=seen_eval.per_class,
                per_class_acc=per_class_acc,
                groups=groups,
                class_counts=protocol.class_counts,
            )

        checkpoint_path = checkpoints_dir / f"model_phase_{phase}.pt"
        checkpoint_payload = {
            "phase": phase,
            "model_state": model.state_dict(),
            "seen_classes": seen_classes,
            "class_to_head_index": class_to_head_index,
            "class_counts": protocol.class_counts,
            "class_order": protocol.class_order,
            "args": _args_to_config(args),
            "run_metadata": run_metadata,
        }
        if method_plugin is not None:
            checkpoint_payload["method_state"] = method_plugin.state_dict()
            checkpoint_payload["method_diagnostics"] = method_diagnostics
        torch.save(checkpoint_payload, checkpoint_path)
        logger.info(
            "phase=%d top1_all_seen=%.4f avg_inc=%.4f forgetting=%.4f checkpoint=%s",
            phase,
            seen_eval.top1,
            metrics["avg_incremental_acc_so_far"],
            metrics["forgetting_so_far"],
            checkpoint_path,
        )

    base_groups = _frequency_groups(protocol, protocol.tasks[0])
    base_class_counts = {int(class_id): protocol.class_counts[int(class_id)] for class_id in protocol.tasks[0]}
    final_summary = {
        "method": args.method,
        "seed": args.seed,
        "rho": args.rho,
        "order": args.order,
        "count_assignment": args.count_assignment,
        "class_count_assignment": args.count_assignment,
        "class_order_source": args.class_order_source,
        "class_order_file": args.class_order_file,
        "cifar_normalization": args.cifar_normalization,
        "normalization_mean": protocol.normalization_mean,
        "normalization_std": protocol.normalization_std,
        "scheduler": args.scheduler,
        "milestones": args.milestones,
        "gamma": args.gamma,
        "epochs": args.epochs,
        "base_epochs": args.base_epochs,
        "inc_epochs": args.inc_epochs,
        "max_phases": args.max_phases,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_exemplars_per_class": args.num_exemplars_per_class,
        "exemplar_selection": args.exemplar_selection,
        "exemplars_per_task": _exemplar_task_summary(protocol.tasks, exemplar_indices_by_class),
        "exemplar_train_samples_added": last_metrics.get("exemplar_train_samples_added"),
        "base_only": args.base_only,
        "num_phases": task_count,
        "final_accuracy": phase_accs[-1] if phase_accs else None,
        "average_incremental_accuracy": sum(phase_accs) / len(phase_accs) if phase_accs else None,
        "paper_acc_include_base": sum(phase_accs) / len(phase_accs) if phase_accs else None,
        "paper_acc_exclude_base": sum(phase_accs[1:]) / len(phase_accs[1:]) if len(phase_accs) > 1 else None,
        "paper_accT": phase_accs[-1] if phase_accs else None,
        "forgetting": forgetting(acc_matrix, task_count - 1) if phase_accs else None,
        "many_acc": last_metrics.get("many_acc"),
        "medium_acc": last_metrics.get("medium_acc"),
        "few_acc": last_metrics.get("few_acc"),
        "balanced_acc": last_metrics.get("balanced_acc"),
        "macro_f1": last_metrics.get("macro_f1"),
        "head_tail_gap": last_metrics.get("head_tail_gap"),
        "base_train_acc_last_epoch": base_train_acc_last_epoch,
        "base_train_loss_last_epoch": base_train_loss_last_epoch,
        "base_test_acc": base_test_acc,
        "base_many_acc": last_metrics.get("many_acc") if args.base_only else None,
        "base_medium_acc": last_metrics.get("medium_acc") if args.base_only else None,
        "base_few_acc": last_metrics.get("few_acc") if args.base_only else None,
        "base_many_class_count": len(base_groups["many"]),
        "base_medium_class_count": len(base_groups["medium"]),
        "base_few_class_count": len(base_groups["few"]),
        "base_classes_list": [int(class_id) for class_id in protocol.tasks[0]],
        "base_class_counts": base_class_counts,
        "task_classes": protocol.tasks,
        "task_frequency_group_counts": _task_frequency_group_counts(protocol),
        "per_task_many_medium_few_count": _task_frequency_group_counts(protocol),
        "per_task_train_samples": _per_task_train_samples(protocol),
        "gpa_anchor_loss_raw": last_metrics.get("gpa_anchor_loss_raw"),
        "gpa_anchor_loss_scaled": last_metrics.get("gpa_anchor_loss_scaled"),
        "gpa_anchor_reduction": args.gpa_anchor_reduction,
        "gpa_bias_mode": args.gpa_bias_mode,
        "ce_loss": last_metrics.get("ce_loss"),
        "anchor_to_ce_ratio": last_metrics.get("anchor_to_ce_ratio"),
        "raw_linear_accT": last_metrics.get("raw_linear_acc"),
        "no_bias_accT": last_metrics.get("no_bias_acc"),
        "normalized_weight_accT": last_metrics.get("normalized_weight_acc"),
        "cosine_eval_accT": last_metrics.get("cosine_eval_acc"),
        "old_new_bias_corrected_accT": last_metrics.get("old_new_bias_corrected_acc"),
        "phase1_logit_forensics": phase1_logit_forensics,
        "phase1_task0_acc_before_expansion": (
            (phase1_logit_forensics or {}).get("before_expansion", {}) or {}
        ).get("old_task_acc"),
        "phase1_task0_acc_after_init": (
            (phase1_logit_forensics or {}).get("after_init", {}) or {}
        ).get("old_task_acc"),
        "phase1_task0_acc_after_train": (
            (phase1_logit_forensics or {}).get("after_train", {}) or {}
        ).get("old_task_acc"),
        "phase1_old_vs_latest_margin_after_init": (
            (phase1_logit_forensics or {}).get("after_init", {}) or {}
        ).get("old_vs_latest_margin_mean"),
        "phase1_old_vs_latest_margin_after_train": (
            (phase1_logit_forensics or {}).get("after_train", {}) or {}
        ).get("old_vs_latest_margin_mean"),
        "phase1_old_vs_new_margin_after_init": (
            (phase1_logit_forensics or {}).get("after_init", {}) or {}
        ).get("old_vs_new_margin_mean"),
        "phase1_old_vs_new_margin_after_train": (
            (phase1_logit_forensics or {}).get("after_train", {}) or {}
        ).get("old_vs_new_margin_mean"),
        "phase1_latest_pred_rate_after_init": (
            ((phase1_logit_forensics or {}).get("after_init", {}) or {}).get("predicted_task_hist_frac", {}) or {}
        ).get("task_1"),
        "phase1_latest_pred_rate_after_train": (
            ((phase1_logit_forensics or {}).get("after_train", {}) or {}).get("predicted_task_hist_frac", {}) or {}
        ).get("task_1"),
        "phase1_dot_margin_after_init": (
            (phase1_logit_forensics or {}).get("after_init", {}) or {}
        ).get("old_vs_new_dot_margin_mean"),
        "phase1_dot_margin_after_train": (
            (phase1_logit_forensics or {}).get("after_train", {}) or {}
        ).get("old_vs_new_dot_margin_mean"),
        "phase1_bias_margin_after_init": (
            (phase1_logit_forensics or {}).get("after_init", {}) or {}
        ).get("old_vs_new_bias_margin_mean"),
        "phase1_bias_margin_after_train": (
            (phase1_logit_forensics or {}).get("after_train", {}) or {}
        ).get("old_vs_new_bias_margin_mean"),
        "feature_norm_mean": last_metrics.get("feature_norm_mean"),
        "feature_norm_std": last_metrics.get("feature_norm_std"),
        "prototype_norm_mean": last_metrics.get("prototype_norm_mean"),
        "prototype_norm_min": last_metrics.get("prototype_norm_min"),
        "prototype_norm_max": last_metrics.get("prototype_norm_max"),
        "old_head_weight_max_delta_after_expansion": last_metrics.get("old_head_weight_max_delta_after_expansion"),
        "old_head_bias_max_delta_after_expansion": last_metrics.get("old_head_bias_max_delta_after_expansion"),
        "old_head_weight_max_delta_before_after_phase": last_metrics.get("old_head_weight_max_delta_before_after_phase"),
        "old_head_bias_max_delta_before_after_phase": last_metrics.get("old_head_bias_max_delta_before_after_phase"),
        "new_weight_cos_to_frozen_proto_mean": last_metrics.get("new_weight_cos_to_frozen_proto_mean"),
        "new_weight_cos_to_frozen_proto_min": last_metrics.get("new_weight_cos_to_frozen_proto_min"),
        "classifier_weight_norm_by_task": last_metrics.get("classifier_weight_norm_by_task"),
        "classifier_bias_mean_by_task": last_metrics.get("classifier_bias_mean_by_task"),
        "predicted_task_hist_over_all_seen": last_metrics.get("predicted_task_hist_over_all_seen"),
        "predicted_task_hist_over_all_seen_frac": last_metrics.get("predicted_task_hist_over_all_seen_frac"),
        "predicted_task_hist_per_eval_task": last_metrics.get("predicted_task_hist_per_eval_task"),
        "predicted_task_hist_per_eval_task_frac": last_metrics.get("predicted_task_hist_per_eval_task_frac"),
        "backbone_param_max_delta_after_phase": last_metrics.get("backbone_param_max_delta_after_phase"),
        "bn_running_mean_max_delta_after_phase": last_metrics.get("bn_running_mean_max_delta_after_phase"),
        "bn_running_var_max_delta_after_phase": last_metrics.get("bn_running_var_max_delta_after_phase"),
        "feature_drift_old_eval_mean_cos": last_metrics.get("feature_drift_old_eval_mean_cos"),
        "old_eval_logit_margin_old_vs_latest": last_metrics.get("old_eval_logit_margin_old_vs_latest"),
        "class_order": protocol.class_order,
        "class_counts": protocol.class_counts,
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoints_dir),
        "config": _args_to_config(args),
    }
    final_summary.update(run_metadata)
    final_summary.update(_method_metadata(args))
    write_json(output_dir / "summary.json", final_summary)
    write_summary_csv(output_dir / "summary.csv", [final_summary])
    logger.info("completed run summary=%s", final_summary)
    tracker.finish()
    return output_dir
