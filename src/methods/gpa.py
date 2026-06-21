"""GPA and TaConCM-GPA plugin logic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.methods.ltconcm import (
    PrototypeMemory,
    ReliabilityAwarePrototypeCalibrator,
    build_tail_aware_structure_anchors,
    tail_anchor_weights,
)


@dataclass
class PrototypeEstimate:
    class_ids: List[int]
    raw: torch.Tensor
    normalized: torch.Tensor
    counts: torch.Tensor
    variance: torch.Tensor
    raw_norms: torch.Tensor


@torch.no_grad()
def compute_class_prototypes(
    model: nn.Module,
    loader: DataLoader,
    class_ids: Sequence[int],
    device: torch.device,
) -> PrototypeEstimate:
    class_ids = [int(class_id) for class_id in class_ids]
    if not class_ids:
        raise ValueError("class_ids must be non-empty")
    class_to_row = {class_id: row for row, class_id in enumerate(class_ids)}
    was_training = model.training
    model.eval()

    feature_dim = int(getattr(model, "feature_dim"))
    sums = torch.zeros(len(class_ids), feature_dim, device=device)
    sums_sq = torch.zeros_like(sums)
    counts = torch.zeros(len(class_ids), dtype=torch.float32, device=device)

    for images, labels_global in loader:
        images = images.to(device)
        labels_global = labels_global.to(device)
        features = model.extract_features(images)
        for class_id in labels_global.unique().detach().cpu().tolist():
            class_id = int(class_id)
            if class_id not in class_to_row:
                continue
            row = class_to_row[class_id]
            mask = labels_global == class_id
            selected = features[mask]
            sums[row] += selected.sum(dim=0)
            sums_sq[row] += selected.pow(2).sum(dim=0)
            counts[row] += float(selected.shape[0])

    if was_training:
        model.train()

    missing = [class_id for row, class_id in enumerate(class_ids) if int(counts[row].item()) == 0]
    if missing:
        raise RuntimeError(f"Prototype computation found zero samples for classes: {missing}")

    raw = sums / counts[:, None].clamp_min(1.0)
    second_moment = sums_sq / counts[:, None].clamp_min(1.0)
    variance = torch.clamp(second_moment - raw.pow(2), min=0.0)
    raw_norms = torch.linalg.vector_norm(raw, dim=1)
    normalized = F.normalize(raw, dim=1)
    return PrototypeEstimate(
        class_ids=class_ids,
        raw=raw,
        normalized=normalized,
        counts=counts,
        variance=variance,
        raw_norms=raw_norms,
    )


def init_new_class_weights(
    classifier: nn.Linear | None,
    class_ids: Sequence[int],
    prototypes: torch.Tensor,
    class_to_head_index: Mapping[int, int],
    counts: torch.Tensor,
    init_bias: bool = True,
    bias_mode: str = "paper",
    old_bias_mean: float | None = None,
    eps: float = 1e-8,
) -> Dict[str, object]:
    if classifier is None:
        raise RuntimeError("Classifier must be initialized before GPA weight init")
    if prototypes.shape[0] != len(class_ids):
        raise ValueError("prototype row count must match class_ids")
    if bias_mode not in {"paper", "zero", "old_mean", "none"}:
        raise ValueError(f"Unsupported GPA bias mode: {bias_mode}")
    n_ref = float(counts.max().detach().cpu().item()) if counts.numel() else 1.0
    bias_values: Dict[int, float] = {}
    formula = "disabled_by_gpa_init_bias_false"
    if init_bias:
        if bias_mode == "paper":
            formula = "-log(N_i / (N_ref + eps) + eps), N_ref=max(new_class_counts)"
        elif bias_mode == "zero":
            formula = "0"
        elif bias_mode == "old_mean":
            formula = "mean(old_classifier_bias)"
        else:
            formula = "leave_expansion_default"
    with torch.no_grad():
        for row, class_id in enumerate(class_ids):
            head_index = int(class_to_head_index[int(class_id)])
            classifier.weight[head_index].copy_(F.normalize(prototypes[row], dim=0))
            if init_bias and classifier.bias is not None and bias_mode != "none":
                if bias_mode == "paper":
                    count = float(counts[row].detach().cpu().item())
                    value = -math.log(count / (n_ref + eps) + eps)
                elif bias_mode == "zero":
                    value = 0.0
                else:
                    value = float(old_bias_mean) if old_bias_mean is not None else 0.0
                classifier.bias[head_index].fill_(value)
                bias_values[int(class_id)] = float(value)
            elif classifier.bias is not None:
                bias_values[int(class_id)] = float(classifier.bias[head_index].detach().cpu().item())
    return {
        "gpa_bias_mode": bias_mode,
        "gpa_bias_n_ref": n_ref,
        "gpa_bias_formula": formula,
        "gpa_new_bias_init_values_per_class": bias_values,
    }


def _mean_for_ids(values: Mapping[int, float], class_ids: Sequence[int]) -> float | None:
    selected = [float(values[int(class_id)]) for class_id in class_ids if int(class_id) in values]
    if not selected:
        return None
    return float(sum(selected) / len(selected))


class GPAPlugin:
    def __init__(self, args) -> None:
        self.method = str(args.method)
        self.lambda_gpa = float(args.lambda_gpa)
        self.gpa_anchor_start_phase = int(args.gpa_anchor_start_phase)
        self.gpa_init_bias = bool(args.gpa_init_bias)
        self.gpa_anchor_reduction = str(getattr(args, "gpa_anchor_reduction", "mean"))
        self.gpa_bias_mode = str(getattr(args, "gpa_bias_mode", "paper"))
        self.use_ltconcm = bool(args.use_ltconcm or args.method == "taconcm_gpa" or args.method.startswith("taconcm_stage"))
        self.calibrate_prototypes = bool(args.ltconcm_calibrate_prototypes)
        self.tail_anchor = bool(args.ltconcm_tail_anchor)
        self.anchor_target = str(args.ltconcm_anchor_target)
        self.anchor_gamma = float(args.ltconcm_anchor_gamma)
        self.anchor_max_weight = float(args.ltconcm_anchor_max_weight)
        self.use_tdsm = bool(args.ltconcm_use_tdsm)
        self.use_match_loss = bool(args.ltconcm_use_match_loss)
        self.match_lambda = float(args.ltconcm_match_lambda)
        self.eps = 1e-8
        self.memory = PrototypeMemory()
        self.anchor_targets: Dict[int, torch.Tensor] = {}
        self.anchor_weights: Dict[int, float] = {}
        self.last_phase_diagnostics: Dict[str, object] = {}
        self.last_prototype_snapshot: Dict[str, object] = {}
        self.calibrator = ReliabilityAwarePrototypeCalibrator(
            top_k_memory=args.ltconcm_memory_topk,
            alpha_a=args.ltconcm_alpha_a,
            alpha_b=args.ltconcm_alpha_b,
            alpha_min=args.ltconcm_alpha_min,
            alpha_max=args.ltconcm_alpha_max,
            eps=self.eps,
        )
        if self.gpa_anchor_reduction not in {"mean", "sum", "cosine"}:
            raise ValueError(f"Unsupported GPA anchor reduction: {self.gpa_anchor_reduction}")
        if self.gpa_bias_mode not in {"paper", "zero", "old_mean", "none"}:
            raise ValueError(f"Unsupported GPA bias mode: {self.gpa_bias_mode}")

    @classmethod
    def maybe_build(cls, args) -> "GPAPlugin | None":
        if bool(getattr(args, "uses_gpa", False)) or bool(args.use_ltconcm):
            return cls(args)
        return None

    def _group_diagnostics(
        self,
        values: torch.Tensor,
        class_ids: Sequence[int],
        groups: Mapping[str, Sequence[int]],
        prefix: str,
    ) -> Dict[str, float | None]:
        by_class = {
            int(class_id): float(values[row].detach().cpu().item())
            for row, class_id in enumerate(class_ids)
        }
        return {
            f"{prefix}_many_mean": _mean_for_ids(by_class, groups.get("many", [])),
            f"{prefix}_medium_mean": _mean_for_ids(by_class, groups.get("medium", [])),
            f"{prefix}_few_mean": _mean_for_ids(by_class, groups.get("few", [])),
        }

    def _refresh_anchor_weights(
        self,
        seen_classes: Sequence[int],
        class_counts: Mapping[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        counts = torch.tensor([class_counts[int(class_id)] for class_id in seen_classes], dtype=torch.float32, device=device)
        if self.use_ltconcm and self.tail_anchor:
            weights = tail_anchor_weights(counts, gamma=self.anchor_gamma, w_max=self.anchor_max_weight, eps=self.eps)
        else:
            weights = torch.ones_like(counts)
        self.anchor_weights = {
            int(class_id): float(weights[row].detach().cpu().item())
            for row, class_id in enumerate(seen_classes)
        }
        return weights

    def _set_anchor_targets_from_memory(
        self,
        seen_classes: Sequence[int],
        class_counts: Mapping[int, int],
        device: torch.device,
        task_classes: Sequence[int] | None = None,
    ) -> Dict[str, float]:
        if not self.use_ltconcm or self.anchor_target == "gpa_raw":
            self.anchor_targets = {}
            self.anchor_weights = {}
            return {}
        if self.anchor_target == "tdsm" and not self.use_tdsm:
            raise RuntimeError("ltconcm anchor_target=tdsm requires use_tdsm=True")

        ids = [int(class_id) for class_id in seen_classes if int(class_id) in self.memory]
        if not ids:
            self.anchor_targets = {}
            return {}

        _, _, proto_calib, counts, _ = self.memory.tensors(ids, device=device)
        anchors = proto_calib
        diag: Dict[str, float] = {}
        if self.anchor_target == "tdsm":
            old_anchor_rows = []
            old_mask = []
            new_set = set(int(class_id) for class_id in (task_classes or []))
            for row, class_id in enumerate(ids):
                old_mask.append(int(class_id) not in new_set)
                if int(class_id) in self.anchor_targets:
                    old_anchor_rows.append(self.anchor_targets[int(class_id)])
                else:
                    old_anchor_rows.append(proto_calib[row].detach().cpu())
            old_anchors = torch.stack(old_anchor_rows).to(device=device)
            old_class_mask = torch.tensor(old_mask, dtype=torch.bool, device=device)
            anchors, diag = build_tail_aware_structure_anchors(
                proto_calib=proto_calib,
                class_counts=counts,
                old_anchors=old_anchors,
                old_class_mask=old_class_mask,
                steps=100,
                lr=0.05,
            )
            self.memory.set_anchors(ids, anchors)

        self.anchor_targets = {
            int(class_id): F.normalize(anchors[row].detach().cpu(), dim=0)
            for row, class_id in enumerate(ids)
        }
        self._refresh_anchor_weights(ids, class_counts, device)
        return diag

    def prepare_incremental_phase(
        self,
        model: nn.Module,
        phase: int,
        task_classes: Sequence[int],
        seen_classes: Sequence[int],
        class_to_head_index: Mapping[int, int],
        class_counts: Mapping[int, int],
        groups: Mapping[str, Sequence[int]],
        estimate: PrototypeEstimate,
    ) -> Dict[str, object]:
        device = estimate.raw.device
        proto_for_init = estimate.normalized
        alpha = torch.ones(len(task_classes), dtype=torch.float32, device=device)
        uncertainty = torch.zeros_like(alpha)
        raw_calib_cosine = torch.ones_like(alpha)
        memory_used = torch.zeros(len(task_classes), dtype=torch.bool, device=device)

        if self.use_ltconcm and self.calibrate_prototypes:
            result = self.calibrator.calibrate(
                proto_raw=estimate.raw,
                class_counts=estimate.counts,
                proto_var=estimate.variance,
                prototype_memory=self.memory,
                class_ids=task_classes,
            )
            proto_for_init = result.proto_calib
            alpha = result.alpha
            uncertainty = result.uncertainty
            raw_calib_cosine = result.raw_calib_cosine
            memory_used = result.memory_used

        self.last_prototype_snapshot = {
            "phase": int(phase),
            "class_ids": [int(class_id) for class_id in task_classes],
            "proto_raw": estimate.raw.detach().cpu(),
            "proto_normalized": estimate.normalized.detach().cpu(),
            "proto_init": proto_for_init.detach().cpu(),
        }

        old_bias_values = []
        if getattr(model, "classifier", None) is not None and model.classifier.bias is not None:
            new_set = set(int(class_id) for class_id in task_classes)
            for class_id in seen_classes:
                class_id = int(class_id)
                if class_id in new_set or class_id not in class_to_head_index:
                    continue
                old_bias_values.append(
                    float(model.classifier.bias[int(class_to_head_index[class_id])].detach().cpu().item())
                )
        old_bias_mean = float(sum(old_bias_values) / len(old_bias_values)) if old_bias_values else None

        bias_diagnostics = init_new_class_weights(
            classifier=getattr(model, "classifier", None),
            class_ids=task_classes,
            prototypes=proto_for_init,
            class_to_head_index=class_to_head_index,
            counts=estimate.counts,
            init_bias=self.gpa_init_bias,
            bias_mode=self.gpa_bias_mode,
            old_bias_mean=old_bias_mean,
            eps=self.eps,
        )

        self.memory.update(
            class_ids=task_classes,
            proto_raw=estimate.raw,
            proto_calib=proto_for_init,
            counts=estimate.counts,
            uncertainty=uncertainty,
            session_id=phase,
        )
        tdsm_diag = self._set_anchor_targets_from_memory(
            seen_classes=seen_classes,
            class_counts=class_counts,
            device=device,
            task_classes=task_classes,
        )

        weight_values = {
            int(class_id): float(self.anchor_weights[int(class_id)])
            for class_id in seen_classes
            if int(class_id) in self.anchor_weights
        }
        diagnostics: Dict[str, object] = {
            "gpa_num_prototypes": len(task_classes),
            "gpa_prototype_raw_norm_mean": float(estimate.raw_norms.mean().detach().cpu().item()),
            "gpa_prototype_raw_norm_min": float(estimate.raw_norms.min().detach().cpu().item()),
            "gpa_prototype_raw_norm_max": float(estimate.raw_norms.max().detach().cpu().item()),
            "ltconcm_num_classes_calibrated": len(task_classes) if self.use_ltconcm else 0,
            "ltconcm_memory_usage_rate": float(memory_used.float().mean().detach().cpu().item()) if memory_used.numel() else 0.0,
            "ltconcm_raw_calib_cosine_mean": float(raw_calib_cosine.mean().detach().cpu().item()),
            "ltconcm_uncertainty_mean": float(uncertainty.mean().detach().cpu().item()) if uncertainty.numel() else 0.0,
            "ltconcm_anchor_target": self.anchor_target,
            "ltconcm_anchor_weight_many_mean": _mean_for_ids(weight_values, groups.get("many", [])),
            "ltconcm_anchor_weight_medium_mean": _mean_for_ids(weight_values, groups.get("medium", [])),
            "ltconcm_anchor_weight_few_mean": _mean_for_ids(weight_values, groups.get("few", [])),
            "gpa_old_bias_mean_for_init": old_bias_mean,
        }
        diagnostics.update(bias_diagnostics)
        if self.use_ltconcm:
            diagnostics.update(self._group_diagnostics(alpha, task_classes, groups, "ltconcm_alpha"))
        diagnostics.update(tdsm_diag)
        self.last_phase_diagnostics = diagnostics
        return diagnostics

    def update_memory_after_phase(
        self,
        phase: int,
        seen_classes: Sequence[int],
        class_counts: Mapping[int, int],
        groups: Mapping[str, Sequence[int]],
        estimate: PrototypeEstimate,
    ) -> Dict[str, object]:
        device = estimate.raw.device
        proto_calib = estimate.normalized
        alpha = torch.ones(len(estimate.class_ids), dtype=torch.float32, device=device)
        uncertainty = torch.zeros_like(alpha)
        raw_calib_cosine = torch.ones_like(alpha)
        memory_used = torch.zeros(len(estimate.class_ids), dtype=torch.bool, device=device)

        if self.use_ltconcm and self.calibrate_prototypes:
            result = self.calibrator.calibrate(
                proto_raw=estimate.raw,
                class_counts=estimate.counts,
                proto_var=estimate.variance,
                prototype_memory=self.memory,
                class_ids=estimate.class_ids,
            )
            proto_calib = result.proto_calib
            alpha = result.alpha
            uncertainty = result.uncertainty
            raw_calib_cosine = result.raw_calib_cosine
            memory_used = result.memory_used

        self.memory.update(
            class_ids=estimate.class_ids,
            proto_raw=estimate.raw,
            proto_calib=proto_calib,
            counts=estimate.counts,
            uncertainty=uncertainty,
            session_id=phase,
        )
        tdsm_diag = self._set_anchor_targets_from_memory(
            seen_classes=seen_classes,
            class_counts=class_counts,
            device=device,
            task_classes=estimate.class_ids,
        )
        diagnostics: Dict[str, object] = {
            "gpa_memory_classes": len(self.memory.entries),
            "ltconcm_memory_usage_rate": float(memory_used.float().mean().detach().cpu().item()) if memory_used.numel() else 0.0,
            "ltconcm_raw_calib_cosine_mean": float(raw_calib_cosine.mean().detach().cpu().item()),
        }
        if self.use_ltconcm:
            diagnostics.update(self._group_diagnostics(alpha, estimate.class_ids, groups, "ltconcm_alpha"))
        diagnostics.update(tdsm_diag)
        self.last_phase_diagnostics.update(diagnostics)
        return diagnostics

    def _batch_anchor_loss(
        self,
        classifier: nn.Linear,
        features: torch.Tensor,
        labels_global: torch.Tensor,
        class_to_head_index: Mapping[int, int],
    ) -> torch.Tensor:
        features_norm = F.normalize(features, dim=1)
        losses = []
        for class_id in labels_global.unique().detach().cpu().tolist():
            class_id = int(class_id)
            if class_id not in class_to_head_index:
                continue
            mask = labels_global == class_id
            proto = F.normalize(features_norm[mask].mean(dim=0), dim=0)
            weight = F.normalize(classifier.weight[int(class_to_head_index[class_id])], dim=0)
            losses.append(self._anchor_distance(weight, proto))
        if not losses:
            return torch.zeros((), dtype=features.dtype, device=features.device)
        return torch.stack(losses).mean()

    def _target_anchor_loss(
        self,
        classifier: nn.Linear,
        class_to_head_index: Mapping[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        losses = []
        weights = []
        for class_id, anchor_cpu in self.anchor_targets.items():
            if class_id not in class_to_head_index:
                continue
            head_index = int(class_to_head_index[class_id])
            weight = F.normalize(classifier.weight[head_index], dim=0)
            anchor = anchor_cpu.to(device=device, dtype=weight.dtype).detach()
            losses.append(self._anchor_distance(weight, anchor))
            weights.append(self.anchor_weights.get(class_id, 1.0))
        if not losses:
            return torch.zeros((), device=device)
        loss_tensor = torch.stack(losses)
        weight_tensor = torch.tensor(weights, dtype=loss_tensor.dtype, device=device)
        return torch.mean(weight_tensor * loss_tensor)

    def _anchor_distance(self, weight: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.gpa_anchor_reduction == "sum":
            return F.mse_loss(weight, target, reduction="sum")
        if self.gpa_anchor_reduction == "cosine":
            return 1.0 - torch.sum(weight * target)
        return F.mse_loss(weight, target, reduction="mean")

    def _match_loss(
        self,
        features: torch.Tensor,
        labels_global: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_match_loss or not self.anchor_targets:
            return torch.zeros((), dtype=features.dtype, device=features.device)
        features_norm = F.normalize(features, dim=1)
        losses = []
        for row, class_id in enumerate(labels_global.detach().cpu().tolist()):
            class_id = int(class_id)
            if class_id not in self.anchor_targets:
                continue
            anchor = self.anchor_targets[class_id].to(device=features.device, dtype=features.dtype)
            weight = self.anchor_weights.get(class_id, 1.0)
            losses.append(float(weight) * (1.0 - torch.sum(features_norm[row] * anchor.detach())))
        if not losses:
            return torch.zeros((), dtype=features.dtype, device=features.device)
        return torch.stack(losses).mean()

    def loss(
        self,
        classifier: nn.Linear,
        features: torch.Tensor,
        labels_global: torch.Tensor,
        class_to_head_index: Mapping[int, int],
    ) -> Dict[str, torch.Tensor]:
        if self.use_ltconcm and self.anchor_targets:
            anchor_loss = self._target_anchor_loss(classifier, class_to_head_index, features.device)
        else:
            anchor_loss = self._batch_anchor_loss(classifier, features, labels_global, class_to_head_index)
        match_loss = self._match_loss(features, labels_global)
        total = self.lambda_gpa * anchor_loss + self.match_lambda * match_loss
        return {
            "gpa_anchor_loss": anchor_loss,
            "ltconcm_match_loss": match_loss,
            "method_extra_loss": total,
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "method": self.method,
            "gpa_anchor_start_phase": self.gpa_anchor_start_phase,
            "gpa_anchor_reduction": self.gpa_anchor_reduction,
            "gpa_bias_mode": self.gpa_bias_mode,
            "use_ltconcm": self.use_ltconcm,
            "ltconcm_calibrate_prototypes": self.calibrate_prototypes,
            "ltconcm_anchor_target": self.anchor_target,
            "ltconcm_tail_anchor": self.tail_anchor,
            "ltconcm_use_tdsm": self.use_tdsm,
            "ltconcm_use_match_loss": self.use_match_loss,
            "memory": self.memory.state_dict(),
            "anchor_targets": {
                str(class_id): anchor for class_id, anchor in sorted(self.anchor_targets.items())
            },
            "anchor_weights": {str(k): v for k, v in sorted(self.anchor_weights.items())},
            "diagnostics": self.last_phase_diagnostics,
            "latest_prototypes": self.last_prototype_snapshot,
        }
