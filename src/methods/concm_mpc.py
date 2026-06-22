"""ConCM-style medical prototype calibration helpers.

This module implements a bounded MPC-lite proxy for the HyperKvasir23 gate.  It
uses only visual memory prototypes already available in the run; it is not the
original semantic MPC module.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F


def load_semantic_prior(
    *,
    json_path: str | Path | None = None,
    csv_path: str | Path | None = None,
    current_class_ids: Sequence[int],
    old_class_ids: Sequence[int],
) -> torch.Tensor | None:
    """Load an optional current-to-old prior matrix without network access.

    JSON format:
      {"current_class_id": {"old_class_id": weight, ...}, ...}

    CSV format:
      current_class,old_class,weight
    """

    if not json_path and not csv_path:
        return None
    current_index = {int(class_id): idx for idx, class_id in enumerate(current_class_ids)}
    old_index = {int(class_id): idx for idx, class_id in enumerate(old_class_ids)}
    prior = torch.zeros((len(current_class_ids), len(old_class_ids)), dtype=torch.float32)

    if json_path:
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("semantic prior JSON must be a mapping of current class to old-class weights")
        for current_key, old_weights in payload.items():
            current_class = int(current_key)
            if current_class not in current_index:
                continue
            if not isinstance(old_weights, Mapping):
                raise ValueError("semantic prior JSON values must be old-class weight mappings")
            for old_key, weight in old_weights.items():
                old_class = int(old_key)
                if old_class in old_index:
                    prior[current_index[current_class], old_index[old_class]] = float(weight)

    if csv_path:
        with Path(csv_path).open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"current_class", "old_class", "weight"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("semantic prior CSV must contain current_class, old_class, weight columns")
            for row in reader:
                current_class = int(row["current_class"])
                old_class = int(row["old_class"])
                if current_class in current_index and old_class in old_index:
                    prior[current_index[current_class], old_index[old_class]] = float(row["weight"])

    return prior.clamp_min(0.0)


def compute_mpc_lite_calibration(
    current_prototypes: torch.Tensor,
    old_prototypes: torch.Tensor,
    *,
    current_class_ids: Sequence[int],
    old_class_ids: Sequence[int],
    alpha: float = 0.6,
    topk: int = 5,
    tau: float = 16.0,
    semantic_prior: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor], List[Dict[str, object]], List[Dict[str, object]]]:
    """Calibrate current prototypes with visual-memory MPC-lite.

    The top-k search is cosine-based.  If a semantic prior is supplied, it is
    used only as an additive log-prior over the same local old-class candidates.
    """

    if current_prototypes.ndim != 2 or old_prototypes.ndim != 2:
        raise ValueError("current_prototypes and old_prototypes must be 2D tensors")
    if current_prototypes.shape[1] != old_prototypes.shape[1]:
        raise ValueError("current and old prototypes must have the same feature dimension")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if int(topk) <= 0:
        raise ValueError("topk must be positive")
    k = min(int(topk), int(old_prototypes.shape[0]))

    current = torch.nan_to_num(current_prototypes.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    old = torch.nan_to_num(old_prototypes.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    current_norm = F.normalize(current, p=2, dim=1)
    old_norm = F.normalize(old, p=2, dim=1)
    cosine = current_norm @ old_norm.transpose(0, 1)
    logits = float(tau) * cosine

    prior_used = False
    if semantic_prior is not None:
        prior = torch.nan_to_num(semantic_prior.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        if prior.shape != logits.shape:
            raise ValueError("semantic_prior shape must be [num_current, num_old]")
        row_sum = prior.sum(dim=1, keepdim=True)
        uniform = torch.full_like(prior, 1.0 / max(int(prior.shape[1]), 1))
        normalized = torch.where(row_sum > 0, prior / row_sum.clamp_min(1e-12), uniform)
        logits = logits + torch.log(normalized.clamp_min(1e-12))
        prior_used = True

    top_values, top_indices = torch.topk(logits, k=k, dim=1)
    weights = torch.softmax(top_values, dim=1)
    memory = torch.sum(old[top_indices] * weights.unsqueeze(-1), dim=1)
    calibrated = float(alpha) * current + (1.0 - float(alpha)) * memory

    topk_rows: List[Dict[str, object]] = []
    stats_rows: List[Dict[str, object]] = []
    for row_idx, current_class in enumerate(current_class_ids):
        raw = current[row_idx]
        cal = calibrated[row_idx]
        shift = cal - raw
        raw_norm = raw.norm().item()
        cal_norm = cal.norm().item()
        memory_norm = memory[row_idx].norm().item()
        raw_to_cal_cos = float(F.cosine_similarity(raw.view(1, -1), cal.view(1, -1)).item())
        stats_rows.append(
            {
                "current_class": int(current_class),
                "alpha": float(alpha),
                "topk": int(k),
                "tau": float(tau),
                "semantic_prior_used": bool(prior_used),
                "raw_norm": float(raw_norm),
                "calibrated_norm": float(cal_norm),
                "memory_norm": float(memory_norm),
                "shift_l2": float(shift.norm().item()),
                "raw_to_calibrated_cosine": raw_to_cal_cos,
                "top1_old_class": int(old_class_ids[int(top_indices[row_idx, 0].item())]),
                "top1_cosine": float(cosine[row_idx, top_indices[row_idx, 0]].item()),
                "top1_weight": float(weights[row_idx, 0].item()),
            }
        )
        for rank in range(k):
            old_idx = int(top_indices[row_idx, rank].item())
            topk_rows.append(
                {
                    "current_class": int(current_class),
                    "rank": int(rank + 1),
                    "old_class": int(old_class_ids[old_idx]),
                    "cosine": float(cosine[row_idx, old_idx].item()),
                    "logit": float(top_values[row_idx, rank].item()),
                    "weight": float(weights[row_idx, rank].item()),
                    "semantic_prior_used": bool(prior_used),
                }
            )

    state = {
        "cosine": cosine,
        "top_indices": top_indices,
        "top_logits": top_values,
        "weights": weights,
        "memory": memory,
    }
    return calibrated, state, stats_rows, topk_rows


def calibrate_current_stds(
    current_stds: torch.Tensor,
    old_stds: torch.Tensor,
    *,
    top_indices: torch.Tensor,
    weights: torch.Tensor,
    gamma: float = 0.6,
) -> tuple[torch.Tensor, List[Dict[str, object]]]:
    """Calibrate diagonal std vectors with weighted old-class memory."""

    current = torch.nan_to_num(current_stds.detach().float().cpu(), nan=1e-6, posinf=1e-6, neginf=1e-6).clamp_min(1e-6)
    old = torch.nan_to_num(old_stds.detach().float().cpu(), nan=1e-6, posinf=1e-6, neginf=1e-6).clamp_min(1e-6)
    selected = old[top_indices.detach().cpu().long()]
    weighted_old = torch.sum(selected * weights.detach().cpu().float().unsqueeze(-1), dim=1)
    calibrated = (float(gamma) * (current + weighted_old)).clamp_min(1e-6)
    rows: List[Dict[str, object]] = []
    for idx in range(current.shape[0]):
        rows.append(
            {
                "current_row": int(idx),
                "gamma": float(gamma),
                "raw_std_mean": float(current[idx].mean().item()),
                "memory_std_mean": float(weighted_old[idx].mean().item()),
                "calibrated_std_mean": float(calibrated[idx].mean().item()),
                "raw_std_l2": float(current[idx].norm().item()),
                "memory_std_l2": float(weighted_old[idx].norm().item()),
                "calibrated_std_l2": float(calibrated[idx].norm().item()),
            }
        )
    return calibrated, rows
