"""Tail-aware ConCM-style prototype calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
from torch.nn import functional as F


@dataclass
class PrototypeMemoryEntry:
    class_id: int
    proto_raw: torch.Tensor
    proto_calib: torch.Tensor
    count: int
    uncertainty: float
    session_id: int
    anchor: torch.Tensor | None = None


@dataclass
class CalibrationResult:
    proto_calib: torch.Tensor
    alpha: torch.Tensor
    uncertainty: torch.Tensor
    memory_used: torch.Tensor
    raw_calib_cosine: torch.Tensor


class PrototypeMemory:
    def __init__(self) -> None:
        self.entries: Dict[int, PrototypeMemoryEntry] = {}

    def __contains__(self, class_id: int) -> bool:
        return int(class_id) in self.entries

    def update(
        self,
        class_ids: Sequence[int],
        proto_raw: torch.Tensor,
        proto_calib: torch.Tensor,
        counts: torch.Tensor,
        uncertainty: torch.Tensor,
        session_id: int,
        anchors: torch.Tensor | None = None,
    ) -> None:
        for row, class_id in enumerate(class_ids):
            anchor = None if anchors is None else anchors[row].detach().cpu()
            self.entries[int(class_id)] = PrototypeMemoryEntry(
                class_id=int(class_id),
                proto_raw=F.normalize(proto_raw[row].detach().cpu(), dim=0),
                proto_calib=F.normalize(proto_calib[row].detach().cpu(), dim=0),
                count=int(counts[row].detach().cpu().item()),
                uncertainty=float(uncertainty[row].detach().cpu().item()),
                session_id=int(session_id),
                anchor=anchor,
            )

    def class_ids(self) -> List[int]:
        return sorted(self.entries)

    def tensors(
        self,
        class_ids: Iterable[int] | None = None,
        device: torch.device | None = None,
    ) -> tuple[List[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = self.class_ids() if class_ids is None else [int(class_id) for class_id in class_ids]
        ids = [class_id for class_id in ids if class_id in self.entries]
        if not ids:
            empty = torch.empty(0, device=device)
            return [], empty, empty, empty, empty
        protos = torch.stack([self.entries[class_id].proto_calib for class_id in ids]).to(device=device)
        raw = torch.stack([self.entries[class_id].proto_raw for class_id in ids]).to(device=device)
        counts = torch.tensor([self.entries[class_id].count for class_id in ids], dtype=torch.float32, device=device)
        uncertainty = torch.tensor(
            [self.entries[class_id].uncertainty for class_id in ids],
            dtype=torch.float32,
            device=device,
        )
        return ids, raw, protos, counts, uncertainty

    def anchors(
        self,
        class_ids: Sequence[int],
        device: torch.device,
    ) -> torch.Tensor:
        values = []
        for class_id in class_ids:
            entry = self.entries[int(class_id)]
            anchor = entry.anchor if entry.anchor is not None else entry.proto_calib
            values.append(anchor)
        return F.normalize(torch.stack(values).to(device=device), dim=1)

    def set_anchors(self, class_ids: Sequence[int], anchors: torch.Tensor) -> None:
        for row, class_id in enumerate(class_ids):
            if int(class_id) in self.entries:
                self.entries[int(class_id)].anchor = F.normalize(anchors[row].detach().cpu(), dim=0)

    def state_dict(self) -> Dict[str, object]:
        return {
            str(class_id): {
                "class_id": entry.class_id,
                "proto_raw": entry.proto_raw,
                "proto_calib": entry.proto_calib,
                "count": entry.count,
                "uncertainty": entry.uncertainty,
                "session_id": entry.session_id,
                "anchor": entry.anchor,
            }
            for class_id, entry in sorted(self.entries.items())
        }


class ReliabilityAwarePrototypeCalibrator:
    def __init__(
        self,
        top_k_memory: int = 5,
        alpha_a: float = 2.0,
        alpha_b: float = 1.0,
        alpha_min: float = 0.15,
        alpha_max: float = 0.95,
        eps: float = 1e-8,
    ) -> None:
        self.top_k_memory = int(top_k_memory)
        self.alpha_a = float(alpha_a)
        self.alpha_b = float(alpha_b)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.eps = float(eps)

    def _uncertainty(self, proto_var: torch.Tensor | None, rows: int, device: torch.device) -> torch.Tensor:
        if proto_var is None:
            return torch.zeros(rows, dtype=torch.float32, device=device)
        values = proto_var.to(device=device, dtype=torch.float32)
        if values.ndim > 1:
            values = values.mean(dim=1)
        values = torch.clamp(values, min=0.0)
        max_value = values.max().clamp_min(self.eps)
        return torch.clamp(values / max_value, min=0.0, max=1.0)

    def calibrate(
        self,
        proto_raw: torch.Tensor,
        class_counts: torch.Tensor,
        proto_var: torch.Tensor | None,
        prototype_memory: PrototypeMemory,
        class_ids: Sequence[int],
    ) -> CalibrationResult:
        if proto_raw.ndim != 2:
            raise ValueError("proto_raw must be a 2D tensor")
        device = proto_raw.device
        proto_visual = F.normalize(proto_raw, dim=1)
        counts = class_counts.to(device=device, dtype=torch.float32).clamp_min(0)
        uncertainty = self._uncertainty(proto_var, proto_raw.shape[0], device)
        log_counts = torch.log(counts + 1.0)
        median_log_count = torch.median(log_counts)
        alpha = torch.sigmoid(self.alpha_a * (log_counts - median_log_count) - self.alpha_b * uncertainty)
        alpha = torch.clamp(alpha, min=self.alpha_min, max=self.alpha_max)

        memory_ids, _, memory_protos, memory_counts, memory_unc = prototype_memory.tensors(device=device)
        calibrated: List[torch.Tensor] = []
        memory_used: List[bool] = []
        memory_id_tensor = torch.tensor(memory_ids, dtype=torch.long, device=device) if memory_ids else None
        memory_rel = None
        if memory_ids:
            memory_log_counts = torch.log(memory_counts + 1.0)
            memory_rel = memory_log_counts / memory_log_counts.max().clamp_min(self.eps)
            memory_rel = memory_rel * (1.0 - memory_unc.clamp(0.0, 1.0))

        for row, class_id in enumerate(class_ids):
            p_v = proto_visual[row]
            p_m = p_v
            used = False
            if memory_ids and memory_id_tensor is not None and memory_rel is not None:
                candidate_mask = memory_id_tensor != int(class_id)
                if torch.any(candidate_mask):
                    candidates = memory_protos[candidate_mask]
                    rel = memory_rel[candidate_mask].clamp_min(self.eps)
                    sims = torch.mv(candidates, p_v)
                    scores = sims * rel
                    k = min(self.top_k_memory, int(scores.numel()))
                    if k > 0:
                        top_scores, top_indices = torch.topk(scores, k=k)
                        weights = F.softmax(top_scores, dim=0)
                        p_m = F.normalize(torch.sum(weights[:, None] * candidates[top_indices], dim=0), dim=0)
                        used = True
            mixed = alpha[row] * p_v + (1.0 - alpha[row]) * p_m
            calibrated.append(F.normalize(mixed, dim=0))
            memory_used.append(used)

        proto_calib = torch.stack(calibrated, dim=0)
        raw_calib_cosine = torch.sum(proto_visual * proto_calib, dim=1)
        return CalibrationResult(
            proto_calib=proto_calib,
            alpha=alpha,
            uncertainty=uncertainty,
            memory_used=torch.tensor(memory_used, dtype=torch.bool, device=device),
            raw_calib_cosine=raw_calib_cosine,
        )


def tail_anchor_weights(
    class_counts: torch.Tensor,
    gamma: float = 0.5,
    w_max: float = 5.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    counts = class_counts.to(dtype=torch.float32).clamp_min(0)
    n_max = counts.max().clamp_min(eps)
    weights = ((n_max + eps) / (counts + eps)) ** float(gamma)
    weights = weights / weights.mean().clamp_min(eps)
    return torch.clamp(weights, min=0.5, max=float(w_max))


def _tailness(class_counts: torch.Tensor, beta: float, eps: float) -> torch.Tensor:
    counts = class_counts.to(dtype=torch.float32).clamp_min(0)
    n_max = counts.max().clamp_min(eps)
    raw = ((n_max + eps) / (counts + eps)) ** float(beta)
    span = raw.max() - raw.min()
    if float(span.detach().cpu()) <= eps:
        return torch.zeros_like(raw)
    return (raw - raw.min()) / span.clamp_min(eps)


def build_tail_aware_structure_anchors(
    proto_calib: torch.Tensor,
    class_counts: torch.Tensor,
    old_anchors: torch.Tensor | None = None,
    old_class_mask: torch.Tensor | None = None,
    steps: int = 100,
    lr: float = 0.05,
    margin_base: float = 0.2,
    margin_tail: float = 0.2,
    beta: float = 0.5,
    lambda_margin: float = 1.0,
    lambda_old: float = 1.0,
    lambda_tail: float = 1.0,
    lambda_old_class: float = 0.5,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, Dict[str, float]]:
    proto = F.normalize(proto_calib.detach(), dim=1)
    if proto.shape[0] <= 1 or steps <= 0:
        return proto, {"tdsm_align_loss": 0.0, "tdsm_margin_loss": 0.0, "tdsm_old_loss": 0.0}

    device = proto.device
    tail = _tailness(class_counts.to(device=device), beta=beta, eps=eps)
    class_weight = 1.0 + float(lambda_tail) * tail
    if old_class_mask is not None:
        class_weight = class_weight + old_class_mask.to(device=device, dtype=torch.float32) * float(lambda_old_class)

    anchors = proto.clone().detach().requires_grad_(True)
    optimizer = torch.optim.SGD([anchors], lr=float(lr))
    diag = {"tdsm_align_loss": 0.0, "tdsm_margin_loss": 0.0, "tdsm_old_loss": 0.0}

    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        anchors_norm = F.normalize(anchors, dim=1)
        align = torch.mean(class_weight * (1.0 - torch.sum(anchors_norm * proto, dim=1)))

        sim = anchors_norm @ anchors_norm.T
        dist = 1.0 - sim
        margin = float(margin_base) + float(margin_tail) * (tail[:, None] + tail[None, :]) / 2.0
        pair_mask = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)
        margin_loss = torch.relu(margin[pair_mask] - dist[pair_mask]).pow(2).mean()

        old_loss = torch.zeros((), device=device)
        if old_anchors is not None and old_class_mask is not None and torch.any(old_class_mask):
            mask = old_class_mask.to(device=device, dtype=torch.bool)
            old_target = F.normalize(old_anchors.to(device=device).detach(), dim=1)
            old_loss = torch.mean((anchors_norm[mask] - old_target[mask]).pow(2).sum(dim=1))

        loss = align + float(lambda_margin) * margin_loss + float(lambda_old) * old_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            anchors.copy_(F.normalize(anchors, dim=1))
        diag = {
            "tdsm_align_loss": float(align.detach().cpu()),
            "tdsm_margin_loss": float(margin_loss.detach().cpu()),
            "tdsm_old_loss": float(old_loss.detach().cpu()),
        }

    return F.normalize(anchors.detach(), dim=1), diag

