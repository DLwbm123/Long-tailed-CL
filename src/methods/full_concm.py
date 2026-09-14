"""Full-session ConCM-style components for long-tailed medical CIL."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.methods.concm_dsm import DSMProjector, compute_dynamic_structure


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def stable_text_embedding(text: str, dim: int = 64) -> torch.Tensor:
    """Deterministic signed hashing fallback with no external model dependency."""

    vector = torch.zeros(int(dim), dtype=torch.float32)
    tokens = re.findall(r"[a-z0-9]+", text.lower()) or ["unknown"]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for offset in range(0, len(digest), 2):
            index = int.from_bytes(digest[offset : offset + 2], "little") % int(dim)
            sign = 1.0 if digest[offset] & 1 else -1.0
            vector[index] += sign
    return F.normalize(vector, dim=0)


def fallback_attributes(class_name: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", class_name.lower()).strip()
    tokens = normalized.split() or ["unknown"]
    attributes = [f"token:{token}" for token in tokens]
    if len(tokens) > 1:
        attributes.append("phrase:" + "_".join(tokens))
    attributes.extend(["hypernym:medical_finding", "domain:gastrointestinal_endoscopy"])
    return sorted(set(attributes))


def build_attribute_mapping(class_names: Mapping[int, str]) -> tuple[Dict[int, list[str]], Dict[str, object]]:
    mapping = {int(class_id): fallback_attributes(name) for class_id, name in class_names.items()}
    return mapping, {
        "source": "deterministic_class_name_fallback",
        "wordnet_available": False,
        "coverage_rate": 0.0,
        "fallback_rate": 1.0,
        "classes": len(mapping),
    }


def build_visual_attribute_pool(
    base_class_ids: Sequence[int],
    base_means: Mapping[int, torch.Tensor],
    attribute_mapping: Mapping[int, Sequence[str]],
    semantic_dim: int = 64,
) -> tuple[list[str], torch.Tensor, torch.Tensor, Dict[str, list[int]]]:
    associations: Dict[str, list[int]] = {}
    for class_id in base_class_ids:
        for attribute in attribute_mapping[int(class_id)]:
            associations.setdefault(str(attribute), []).append(int(class_id))
    names = sorted(associations)
    visual = torch.stack(
        [torch.stack([base_means[class_id].float() for class_id in associations[name]]).mean(dim=0) for name in names]
    )
    semantic = torch.stack([stable_text_embedding(name, semantic_dim) for name in names])
    return names, visual, semantic, associations


def class_semantic_embedding(attributes: Sequence[str], dim: int = 64) -> torch.Tensor:
    return F.normalize(torch.stack([stable_text_embedding(value, dim) for value in attributes]).mean(dim=0), dim=0)


class MPCNetwork(nn.Module):
    """Prototype completion with joint visual/semantic attribute attention."""

    def __init__(self, feature_dim: int, semantic_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.semantic_dim = int(semantic_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.visual_key = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.semantic_query = nn.Linear(semantic_dim, hidden_dim, bias=False)
        self.semantic_key = nn.Linear(semantic_dim, hidden_dim, bias=False)
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(
        self,
        biased_prototype: torch.Tensor,
        class_semantic: torch.Tensor,
        attribute_visual: torch.Tensor,
        attribute_semantic: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(biased_prototype)
        visual_score = encoded @ self.visual_key(attribute_visual).T / math.sqrt(self.hidden_dim)
        semantic_score = self.semantic_query(class_semantic) @ self.semantic_key(attribute_semantic).T
        semantic_score = semantic_score / math.sqrt(self.hidden_dim)
        attention = torch.softmax(visual_score + semantic_score, dim=-1)
        attribute_encoded = self.encoder(attribute_visual)
        aggregate = attention @ attribute_encoded
        completed = self.decoder(torch.cat([encoded, aggregate], dim=-1))
        return completed, attention

    def config(self) -> Dict[str, int]:
        return {
            "feature_dim": self.feature_dim,
            "semantic_dim": self.semantic_dim,
            "hidden_dim": self.hidden_dim,
        }


@dataclass
class PrototypeRecord:
    class_id: int
    session_id: int
    mean: torch.Tensor
    covariance_diag: torch.Tensor
    raw_mean: torch.Tensor
    frequency: int
    frequency_group: str
    calibrated: bool

    def cpu_dict(self) -> Dict[str, object]:
        return {
            "class_id": int(self.class_id),
            "session_id": int(self.session_id),
            "mean": self.mean.detach().cpu(),
            "covariance_diag": self.covariance_diag.detach().cpu(),
            "raw_mean": self.raw_mean.detach().cpu(),
            "frequency": int(self.frequency),
            "frequency_group": self.frequency_group,
            "calibrated": bool(self.calibrated),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "PrototypeRecord":
        return cls(
            class_id=int(value["class_id"]),
            session_id=int(value["session_id"]),
            mean=torch.as_tensor(value["mean"]).float(),
            covariance_diag=torch.as_tensor(value["covariance_diag"]).float(),
            raw_mean=torch.as_tensor(value["raw_mean"]).float(),
            frequency=int(value["frequency"]),
            frequency_group=str(value["frequency_group"]),
            calibrated=bool(value["calibrated"]),
        )


def train_mpc_episodic(
    network: MPCNetwork,
    base_features: Mapping[int, torch.Tensor],
    class_semantics: Mapping[int, torch.Tensor],
    attribute_visual: torch.Tensor,
    attribute_semantic: torch.Tensor,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
    k_shot: int = 5,
) -> list[Dict[str, float | int]]:
    network.to(device)
    attribute_visual = attribute_visual.to(device)
    attribute_semantic = attribute_semantic.to(device)
    optimizer = torch.optim.SGD(network.parameters(), lr=float(lr), momentum=0.9, weight_decay=1e-4)
    trace = []
    class_ids = sorted(int(value) for value in base_features)
    targets = torch.stack([base_features[class_id].float().mean(dim=0) for class_id in class_ids]).to(device)
    for epoch in range(int(epochs)):
        generator = torch.Generator().manual_seed(int(seed) * 100003 + int(epoch))
        biased = []
        replacement_classes = 0
        for class_id in class_ids:
            values = base_features[class_id].float()
            replacement = values.shape[0] < int(k_shot)
            replacement_classes += int(replacement)
            indices = (
                torch.randint(values.shape[0], (int(k_shot),), generator=generator)
                if replacement
                else torch.randperm(values.shape[0], generator=generator)[: int(k_shot)]
            )
            biased.append(values[indices].mean(dim=0))
        biased_tensor = torch.stack(biased).to(device)
        semantics = torch.stack([class_semantics[class_id] for class_id in class_ids]).to(device)
        optimizer.zero_grad(set_to_none=True)
        completed, attention = network(biased_tensor, semantics, attribute_visual, attribute_semantic)
        loss = F.mse_loss(completed, targets)
        loss.backward()
        optimizer.step()
        trace.append(
            {
                "epoch": epoch + 1,
                "mse": float(loss.detach().cpu().item()),
                "attention_entropy": float((-(attention * attention.clamp_min(1e-12).log()).sum(dim=1).mean()).detach().cpu()),
                "replacement_classes": replacement_classes,
            }
        )
    network.eval()
    return trace


@torch.no_grad()
def calibrate_prototype(
    network: MPCNetwork,
    raw_prototype: torch.Tensor,
    class_semantic: torch.Tensor,
    attribute_visual: torch.Tensor,
    attribute_semantic: torch.Tensor,
    alpha: float,
    device: torch.device,
) -> tuple[torch.Tensor, Dict[str, float]]:
    completed, _ = network(
        raw_prototype.view(1, -1).to(device),
        class_semantic.view(1, -1).to(device),
        attribute_visual.to(device),
        attribute_semantic.to(device),
    )
    completed = completed[0].cpu()
    calibrated = float(alpha) * raw_prototype.cpu() + (1.0 - float(alpha)) * completed
    return calibrated, {
        "displacement": float(torch.linalg.vector_norm(calibrated - raw_prototype.cpu()).item()),
        "raw_calibrated_cosine": float(F.cosine_similarity(raw_prototype.cpu(), calibrated, dim=0).item()),
    }


def blend_current_covariance(
    observed: torch.Tensor,
    calibrated_mean: torch.Tensor,
    base_records: Sequence[PrototypeRecord],
    gamma: float = 16.0,
    beta: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_means = torch.stack([record.mean for record in base_records]).float()
    base_covariances = torch.stack([record.covariance_diag for record in base_records]).float()
    similarities = F.normalize(base_means, dim=1) @ F.normalize(calibrated_mean.float(), dim=0)
    weights = torch.softmax(float(gamma) * similarities, dim=0)
    covariance = float(beta) * (observed.float() + weights @ base_covariances)
    return covariance.clamp_min(1e-8), weights


def resample_repository(
    records: Sequence[PrototypeRecord],
    current_class_ids: Sequence[int],
    seed: int,
    epoch: int,
    sample_num_old: int = 100,
    sample_num_current: int = 50,
) -> tuple[torch.Tensor, torch.Tensor, list[Dict[str, object]]]:
    current = {int(value) for value in current_class_ids}
    feature_rows = []
    label_rows = []
    stats = []
    for head, record in enumerate(records):
        count = int(sample_num_current) if record.class_id in current else int(sample_num_old)
        generator = torch.Generator().manual_seed(int(seed) * 1000003 + int(epoch) * 1009 + int(record.class_id))
        samples = torch.normal(
            record.mean.float().view(1, -1).expand(count, -1),
            record.covariance_diag.float().sqrt().view(1, -1).expand(count, -1),
            generator=generator,
        )
        if not bool(torch.isfinite(samples).all()):
            raise FloatingPointError(f"Non-finite Gaussian sample for class {record.class_id}")
        feature_rows.append(samples)
        label_rows.append(torch.full((count,), int(head), dtype=torch.long))
        stats.append(
            {
                "class_id": record.class_id,
                "head_idx": head,
                "sample_count": count,
                "covariance_norm": float(torch.linalg.vector_norm(record.covariance_diag).item()),
                "sampled_feature_norm": float(torch.linalg.vector_norm(samples, dim=1).mean().item()),
                "finite": True,
                "checksum": tensor_sha256(samples),
            }
        )
    return torch.cat(feature_rows), torch.cat(label_rows), stats


def class_balanced_mean(losses: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    per_class = []
    for class_index in range(int(num_classes)):
        mask = labels == class_index
        if not bool(mask.any()):
            raise ValueError(f"Missing samples for class index {class_index}")
        per_class.append(losses[mask].mean())
    return torch.stack(per_class).mean()


def structure_anchor_contrastive(
    z: torch.Tensor,
    labels: torch.Tensor,
    geometry: torch.Tensor,
    current_heads: Sequence[int],
    temperature: float = 0.07,
) -> torch.Tensor:
    current_set = {int(value) for value in current_heads}
    anchor_indices = torch.tensor(
        [index for index, label in enumerate(labels.tolist()) if int(label) in current_set],
        dtype=torch.long,
        device=z.device,
    )
    if anchor_indices.numel() == 0:
        return z.new_zeros(())
    z = F.normalize(z, dim=1)
    geometry = F.normalize(geometry.detach(), dim=1)
    contrast = torch.cat([z, geometry], dim=0)
    contrast_labels = torch.cat([labels, torch.arange(geometry.shape[0], device=z.device)])
    anchors = z[anchor_indices]
    anchor_labels = labels[anchor_indices]
    logits = anchors @ contrast.T / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.zeros_like(logits, dtype=torch.bool)
    self_mask[torch.arange(anchor_indices.numel(), device=z.device), anchor_indices] = True
    positive = anchor_labels[:, None].eq(contrast_labels[None, :]) & ~self_mask
    denominator = (torch.exp(logits) * (~self_mask)).sum(dim=1).clamp_min(1e-12)
    per_anchor = -(positive * (logits - denominator.log()[:, None])).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
    per_class = []
    for head in sorted(current_set):
        per_class.append(per_anchor[anchor_labels == head].mean())
    return torch.stack(per_class).mean()


def fixed_nc_geometry(total_classes: int, output_dim: int) -> torch.Tensor:
    if int(output_dim) <= int(total_classes):
        raise ValueError("fixed NC geometry requires output_dim > total_classes")
    centering = torch.eye(total_classes) - torch.ones(total_classes, total_classes) / float(total_classes)
    simplex = math.sqrt(float(total_classes) / float(total_classes - 1)) * centering
    output = torch.zeros(total_classes, output_dim)
    output[:, :total_classes] = simplex
    return F.normalize(output, dim=1)


def train_projector_session(
    projector: DSMProjector,
    records: Sequence[PrototypeRecord],
    current_class_ids: Sequence[int],
    fixed_geometry: torch.Tensor,
    mode: str,
    epochs: int,
    lr: float,
    seed: int,
    session: int,
    device: torch.device,
    lambda_dsm: float = 0.2,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, list[Dict[str, object]], list[Dict[str, object]]]:
    if mode not in {"locked_nc", "full_dynamic", "nc_anchored"}:
        raise ValueError(f"Unknown full ConCM mode: {mode}")
    projector.to(device)
    optimizer = torch.optim.SGD(projector.parameters(), lr=float(lr), momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))
    means = torch.stack([record.mean for record in records]).to(device)
    class_to_head = {record.class_id: head for head, record in enumerate(records)}
    current_heads = [class_to_head[int(class_id)] for class_id in current_class_ids]
    trace = []
    augmentation_rows = []
    geometry = None
    for epoch in range(int(epochs)):
        features, labels, sample_stats = resample_repository(records, current_class_ids, seed + session * 100, epoch)
        for row in sample_stats:
            row.update({"session": session, "epoch": epoch + 1})
            augmentation_rows.append(row)
        features = features.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            geometry, geometry_stats = compute_dynamic_structure(projector(means))
            if float(geometry_stats["etf_residual"]) > 1e-4:
                raise RuntimeError(f"ETF residual exceeds gate: {geometry_stats['etf_residual']}")
        optimizer.zero_grad(set_to_none=True)
        z = projector(features)
        dynamic_logits = z @ geometry.T
        dynamic_losses = F.cross_entropy(dynamic_logits, labels, reduction="none")
        match_loss = class_balanced_mean(dynamic_losses, labels, len(records))
        cont_loss = structure_anchor_contrastive(z, labels, geometry, current_heads, temperature)
        nc_logits = z @ fixed_geometry[: len(records)].to(device).T
        nc_losses = F.cross_entropy(nc_logits, labels, reduction="none")
        nc_loss = class_balanced_mean(nc_losses, labels, len(records))
        projector_loss = match_loss + cont_loss
        if mode == "locked_nc":
            loss = nc_loss
        elif mode == "full_dynamic":
            loss = projector_loss
        else:
            loss = nc_loss + float(lambda_dsm) * projector_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        trace.append(
            {
                "session": session,
                "epoch": epoch + 1,
                "loss": float(loss.detach().cpu()),
                "nc_loss": float(nc_loss.detach().cpu()),
                "match_loss": float(match_loss.detach().cpu()),
                "cont_loss": float(cont_loss.detach().cpu()),
                "old_block_weight": float((len(records) - len(current_heads)) / max(len(records), 1)),
                "current_block_weight": float(len(current_heads) / max(len(records), 1)),
                "etf_residual": float(geometry_stats["etf_residual"]),
                "resampling_checksum": hashlib.sha256("".join(row["checksum"] for row in sample_stats).encode()).hexdigest(),
            }
        )
    projector.eval()
    with torch.no_grad():
        geometry, _ = compute_dynamic_structure(projector(means))
    return geometry.detach(), trace, augmentation_rows


def validate_continuity(
    previous_seen: Sequence[int],
    current_seen: Sequence[int],
    previous_mapping: Mapping[int, int],
    current_mapping: Mapping[int, int],
) -> None:
    if len(current_seen) <= len(previous_seen):
        raise RuntimeError("seen class count must increase between incremental sessions")
    if list(current_seen[: len(previous_seen)]) != list(previous_seen):
        raise RuntimeError("previous seen-class order changed")
    for class_id, head in previous_mapping.items():
        if int(current_mapping[int(class_id)]) != int(head):
            raise RuntimeError(f"class mapping changed for class {class_id}")

