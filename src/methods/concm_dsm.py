"""Original-ConCM-style Dynamic Structure Matching utilities.

This module is intentionally isolated from the main training engine.  It works
on frozen-backbone feature tensors and trains only a small projector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class DSMProjector(nn.Module):
    """Two-layer ConCM-style projector: normalize, MLP, normalize."""

    def __init__(self, input_dim: int, hidden_dim: int = 2048, output_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.projector = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=-1)
        x = self.projector(x)
        return F.normalize(x, p=2, dim=-1)

    def config(self) -> Dict[str, int]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
        }


def compute_dynamic_structure(projected_prototypes: torch.Tensor) -> tuple[torch.Tensor, Dict[str, object]]:
    """Build ConCM ETF-like geometry vectors with torch.linalg.svd.

    This follows the official helper:
    M = I - 1/N * 11^T
    orth_vec = U @ Vh for SVD(prototypes.T @ M)
    geometry = sqrt(N/(N-1)) * (orth_vec @ M).T
    """

    if projected_prototypes.ndim != 2:
        raise ValueError("projected_prototypes must be a 2D tensor [num_classes, proj_dim]")
    num_classes, proj_dim = projected_prototypes.shape
    if int(num_classes) < 2:
        raise ValueError("compute_dynamic_structure requires at least two classes")
    if int(proj_dim) <= int(num_classes):
        raise ValueError(
            "compute_dynamic_structure requires projection dimension dg > number of classes N "
            f"(got dg={int(proj_dim)}, N={int(num_classes)})"
        )
    device = projected_prototypes.device
    dtype = projected_prototypes.dtype
    prototypes = F.normalize(torch.nan_to_num(projected_prototypes), p=2, dim=1)
    eye = torch.eye(num_classes, device=device, dtype=dtype)
    mean = torch.full((num_classes, num_classes), 1.0 / float(num_classes), device=device, dtype=dtype)
    centering = eye - mean
    centered = prototypes.transpose(0, 1) @ centering
    u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    orth_vec = F.normalize(u @ vh, p=2, dim=0)
    geometry = (float(num_classes) / float(num_classes - 1)) ** 0.5 * (orth_vec @ centering).transpose(0, 1)
    geometry = F.normalize(torch.nan_to_num(geometry), p=2, dim=1)

    dots = geometry @ geometry.transpose(0, 1)
    diag = torch.diagonal(dots)
    offdiag_mask = ~torch.eye(num_classes, device=device, dtype=torch.bool)
    offdiag = dots[offdiag_mask]
    target = -1.0 / float(num_classes - 1)
    gram_target = (
        float(num_classes) / float(num_classes - 1) * eye
        - torch.ones_like(eye) / float(num_classes - 1)
    )
    etf_residual = torch.linalg.vector_norm(dots - gram_target) / torch.linalg.vector_norm(gram_target).clamp_min(1e-12)
    tolerance = max(centered.shape) * torch.finfo(singular_values.dtype).eps * singular_values.max().clamp_min(1e-12)
    nonzero = singular_values[singular_values > tolerance]
    probabilities = singular_values / singular_values.sum().clamp_min(1e-12)
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    stats: Dict[str, object] = {
        "num_classes": int(num_classes),
        "proj_dim": int(proj_dim),
        "etf_residual": float(etf_residual.detach().cpu().item()),
        "pairwise_diag_mean": float(diag.mean().detach().cpu().item()),
        "pairwise_offdiag_mean": float(offdiag.mean().detach().cpu().item()) if offdiag.numel() else 0.0,
        "pairwise_offdiag_target": target,
        "max_abs_offdiag_error": float((offdiag - target).abs().max().detach().cpu().item()) if offdiag.numel() else 0.0,
        "singular_values": [float(value) for value in singular_values.detach().cpu().tolist()],
        "largest_singular_value": float(singular_values.max().detach().cpu().item()),
        "smallest_nonzero_singular_value": float(nonzero.min().detach().cpu().item()) if nonzero.numel() else 0.0,
        "numerical_rank": int(nonzero.numel()),
        "effective_rank": float(effective_rank.detach().cpu().item()),
        "condition_number": (
            float((singular_values.max() / nonzero.min()).detach().cpu().item()) if nonzero.numel() else float("inf")
        ),
        "singular_value_gap": (
            float((nonzero[-2] - nonzero[-1]).detach().cpu().item()) if nonzero.numel() >= 2 else None
        ),
        "rank_tolerance": float(tolerance.detach().cpu().item()),
    }
    return geometry, stats


class GaussianFeatureDataset(Dataset):
    """Feature-level Gaussian augmentation with head-index labels."""

    def __init__(
        self,
        means: torch.Tensor,
        stds: torch.Tensor,
        labels: Sequence[int] | torch.Tensor | None = None,
        sample_num_old: int = 100,
        sample_num_current: int = 50,
        num_old_classes: int | None = None,
        old_class_ids: Iterable[int] | None = None,
        real_features: torch.Tensor | None = None,
        real_labels: Sequence[int] | torch.Tensor | None = None,
        seed: int = 0,
    ) -> None:
        if means.ndim != 2 or stds.ndim != 2:
            raise ValueError("means and stds must be 2D tensors")
        if means.shape != stds.shape:
            raise ValueError("means and stds must have matching shapes")
        self.means = torch.nan_to_num(means.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        self.stds = torch.nan_to_num(stds.detach().float().cpu(), nan=1e-6, posinf=1e-6, neginf=1e-6).clamp_min(1e-6)
        if labels is None:
            label_tensor = torch.arange(self.means.shape[0], dtype=torch.long)
        else:
            label_tensor = torch.as_tensor(labels, dtype=torch.long).detach().cpu()
        if int(label_tensor.numel()) != int(self.means.shape[0]):
            raise ValueError("labels length must match means rows")
        if num_old_classes is None:
            if old_class_ids is None:
                num_old_classes = int(self.means.shape[0])
            else:
                num_old_classes = len(list(old_class_ids))
        self.num_old_classes = int(num_old_classes)

        generator = torch.Generator()
        generator.manual_seed(int(seed))
        feature_rows: List[torch.Tensor] = []
        label_rows: List[torch.Tensor] = []
        for row, head_label in enumerate(label_tensor.tolist()):
            # Explicit head-index split; avoid official label <= label_base off-by-one.
            sample_count = int(sample_num_old) if int(head_label) < self.num_old_classes else int(sample_num_current)
            if sample_count <= 0:
                continue
            samples = torch.normal(
                mean=self.means[row].unsqueeze(0).expand(sample_count, -1),
                std=self.stds[row].unsqueeze(0).expand(sample_count, -1),
                generator=generator,
            )
            feature_rows.append(torch.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0))
            label_rows.append(torch.full((sample_count,), int(head_label), dtype=torch.long))

        if real_features is not None:
            if real_labels is None:
                raise ValueError("real_labels must be provided with real_features")
            real_x = torch.nan_to_num(real_features.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
            real_y = torch.as_tensor(real_labels, dtype=torch.long).detach().cpu()
            if int(real_x.shape[0]) != int(real_y.numel()):
                raise ValueError("real_features rows must match real_labels length")
            feature_rows.append(real_x)
            label_rows.append(real_y)

        if not feature_rows:
            raise ValueError("GaussianFeatureDataset has no samples")
        self.features = torch.cat(feature_rows, dim=0)
        self.labels = torch.cat(label_rows, dim=0)

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


@dataclass
class FeatureClassStats:
    features: torch.Tensor
    original_labels: torch.Tensor
    head_labels: torch.Tensor
    means_by_head: Dict[int, torch.Tensor]
    stds_by_head: Dict[int, torch.Tensor]
    counts_by_head: Dict[int, int]
    class_to_head: Dict[int, int]
    head_to_class: Dict[int, int]


@torch.no_grad()
def extract_features_by_class(
    model: nn.Module,
    dataloader: DataLoader,
    class_to_head: Mapping[int, int],
    device: torch.device | str,
) -> FeatureClassStats:
    """Extract frozen-backbone features and summarize them by head label."""

    if not hasattr(model, "extract_features"):
        raise RuntimeError("Checkpoint model does not expose extract_features(images); check src/models/resnet_cifar.py.")
    device = torch.device(device)
    model.eval()
    features_list: List[torch.Tensor] = []
    original_list: List[torch.Tensor] = []
    head_list: List[torch.Tensor] = []
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        features = model.extract_features(images)
        mapped = []
        for value in labels.detach().cpu().tolist():
            class_id = int(value)
            if class_id not in class_to_head:
                raise KeyError(f"Label {class_id} is not present in class_to_head")
            mapped.append(int(class_to_head[class_id]))
        features_list.append(torch.nan_to_num(features.detach().cpu().float(), nan=0.0, posinf=0.0, neginf=0.0))
        original_list.append(labels.detach().cpu().long())
        head_list.append(torch.tensor(mapped, dtype=torch.long))

    if not features_list:
        raise ValueError("No features were extracted; dataloader is empty")
    features_all = torch.cat(features_list, dim=0)
    original_all = torch.cat(original_list, dim=0)
    head_all = torch.cat(head_list, dim=0)
    means_by_head: Dict[int, torch.Tensor] = {}
    stds_by_head: Dict[int, torch.Tensor] = {}
    counts_by_head: Dict[int, int] = {}
    for head in sorted(set(head_all.tolist())):
        mask = head_all == int(head)
        values = features_all[mask]
        counts_by_head[int(head)] = int(values.shape[0])
        means_by_head[int(head)] = values.mean(dim=0)
        stds_by_head[int(head)] = values.std(dim=0, unbiased=False).clamp_min(1e-6)
    head_to_class = {int(head): int(class_id) for class_id, head in class_to_head.items()}
    return FeatureClassStats(
        features=features_all,
        original_labels=original_all,
        head_labels=head_all,
        means_by_head=means_by_head,
        stds_by_head=stds_by_head,
        counts_by_head=counts_by_head,
        class_to_head={int(k): int(v) for k, v in class_to_head.items()},
        head_to_class=head_to_class,
    )


def supervised_contrastive_with_anchors(
    z: torch.Tensor,
    labels: torch.Tensor,
    geometry_anchors: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Stable anchor-augmented supervised contrastive loss.

    The original batch samples are anchors for the loss.  Geometry vectors are
    added to the contrast set as same-label positives, so every class with an
    anchor can contribute even when the minibatch has one sample per class.
    """

    if z.numel() == 0:
        return z.new_zeros(())
    z = F.normalize(z, p=2, dim=1)
    anchors = F.normalize(geometry_anchors.detach(), p=2, dim=1).to(device=z.device, dtype=z.dtype)
    labels = labels.to(device=z.device, dtype=torch.long)
    anchor_labels = torch.arange(anchors.shape[0], device=z.device, dtype=torch.long)
    contrast = torch.cat([z, anchors], dim=0)
    contrast_labels = torch.cat([labels, anchor_labels], dim=0)
    logits = (z @ contrast.transpose(0, 1)) / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    positive_mask = labels.view(-1, 1).eq(contrast_labels.view(1, -1))
    self_mask = torch.zeros_like(positive_mask)
    self_mask[:, : z.shape[0]] = torch.eye(z.shape[0], device=z.device, dtype=torch.bool)
    positive_mask = positive_mask & ~self_mask
    logits_mask = ~self_mask
    exp_logits = torch.exp(logits) * logits_mask.to(dtype=z.dtype)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not bool(valid.any()):
        return z.new_zeros(())
    mean_log_prob = (positive_mask.to(dtype=z.dtype) * log_prob).sum(dim=1)[valid] / positive_count[valid].to(dtype=z.dtype)
    return -mean_log_prob.mean()


def train_dsm_projector(
    means: torch.Tensor,
    stds: torch.Tensor,
    projector: DSMProjector,
    epochs: int,
    lr: float,
    batch_size: int,
    sample_num_old: int,
    sample_num_current: int,
    num_old_classes: int,
    cont_weight: float = 1.0,
    no_cont_loss: bool = False,
    real_features: torch.Tensor | None = None,
    real_labels: torch.Tensor | None = None,
    labels: Sequence[int] | torch.Tensor | None = None,
    device: torch.device | str = "cpu",
    max_train_batches: int | None = None,
    seed: int = 0,
) -> tuple[DSMProjector, torch.Tensor, List[Dict[str, float | int]], Dict[str, float | int]]:
    """Train DSM projector on feature-level Gaussian augmentation."""

    device = torch.device(device)
    means = torch.nan_to_num(means.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    stds = torch.nan_to_num(stds.detach().float(), nan=1e-6, posinf=1e-6, neginf=1e-6).clamp_min(1e-6)
    label_tensor = torch.arange(means.shape[0], dtype=torch.long) if labels is None else torch.as_tensor(labels, dtype=torch.long)
    dataset = GaussianFeatureDataset(
        means=means,
        stds=stds,
        labels=label_tensor,
        sample_num_old=int(sample_num_old),
        sample_num_current=int(sample_num_current),
        num_old_classes=int(num_old_classes),
        real_features=real_features,
        real_labels=real_labels,
        seed=int(seed),
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    projector.to(device)
    optimizer = torch.optim.SGD([{"params": projector.parameters()}], lr=float(lr), momentum=0.9, weight_decay=1e-4)
    trace: List[Dict[str, float | int]] = []
    labels_device = label_tensor.to(device=device)
    means_device = means.to(device=device)

    for epoch in range(int(epochs)):
        projector.train()
        with torch.no_grad():
            projected_means = projector(means_device)
            geometry, geometry_stats = compute_dynamic_structure(projected_means)
            geometry = geometry.detach()
        sample_total = 0
        correct = 0
        loss_sum = 0.0
        match_sum = 0.0
        cont_sum = 0.0
        batches = 0
        for features, head_labels in loader:
            if max_train_batches is not None and batches >= int(max_train_batches):
                break
            features = features.to(device=device, dtype=torch.float32)
            head_labels = head_labels.to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            z = projector(features)
            logits = z @ geometry.transpose(0, 1)
            match_loss = F.cross_entropy(logits, head_labels)
            if bool(no_cont_loss) or float(cont_weight) <= 0.0:
                cont_loss = match_loss.new_zeros(())
            else:
                cont_loss = supervised_contrastive_with_anchors(z, head_labels, geometry)
            loss = match_loss + float(cont_weight) * cont_loss
            loss.backward()
            optimizer.step()

            batch = int(head_labels.numel())
            sample_total += batch
            correct += int(logits.detach().argmax(dim=1).eq(head_labels).sum().item())
            loss_sum += float(loss.detach().cpu().item()) * batch
            match_sum += float(match_loss.detach().cpu().item()) * batch
            cont_sum += float(cont_loss.detach().cpu().item()) * batch
            batches += 1

        trace.append(
            {
                "epoch": int(epoch + 1),
                "loss": loss_sum / max(sample_total, 1),
                "match_loss": match_sum / max(sample_total, 1),
                "cont_loss": cont_sum / max(sample_total, 1),
                "train_acc": 100.0 * float(correct) / float(max(sample_total, 1)),
                "samples": int(sample_total),
                "batches": int(batches),
                "geometry_offdiag_error": float(geometry_stats["max_abs_offdiag_error"]),
                "geometry_offdiag_mean": float(geometry_stats["pairwise_offdiag_mean"]),
                "num_classes": int(labels_device.numel()),
            }
        )

    projector.eval()
    with torch.no_grad():
        final_geometry, final_stats = compute_dynamic_structure(projector(means_device))
    return projector, final_geometry.detach(), trace, final_stats


@torch.no_grad()
def evaluate_dsm(
    features: torch.Tensor,
    head_labels: torch.Tensor,
    projector: DSMProjector,
    geometry_vectors: torch.Tensor,
    old_head_indices: Sequence[int],
    current_head_indices: Sequence[int],
    device: torch.device | str = "cpu",
) -> Dict[str, object]:
    """Evaluate projected features with the geometric classifier."""

    device = torch.device(device)
    projector.eval()
    features = features.to(device=device, dtype=torch.float32)
    head_labels = head_labels.to(device=device, dtype=torch.long)
    geometry = F.normalize(geometry_vectors.to(device=device, dtype=torch.float32), p=2, dim=1)
    logits = projector(features) @ geometry.transpose(0, 1)
    preds = logits.argmax(dim=1)
    matches = preds.eq(head_labels)
    old_set = {int(item) for item in old_head_indices}
    current_set = {int(item) for item in current_head_indices}
    old_mask = torch.tensor([int(item) in old_set for item in head_labels.detach().cpu().tolist()], device=device)
    current_mask = torch.tensor([int(item) in current_set for item in head_labels.detach().cpu().tolist()], device=device)
    pred_old_mask = torch.tensor([int(item) in old_set for item in preds.detach().cpu().tolist()], device=device)
    pred_current_mask = torch.tensor([int(item) in current_set for item in preds.detach().cpu().tolist()], device=device)
    num_classes = int(geometry.shape[0])
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for label, pred in zip(head_labels.detach().cpu().tolist(), preds.detach().cpu().tolist()):
        confusion[int(label), int(pred)] += 1

    per_class: List[Dict[str, float | int | None]] = []
    recalls: List[float] = []
    f1_values: List[float] = []
    for head in range(num_classes):
        total = int(confusion[head].sum().item())
        correct = int(confusion[head, head].item())
        predicted = int(confusion[:, head].sum().item())
        precision = 100.0 * correct / predicted if predicted else 0.0
        recall = 100.0 * correct / total if total else None
        f1 = 0.0
        if recall is not None and precision + recall > 0.0:
            f1 = 2.0 * precision * recall / (precision + recall)
        if recall is not None:
            recalls.append(recall)
            f1_values.append(f1)
        per_class.append(
            {
                "head_idx": int(head),
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "correct": correct,
                "total": total,
                "predicted": predicted,
            }
        )

    def pct(numerator: int, denominator: int) -> float | None:
        return 100.0 * float(numerator) / float(denominator) if denominator > 0 else None

    total = int(head_labels.numel())
    old_total = int(old_mask.sum().item())
    current_total = int(current_mask.sum().item())
    old_correct = int(matches[old_mask].sum().item())
    current_correct = int(matches[current_mask].sum().item())
    return {
        "AccT": pct(int(matches.sum().item()), total),
        "balanced_acc": sum(recalls) / len(recalls) if recalls else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "old_acc": pct(old_correct, old_total),
        "current_acc": pct(current_correct, current_total),
        "old_to_current_rate": pct(int((old_mask & pred_current_mask).sum().item()), old_total),
        "current_to_old_rate": pct(int((current_mask & pred_old_mask).sum().item()), current_total),
        "total": total,
        "old_total": old_total,
        "current_total": current_total,
        "confusion_matrix": confusion,
        "per_class": per_class,
    }
