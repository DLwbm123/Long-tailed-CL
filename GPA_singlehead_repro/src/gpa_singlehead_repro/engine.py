import csv
import json
import math
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import normalize_rows


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    raise TypeError("Object of type {} is not JSON serializable".format(type(obj).__name__))


def task_sizes_until(taskcla, t):
    return [int(taskcla[i][1]) for i in range(t + 1)]


def class_range_for_task(taskcla, t):
    start = sum(int(taskcla[i][1]) for i in range(t))
    stop = start + int(taskcla[t][1])
    return list(range(start, stop))


def seen_class_count(taskcla, upto_task):
    return int(sum(int(taskcla[i][1]) for i in range(upto_task + 1)))


def task_offsets(taskcla, upto_task):
    sizes = [int(taskcla[i][1]) for i in range(upto_task + 1)]
    return np.cumsum(sizes)


def make_train_loader_with_exemplars(trn_loader, exemplars):
    if exemplars is not None and len(exemplars) > 0:
        dataset = trn_loader.dataset + exemplars
    else:
        dataset = trn_loader.dataset
    return DataLoader(
        dataset,
        batch_size=trn_loader.batch_size,
        shuffle=True,
        num_workers=trn_loader.num_workers,
        pin_memory=trn_loader.pin_memory,
    )


def estimate_prototypes(model, loader, class_ids, device):
    """Estimate global-label prototypes with a frozen/eval model."""
    sums = {class_id: torch.zeros(model.feature_dim, device=device) for class_id in class_ids}
    counts = {class_id: 0 for class_id in class_ids}
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            targets = targets.to(device)
            _, features = model(images.to(device), return_features=True)
            for class_id in class_ids:
                mask = targets == class_id
                if mask.any():
                    sums[class_id] += features[mask].sum(dim=0)
                    counts[class_id] += int(mask.sum().item())
    protos = []
    count_values = []
    for class_id in class_ids:
        if counts[class_id] == 0:
            raise RuntimeError("No samples found for class {} during prototype estimation".format(class_id))
        protos.append(sums[class_id] / counts[class_id])
        count_values.append(counts[class_id])
    return torch.stack(protos, dim=0), torch.tensor(count_values, dtype=torch.float32, device=device)


def count_targets_in_loader(loader, device=None):
    """Count global labels in a loader. This is used only for diagnostics / global bias ablations."""
    counts = defaultdict(int)
    with torch.no_grad():
        for _images, targets in loader:
            for y in targets.detach().cpu().tolist():
                counts[int(y)] += 1
    return dict(counts)


def make_gpa_bias(counts, nref_mode, eps):
    if nref_mode == "max_current":
        n_ref = counts.max()
    elif nref_mode == "mean_current":
        n_ref = counts.mean()
    elif nref_mode == "first_current":
        n_ref = counts[0]
    else:
        raise ValueError("Unsupported nref_mode: {}".format(nref_mode))
    return -torch.log(counts / n_ref + eps), float(n_ref.item())


def make_bias_for_mode(mode, counts, nref_mode, eps, old_bias=None):
    """Return (bias tensor or None, diagnostic dict) for the new rows.

    `paper` implements Eq. 4. `zero` and `old_mean` are diagnostic ambiguity checks.
    `none` leaves the newly allocated classifier bias unchanged.
    """
    diag = {"mode": mode}
    if mode == "paper":
        bias, n_ref = make_gpa_bias(counts, nref_mode, eps)
        diag.update({"n_ref": n_ref, "values": bias.detach().cpu().tolist()})
        return bias, diag
    if mode == "zero":
        bias = torch.zeros_like(counts)
        diag.update({"n_ref": None, "values": bias.detach().cpu().tolist()})
        return bias, diag
    if mode == "old_mean":
        if old_bias is None or old_bias.numel() == 0:
            value = torch.zeros((), device=counts.device)
        else:
            value = old_bias.detach().to(counts.device).mean()
        bias = torch.full_like(counts, float(value.item()))
        diag.update({"n_ref": None, "values": bias.detach().cpu().tolist(), "old_bias_mean": float(value.item())})
        return bias, diag
    if mode == "none":
        diag.update({"n_ref": None, "values": None})
        return None, diag
    raise ValueError("Unsupported gpa bias mode: {}".format(mode))


def _logits_for_eval(model, images, eval_mode="raw", return_features=False):
    logits, features = model(images, return_features=True)
    if eval_mode == "raw":
        out = logits
    else:
        weight = model.classifier.weight
        bias = model.classifier.bias
        if eval_mode == "no_bias":
            out = features.matmul(weight.t())
        elif eval_mode == "norm_weight_eval":
            out = features.matmul(normalize_rows(weight).t())
            if bias is not None:
                out = out + bias
        elif eval_mode == "cosine_eval":
            out = F.normalize(features, p=2, dim=1, eps=1e-12).matmul(normalize_rows(weight).t())
        else:
            raise ValueError("Unsupported eval_mode: {}".format(eval_mode))
    if return_features:
        return out, features
    return out


def evaluate_seen(model, loaders, taskcla, upto_task, device, eval_mode="raw"):
    num_tasks = upto_task + 1
    acc_by_task = np.zeros(num_tasks, dtype=np.float64)
    loss_by_task = np.zeros(num_tasks, dtype=np.float64)
    pred_hist_by_task = []
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    model.eval()
    seen_classes = seen_class_count(taskcla, upto_task)
    offsets = task_offsets(taskcla, upto_task)
    with torch.no_grad():
        for task_id in range(num_tasks):
            total = 0
            correct = 0
            loss_sum = 0.0
            pred_hist = np.zeros(num_tasks, dtype=np.int64)
            for images, targets in loaders[task_id]:
                targets = targets.to(device)
                logits = _logits_for_eval(model, images.to(device), eval_mode=eval_mode)[:, :seen_classes]
                loss = F.cross_entropy(logits, targets)
                pred = logits.argmax(dim=1)
                correct += int((pred == targets).sum().item())
                total += int(targets.numel())
                loss_sum += float(loss.item()) * int(targets.numel())
                pred_np = pred.detach().cpu().numpy()
                for p in pred_np:
                    pred_hist[int((p >= offsets).sum())] += 1
                for y, p in zip(targets.detach().cpu().numpy(), pred_np):
                    per_class_total[int(y)] += 1
                    if int(y) == int(p):
                        per_class_correct[int(y)] += 1
            acc_by_task[task_id] = correct / total if total else 0.0
            loss_by_task[task_id] = loss_sum / total if total else 0.0
            pred_hist_by_task.append(pred_hist.tolist())

    weights = np.array([int(taskcla[i][1]) for i in range(num_tasks)], dtype=np.float64)
    weighted_seen = float((acc_by_task * weights).sum() / weights.sum())
    per_class_acc = {
        str(k): (per_class_correct[k] / per_class_total[k] if per_class_total[k] else 0.0)
        for k in sorted(per_class_total)
    }
    return {
        "eval_mode": eval_mode,
        "acc_by_task": acc_by_task.tolist(),
        "loss_by_task": loss_by_task.tolist(),
        "weighted_seen_acc": weighted_seen,
        "predicted_task_hist_per_eval_task": pred_hist_by_task,
        "per_class_acc": per_class_acc,
    }


def old_new_logit_diagnostics(model, loaders, taskcla, upto_task, device, eval_mode="raw"):
    """Diagnostics on all old-task evaluation samples at an incremental phase."""
    if upto_task <= 0:
        return {}
    old_dim = sum(int(taskcla[i][1]) for i in range(upto_task))
    seen_dim = old_dim + int(taskcla[upto_task][1])
    old_total = 0
    old_correct = 0
    latest_pred = 0
    max_old_logit = []
    max_new_logit = []
    old_dot_max = []
    new_dot_max = []
    old_bias_arg = []
    new_bias_arg = []

    model.eval()
    with torch.no_grad():
        for task_id in range(upto_task):
            for images, targets in loaders[task_id]:
                images = images.to(device)
                targets = targets.to(device)
                logits_eval, features = _logits_for_eval(model, images, eval_mode=eval_mode, return_features=True)
                logits_eval = logits_eval[:, :seen_dim]
                pred = logits_eval.argmax(dim=1)
                old_total += int(targets.numel())
                old_correct += int((pred == targets).sum().item())
                latest_pred += int((pred >= old_dim).sum().item())

                old_logits = logits_eval[:, :old_dim]
                new_logits = logits_eval[:, old_dim:seen_dim]
                old_vals, old_idx = old_logits.max(dim=1)
                new_vals, new_idx = new_logits.max(dim=1)
                max_old_logit.append(old_vals.detach().cpu())
                max_new_logit.append(new_vals.detach().cpu())

                weight = model.classifier.weight[:seen_dim]
                bias = model.classifier.bias[:seen_dim]
                old_dot = features.matmul(weight[:old_dim].t())
                new_dot = features.matmul(weight[old_dim:seen_dim].t())
                old_dot_vals, old_dot_idx = old_dot.max(dim=1)
                new_dot_vals, new_dot_idx = new_dot.max(dim=1)
                old_dot_max.append(old_dot_vals.detach().cpu())
                new_dot_max.append(new_dot_vals.detach().cpu())
                old_bias_arg.append(bias[:old_dim][old_dot_idx].detach().cpu())
                new_bias_arg.append(bias[old_dim:seen_dim][new_dot_idx].detach().cpu())

    if old_total == 0:
        return {}

    def cat_mean(values):
        if not values:
            return math.nan
        return float(torch.cat(values).float().mean().item())

    old_logit_mean = cat_mean(max_old_logit)
    new_logit_mean = cat_mean(max_new_logit)
    old_dot_mean = cat_mean(old_dot_max)
    new_dot_mean = cat_mean(new_dot_max)
    old_bias_mean = cat_mean(old_bias_arg)
    new_bias_mean = cat_mean(new_bias_arg)
    return {
        "eval_mode": eval_mode,
        "old_acc": old_correct / old_total,
        "latest_task_prediction_rate_on_old": latest_pred / old_total,
        "max_old_logit_mean": old_logit_mean,
        "max_new_logit_mean": new_logit_mean,
        "old_vs_new_margin": old_logit_mean - new_logit_mean,
        "old_dot_max_mean": old_dot_mean,
        "new_dot_max_mean": new_dot_mean,
        "dot_margin": old_dot_mean - new_dot_mean,
        "old_bias_at_argmax_mean": old_bias_mean,
        "new_bias_at_argmax_mean": new_bias_mean,
        "bias_margin": old_bias_mean - new_bias_mean,
        "old_total": old_total,
    }


def classifier_stats(model, old_dim=0, seen_dim=None):
    if model.classifier is None:
        return {}
    if seen_dim is None:
        seen_dim = model.classifier.out_features
    with torch.no_grad():
        weight = model.classifier.weight[:seen_dim].detach().cpu()
        bias = model.classifier.bias[:seen_dim].detach().cpu()
        stats = {}
        if old_dim > 0:
            old_norm = weight[:old_dim].norm(dim=1)
            stats.update({
                "old_weight_norm_mean": float(old_norm.mean().item()),
                "old_weight_norm_std": float(old_norm.std(unbiased=False).item()),
                "old_weight_norm_min": float(old_norm.min().item()),
                "old_weight_norm_max": float(old_norm.max().item()),
                "old_bias_mean": float(bias[:old_dim].mean().item()),
                "old_bias_std": float(bias[:old_dim].std(unbiased=False).item()),
                "old_bias_min": float(bias[:old_dim].min().item()),
                "old_bias_max": float(bias[:old_dim].max().item()),
            })
        if old_dim < seen_dim:
            new_norm = weight[old_dim:seen_dim].norm(dim=1)
            stats.update({
                "new_weight_norm_mean": float(new_norm.mean().item()),
                "new_weight_norm_std": float(new_norm.std(unbiased=False).item()),
                "new_weight_norm_min": float(new_norm.min().item()),
                "new_weight_norm_max": float(new_norm.max().item()),
                "new_bias_mean": float(bias[old_dim:seen_dim].mean().item()),
                "new_bias_std": float(bias[old_dim:seen_dim].std(unbiased=False).item()),
                "new_bias_min": float(bias[old_dim:seen_dim].min().item()),
                "new_bias_max": float(bias[old_dim:seen_dim].max().item()),
            })
        return stats


def _update_anchors_from_batch(anchors: Dict[int, torch.Tensor], class_ids: Sequence[int], targets, features, momentum: float):
    """Update raw EMA feature centroids. No gradient is propagated through anchors."""
    with torch.no_grad():
        for class_id in class_ids:
            mask = targets == class_id
            if mask.any():
                centroid = features[mask].detach().mean(dim=0).cpu()
                if class_id not in anchors:
                    anchors[class_id] = centroid.clone()
                else:
                    anchors[class_id].mul_(momentum).add_(centroid, alpha=1.0 - momentum)


def _gpa_weight_anchor_loss(model, anchors: Dict[int, torch.Tensor], class_ids: Sequence[int], device, reduction="sum"):
    """Paper-style GPA anchor: || W_c - normalize(mu_c^(t)) ||_2^2.

    The previous scaffold accidentally penalized feature centroids against frozen anchors. GPA Eq. 5
    regularizes classifier rows against the moving-average current prototype instead.
    """
    terms = []
    for class_id in class_ids:
        if class_id not in anchors:
            continue
        target = F.normalize(anchors[class_id].to(device).view(1, -1), p=2, dim=1, eps=1e-12).view(-1)
        weight = model.classifier.weight[class_id]
        terms.append(torch.sum((weight - target.detach()) ** 2))
    if not terms:
        return torch.zeros((), device=device)
    values = torch.stack(terms)
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError("Unsupported GPA anchor reduction: {}".format(reduction))


def train_one_phase(
    model,
    loader,
    taskcla,
    task_id,
    device,
    epochs,
    lr,
    lr_factor,
    schedule_step,
    momentum,
    weight_decay,
    clipgrad,
    method,
    lambda_gpa,
    anchors,
    anchor_momentum,
    freeze_old_rows,
    restore_old_rows,
    anchor_reduction="sum",
):
    old_dim = sum(int(taskcla[i][1]) for i in range(task_id))
    old_weight = None
    old_bias = None
    if old_dim > 0:
        old_weight = model.classifier.weight[:old_dim].detach().clone()
        old_bias = model.classifier.bias[:old_dim].detach().clone()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    current_lr = lr
    class_ids = class_range_for_task(taskcla, task_id)
    last_stats = {}

    for epoch in range(epochs):
        model.train()
        total_ce = 0.0
        total_anchor = 0.0
        total_loss = 0.0
        total_num = 0
        correct = 0
        started = time.time()
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits, features = model(images, return_features=True)
            ce_loss = F.cross_entropy(logits, targets)
            anchor_loss = torch.zeros((), device=device)

            if method == "finetune_gpa" and task_id > 0 and lambda_gpa > 0:
                _update_anchors_from_batch(anchors, class_ids, targets, features, anchor_momentum)
                anchor_loss = _gpa_weight_anchor_loss(
                    model=model,
                    anchors=anchors,
                    class_ids=class_ids,
                    device=device,
                    reduction=anchor_reduction,
                )

            loss = ce_loss + lambda_gpa * anchor_loss
            optimizer.zero_grad()
            loss.backward()
            if method == "finetune_gpa" and freeze_old_rows and old_dim > 0:
                if model.classifier.weight.grad is not None:
                    model.classifier.weight.grad[:old_dim].zero_()
                if model.classifier.bias.grad is not None:
                    model.classifier.bias.grad[:old_dim].zero_()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipgrad)
            optimizer.step()
            if method == "finetune_gpa" and restore_old_rows and old_dim > 0:
                with torch.no_grad():
                    model.classifier.weight[:old_dim].copy_(old_weight)
                    model.classifier.bias[:old_dim].copy_(old_bias)

            batch_num = int(targets.numel())
            total_num += batch_num
            total_ce += float(ce_loss.item()) * batch_num
            total_anchor += float(anchor_loss.item()) * batch_num
            total_loss += float(loss.item()) * batch_num
            correct += int((logits.argmax(dim=1) == targets).sum().item())

        if (epoch + 1) in schedule_step:
            current_lr /= lr_factor
            for group in optimizer.param_groups:
                group["lr"] = current_lr

        ce = total_ce / max(total_num, 1)
        anchor = total_anchor / max(total_num, 1)
        last_stats = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(total_num, 1),
            "ce_loss": ce,
            "gpa_anchor_loss_raw": anchor,
            "gpa_anchor_loss_scaled": lambda_gpa * anchor,
            "anchor_to_ce_ratio": (lambda_gpa * anchor / ce) if ce > 0 else math.nan,
            "train_acc": correct / max(total_num, 1),
            "lr": current_lr,
            "seconds": time.time() - started,
            "anchor_reduction": anchor_reduction,
        }
        print(
            "| Phase {} Epoch {:3d} | loss={:.4f} ce={:.4f} anchor={:.4f} acc={:.2f}% lr={:.2e} |".format(
                task_id,
                epoch + 1,
                last_stats["train_loss"],
                ce,
                anchor,
                100.0 * last_stats["train_acc"],
                current_lr,
            ),
            flush=True,
        )

    delta_weight = 0.0
    delta_bias = 0.0
    if old_dim > 0:
        delta_weight = float((model.classifier.weight[:old_dim].detach().cpu() - old_weight.cpu()).abs().max().item())
        delta_bias = float((model.classifier.bias[:old_dim].detach().cpu() - old_bias.cpu()).abs().max().item())
    last_stats["old_head_weight_max_delta_after_phase"] = delta_weight
    last_stats["old_head_bias_max_delta_after_phase"] = delta_bias
    return last_stats


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=json_default)
