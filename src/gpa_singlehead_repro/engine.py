import csv
import json
import math
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


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


def make_gpa_bias(counts, nref_mode, eps, bias_mode="paper", old_bias_mean=0.0):
    if bias_mode == "zero":
        return torch.zeros_like(counts), None
    if bias_mode == "old_mean":
        return torch.full_like(counts, float(old_bias_mean)), None
    if bias_mode == "none":
        return None, None
    if bias_mode != "paper":
        raise ValueError("Unsupported bias_mode: {}".format(bias_mode))
    if nref_mode == "max_current":
        n_ref = counts.max()
    elif nref_mode == "mean_current":
        n_ref = counts.mean()
    elif nref_mode == "first_current":
        n_ref = counts[0]
    else:
        raise ValueError("Unsupported nref_mode: {}".format(nref_mode))
    return -torch.log(counts / n_ref + eps), float(n_ref.item())


def _calibrated_logits(model, images, seen_classes, mode):
    logits, features = model(images, return_features=True)
    logits = logits[:, :seen_classes]
    weight = model.classifier.weight[:seen_classes]
    bias = model.classifier.bias[:seen_classes]
    if mode == "raw":
        return logits
    if mode == "no_bias":
        return logits - bias.view(1, -1)
    if mode == "task_bias_center":
        calibrated = logits.clone()
        offsets = [0] + model.task_cls.cumsum(0).detach().cpu().tolist()
        for start, stop in zip(offsets[:-1], offsets[1:]):
            calibrated[:, start:stop] -= bias[start:stop].mean().view(1, 1)
        return calibrated
    if mode == "cosine_eval":
        return F.linear(F.normalize(features, dim=1), F.normalize(weight, dim=1))
    if mode == "norm_weight_eval":
        return F.linear(features, F.normalize(weight, dim=1))
    raise ValueError("Unsupported eval calibration mode: {}".format(mode))


def evaluate_seen_with_mode(model, loaders, taskcla, upto_task, device, mode="raw"):
    num_tasks = upto_task + 1
    acc_by_task = np.zeros(num_tasks, dtype=np.float64)
    pred_hist_by_task = []
    model.eval()
    seen_classes = sum(int(taskcla[i][1]) for i in range(num_tasks))
    with torch.no_grad():
        offsets = np.cumsum([int(taskcla[i][1]) for i in range(num_tasks)])
        for task_id in range(num_tasks):
            total = 0
            correct = 0
            pred_hist = np.zeros(num_tasks, dtype=np.int64)
            for images, targets in loaders[task_id]:
                targets = targets.to(device)
                logits = _calibrated_logits(model, images.to(device), seen_classes, mode)
                pred = logits.argmax(dim=1)
                correct += int((pred == targets).sum().item())
                total += int(targets.numel())
                for p in pred.detach().cpu().numpy():
                    pred_hist[int((p >= offsets).sum())] += 1
            acc_by_task[task_id] = correct / total if total else 0.0
            pred_hist_by_task.append(pred_hist.tolist())
    weights = np.array([int(taskcla[i][1]) for i in range(num_tasks)], dtype=np.float64)
    return {
        "eval_mode": mode,
        "acc_by_task": acc_by_task.tolist(),
        "weighted_seen_acc": float((acc_by_task * weights).sum() / weights.sum()),
        "predicted_task_hist_per_eval_task": pred_hist_by_task,
    }


def evaluate_seen(model, loaders, taskcla, upto_task, device):
    num_tasks = upto_task + 1
    acc_by_task = np.zeros(num_tasks, dtype=np.float64)
    loss_by_task = np.zeros(num_tasks, dtype=np.float64)
    pred_hist_by_task = []
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    model.eval()
    seen_classes = sum(int(taskcla[i][1]) for i in range(num_tasks))
    with torch.no_grad():
        for task_id in range(num_tasks):
            total = 0
            correct = 0
            loss_sum = 0.0
            pred_hist = np.zeros(num_tasks, dtype=np.int64)
            offsets = np.cumsum([int(taskcla[i][1]) for i in range(num_tasks)])
            for images, targets in loaders[task_id]:
                targets = targets.to(device)
                logits = model(images.to(device))[:, :seen_classes]
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
        "acc_by_task": acc_by_task.tolist(),
        "loss_by_task": loss_by_task.tolist(),
        "weighted_seen_acc": weighted_seen,
        "predicted_task_hist_per_eval_task": pred_hist_by_task,
        "per_class_acc": per_class_acc,
    }


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
    anchor_reduction,
    freeze_old_rows,
    restore_old_rows,
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
                terms = []
                for class_id in class_ids:
                    mask = targets == class_id
                    if mask.any():
                        centroid = features[mask].mean(dim=0)
                        target = anchors[class_id].to(device)
                        terms.append(torch.sum((centroid - target.detach()) ** 2))
                        with torch.no_grad():
                            anchors[class_id].mul_(anchor_momentum).add_(
                                centroid.detach().cpu(), alpha=1.0 - anchor_momentum
                            )
                if terms:
                    stacked_terms = torch.stack(terms)
                    if anchor_reduction == "sum":
                        anchor_loss = stacked_terms.sum()
                    elif anchor_reduction == "mean":
                        anchor_loss = stacked_terms.mean()
                    elif anchor_reduction == "cosine":
                        cosine_terms = []
                        for class_id in class_ids:
                            mask = targets == class_id
                            if mask.any():
                                centroid = features[mask].mean(dim=0)
                                target = anchors[class_id].to(device)
                                cosine_terms.append(1.0 - F.cosine_similarity(centroid, target.detach(), dim=0))
                        if cosine_terms:
                            anchor_loss = torch.stack(cosine_terms).mean()
                    else:
                        raise ValueError("Unsupported anchor_reduction: {}".format(anchor_reduction))

            loss = ce_loss + lambda_gpa * anchor_loss
            optimizer.zero_grad()
            loss.backward()
            if method == "finetune_gpa" and freeze_old_rows and old_dim > 0:
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
