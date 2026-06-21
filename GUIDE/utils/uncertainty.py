# ==============================================================================
# BEGIN FILE: utils/uncertainty.py (A-MUSE 新增模块)
# ==============================================================================
import torch
import torch.nn.functional as F
import numpy as np


def softmax_with_temperature(logits, T=1.0):
    return (logits / T).softmax(dim=-1)


def entropy(p, eps=1e-8):
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def get_uncertainty_metrics(probs_per_expert):
    """

    Args:
        probs_per_expert (torch.Tensor):  [num_experts, batch_size, num_classes]

    Returns:
        tuple[torch.Tensor, torch.Tensor]: aleatoric_uncertainty, epistemic_uncertainty
                                           [batch_size]。
    """

    aleatoric_uncertainty = entropy(probs_per_expert).mean(dim=0)


    p_bar = probs_per_expert.mean(dim=0)
    total_uncertainty = entropy(p_bar)

    epistemic_uncertainty = (total_uncertainty - aleatoric_uncertainty).clamp_min(0.0)

    return aleatoric_uncertainty, epistemic_uncertainty


class ClasswiseEMA:

    def __init__(self, num_classes, momentum=0.9, device="cuda"):
        self.num_classes = num_classes
        self.m = momentum
        self.device = device

        self.ale = torch.zeros(num_classes, device=device)
        self.epi = torch.zeros(num_classes, device=device)

        self.count = torch.zeros(num_classes, device=device)

    @torch.no_grad()
    def update_batch(self, ale_batch, epi_batch, targets):

        for cls_idx in targets.unique():
            mask = (targets == cls_idx)
            if mask.any():
                ale_cls_mean = ale_batch[mask].mean()
                epi_cls_mean = epi_batch[mask].mean()

                if self.count[cls_idx] > 0:  # 如果不是第一次见到该类别
                    self.ale[cls_idx] = self.m * self.ale[cls_idx] + (1 - self.m) * ale_cls_mean
                    self.epi[cls_idx] = self.m * self.epi[cls_idx] + (1 - self.m) * epi_cls_mean
                else:
                    self.ale[cls_idx] = ale_cls_mean
                    self.epi[cls_idx] = epi_cls_mean

                self.count[cls_idx] += 1

    @torch.no_grad()
    def get_stats(self):
        return {"ale": self.ale.clone(), "epi": self.epi.clone()}