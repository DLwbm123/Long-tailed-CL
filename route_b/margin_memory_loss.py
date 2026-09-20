"""ACTM class-balanced query risk and covariance-aware old-memory margin."""
from __future__ import annotations

from typing import Any


def memory_margin_loss(W: Any, old_mean: Any, old_cov: Any, *, delta: float = .1,
                       kappa: float = .5, tau: float = .1):
    """Return ``tau*log(1+sum_j exp(margin_j/tau))`` for every old class."""
    import torch
    if not all(isinstance(x, torch.Tensor) for x in (W, old_mean, old_cov)):
        raise TypeError("BLOCKED_TORCH_INPUT")
    if W.ndim != 2 or old_mean.ndim != 2 or old_cov.shape != (W.shape[0], W.shape[0]):
        raise ValueError("BLOCKED_MEMORY_SHAPE")
    K_old, K = old_mean.shape[1], W.shape[1]
    if K_old == 0 or K <= 1:
        return W.new_zeros(())
    logits = old_mean.T @ W
    terms = []
    for c in range(K_old):
        q = W[:, c, None] - W
        # Full covariance quadratic form; the off-diagonal cross block matters.
        var = torch.einsum("ik,ij,jk->k", q, old_cov, q).clamp_min(0)
        margin = delta + kappa * torch.sqrt(var + 1e-8) - (logits[c, c] - logits[c])
        competitors = torch.cat((margin[:c], margin[c + 1:]))
        if competitors.numel():
            terms.append(tau * torch.logsumexp(torch.cat((margin.new_zeros(1), competitors / tau)), dim=0))
    return torch.stack(terms).mean() if terms else W.new_zeros(())


def current_query_risk(logits: Any, labels: Any):
    """Class-balanced squared error over the K-way one-hot target."""
    import torch
    if not isinstance(logits, torch.Tensor):
        raise TypeError("BLOCKED_TORCH_INPUT")
    labels = labels.to(device=logits.device, dtype=torch.long)
    target = torch.nn.functional.one_hot(labels, num_classes=logits.shape[1]).to(logits.dtype)
    per_sample = (logits - target).square().sum(dim=1)
    classes = torch.unique(labels, sorted=True)
    return torch.stack([per_sample[labels == c].mean() for c in classes]).mean()


def total_actm_loss(query_logits: Any, labels: Any, W: Any, old_mean: Any, old_cov: Any,
                    fd_loss: Any = 0., *, memory: bool = True, fd_weight: float = 1.0):
    risk = current_query_risk(query_logits, labels)
    mem = memory_margin_loss(W, old_mean, old_cov) if memory else risk.new_zeros(())
    return risk + mem + fd_weight * fd_loss, {"query": risk, "memory": mem, "fd": fd_loss}
