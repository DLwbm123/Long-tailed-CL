"""ACTM current-query risk and covariance-aware old-memory margin."""
from __future__ import annotations

from typing import Any


def memory_margin_loss(W: Any, old_mean: Any, old_cov: Any, *, delta: float = .1,
                       kappa: float = .5, tau: float = .1):
    """Return a differentiable old-memory margin term.

    ``W`` is D x K and ``old_mean`` is D x Ko.  The covariance is shared; the
    diagonal self logit is omitted from each log-sum-exp competitor set.
    """
    import torch
    if not all(isinstance(x, torch.Tensor) for x in (W, old_mean, old_cov)):
        raise TypeError("BLOCKED_TORCH_INPUT")
    logits = old_mean.T @ W
    K_old = old_mean.shape[1]
    if K_old < 2:
        return W.new_zeros(())
    terms = []
    for c in range(K_old):
        q = W[:, c, None] - W
        var = torch.einsum("dk,dd,dk->k", q, old_cov, q).clamp_min(0)
        margin = delta + kappa * torch.sqrt(var + 1e-12) - (logits[c, c] - logits[c])
        margin = torch.cat((margin[:c], margin[c + 1:]))
        terms.append(tau * torch.logsumexp(margin / tau, dim=0))
    return torch.stack(terms).mean()


def current_query_risk(logits: Any, labels: Any):
    import torch
    return torch.nn.functional.cross_entropy(logits, labels, reduction="mean")


def total_actm_loss(query_logits: Any, labels: Any, W: Any, old_mean: Any, old_cov: Any,
                    fd_loss: Any = 0., *, memory: bool = True, fd_weight: float = 1.0):
    risk = current_query_risk(query_logits, labels)
    mem = memory_margin_loss(W, old_mean, old_cov) if memory else risk.new_zeros(())
    return risk + mem + fd_weight * fd_loss, {"query": risk, "memory": mem, "fd": fd_loss}
