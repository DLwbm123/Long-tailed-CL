"""SPHERE-inspired single-source-marginal transport, not Sinkhorn."""
import torch
import torch.nn.functional as F


def geometry(features, eps_s=.2):
    assert len(features) > 0 and eps_s > 0
    h = F.normalize(features, dim=-1)
    C = torch.cdist(h, h)
    nonzero = C[C > 1e-10]
    if len(nonzero):
        C = C / nonzero.median()
    else:
        C = torch.zeros_like(C)
    K = torch.exp(-C / eps_s)
    return C.detach(), (K / K.sum(1, keepdim=True)).detach()


def transport(delta, C, S, eps_T=.5, alpha=.5, sigma_floor=.02, tau=None):
    assert delta.ndim == 1 and len(delta) > 0
    assert C.shape == S.shape == (len(delta), len(delta))
    assert eps_T > 0 and 0 <= alpha < 1 and sigma_floor > 0
    signed = S @ delta
    v = signed.relu()
    if tau is None:
        tau = v.detach().std(unbiased=False).clamp_min(sigma_floor) / eps_T
    tau = torch.as_tensor(tau, dtype=delta.dtype, device=delta.device).detach()
    assert float(tau) > 0
    logits = (v[:, None] - tau * C) / (tau * eps_T)
    Pi = logits.softmax(0) / len(delta)
    q = Pi.sum(1)
    w = S.T @ (q * (signed > 0).to(q.dtype))
    mixed = (1 - alpha) / len(delta) + alpha * w
    p = mixed / mixed.sum()
    value = tau * eps_T * (torch.logsumexp(logits, dim=0) - 2 * __import__('math').log(len(delta))).mean()
    assert torch.isfinite(p).all()
    return dict(p=p, S=S, C=C, signed=signed, Pi=Pi, q=q, w=w, value=value, tau=tau)

