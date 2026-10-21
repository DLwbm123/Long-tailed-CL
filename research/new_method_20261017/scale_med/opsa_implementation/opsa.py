"""Paper-derived OPSA forward and an explicitly local row-any MAPC mapping.

Reference: https://openreview.net/forum?id=msP4WP1nA2
Not author code. See README.md for the unresolved optimizer interpretation.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def quadrature(a):
    """Apply block-diagonal [[0, -1], [1, 0]] to last-axis pairs."""
    if a.shape[-1] % 2:
        raise ValueError("Quadrature requires an even latent dimension")
    pairs = a.reshape(*a.shape[:-1], -1, 2)
    return torch.stack((-pairs[..., 1], pairs[..., 0]), -1).flatten(-2)


class OPSAAdapter(nn.Module):
    """Residual adapter for [..., width] tokens; no backbone hooks installed.

    collect_mapc=True is for training forwards only. Multiple microbatches OR
    their boundary rows until reset_mapc() or clear_mapc() consumes them.
    """

    def __init__(self, width, rank=16, kappa=1.0, gamma=8.0, delta=0.25):
        super().__init__()
        if width <= 0 or rank <= 0 or rank % 2:
            raise ValueError("Positive width and positive even rank required")
        if not all(math.isfinite(v) for v in (kappa, gamma, delta)):
            raise ValueError("Constants must be finite")
        if kappa <= 0 or gamma <= 0 or not 0 < delta < math.pi / (2 * kappa):
            raise ValueError("Invalid phase/radius constants")
        self.width, self.rank = width, rank
        self.kappa, self.gamma, self.delta = kappa, gamma, delta
        self.down = nn.Parameter(torch.empty(rank, width))
        self.up = nn.Parameter(torch.empty(width, rank))
        nn.init.normal_(self.down, std=0.1 / math.sqrt(width * rank))
        # Local convention: the paper specifies V != 0, not its initialization.
        nn.init.normal_(self.up, std=1.0 / math.sqrt(width * rank))
        self.register_buffer("boundary_rows", torch.zeros(rank, dtype=torch.bool),
                             persistent=False)
        self._collected = False

    def effective_up(self):
        norm = self.up.norm()
        if not torch.isfinite(norm) or norm == 0:
            raise ValueError("Vup must have a finite nonzero Frobenius norm")
        return self.gamma * self.up / norm

    def forward(self, x, *, collect_mapc=False):
        if x.shape[-1] != self.width or not torch.isfinite(x).all():
            raise ValueError("Expected finite tokens with last dimension width")
        if collect_mapc and (not self.training or not torch.is_grad_enabled()):
            raise ValueError("MAPC collection requires a training gradient forward")
        norm = x.norm(dim=-1, keepdim=True)
        if (norm == 0).any():
            raise ValueError("Paper token normalization is undefined for zero tokens")
        a = F.linear(x / norm * math.sqrt(self.width), self.down)
        phi = quadrature(a)
        z = a * (self.kappa * phi).cos() + phi * (self.kappa * a).cos()
        if collect_mapc:
            with torch.no_grad():
                rows = (a.detach().abs() >= math.pi / (2 * self.kappa) - self.delta)
                self.boundary_rows.logical_or_(rows.reshape(-1, self.rank).any(0))
                self._collected = True
        return x + F.linear(z, self.effective_up())

    def clear_mapc(self):
        self.boundary_rows.zero_()
        self._collected = False

    @torch.no_grad()
    def reset_mapc(self, optimizer):
        """Call after backward/unscale/clip, immediately before AdamW.step().

        Local row-any mapping; A.13 pre-accumulation reset interpretation.
        Returns number of first-moment entries reset; second moment untouched.
        No reset on an uninitialized Adam state. Reject nonfinite gradients.
        """
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError("This contract supports torch.optim.AdamW only")
        groups = [g for g in optimizer.param_groups if any(p is self.down for p in g['params'])]
        if len(groups) != 1 or groups[0].get('maximize', False):
            raise ValueError("Down parameter must belong to one minimizing AdamW group")
        if not self._collected or self.down.grad is None:
            raise RuntimeError("Collect a training forward and backward before MAPC")
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for g in optimizer.param_groups for p in g['params']):
            self.clear_mapc()
            raise FloatingPointError("Nonfinite gradient: skip optimizer step")
        state = optimizer.state.get(self.down, {})
        count = 0
        if 'exp_avg' in state:
            moment = state['exp_avg']
            mask = self.boundary_rows[:, None] & (self.down.grad.sign() != moment.sign())
            count = int(mask.sum())
            moment.masked_fill_(mask, 0)
        self.clear_mapc()
        return count
