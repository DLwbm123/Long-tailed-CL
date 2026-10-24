"""Frozen local/global representation objective; no experiment entry."""
import torch
from torch import nn


def redundancy_loss(first, second):
    """Batch axis is patients/draws; optional middle axis is matched raster cells."""
    assert first.shape == second.shape and first.ndim in (2, 3)
    n, d = first.shape[0], first.shape[-1]
    assert n > 1 and d > 1
    invariance = (first - second).square().mean()
    variance = first.new_zeros(())
    covariance = first.new_zeros(())
    for value in (first, second):
        centered = value - value.mean(dim=0, keepdim=True)
        cov = torch.einsum('n...d,n...e->...de', centered, centered) / (n - 1)
        diagonal = cov.diagonal(dim1=-2, dim2=-1)
        variance = variance + torch.relu(1 - torch.sqrt(diagonal + 1e-4)).mean() / 2
        covariance = covariance + ((cov.square().sum(dim=(-2, -1)) - diagonal.square().sum(dim=-1)) / d).mean()
    return 25 * invariance + 25 * variance + covariance


class ProjectionHeads(nn.Module):
    def __init__(self):
        super().__init__()
        # No running statistics or stochastic layers: replay must use the same function.
        self.global_map = nn.Sequential(nn.Linear(1536, 512), nn.GELU(approximate='none'), nn.Linear(512, 256))
        self.local_map = nn.Sequential(nn.Linear(768, 128), nn.GELU(approximate='none'), nn.Linear(128, 64))

    def forward(self, native_tokens):
        assert native_tokens.ndim == 3 and native_tokens.shape[1:] == (197, 768)
        descriptor = torch.cat((native_tokens[:, 0], native_tokens[:, 1:].mean(dim=1)), dim=-1)
        return self.global_map(descriptor), self.local_map(native_tokens[:, 1:])


def paired_objective(heads, first, second, first_cells, second_cells, local):
    global_first, local_first = heads(first)
    global_second, local_second = heads(second)
    global_loss = redundancy_loss(global_first, global_second)
    if not local:
        return global_loss
    assert first_cells.ndim == second_cells.ndim == 1
    assert first_cells.shape == second_cells.shape and first_cells.numel() > 0
    local_loss = redundancy_loss(local_first[:, first_cells], local_second[:, second_cells])
    return (global_loss + local_loss) / 2


class NativeUpper(nn.Module):
    def __init__(self, backbone):
        import copy
        super().__init__()
        assert len(backbone.blocks) == 12
        self.blocks = nn.Sequential(*copy.deepcopy(list(backbone.blocks.children())[8:]))
        self.norm = copy.deepcopy(backbone.norm)
        self.requires_grad_(True)
        self.eval()

    def forward(self, lower_tokens):
        return self.norm(self.blocks(lower_tokens))


def native_descriptor(tokens):
    from torch.nn import functional as F
    return F.normalize(torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=-1), dim=-1)


def affine_descriptor(tokens):
    import math
    descriptor = native_descriptor(tokens)
    return torch.cat((descriptor, torch.ones_like(descriptor[:, :1])), dim=-1) / math.sqrt(2)
