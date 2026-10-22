"""Draft native upper-ViT learner. No experiment has been frozen or started."""
import copy
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ['FROZEN_UPPER', 'PLASTIC_UPPER']


class Upper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        assert len(backbone.blocks) == 12
        self.blocks = copy.deepcopy(backbone.blocks[10:])
        self.norm = copy.deepcopy(backbone.norm)
        assert isinstance(self.norm, nn.LayerNorm)
        for block in self.blocks:
            assert isinstance(block.attn.q_norm, nn.Identity)
            assert isinstance(block.attn.k_norm, nn.Identity)
            assert isinstance(block.ls1, nn.Identity) and isinstance(block.ls2, nn.Identity)
            assert isinstance(block.mlp.norm, nn.Identity)
            assert isinstance(block.mlp.act, nn.GELU) and block.mlp.act.approximate == 'none'
            assert block.norm1.eps == block.norm2.eps == self.norm.eps
            # Explicit attention also permits an independent NumPy implementation.
            block.attn.fused_attn = False
        self.eval()

    def forward(self, tokens):
        tokens = self.norm(self.blocks(tokens))
        return F.normalize(torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], 1), dim=1)


class Model(nn.Module):
    def __init__(self, upper, means, plastic):
        super().__init__()
        self.upper = copy.deepcopy(upper)
        self.upper.requires_grad_(plastic)
        self.head = nn.Parameter(F.normalize(means.detach().clone(), dim=1))

    def forward(self, tokens):
        return 25. * self.upper(tokens) @ F.normalize(self.head, dim=1).T

    def expand(self, means):
        self.head = nn.Parameter(torch.cat([self.head.detach(), F.normalize(means.detach(), dim=1)]))


def np_norm(x, weight, bias, eps):
    centered = x - x.mean(axis=-1, keepdims=True)
    return centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + eps) * weight + bias


def np_upper(tokens, state, eps, heads=12):
    """Independent explicit attention/MLP; state comes from a sealed Upper only."""
    import math
    from scipy.special import erf
    x = np.asarray(tokens, dtype=np.float64)
    s = {k: np.asarray(v, dtype=np.float64) for k, v in state.items()}
    for i in range(2):
        p = f'blocks.{i}.'
        z = np_norm(x, s[p+'norm1.weight'], s[p+'norm1.bias'], eps)
        qkv = z @ s[p+'attn.qkv.weight'].T + s[p+'attn.qkv.bias']
        b, n, triple = qkv.shape
        d = triple // (3 * heads)
        q, k, v = qkv.reshape(b, n, 3, heads, d).transpose(2, 0, 3, 1, 4)
        a = (q @ k.swapaxes(-1, -2)) / math.sqrt(d)
        a = np.exp(a - a.max(axis=-1, keepdims=True))
        a /= a.sum(axis=-1, keepdims=True)
        z = (a @ v).transpose(0, 2, 1, 3).reshape(b, n, heads*d)
        x = x + z @ s[p+'attn.proj.weight'].T + s[p+'attn.proj.bias']
        z = np_norm(x, s[p+'norm2.weight'], s[p+'norm2.bias'], eps)
        z = z @ s[p+'mlp.fc1.weight'].T + s[p+'mlp.fc1.bias']
        # Exact GELU, not the tanh approximation.
        z = .5 * z * (1. + erf(z / math.sqrt(2.)))
        x = x + z @ s[p+'mlp.fc2.weight'].T + s[p+'mlp.fc2.bias']
    x = np_norm(x, s['norm.weight'], s['norm.bias'], eps)
    z = np.concatenate([x[:, 0], x[:, 1:].mean(axis=1)], axis=1)
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)


def key(seed,c,g,path=''):
    return hashlib.sha256(f'{seed}:{c}:{g}:{path}'.encode()).digest()

def memory_indices(rows,cap,seed):
    out=[]
    for c in sorted({r['label'] for r in rows}):
        groups={}
        for i,r in enumerate(rows):
            if r['label']==c:groups.setdefault(r['group'],[]).append(i)
        for g in sorted(groups,key=lambda g:(key(seed,c,g),g))[:cap]:
            out.append(min(groups[g],key=lambda i:(key(seed,c,g,rows[i]['relative_path']),i)))
    return out



def parameters_for_adam(model, cfg):
    groups = [{'params': [model.head], 'lr': cfg['head_learning_rate']}]
    upper = [p for p in model.upper.parameters() if p.requires_grad]
    if upper:
        groups.append({'params': upper, 'lr': cfg['upper_learning_rate']})
    return torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])


def update(model, optimizer, tokens, labels, cfg, cost):
    model.eval()  # Native stochastic layers disabled in both arms.
    optimizer.zero_grad(set_to_none=True)
    loss_value = 0.
    for begin in range(0, len(tokens), cfg['microbatch']):
        x = tokens[begin:begin+cfg['microbatch']].cuda().float()
        y = labels[begin:begin+cfg['microbatch']].cuda()
        loss = F.cross_entropy(model(x), y, reduction='sum') / len(tokens)
        assert torch.isfinite(loss)
        loss.backward()
        loss_value += float(loss.detach())
        cost['head_forward_rows'] += len(x)
        cost['head_backward_rows'] += len(x)
        cost['upper_forward_rows'] += len(x)
        if any(p.requires_grad for p in model.upper.parameters()):
            cost['upper_backward_rows'] += len(x)
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg['gradient_clip'])
    assert torch.isfinite(norm)
    optimizer.step()
    cost['updates'] += 1
    cost['logical_draws'] += len(tokens)
    return dict(CE=loss_value, gradient_norm_before_clip=float(norm))


def empty_arm_cost():
    return dict(updates=0, logical_draws=0, head_forward_rows=0, head_backward_rows=0,
                upper_forward_rows=0, upper_backward_rows=0, initialization_rows=0,
                FP64_probe_rows=0, validation_rows=0, numpy_probe_rows=0)
