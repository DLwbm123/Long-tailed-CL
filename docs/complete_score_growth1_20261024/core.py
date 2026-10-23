"""Draft class-score growth core. No qualification or scientific execution evidence yet."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class Upper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        assert len(backbone.blocks) == 12
        # Sequential slicing retains10/11 names; serialization and NumPy use local0/1.
        self.blocks = nn.Sequential(*copy.deepcopy(list(backbone.blocks.children())[10:]))
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


class Expert(nn.Module):
    def __init__(self,native_upper,means):
        super().__init__()
        self.upper=copy.deepcopy(native_upper)
        self.head=nn.Parameter(F.normalize(means.detach().clone(),dim=1))

    def forward(self,tokens,features=None):
        z=self.upper(tokens) if features is None else features
        return 25*z@F.normalize(self.head,dim=1).T


class Growing(nn.Module):
    """Each class has one permanent output owner; every input sees every output."""
    def __init__(self,freeze_old_scores):
        super().__init__()
        self.experts=nn.ModuleList();self.groups=[];self.freeze_old_scores=freeze_old_scores

    def append(self,native_upper,current_means,arrived_labels):
        assert len(arrived_labels)==len(current_means)
        assert not set(arrived_labels)&{c for group in self.groups for c in group}
        for expert in self.experts:
            expert.upper.requires_grad_(False).eval()
            expert.head.requires_grad_(not self.freeze_old_scores)
        self.experts.append(Expert(native_upper,current_means))
        self.groups.append(list(arrived_labels))

    def forward(self,tokens,old_features=None):
        if old_features is not None:assert len(old_features)==len(self.experts)-1
        logits=[]
        for j,expert in enumerate(self.experts):
            if j==len(self.experts)-1:logits.append(expert(tokens))
            else:
                if old_features is None:
                    with torch.no_grad():z=expert.upper(tokens)
                else:
                    z=old_features[j].detach();assert z.shape==(len(tokens),1536)
                logits.append(expert(tokens,z))
        return torch.cat(logits,dim=1)

    @property
    def seen(self):return [c for group in self.groups for c in group]


def np_scores(tokens,experts,eps=1e-6):
    result=[]
    for weights in experts:
        upper={k.removeprefix('upper.'):v for k,v in weights.items() if k.startswith('upper.')}
        z=np_upper(tokens,upper,eps)
        h=np.asarray(weights['head'],dtype=np.float64);h=h/np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-12)
        result.append(25*z@h.T)
    return np.concatenate(result,axis=1)
