"""Approved V2 input capacity, fixed weight loading and effective optimizer profile."""
import hashlib
import torch
from torch import nn
import timm
from safetensors.torch import load_file

WEIGHT_SHA = 'c401d219603ac3e20b6373c7b198c78d3a733f80b755d148bda3bc320ae69800'

def verified_state(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    assert h.hexdigest() == WEIGHT_SHA, 'BLOCKED_WEIGHT_SHA'
    state = load_file(str(path), device='cpu')
    assert state.pop('head.weight').shape == (1000, 768)
    assert state.pop('head.bias').shape == (1000,)
    return state

def load_adapter(model, path):
    state = verified_state(path)
    mapped = {}
    for key, value in state.items():
        if '.qkv.' in key:
            for name, part in zip(('q_proj', 'k_proj', 'v_proj'), value.chunk(3, dim=0)):
                mapped[key.replace('qkv', name)] = part
        else:
            mapped[key.replace('.mlp.fc', '.fc')] = value
    own = model.state_dict()
    missing = sorted(set(own) - set(mapped))
    allowed = {f'{pool}.pool.{i}.{layer}.{field}' for pool in ('pool','pool_few')
               for i in range(60) for layer in ('down_proj','up_proj') for field in ('weight','bias')}
    allowed |= {'pool.prompt_key','pool_few.prompt_key','assigner.cls_emb.weight'}
    allowed |= {f'{module}.{field}' for module in ('assigner.assign','assigner.ins_linear','head','head_few') for field in ('weight','bias')}
    assert set(missing)==allowed, ('BLOCKED_CORE_MISSING', sorted(set(missing)^allowed))
    assert not set(mapped) - set(own), 'BLOCKED_UNEXPECTED_PRETRAINED'
    result = model.load_state_dict(mapped, strict=False)
    assert sorted(result.missing_keys) == missing and not result.unexpected_keys
    own = model.state_dict()
    assert all(torch.equal(own[k], v) for k,v in mapped.items()), 'BLOCKED_TENSOR_MAPPING'
    for name, p in model.named_parameters(): p.requires_grad_(name in missing)
    model.weight_load_audit = dict(core_tensors=len(mapped), allowed_missing=missing, tensor_equality=True)
    return model

class FrozenOriginal(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        state = verified_state(path)
        self.model.load_state_dict(state, strict=True)
        assert all(torch.equal(self.model.state_dict()[k], v) for k,v in state.items())
        self.requires_grad_(False)
    def forward(self, x): return {'pre_logits': self.model(x)}

def extend_embedding(assigner, seed):
    old = assigner.cls_emb
    assert old.weight.shape == (501, 16) and old.weight.device.type == 'cpu'
    generator = torch.Generator().manual_seed(seed + 1000003)
    weights = torch.cat([old.weight.detach().clone(), torch.randn(12225, 16, generator=generator)])
    assigner.cls_emb = nn.Embedding.from_pretrained(weights, freeze=False)
    assert torch.equal(assigner.cls_emb.weight[:501], old.weight)

def effective_optimizer(backbone):
    pool=[];other=[]
    for name,p in backbone.named_parameters():
        if p.requires_grad: (pool if 'pool' in name else other).append(p)
    return torch.optim.AdamW([{'params':pool,'lr':.003 * .1},{'params':other,'lr':.003}],
                             weight_decay=.01, betas=(.9,.999), eps=1e-8)

def effective_scheduler(optimizer, session):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5) if session == 0 else None
