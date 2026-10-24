"""Reused frozen native encoder, compact persistence and independent full12-layer reference."""
import json
import numpy as np
import torch

def write(p,v):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)

def compact(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu().clone().contiguous()
    if isinstance(v,dict):return {k:compact(x) for k,x in v.items()}
    if isinstance(v,list):return [compact(x) for x in v]
    return v

def save(p,v):
    v=compact(v)
    def check(x):
        if isinstance(x,torch.Tensor):assert x.untyped_storage().nbytes()==x.numel()*x.element_size()
        elif isinstance(x,dict):
            for y in x.values():check(y)
        elif isinstance(x,list):
            for y in x:check(y)
    check(v);t=p.with_suffix(p.suffix+'.tmp');torch.save(v,t);t.replace(p)

def prefix(b,images):return b.norm_pre(b.patch_drop(b._pos_embed(b.patch_embed(images))))

def build_backbone(cfg,device):
    import timm
    from safetensors.torch import load_file
    b=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
    b.load_state_dict({k:v for k,v in load_file(cfg['weight']).items() if not k.startswith('head.')},strict=True)
    b.eval().requires_grad_(False)
    assert isinstance(b.norm_pre,torch.nn.Identity) and isinstance(b.patch_embed.norm,torch.nn.Identity) and isinstance(b.patch_drop,torch.nn.Identity)
    for block in b.blocks:
        block.attn.fused_attn=False
        assert isinstance(block.attn.q_norm,torch.nn.Identity) and isinstance(block.attn.k_norm,torch.nn.Identity)
        assert isinstance(block.ls1,torch.nn.Identity) and isinstance(block.ls2,torch.nn.Identity)
        assert isinstance(block.mlp.norm,torch.nn.Identity) and block.mlp.act.approximate=='none'
        assert block.norm1.eps==block.norm2.eps==b.norm.eps==1e-6
    for m in b.modules():
        if isinstance(m,torch.nn.Dropout):assert m.p==0
        if hasattr(m,'drop_prob'):assert m.drop_prob==0
    return b.to(device=device,dtype=torch.float64)
