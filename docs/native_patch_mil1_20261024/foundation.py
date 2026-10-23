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
    for block in b.blocks:
        block.attn.fused_attn=False
        assert isinstance(block.attn.q_norm,torch.nn.Identity) and isinstance(block.attn.k_norm,torch.nn.Identity)
        assert isinstance(block.ls1,torch.nn.Identity) and isinstance(block.ls2,torch.nn.Identity)
        assert isinstance(block.mlp.norm,torch.nn.Identity) and block.mlp.act.approximate=='none'
        assert block.norm1.eps==block.norm2.eps==b.norm.eps==1e-6
    for m in b.modules():
        if isinstance(m,torch.nn.Dropout):assert m.p==0
        if hasattr(m,'drop_prob'):assert m.drop_prob==0
    return b.to(device)

def body_state(b):return {k:v.detach().cpu().numpy() for k,v in b.state_dict().items() if k.startswith(('blocks.','norm.'))}

def np_tokens(tokens,state,prompts=None,heads=12,eps=1e-6):
    """Independent NumPy attention, exact GELU and deep-prompt routing."""
    from scipy.special import erf
    s={k:np.asarray(v,dtype=np.float64) for k,v in state.items()};x=np.asarray(tokens,dtype=np.float64)
    def norm(z,p):
        d=z-z.mean(-1,keepdims=True)
        return d/np.sqrt((d*d).mean(-1,keepdims=True)+eps)*s[p+'weight']+s[p+'bias']
    depth=len([k for k in s if k.startswith('blocks.') and k.endswith('.attn.qkv.weight')])
    for j in range(depth):
        n=0
        if prompts is not None and j<len(prompts):
            n=len(prompts[j]);x=np.concatenate([x[:,:1],np.broadcast_to(np.asarray(prompts[j],dtype=np.float64),(len(x),n,x.shape[-1])),x[:,1:]],1)
        p=f'blocks.{j}.';z=norm(x,p+'norm1.');qkv=z@s[p+'attn.qkv.weight'].T+s[p+'attn.qkv.bias'];b,t,triple=qkv.shape;d=triple//(3*heads)
        q,k,v=qkv.reshape(b,t,3,heads,d).transpose(2,0,3,1,4);a=(q@k.swapaxes(-1,-2))/np.sqrt(d);a=np.exp(a-a.max(-1,keepdims=True));a/=a.sum(-1,keepdims=True)
        z=(a@v).transpose(0,2,1,3).reshape(b,t,heads*d);x=x+z@s[p+'attn.proj.weight'].T+s[p+'attn.proj.bias']
        z=norm(x,p+'norm2.');z=z@s[p+'mlp.fc1.weight'].T+s[p+'mlp.fc1.bias'];z=.5*z*(1+erf(z/np.sqrt(2)));x=x+z@s[p+'mlp.fc2.weight'].T+s[p+'mlp.fc2.bias']
        if n:x=np.concatenate([x[:,:1],x[:,1+n:]],1)
    return norm(x,'norm.')

