"""Native-token global bypass and image/class-conditioned MIL evidence."""
import math
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS=['NATIVE_RIDGE','MEAN','SHARED_ATTN','CLASS_ATTN']

class Model(nn.Module):
    def __init__(self,global_means,patch_means,arm):
        super().__init__()
        assert arm in ARMS[1:]
        self.arm=arm
        self.global_head=nn.Parameter(F.normalize(global_means.clone(),dim=1))
        self.local_head=nn.Parameter(torch.zeros_like(patch_means))
        if arm=='CLASS_ATTN':self.query=nn.Parameter(F.normalize(patch_means.clone(),dim=1))
        elif arm=='SHARED_ATTN':self.query=nn.Parameter(F.normalize(patch_means.mean(0,keepdim=True),dim=1))
        else:self.register_parameter('query',None)

    def forward(self,tokens):
        patches=F.normalize(tokens[:,1:],dim=-1)
        global_z=F.normalize(torch.cat([tokens[:,0],tokens[:,1:].mean(1)],dim=1),dim=1)
        score=25*global_z@F.normalize(self.global_head,dim=1).T
        if self.arm=='MEAN':local=patches.mean(1)@self.local_head.T
        else:
            attention=torch.softmax(10*torch.einsum('bpd,cd->bcp',patches,F.normalize(self.query,dim=1)),dim=-1)
            pooled=torch.einsum('bcp,bpd->bcd',attention,patches)
            local=(pooled*self.local_head[None]).sum(-1) if self.arm=='CLASS_ATTN' else pooled[:,0]@self.local_head.T
        return score+25/math.sqrt(patches.shape[-1])*local

    def expand(self,global_means,patch_means):
        self.global_head=nn.Parameter(torch.cat([self.global_head.detach(),F.normalize(global_means,dim=1)]))
        self.local_head=nn.Parameter(torch.cat([self.local_head.detach(),torch.zeros_like(patch_means)]))
        if self.arm=='CLASS_ATTN':self.query=nn.Parameter(torch.cat([self.query.detach(),F.normalize(patch_means,dim=1)]))


def np_scores(tokens,state,arm):
    """Independent NumPy implementation for sealed FP64 readouts."""
    x=np.asarray(tokens,dtype=np.float64)
    def unit(a):return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
    patches=unit(x[:,1:]);global_z=unit(np.concatenate([x[:,0],x[:,1:].mean(1)],1))
    score=25*global_z@unit(state['global_head']).T
    if arm=='MEAN':local=patches.mean(1)@state['local_head'].T
    else:
        logits=10*np.einsum('bpd,cd->bcp',patches,unit(state['query']))
        attention=np.exp(logits-logits.max(-1,keepdims=True));attention/=attention.sum(-1,keepdims=True)
        pooled=np.einsum('bcp,bpd->bcd',attention,patches)
        local=np.sum(pooled*state['local_head'][None],-1) if arm=='CLASS_ATTN' else pooled[:,0]@state['local_head'].T
    return score+25/math.sqrt(patches.shape[-1])*local


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


def global_features(tokens):
    return F.normalize(torch.cat([tokens[:,0],tokens[:,1:].mean(1)],1),dim=1)


def affine(z):
    z=F.normalize(z.double(),dim=1)
    return torch.cat([z,torch.ones(len(z),1,dtype=z.dtype,device=z.device)],1)/math.sqrt(2)


def weights(rows):
    from collections import Counter
    counts=Counter(r['group'] for r in rows)
    return np.array([1/(len(counts)*counts[r['group']]) for r in rows]),len(counts)


def update(model,opt,tokens,labels,cfg,cost):
    device=next(model.parameters()).device;opt.zero_grad(set_to_none=True);value=0.
    for i in range(0,len(tokens),cfg['microbatch']):
        x=tokens[i:i+cfg['microbatch']].to(device).float();y=labels[i:i+cfg['microbatch']].to(device)
        loss=F.cross_entropy(model(x),y,reduction='sum')/len(tokens);assert torch.isfinite(loss)
        loss.backward();value+=float(loss.detach());cost['forward_rows']+=len(x);cost['backward_rows']+=len(x)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);assert torch.isfinite(norm)
    opt.step();cost['updates']+=1;return dict(CE=value,gradient_norm=float(norm))


def cpu_scores(tokens,state,arm,micro=8):
    model=Model(state['global_head'].float(),torch.zeros_like(state['local_head']).float(),arm).double()
    model.load_state_dict(state)
    with torch.no_grad():return torch.cat([model(tokens[i:i+micro].double()) for i in range(0,len(tokens),micro)])
