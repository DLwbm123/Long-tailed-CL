"""Canonical micro8 objective, legal patient memory and same-gradient AdamW check."""
import hashlib,time
import numpy as np
import torch
from torch.nn import functional as F

from core import ARMS
def state(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def counter():return dict(updates=0,forward_rows=0,backward_rows=0)
def optimizer(m,cfg):
    return torch.optim.AdamW([v for v in m.parameters() if v.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],eps=1e-8,betas=(.9,.999),foreach=False,fused=False)
def selected(rows,cap,seed):
    key=lambda c,g,path='':hashlib.sha256(f'{seed}:{c}:{g}:{path}'.encode()).digest()
    out=[]
    for c in sorted({r['label'] for r in rows}):
        groups={}
        for i,r in enumerate(rows):
            if r['label']==c:groups.setdefault(r['group'],[]).append(i)
        for g in sorted(groups,key=lambda g:(key(c,g),g))[:cap]:
            out.append(min(groups[g],key=lambda i:(key(c,g,rows[i]['relative_path']),i)))
    return out

def weights(rows):
    from collections import Counter
    n=Counter(r['group'] for r in rows)
    return np.array([1/(len(n)*n[r['group']]) for r in rows]),len(n)
def affine(z):return torch.cat([F.normalize(z.double(),dim=1),torch.ones(len(z),1,dtype=torch.float64)],1)/np.sqrt(2)

def update(model,opt,tokens,labels,cfg,cost,record,audit=None):
    """The only canonical update: FP64 micro8, mean per-row declared loss, one clip."""
    import time
    params=[v for v in model.parameters() if v.requires_grad];ref=params[0]
    opt.zero_grad(set_to_none=True);value=0.
    if ref.is_cuda:torch.cuda.synchronize()
    tick=time.perf_counter()
    for i in range(0,len(tokens),cfg['microbatch']):
        x=tokens[i:i+cfg['microbatch']].to(ref);y=labels[i:i+len(x)].to(ref.device)
        losses=model.loss(x,y)
        if not bool(torch.isfinite(losses).all()):record(dict(check='finite_objective',PASS=False,micro_begin=i,values=str(losses.detach().cpu().tolist()),completed_counts=dict(cost)))
        assert torch.isfinite(losses).all()
        loss=losses.sum()/len(tokens);loss.backward();value+=float(loss.detach())
        cost['forward_rows']+=len(x);cost['backward_rows']+=len(x)
    norm=torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip'])
    if not bool(torch.isfinite(norm)):record(dict(check='finite_gradient',PASS=False,norm=str(float(norm)),completed_counts=dict(cost)))
    assert torch.isfinite(norm)
    if ref.is_cuda:torch.cuda.synchronize()
    gradient_seconds=time.perf_counter()-tick
    if audit is not None:
        expected=[]
        for v in params:
            st=opt.state[v];step=int(st.get('step',0))+1
            w=v.detach().cpu().numpy().copy();g=v.grad.detach().cpu().numpy()
            m=st['exp_avg'].detach().cpu().numpy() if st else np.zeros_like(w)
            u=st['exp_avg_sq'].detach().cpu().numpy() if st else np.zeros_like(w)
            m=.9*m+.1*g;u=.999*u+.001*g*g
            expected.append(w*(1-cfg['learning_rate']*cfg['weight_decay'])-cfg['learning_rate']*(m/(1-.9**step))/(np.sqrt(u/(1-.999**step))+1e-8))
    tick=time.perf_counter();opt.step()
    if ref.is_cuda:torch.cuda.synchronize()
    seconds=gradient_seconds+time.perf_counter()-tick;cost['updates']+=1
    if audit is not None:
        audit.update(absolute_error=max(float(abs(v.detach().cpu().numpy()-e).max()) for v,e in zip(params,expected)),normalized_error=max(float((abs(v.detach().cpu().numpy()-e)/(cfg['operator_tolerance']+cfg['operator_relative_tolerance']*abs(e))).max()) for v,e in zip(params,expected)),completed_updates=cost['updates'],objective=value,gradient_norm=float(norm))
    return dict(objective=value,gradient_norm=float(norm),canonical_step_seconds=seconds)
