"""Canonical micro8 objective, legal patient memory and same-gradient AdamW check."""
import hashlib,time
import numpy as np
import torch
from torch.nn import functional as F

ARMS=['NATIVE_RIDGE','FEATURE_FROZEN','FULL_SCORE_FROZEN']
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

def update(model,opt,tokens,labels,cfg,cost,old_features=None,audit=None):
    params=[v for v in model.parameters() if v.requires_grad];device=params[0].device
    opt.zero_grad(set_to_none=True);loss_sum=0.
    if device.type=='cuda':torch.cuda.synchronize()
    tick=time.perf_counter()
    for i in range(0,len(tokens),cfg['microbatch']):
        xx=tokens[i:i+cfg['microbatch']].to(device).float();yy=labels[i:i+cfg['microbatch']].to(device)
        zz=None if old_features is None else [z[i:i+len(xx)].to(device).float() for z in old_features]
        loss=F.cross_entropy(model(xx,zz),yy,reduction='sum')/len(tokens)
        assert torch.isfinite(loss);loss.backward();loss_sum+=float(loss.detach())
        cost['forward_rows']+=len(xx);cost['backward_rows']+=len(xx)
    norm=torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip']);assert torch.isfinite(norm)
    if device.type=='cuda':torch.cuda.synchronize()
    gradient_seconds=time.perf_counter()-tick
    if audit is not None:
        # Same already-clipped gradient and same optimizer initial state, all coordinates.
        expected=[]
        for v in params:
            st=opt.state[v];step=int(st.get('step',0))+1
            w=v.detach().cpu().double().numpy();g=v.grad.detach().cpu().double().numpy()
            m=st['exp_avg'].detach().cpu().double().numpy() if st else np.zeros_like(w)
            u=st['exp_avg_sq'].detach().cpu().double().numpy() if st else np.zeros_like(w)
            m=.9*m+.1*g;u=.999*u+.001*g*g
            expected.append(w*(1-cfg['learning_rate']*cfg['weight_decay'])-cfg['learning_rate']*(m/(1-.9**step))/(np.sqrt(u/(1-.999**step))+1e-8))
    tick=time.perf_counter();opt.step()
    if device.type=='cuda':torch.cuda.synchronize()
    canonical_seconds=gradient_seconds+time.perf_counter()-tick;cost['updates']+=1
    if audit is not None:
        error=max(float(np.max(abs(v.detach().cpu().double().numpy()-e))) for v,e in zip(params,expected))
        ratio=max(float(np.max(abs(v.detach().cpu().double().numpy()-e)/(cfg['operator_tolerance']+cfg['operator_relative_tolerance']*abs(e)))) for v,e in zip(params,expected))
        audit['AdamW_normalized_error']=ratio
        audit['AdamW_same_gradient_error']=error;audit['gradient_norm']=float(norm);audit['loss']=loss_sum
    return dict(CE=loss_sum,gradient_norm=float(norm),canonical_step_seconds=canonical_seconds)


def forward(model,tokens,micro=8):
    dev=next(model.parameters()).device;dtype=next(model.parameters()).dtype
    with torch.no_grad():return torch.cat([model(tokens[i:i+micro].to(dev,dtype=dtype)).cpu() for i in range(0,len(tokens),micro)])

def cached(model,tokens,micro=8):
    out=[]
    for expert in list(model.experts)[:-1]:
        with torch.no_grad():out.append(torch.cat([expert.upper(tokens[i:i+micro].to(expert.head.device).float()).cpu() for i in range(0,len(tokens),micro)]))
    return out

def restore(template,groups,states,freeze):
    from core import Growing
    m=Growing(freeze)
    for labels,s in zip(groups,states):
        m.append(template,s['head'].float(),labels);m.experts[-1].load_state_dict(s)
    return m
