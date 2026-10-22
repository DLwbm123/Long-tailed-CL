"""Class-risk feasible transitions, not historical target distillation."""
import hashlib
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ARMS=['ER','PROJECT','FEASIBLE']

class Model(nn.Module):
    def __init__(self, dimension, rank, means):
        super().__init__()
        self.down=nn.Linear(dimension,rank);self.up=nn.Linear(rank,dimension)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
        self.head=nn.Parameter(F.normalize(means.clone(),dim=1))
        self.bias=nn.Parameter(torch.zeros(len(means),device=means.device,dtype=means.dtype))
    def forward(self,x,scale):
        x=F.normalize(x,dim=1)
        z=F.normalize(x+self.up(F.silu(self.down(x))),dim=1)
        return z,scale*z@F.normalize(self.head,dim=1).T+self.bias
    def expand(self,means,memory,scale,allowance):
        """Common admission initializer; each old row's CE increases at most allowance."""
        with torch.no_grad():
            old=self(memory,scale)[1]
            oldk=len(self.head)
            self.head=nn.Parameter(torch.cat([self.head.detach(),F.normalize(means,dim=1)]))
            self.bias=nn.Parameter(torch.cat([self.bias.detach(),torch.zeros(len(means),device=means.device)]))
            new=self(memory,scale)[1][:,oldk:]
            shift=torch.minimum(torch.tensor(0.,device=means.device),
                (torch.logsumexp(old,1)-torch.logsumexp(new,1)).min()+np.log(np.expm1(allowance)))
            self.bias[oldk:]=shift
        return float(shift)

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

def state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def flat(parameters):
    return torch.cat([p.detach().flatten() for p in parameters])

def setflat(model,v):
    start=0
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(v[start:start+p.numel()].reshape_as(p));start+=p.numel()
    assert start==v.numel()

def risks(model,x,y,scale):
    if not len(x):return x.new_empty(0)
    loss=F.cross_entropy(model(x,scale)[1],y,reduction='none')
    return torch.stack([loss[y==c].mean() for c in range(int(y.max())+1)])

def project(g,G,cfg):
    """Dual nonnegative coordinate descent. Normalizing constraints changes no halfspace."""
    if not len(G):return g,dict(sweeps=0,residual=0.,correction_norm=0.)
    norms=G.norm(dim=1);keep=norms>1e-12
    if not bool(keep.any()):return g,dict(sweeps=0,residual=0.,correction_norm=0.)
    H=G[keep]/norms[keep,None]
    gram=(H.double()@H.double().T).cpu().numpy();b=(H.double()@g.double()).cpu().numpy()
    lam=np.zeros(len(b));q=b.copy()
    for sweep in range(cfg['projection_max_sweeps']):
        for j in range(len(b)):
            new=max(0.,lam[j]-q[j]/max(gram[j,j],1e-15))
            delta=new-lam[j];lam[j]=new;q+=gram[:,j]*delta
        kkt=np.where(lam>1e-10,abs(q),np.maximum(-q,0.)).max()
        if kkt<=cfg['projection_solver_tolerance']:break
    d=g+torch.tensor(lam,device=g.device,dtype=g.dtype)@H
    residual=float(torch.clamp(-(H@d),min=0).max())
    assert np.isfinite(kkt) and kkt<=cfg['projection_solver_tolerance']*2, 'projection solver qualification failed'
    assert residual<=cfg['projection_direction_tolerance'], 'infeasible first-order direction'
    return d,dict(sweeps=sweep+1,residual=residual,correction_norm=float((d-g).norm()))

def step(model,x,y,mx,my,ceiling,arm,cfg,cost,capture=False):
    model.zero_grad(set_to_none=True)
    loss=F.cross_entropy(model(x,cfg['logit_scale'])[1],y);loss.backward()
    cost['batch_forward_rows']+=len(x);cost['batch_backward_rows']+=len(x)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
    assert torch.isfinite(loss) and torch.isfinite(norm)
    g=torch.cat([p.grad.detach().flatten() for p in model.parameters()])
    G=None
    original=flat(model.parameters());projection=dict(sweeps=0,residual=0.,correction_norm=0.)
    if arm!='ER' and len(mx):
        rs=risks(model,mx,my,cfg['logit_scale']);G=[]
        cost['constraint_forward_rows']+=len(mx)
        for j,r in enumerate(rs):
            grads=torch.autograd.grad(r,list(model.parameters()),retain_graph=j+1<len(rs))
            G.append(torch.cat([v.flatten() for v in grads]))
            # Full graph backward visits the full memory batch for each scalar risk.
            cost['constraint_backward_rows']+=len(mx)
        G=torch.stack(G)
        d,projection=project(g,G,cfg)
    else:d=g
    cost['opportunities']+=1
    candidates=cfg['backtracking_candidates'] if arm=='FEASIBLE' and len(mx) else 1
    accepted=False;after=[];factor=0.
    for k in range(candidates):
        factor=2.**(-k)
        setflat(model,original-cfg['learning_rate']*factor*d)
        if arm=='FEASIBLE' and len(mx):
            with torch.no_grad():r=risks(model,mx,my,cfg['logit_scale'])
            cost['feasibility_forward_rows']+=len(mx);cost['candidate_checks']+=1
            after=r.cpu().tolist()
            if bool((r<=ceiling+cfg['risk_training_tolerance']).all()):accepted=True;break
        else:accepted=True;break
    if not accepted:setflat(model,original);factor=0.
    else:cost['accepted_updates']+=1
    if arm=='FEASIBLE' and len(mx):
        with torch.no_grad():r=risks(model,mx,my,cfg['logit_scale'])
        cost['feasibility_forward_rows']+=len(mx)
        assert bool((r<=ceiling+cfg['risk_training_tolerance']).all()), 'old-memory finite-step feasibility lost'
        after=r.cpu().tolist()
    result=dict(CE=float(loss.detach()),gradient_norm_before_clip=float(norm),accepted=accepted,
                step_factor=factor,projection=projection,risks_after=after)
    if capture:result['_witness']=dict(direction=d.detach().cpu(),gradient=g.detach().cpu(),constraint_gradients=G.detach().cpu() if G is not None else None)
    return result
