"""Paired patient crossfit, differentiable analytic readout and exact VJP recomputation."""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ARMS=['FROZEN_NATIVE','CE_RC','RIDGE_IN','RIDGE_OUT']
def side_for_patient(seed,group):
    return 'S' if hashlib.sha256(f'{seed}:patient_crossfit:{group}'.encode()).digest()[0]<128 else 'Q'
def pools(rows,classes,seed):
    out={c:{'S':{},'Q':{}} for c in classes}
    for i,r in enumerate(rows):out[r['label']][side_for_patient(seed,r['group'])].setdefault(r['group'],[]).append(i)
    assert all(v['S'] and v['Q'] for v in out.values())
    left={g for v in out.values() for g in v['S']};right={g for v in out.values() for g in v['Q']};assert not left&right
    return out
def episode(rng,pp,classes):
    selected=[int(c) for c in rng.choice(classes,8,replace=False)];ix={'S':[],'Q':[]};labels={'S':[],'Q':[]};weights={'S':[],'Q':[]}
    for label,c in enumerate(selected):
        n=min(4,len(pp[c]['S']),len(pp[c]['Q']))
        for side in ['S','Q']:
            patients=list(pp[c][side])
            for j in rng.choice(len(patients),n,replace=False):
                ix[side].append(int(rng.choice(pp[c][side][patients[int(j)]])));labels[side].append(classes.index(c));weights[side].append(1/(8*n))
    return selected,ix,labels,weights

def affine(z):
    z=F.normalize(z.double(),dim=1)
    return torch.cat([z,torch.ones(len(z),1,device=z.device,dtype=z.dtype)],1)/np.sqrt(2)
def dual_head(phi,labels,weights,ridge,classes):
    weights=torch.as_tensor(weights,dtype=phi.dtype,device=phi.device);labels=torch.as_tensor(labels,device=phi.device)
    z=phi*weights.sqrt()[:,None];target=F.one_hot(labels,classes).to(phi.dtype)*weights.sqrt()[:,None]
    # Differentiable solve: no explicit inverse or detached inner head.
    return z.T@torch.linalg.solve(z@z.T+ridge*torch.eye(len(z),dtype=z.dtype,device=z.device),target)
def representation_objective(z,teacher,labels,weights,arm,ridge,beta,temphead=None):
    n=len(z)//2;assert len(z)%2==0
    f=affine(z);ls,lq=labels[:n],labels[n:];ws,wq=weights[:n],weights[n:];classes=10
    if arm=='CE_RC':
        logits=25*z@F.normalize(temphead,dim=1).T
        risk=(weights*F.cross_entropy(logits,labels,reduction='none')).sum()/2
    else:
        hs=dual_head(f[:n],ls,ws,ridge,classes);hq=dual_head(f[n:],lq,wq,ridge,classes)
        ys=F.one_hot(ls,classes).to(f.dtype);yq=F.one_hot(lq,classes).to(f.dtype)
        if arm=='RIDGE_IN':rs=(f[:n]@hs-ys).square().sum(1);rq=(f[n:]@hq-yq).square().sum(1)
        else:
            assert arm=='RIDGE_OUT';rs=(f[:n]@hq-ys).square().sum(1);rq=(f[n:]@hs-yq).square().sum(1)
        risk=((ws*rs).sum()+(wq*rq).sum())/2
    rc=(weights*(1-(z*teacher.detach()).sum(1))).sum()/2
    return risk+beta*rc,risk,rc

def cached_update(model,optimizer,tokens,teacher,labels,weights,arm,cfg,cost,temphead=None):
    device=next(model.parameters()).device;micro=cfg['microbatch'];optimizer.zero_grad(set_to_none=True)
    # Exact vector-Jacobian recomputation for sample-coupled inner solves; no dropout.
    with torch.no_grad():cached=torch.cat([model(tokens[i:i+micro].to(device).float()) for i in range(0,len(tokens),micro)])
    cost['cached_body_forward_rows']+=len(tokens);leaf=cached.detach().requires_grad_(True)
    loss,risk,rc=representation_objective(leaf,teacher.to(device),labels.to(device),weights.to(device),arm,cfg['ridge'],cfg['consistency_weight'],temphead)
    loss.backward();grad=leaf.grad.detach()
    for i in range(0,len(tokens),micro):
        z=model(tokens[i:i+micro].to(device).float());z.backward(grad[i:i+micro]);cost['recomputed_body_forward_rows']+=len(z);cost['body_backward_rows']+=len(z)
    params=[model.prompts]+([] if temphead is None else [temphead]);norm=torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip']);assert torch.isfinite(norm);optimizer.step();cost['updates']+=1;cost['logical_rows']+=len(tokens)
    if arm.startswith('RIDGE'):cost['inner_dual_solves']+=2
    return dict(objective=float(loss.detach()),outer_risk=float(risk.detach()),response_consistency=float(rc.detach()),gradient_norm=float(norm))


class NativeEvidence(nn.Module):
    def __init__(self,backbone,prompts=None):
        super().__init__();self.backbone=backbone.eval().requires_grad_(False)
        if prompts is None:self.register_parameter('prompts',None)
        else:self.prompts=nn.Parameter(prompts.clone())
    def train(self,mode=True):
        super().train(mode);self.backbone.eval();return self
    def forward(self,tokens):
        x=tokens
        for j,block in enumerate(self.backbone.blocks):
            if self.prompts is not None and j<len(self.prompts):
                n=len(self.prompts[j]);pt=self.prompts[j][None].expand(len(x),-1,-1)
                x=block(torch.cat([x[:,:1],pt,x[:,1:]],1));x=torch.cat([x[:,:1],x[:,1+n:]],1)
            else:x=block(x)
        x=self.backbone.norm(x)
        return F.normalize(torch.cat([x[:,0],x[:,1:].mean(1)],1),dim=1)

def np_evidence(tokens,state,prompts=None,heads=12,eps=1e-6):
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
    x=norm(x,'norm.');z=np.concatenate([x[:,0],x[:,1:].mean(1)],1)
    return z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)

def patient_weights(rows):
    counts={}
    for r in rows:counts[r['group']]=counts.get(r['group'],0)+1
    return np.array([1/(len(counts)*counts[r['group']]) for r in rows]),len(counts)
def moments(phi,weights):
    w=torch.as_tensor(weights,dtype=torch.float64,device=phi.device)
    return phi.T@(w[:,None]*phi),(w[:,None]*phi).sum(0)
def solve(records,arm,ridge):
    gram=sum(r[arm]['gram'] for r in records)/len(records);rhs=torch.stack([r[arm]['mean']/len(records) for r in records],1)
    mat=gram+ridge*torch.eye(len(gram),dtype=torch.float64,device=gram.device);head=torch.linalg.solve(mat,rhs)
    error=float((mat@head-rhs).abs().max());assert error<1e-9 and torch.isfinite(head).all()
    return head,error
