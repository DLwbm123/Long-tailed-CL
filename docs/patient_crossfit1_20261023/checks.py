"""Independent dual objectives, finite differences and coupled-gradient recomputation."""
import copy,time
import numpy as np
import torch
from torch.nn import functional as F
from core import NativeEvidence,representation_objective,cached_update,affine

def np_objective(z,teacher,y,w,arm,ridge,beta,head=None):
    z=np.asarray(z,dtype=np.float64);n=len(z)//2;norm=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12);phi=np.concatenate([norm,np.ones((len(z),1))],1)/np.sqrt(2)
    def fit(ix):
        a=phi[ix]*np.sqrt(w[ix,None]);t=np.eye(10)[y[ix]]*np.sqrt(w[ix,None]);return a.T@np.linalg.solve(a@a.T+ridge*np.eye(len(a)),t)
    if arm=='CE_RC':
        h=head/np.maximum(np.linalg.norm(head,axis=1,keepdims=True),1e-12);v=25*z@h.T;v-=v.max(1,keepdims=True);risk=np.sum(w*(np.log(np.exp(v).sum(1))-v[np.arange(len(y)),y]))/2
    else:
        hs,hq=fit(np.arange(n)),fit(np.arange(n,len(z)));s=phi[:n]@(hs if arm=='RIDGE_IN' else hq);q=phi[n:]@(hq if arm=='RIDGE_IN' else hs);risk=(np.sum(w[:n]*np.sum((s-np.eye(10)[y[:n]])**2,1))+np.sum(w[n:]*np.sum((q-np.eye(10)[y[n:]])**2,1)))/2
    return risk+beta*np.sum(w*(1-np.sum(z*teacher,1)))/2

def math_checks(cfg):
    from functools import partial
    from timm.models.vision_transformer import Block
    torch.manual_seed(cfg['seed']);start=time.time();z=F.normalize(torch.randn(8,24,dtype=torch.float64),dim=1).requires_grad_();teacher=F.normalize(torch.randn_like(z),dim=1);labels=torch.tensor([0,1,2,3]*2);w=torch.full((8,),.25,dtype=torch.float64);head=torch.randn(10,24,dtype=torch.float64)
    objective_error=0.;fd_error=0.
    for arm in cfg['arms'][1:]:
        loss,_,_=representation_objective(z,teacher,labels,w,arm,cfg['ridge'],cfg['consistency_weight'],head);g=torch.autograd.grad(loss,z)[0];a=z.detach().numpy().copy();f=lambda x:np_objective(x,teacher.numpy(),labels.numpy(),w.numpy(),arm,cfg['ridge'],cfg['consistency_weight'],head.numpy());objective_error=max(objective_error,abs(f(a)-float(loss)))
        plus=a.copy();minus=a.copy();plus[0,2]+=1e-5;minus[0,2]-=1e-5;fd_error=max(fd_error,abs((f(plus)-f(minus))/2e-5-float(g[0,2])))
    assert objective_error<1e-9 and fd_error<2e-6
    norm=partial(torch.nn.LayerNorm,eps=1e-6);b=torch.nn.Module();b.blocks=torch.nn.Sequential(*[Block(12,3,mlp_ratio=2,qkv_bias=True,norm_layer=norm) for _ in range(2)]);b.norm=norm(12);m=NativeEvidence(b,torch.randn(2,2,12)*.02);x=torch.randn(8,5,12);t=F.normalize(torch.randn(8,24),dim=1);w=w.float();grad_error=0.;update_error=0.
    for arm in cfg['arms'][1:]:
        direct=copy.deepcopy(m);cached=copy.deepcopy(m);h1=torch.nn.Parameter(head.float().clone());h2=torch.nn.Parameter(head.float().clone());ps1=[direct.prompts]+([h1] if arm=='CE_RC' else []);ps2=[cached.prompts]+([h2] if arm=='CE_RC' else []);o1=torch.optim.AdamW(ps1,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);o2=torch.optim.AdamW(ps2,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);loss,_,_=representation_objective(direct(x),t,labels,w,arm,cfg['ridge'],cfg['consistency_weight'],h1 if arm=='CE_RC' else None);loss.backward();expected=direct.prompts.grad.detach().clone();torch.nn.utils.clip_grad_norm_(ps1,1e9);o1.step()
        cc=dict(cached_body_forward_rows=0,recomputed_body_forward_rows=0,body_backward_rows=0,updates=0,logical_rows=0,inner_dual_solves=0);small={**cfg,'microbatch':2,'gradient_clip':1e9};cached_update(cached,o2,x,t,labels,w,arm,small,cc,h2 if arm=='CE_RC' else None);grad_error=max(grad_error,float((expected-cached.prompts.grad).abs().max()));update_error=max(update_error,float((direct.prompts-cached.prompts).abs().max()))
    assert grad_error<3e-5 and update_error<3e-6
    return dict(state='PASS',independent_objective_error=objective_error,finite_difference_error=fd_error,full_graph_vs_cached_prompt_gradient_error=grad_error,full_graph_vs_cached_AdamW_update_error=update_error,CPU_objective_rows=24,CPU_fullgraph_rows=24,CPU_cached_forward_rows=24,CPU_recomputed_forward_backward_rows=24,CPU_discarded_AdamW_updates=6,NumPy_objectives=9,NumPy_objective_rows=72,internal_seconds=time.time()-start,new_images=0)

def real_gradient_check(backbone,prompts,tokens,teacher,labels,cfg):
    # Eight real rows: four globally separated support/query patient pairs.
    device=next(backbone.parameters()).device;x=tokens.to(device).float();t=teacher.to(device).float();y=labels.to(device);w=torch.full((8,),.25,device=device);d=NativeEvidence(backbone,prompts.to(device));c=NativeEvidence(backbone,prompts.to(device));o1=torch.optim.AdamW([d.prompts],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);o2=torch.optim.AdamW([c.prompts],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);loss,_,_=representation_objective(d(x),t,y,w,'RIDGE_OUT',cfg['ridge'],cfg['consistency_weight']);loss.backward();torch.nn.utils.clip_grad_norm_([d.prompts],cfg['gradient_clip']);g=d.prompts.grad.detach().clone();o1.step();cost=dict(cached_body_forward_rows=0,recomputed_body_forward_rows=0,body_backward_rows=0,updates=0,logical_rows=0,inner_dual_solves=0);cached_update(c,o2,tokens,teacher,labels,w.cpu(),'RIDGE_OUT',{**cfg,'microbatch':2},cost);ge=float((g-c.prompts.grad).abs().max());ue=float((d.prompts-c.prompts).abs().max());assert ge<3e-4 and ue<2e-5
    return dict(state='PASS',full_vs_cache_clipped_gradient_error=ge,AdamW_update_error=ue,real_body_forward_rows=24,real_body_backward_rows=16,discarded_updates=2,inner_dual_solves=4)
