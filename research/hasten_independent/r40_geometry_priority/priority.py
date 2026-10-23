"""Filter only opposing native geometry gradient; keep original protected space."""
import torch
MAPPING='learned_lorentz_exp.exp_map.0.weight'
def prepare_priority(learner,contrast):
 a=learner.audit.setdefault('geometry_priority',dict(prepare_calls=0,calls=0,conflict_batches=0,zero_contrast_batches=0,tiny_geometry_residual_batches=0,rng_first_prepare_preserved=False,min_geometry_contrast_cosine_before=1.,min_geometry_contrast_cosine_after=1.,max_removed_geometry_fraction=0.,max_mapping_projection_relative_error=0.))
 if a['prepare_calls']==0:cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
 named=[(n,p) for n,p in learner._network.named_parameters() if p.requires_grad]
 grads=torch.autograd.grad(contrast,[p for n,p in named],retain_graph=True,allow_unused=True);saved=[]
 for (name,p),g in zip(named,grads):
  g=torch.zeros_like(p) if g is None else g.detach().clone()
  if learner._cur_task>0 and name==MAPPING:g=learner._network.learned_lorentz_exp.project_gradient(g)
  assert torch.isfinite(g).all();saved.append(g)
 learner._priority_named=named;learner._priority_contrast=saved;a['prepare_calls']+=1
 if a['prepare_calls']==1:
  assert torch.equal(cpu,torch.get_rng_state())
  if gpu:assert all(torch.equal(x,y) for x,y in zip(gpu,torch.cuda.get_rng_state_all()))
  a['rng_first_prepare_preserved']=True
@torch.no_grad()
def apply_priority(learner):
 a=learner.audit['geometry_priority'];named=learner._priority_named;c=learner._priority_contrast
 assert [n for n,p in named]==[n for n,p in learner._network.named_parameters() if p.requires_grad]
 t=[torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for n,p in named]
 dot=sum(float(((tt.double()-cc.double())*cc.double()).sum()) for tt,cc in zip(t,c));cn=sum(float(cc.double().square().sum()) for cc in c);gn=sum(float((tt.double()-cc.double()).square().sum()) for tt,cc in zip(t,c))
 assert all(torch.isfinite(v).all() for v in t) and torch.isfinite(torch.tensor([dot,cn,gn],dtype=torch.float64)).all()
 a['calls']+=1;before=dot/(cn*gn)**.5 if cn>0 and gn>0 else 0.;a['min_geometry_contrast_cosine_before']=min(a['min_geometry_contrast_cosine_before'],before)
 if cn<=1e-24:a['zero_contrast_batches']+=1
 elif dot<0:
  coefficient=dot/cn;a['conflict_batches']+=1
  fraction=abs(coefficient)*(cn/gn)**.5 if gn>0 else 0.;assert fraction<=1.0001;a['max_removed_geometry_fraction']=max(a['max_removed_geometry_fraction'],fraction)
  for (name,p),tt,cc in zip(named,t,c):
   z=(tt.double()-coefficient*cc.double()).to(tt.dtype)
   if p.grad is not None:p.grad.copy_(z)
   else:assert torch.count_nonzero(z)==0
 actual=[torch.zeros_like(p) if p.grad is None else p.grad for n,p in named]
 after_dot=sum(float(((tt.double()-cc.double())*cc.double()).sum()) for tt,cc in zip(actual,c));after_gn=sum(float((tt.double()-cc.double()).square().sum()) for tt,cc in zip(actual,c))
 # Vanishing geometry residual has no reliable cosine; preserve absolute tolerance.
 tiny=after_gn<=1e-12*max(cn,gn,1e-30)
 a['tiny_geometry_residual_batches']+=int(tiny)
 after=after_dot/(cn*after_gn)**.5 if cn>1e-24 and after_gn>0 and not tiny else 0.
 assert after>=-1e-4,(after,before)
 a['min_geometry_contrast_cosine_after']=min(a['min_geometry_contrast_cosine_after'],after)
 for (name,p),g in zip(named,actual):
  assert torch.isfinite(g).all()
  if learner._cur_task>0 and name==MAPPING:
   projected=learner._network.learned_lorentz_exp.project_gradient(g);err=float((g.double()-projected.double()).norm()/g.double().norm().clamp_min(1e-30));assert err<=1e-4
   a['max_mapping_projection_relative_error']=max(a['max_mapping_projection_relative_error'],err)
 del learner._priority_named,learner._priority_contrast
