"""Fixed factorized decision and state/RNG-neutral convergence observations."""
import json,math,time
from pathlib import Path
import torch
import torch.nn.functional as F
def balanced(v,y):return torch.stack([v[y==c].mean() for c in y.unique()]).mean()
def configure(ns,cfg):
 mode=cfg['decision_mode'];INC=ns['INC'];GEO=ns['GEO'];L=ns['CheckedLearner'];types=ns['TYPE'];parents=ns['PARENTS'];base_args=ns['make_args'];base_pred=ns['predictions'];old_update=INC.hyper_net.update_task
 def args(seed,arm):
  v=base_args(seed,arm);v.update(decision_mode=mode);return v
 ns['make_args']=args
 def update(net):
  old_update(net)
  if mode=='factorized' and not hasattr(net,'coarse_prototypes'):
   net.coarse_prototypes=torch.nn.Parameter(torch.zeros(len(parents),512,device='cuda'));net.register_buffer('coarse_initialized',torch.zeros(len(parents),device='cuda',dtype=torch.bool))
 INC.hyper_net.update_task=update
 def grouping(n):
  raw=torch.tensor(types[:n],device='cuda');active=raw.unique(sorted=True);local=torch.searchsorted(active,raw);prior=torch.bincount(local,minlength=len(active)).float().log();return raw,active,local,prior
 def posterior(net,raw,z):
  _,active,groups,prior=grouping(z.shape[1]);gp=net.coarse_prototypes[active];coarse=net.logit_scale.exp()*(F.normalize(raw,dim=1)@F.normalize(gp,dim=1).T)+prior;lg=F.log_softmax(coarse,dim=1)
  p=torch.stack([lg[:,groups[c]]+z[:,c]-torch.logsumexp(z[:,groups==groups[c]],dim=1) for c in range(z.shape[1])],dim=1);assert torch.allclose(p.exp().sum(1),torch.ones(len(p),device=p.device,dtype=p.dtype),atol=1e-6);return p,coarse,groups
 def predictions(learner,x):
  p=base_pred(learner,x)
  if mode!='native':
   net=learner._network;z=p['hyper'].cuda()*net.logit_scale.exp()
   p['final']=(F.log_softmax(z,dim=1) if mode=='flat' else posterior(net,x.cuda(),z)[0]).cpu()
  return p
 ns['predictions']=predictions
 def summarize(learner,x,y):
  p=predictions(learner,x)['final'];prob=p if mode=='native' else p.exp();assert torch.isfinite(prob).all() and torch.allclose(prob.sum(1),torch.ones(len(prob),dtype=prob.dtype),atol=1e-6)
  n=p.shape[1];correct=torch.bincount(y[p.argmax(1)==y],minlength=n);counts=torch.bincount(y,minlength=n);cc=[c for c in range(n) if counts[c]>0];tail=[c for c in cc if learner._curve_train_counts[c]<250];nll=-prob[torch.arange(len(y)),y].clamp_min(1e-30).log()
  return dict(counts=counts.tolist(),correct=correct.tolist(),macro_BA=100*sum(float(correct[c]/counts[c]) for c in cc)/len(cc),tail_BA=None if not tail else 100*sum(float(correct[c]/counts[c]) for c in tail)/len(tail),macro_NLL=float(torch.stack([nll[y==c].mean() for c in cc]).mean()),tail_NLL=None if not tail else float(torch.stack([nll[y==c].mean() for c in tail]).mean()))
 def epoch(self,epoch,objective,lr):
  net=self._network;cpu=torch.get_rng_state().clone();cuda=[v.clone() for v in torch.cuda.get_rng_state_all()];state={k:v.detach().clone() for k,v in net.state_dict().items()};training={m:m.training for m in net.modules()};dm=self.data_manager;n=self._total_classes
  if not hasattr(self,'_curve_train_counts'):
   self._curve_train_counts=torch.bincount(dm.y[dm.tr],minlength=len(types)).tolist()
  ids=[]
  for c in range(n):ids.extend([i for i in dm.tr if int(dm.y[i])==c][:64])
  with torch.no_grad():
   fit=summarize(self,dm.x[ids],dm.y[ids]);hold=None
   if hasattr(self,'_curve_holdout'):hold=summarize(self,*self._curve_holdout)
  net.load_state_dict(state,strict=True)
  for module,flag in training.items():module.training=flag
  assert torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()));assert all(torch.equal(v,net.state_dict()[k]) for k,v in state.items())
  rec=dict(seed=self.seed,stage=self._cur_task,mode=mode,epoch=epoch+1,optimizer_steps=self.audit['steps'],objective_mean=objective,lr=lr,fit=fit,holdout=hold,observation_state_rng_restored=True)
  target=Path(cfg['output'])/'curves'/f'{self.seed}_stage{self._cur_task}.jsonl';target.parent.mkdir(exist_ok=True)
  with target.open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
 def prepare(self):
  net=self._network;n=self._total_classes
  if mode=='factorized':
   raw,active,_,_=grouping(n);before=net.coarse_prototypes.detach().clone();new=[]
   with torch.no_grad():
    for g in active.tolist():
     if not net.coarse_initialized[g]:net.coarse_prototypes[g].copy_(F.normalize(net.fc.weight[:n][raw==g].detach().mean(0),dim=0));net.coarse_initialized[g]=True;new.append(g)
   assert all(torch.equal(before[g],net.coarse_prototypes[g]) for g in active.tolist() if g not in new);assert all(float(net.coarse_prototypes[g].abs().max())==0 for g in range(len(parents)) if g not in active.tolist());self.audit['coarse_new_groups']=new;self.audit['coarse_groups']=active.tolist()
  epoch(self,-1,None,None)
 def loss(self,raw,replay,logits,targets):
  net=self._network;z=net.logit_scale.exp()*logits;self.audit['decision_calls']=self.audit.get('decision_calls',0)+1
  if mode=='flat':return balanced(F.cross_entropy(z,targets,reduction='none'),targets)
  if replay is not None:raw=torch.cat([raw,replay],dim=0)
  assert len(raw)==len(targets);p,g,groups=posterior(net,raw,z);fine=-z[torch.arange(len(targets),device='cuda'),targets]+torch.stack([torch.logsumexp(z[k,groups==groups[targets[k]]],dim=0) for k in range(len(targets))]);coarse=F.cross_entropy(g,groups[targets],reduction='none')
  if self.audit['decision_calls']==1:
   params=[v for key,v in net.named_parameters() if v.requires_grad and key!='coarse_prototypes'];gr=torch.autograd.grad(balanced(coarse,targets),params,retain_graph=True,allow_unused=True);assert all(t is None for t in gr);assert torch.autograd.grad(balanced(fine,targets),net.coarse_prototypes,retain_graph=True,allow_unused=True)[0] is None;self.audit['coarse_fine_gradient_separated']=True
  return balanced(-p[torch.arange(len(targets),device='cuda'),targets],targets)
 before=L._before_step
 def inspect(self):
  before(self)
  if mode=='factorized':
   g=self._network.coarse_prototypes.grad;assert g is not None and torch.isfinite(g).all();self.audit['coarse_gradient_calls']=self.audit.get('coarse_gradient_calls',0)+1;self.audit['coarse_gradient_max']=max(self.audit.get('coarse_gradient_max',0.),float(g.norm()))
 L._before_step=inspect
 L._prepare_decision=prepare;L._decision_loss=loss;L._epoch_done=epoch
 # Group slots exist before strict checkpoint load, including not-yet-active groups (metadata only).
 ns['_decision_posterior']=posterior
