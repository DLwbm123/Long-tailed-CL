"""Whole-memory finite-action full-feedback improvement and matched regression."""
import torch
from torch import nn
class Head(nn.Module):
 def __init__(self,seed):
  super().__init__()
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(seed);self.net=nn.Sequential(nn.Linear(48,32),nn.Tanh(),nn.Linear(32,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
  assert sum(p.numel() for p in self.parameters())==1601
 def forward(self,x):return self.net(x).squeeze(-1)

def prior(n,device):
 p=torch.full((n,),.5/max(1,n-1),device=device,dtype=torch.double);p[0]=.5 if n>1 else 1.;return p

def logprob(net,x):
 f=net(x).double();return (prior(len(x),x.device).log()+f-f[0]).log_softmax(0)

def objective(lp,old,R):
 b=(old*R).sum();sig=R.std(unbiased=False).clamp_min(.02);A=((R-b)/sig).detach()
 if bool((R==R[0]).all()):A=torch.zeros_like(A)
 ratio=lp.exp()/old;sur=(old*torch.minimum(ratio*A,ratio.clamp(.8,1.2)*A)).sum();kl=(old*(old.log()-lp)).sum();return -sur+.01*kl,dict(KL_old_new=float(kl.detach()),max_ratio_deviation=float((ratio-1).abs().max().detach()),clip_fraction=float(((ratio<.8)|(ratio>1.2)).double().mean()),equal_returns=bool((R==R[0]).all()))

def update(net,opt,x,returns,kind,rng,counter,prefix):
 R=torch.tensor(returns,device=x.device,dtype=torch.double);perm=torch.randperm(len(R),device=x.device,generator=rng) if kind=='REWARD_SHUFFLED' else torch.arange(len(R),device=x.device);R=R[perm];old=logprob(net,x).exp().detach().clone();target=((R-R[0])/.05).detach();stats=[]
 for step in range(2):
  opt.zero_grad(set_to_none=True)
  if kind=='SUP_GREEDY':
   f=net(x).double();loss=(f-f[0]-target).square().mean();s={'regression_MSE':float(loss.detach())}
  else:
   loss,s=objective(logprob(net,x),old,R)
   if step==0:assert s['max_ratio_deviation']<1e-10
  assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(grad);opt.step();counter[prefix+('_supervised_optimizer' if kind=='SUP_GREEDY' else '_policy_optimizer')]=counter.get(prefix+('_supervised_optimizer' if kind=='SUP_GREEDY' else '_policy_optimizer'),0)+1;stats.append(dict(s,gradient_norm=float(grad),loss=float(loss.detach())))
 return {'updates':stats,'permutation':perm.cpu().tolist(),'old_probability':old.cpu().tolist()}

@torch.no_grad()
def behavior(net,x,kind,seed,formal_rng=None):
 p0=prior(len(x),x.device)
 if kind=='SUP_GREEDY':
  f=net(x).double();g=f-f[0];ix=int(g.argmax()) if float(g.max())>0 else 0;p=torch.zeros_like(p0);p[ix]=1
 elif kind in ['BASE_ZERO','BALANCED']:p=torch.zeros_like(p0);p[0]=1
 elif kind=='RANDOM_REPAIR':p=p0
 else:p=logprob(net,x).exp()
 rng=torch.Generator(device=x.device).manual_seed(seed);draw=torch.multinomial(p,32,replacement=True,generator=rng);chosen=int(torch.multinomial(p,1,generator=formal_rng)) if formal_rng is not None else None
 nz=p>0;return chosen,dict(noop_probability=float(p[0]),KL_vs_prior=float((p[nz]*(p[nz].log()-p0[nz].log())).sum()),TV_vs_prior=float((p-p0).abs().sum()/2),candidate_entropy=float(-(p[nz]*p[nz].log()).sum()),draw32_counts=torch.bincount(draw,minlength=len(p)).cpu().tolist(),probability=p.cpu().tolist())
