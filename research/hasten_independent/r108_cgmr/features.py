"""Deterministic detached 48D set input; never accepts any query data."""
import math
import torch

def scarce(counts):
 x=torch.as_tensor(counts,dtype=torch.float64);return x<=torch.quantile(x,1/3,interpolation='lower')

@torch.no_grad()
def maxcover(a,b):
 if not len(a) or not len(b):return 0.
 return float(torch.cat([((x@b.T).clamp(-1,1).max(-1).values+1)/2 for x in a.split(256)]).mean())

@torch.no_grad()
def psi(pool,memory,model,counts,birth,stage):
 C=model.C;dev=pool.z.device;T=scarce(counts).to(dev);g=memory.gidx;poolg=pool.gidx;cover=[];redund=[];single=0
 for c in range(C):
  chosen=memory.z[g[memory.y[g]==c]];assert len(chosen);cover.append(maxcover(pool.z[poolg[pool.y[poolg]==c]],chosen))
  if len(chosen)<2:redund.append(0.);single+=1
  else:
   sim=chosen@chosen.T;redund.append(float(sim[~torch.eye(len(chosen),device=dev,dtype=torch.bool)].mean().clamp(-1,1)))
 local=[]
 for parent in memory.parents:
  a=pool.z[(pool.pid==parent)&(pool.view>0)];b=memory.z[(memory.pid==parent)&(memory.view>0)];local.append(maxcover(a,b))
 n=torch.bincount(memory.y[g],minlength=C).double();f=n/n.sum();x,y,_=memory.bags();logits=model(x).double();p=logits.softmax(-1);true=p.gather(1,y[:,None]).squeeze(1);other=logits.clone();other.scatter_(1,y[:,None],-1e6);margin=((logits.gather(1,y[:,None]).squeeze(1)-other.max(-1).values)/40).clamp(-1,1);margin.zero_() if C==1 else None;entropy=-(p*p.clamp_min(1e-30).log()).sum(-1)/max(1e-30,math.log(C));entropy.zero_() if C==1 else None
 macro=lambda a:float(torch.stack([a[y==c].mean() for c in range(C)]).mean())
 births=torch.tensor(birth,device=dev);age=(stage-births[y]).double();cv=torch.tensor(cover,device=dev)
 v=[sum(cover)/C,float(cv[T].mean()),sum(local)/len(local),sum(redund)/C,float(-(f*f.log()).sum()/math.log(C)) if C>1 else 0.,float(n.min()/n.mean()),float(f[T].sum()),float((births[y]<stage).double().mean()),macro(true),macro(margin),macro(entropy),float((age/(1+age)).mean())]
 out=torch.tensor(v,device=dev,dtype=torch.float);assert out.shape==(12,) and torch.isfinite(out).all();return out,{'singleton_global_class_fraction':single/C,'scarce_class_count':int(T.sum())}

@torch.no_grad()
def describe(pool,memories,edits,model,counts,birth,stage,new_parents):
 C=model.C;assert len(counts)==len(birth)==C;dev=pool.z.device;vals=[];aud=[]
 for M in memories:
  a,b=psi(pool,M,model,counts,birth,stage);vals.append(a);aud.append(b)
 v=torch.stack(vals);B=len(memories[0]);Ng=len(memories[0].gidx);nc=torch.bincount(pool.y[pool.gidx],minlength=C).double();n=torch.as_tensor(counts,device=dev,dtype=torch.double);state=torch.tensor([C/23,B/256,B/len(pool),Ng/B,float(torch.isin(pool.parents,torch.tensor(list(new_parents),device=dev,dtype=torch.long)).double().mean()),min(1.,math.log1p(float(nc.min()))/math.log(257)),float(torch.log1p(n).mean()/torch.log1p(n.max())),stage/(1+stage)],device=dev)
 edit=torch.tensor([[float(i==0),e['removed_tokens']/B,e['replaced_parents']/max(1,Ng),e['operations']/4] for i,e in enumerate(edits)],device=dev);x=torch.cat([v,v-v[0],v[0].expand_as(v),state.expand(len(v),-1),edit],-1).detach();assert x.shape==(len(memories),48) and torch.isfinite(x).all();return x,aud
