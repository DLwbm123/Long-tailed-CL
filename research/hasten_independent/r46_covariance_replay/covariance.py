"""Dimension/sample-count shrinkage for replay only; original stored stats stay raw."""
import torch
from collections import Counter

def prepare_replay_covariance(learner,classes,arrived_counts):
 n=learner._known_classes;assert n>0 and sorted(classes)==list(range(n)) and len(learner.cov)==n
 key=(learner._cur_task,n)
 if getattr(learner,'_replay_covariance_key',None)!=key:
  cpu=torch.get_rng_state().clone();gpu=[v.clone() for v in torch.cuda.get_rng_state_all()]
  raw=[v.detach().clone() for v in learner.cov];means=[v.detach().clone() for v in learner.prototype];d=raw[0].shape[0];eye=torch.eye(d,device=raw[0].device,dtype=torch.double)
  pooled=sum(v.double()-1e-5*eye for v in raw)/n;derived=[];lambdas={};counts={}
  for c in range(n):
   count=int(arrived_counts[c]);assert count>0;lam=d/(d+max(count-1,0));cov=(1-lam)*(raw[c].double()-1e-5*eye)+lam*pooled+1e-5*eye
   assert torch.isfinite(cov).all();torch.linalg.cholesky(cov);derived.append(cov);lambdas[c]=lam;counts[c]=count
  assert all(torch.equal(a,b) for a,b in zip(raw,learner.cov)) and all(torch.equal(a,b) for a,b in zip(means,learner.prototype))
  assert torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(gpu,torch.cuda.get_rng_state_all()))
  learner.replay_covariance=derived;learner._replay_covariance_key=key
  learner.audit['covariance_shrinkage']=dict(calls=0,examples=0,by_class=Counter(),preparations=1,pooled_classes=n,lambda_by_class=lambdas,train_counts=counts,regularizer=1e-5,rng_first_prepare_preserved=True,raw_covariance_unchanged=True,means_unchanged=True)
 a=learner.audit['covariance_shrinkage'];a['calls']+=1;a['examples']+=len(classes);a['by_class'].update(classes)
