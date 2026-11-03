"""3 matched objectives x 3 fixed preservation rules; no reward changes."""
import torch
import torch.nn.functional as F
from prototype_policy import build_prototypes,gaussian_logp,gaussian_KL,SIGMA
from utility import class_values,utility,counterfactual_credit
from constraints import behavior_KL,preservation
ARMS=tuple(o+'_'+k for o in ('SC','CF','PATH') for k in ('ACT','BHV','NONE'))

def components(arm,model,reference,ep,eps,counter=None):
 objective,kind=arm.split('_');assert objective in ('SC','CF','PATH');mu=model(ep['state']);p,w=build_prototypes(mu[None],ep,True);p=p[0];score=20*ep['query']@p.T;ce=F.cross_entropy(score+ep['nB'].log()[None],ep['y']);active=ep['active'];aux=ce*0;C=len(mu);Q=len(ep['y']);builds=C;scorerows=Q;calls=0;cf_calls=0;joint=0;acts=0;stats={}
 with torch.no_grad():rm=reference(ep['state']);rp=build_prototypes(rm[None],ep)[0];rs=20*ep['query']@rp.T
 builds+=C;scorerows+=Q;hold=preservation(kind,mu,rm,score,rs,ep['y'],active,ep['kappa'])
 if objective!='SC':
  with torch.no_grad():rv=class_values(rs[None],ep['y'])[0]
  action=mu[None]+SIGMA*eps
  if objective=='CF':action=action.detach()
  pg=build_prototypes(action,ep);sg=20*torch.einsum('qd,gcd->gqc',ep['query'],pg);builds+=8*C;scorerows+=8*Q;joint=8;acts=8*int(active.sum())
  if objective=='PATH':U=utility(sg,ep['y'],rv,ep['protected']);aux=-U.mean();D=None;calls=8
  else:
   with torch.no_grad():U,D=counterfactual_credit(sg,rs,ep['y'],rv,ep['protected'],active)
   aux=-(D[:,active].detach()*gaussian_logp(mu,action)[:,active]).sum(-1).mean();cf_calls=8*int(active.sum());calls=8+cf_calls
  stats.update(U_mean=float(U.detach().mean()),U_std=float(U.detach().std(correction=0)))
  if D is not None:
   v=D[:,active];stats.update(D_mean=float(v.mean()) if v.numel() else 0.,D_std=float(v.std(correction=0)) if v.numel() else 0.,D_positive=float((v>0).double().mean()) if v.numel() else 0.,D_negative=float((v<0).double().mean()) if v.numel() else 0.)
 total=ce+aux+hold;assert torch.isfinite(total)
 with torch.no_grad():
  pred=score.argmax(1);refpred=rs.argmax(1);y=ep['y'];cor=torch.bincount(y[pred==y],minlength=C).double();den=torch.bincount(y,minlength=C);rec=cor/den;tail=ep['tail'];stats.update(CE=float(ce),RL_aux=float(aux),preservation=float(hold),raw_action_KL=float(gaussian_KL(mu,rm,active)),raw_behavior_KL=float(behavior_KL(score,rs,y)),kappa=ep['kappa'],B_micro=float((pred==y).double().mean()*100),B_macro=float(rec.mean()*100),B_tail=float(rec[tail].mean()*100),B_prediction_changes=int((pred!=refpred).sum()),B_rescue=int(((pred==y)&(refpred!=y)).sum()),B_harm=int(((pred!=y)&(refpred==y)).sum()),mu_saturated=float((mu.abs()>1.99).float().mean()),ESS_fraction=float(torch.stack([(1/v.square().sum(-1))/len(z) for v,z in zip(w,ep['support'])]).mean()),max_weight=max(float(v.max()) for v in w),prototype_angle_mean=float(torch.acos((p*ep['p0']).sum(1).clamp(-1,1)).mean()))
 if counter is not None:
  for k,v in {'episodes':1,'joint_rollouts':joint,'active_class_actions':acts,'policy_class_prototype_builds':builds,'raw_score_rows':scorerows,'replacement_utility_contexts':cf_calls,'factual_utility_contexts':8 if objective!='SC' else 0,'utility_contexts':calls}.items():counter[k]=counter.get(k,0)+v
 return total,ce,aux,hold,stats
