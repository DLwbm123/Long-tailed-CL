"""SC/RLP/CF/GLOBAL/PATH: identical CE and Gaussian reference KL."""
import torch
import torch.nn.functional as F
from prototype_policy import build_prototypes,gaussian_logp,gaussian_KL,SIGMA
from utility import class_values,utility,counterfactual_credit
ARMS=('SC','RLP','CF','GLOBAL','PATH')

def loss_for_arm(arm,model,reference,ep,eps,counter=None):
 mu=model(ep['state']);p,w=build_prototypes(mu[None],ep,True);p=p[0];score=20*ep['query']@p.T;ce=F.cross_entropy(score+ep['nB'].log()[None],ep['y']);active=ep['active'];aux=ce*0;kl=ce*0;stats={};C=len(mu);Q=len(ep['y']);builds=C;scorerows=Q;calls=0;joint=0;acts=0
 if arm!='WARM':
  with torch.no_grad():rm=reference(ep['state']);rp=build_prototypes(rm[None],ep)[0];rs=20*ep['query']@rp.T;rv=class_values(rs[None],ep['y'])[0]
  builds+=C;scorerows+=Q;kl=gaussian_KL(mu,rm,active)
  if arm=='RLP':
   z=torch.cat(ep['support']);yy=torch.cat([torch.full((len(v),),c,device=z.device,dtype=torch.long) for c,v in enumerate(ep['support'])]);sim=z@p.T;own=sim.gather(1,yy[:,None]).squeeze(1);wrong=sim.masked_fill(F.one_hot(yy,C).bool(),-torch.inf).max(1).values;pc=p@p.T;collapse=(pc.sum(1)-pc.diag())/(C-1);r=own-wrong-.1*collapse[yy];adv=((r-r.mean())/(r.std(correction=0)+1e-8)).detach();aux=.1*(-adv*torch.cat([v[0] for v in w]).log()).mean();scorerows+=len(z)
  elif arm in ('CF','GLOBAL','PATH'):
   action=mu[None]+SIGMA*eps
   if arm!='PATH':action=action.detach()
   pg=build_prototypes(action,ep);sg=20*torch.einsum('qd,gcd->gqc',ep['query'],pg);builds+=8*C;scorerows+=8*Q;joint=8;acts=8*int(active.sum())
   if arm=='PATH':
    U=utility(sg,ep['y'],rv,ep['protected']);aux=-U.mean();D=None;calls=8
   else:
    with torch.no_grad():
     if arm=='CF':U,D=counterfactual_credit(sg,rs,ep['y'],rv,ep['protected'],active);calls=8*(1+int(active.sum()))
     else:U=utility(sg,ep['y'],rv,ep['protected']);D=(U-utility(rs[None],ep['y'],rv,ep['protected'])[0])[:,None].expand(-1,C);calls=9
    aux=-(D[:,active].detach()*gaussian_logp(mu,action)[:,active]).sum(-1).mean()
   stats['U_mean']=float(U.detach().mean());stats['U_std']=float(U.detach().std(correction=0))
   if D is not None:
    v=D[:,active];stats.update(D_mean=float(v.mean()),D_std=float(v.std(correction=0)),D_positive=float((v>0).double().mean()),D_negative=float((v<0).double().mean()))
 total=ce+aux+.04*kl
 assert torch.isfinite(total)
 with torch.no_grad():
  stats.update(CE=float(ce),aux=float(aux),KL=float(kl),mu_saturated=float((mu.abs()>1.99).float().mean()),ESS=float(torch.stack([1/v.square().sum(-1) for v in w]).mean()),ESS_fraction=float(torch.stack([(1/v.square().sum(-1))/len(z) for v,z in zip(w,ep['support'])]).mean()),max_weight=max(float(v.max()) for v in w),prototype_angle_mean=float(torch.acos((p*ep['p0']).sum(1).clamp(-1,1)).mean()),B_micro=float((score.argmax(1)==ep['y']).float().mean()))
 if counter is not None:
  for k,v in {'episodes':1,'joint_rollouts':joint,'active_class_actions':acts,'prototype_builds':builds,'raw_score_rows':scorerows,'counterfactual_utility_calls':calls if arm=='CF' else 0,'utility_calls':calls}.items():counter[k]=counter.get(k,0)+v
 return total,ce,aux+.04*kl,stats
