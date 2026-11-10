"""Frozen single-transition causal controls and CPU mathematical checks."""
import json,time
from pathlib import Path
ARMS=['RAPO','ACC_CTAN','ACC_FULL_KL','ACC_OLD_COND','CE_OLD_COND']

def write(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def coefficients(scores,target,old_count,conditional):
 import torch
 q=target.detach().softmax(-1);p=scores.softmax(-1);n=len(scores)
 if not conditional:
  loss=(q*(q.clamp_min(1e-30).log()-scores.log_softmax(-1))).sum(-1).mean();coef=(p-q)/n
 else:
  qo=q[:,:old_count].sum(-1).detach();qc=target[:,:old_count].detach().softmax(-1);pc=scores[:,:old_count].softmax(-1)
  loss=(qo*(qc*(qc.clamp_min(1e-30).log()-scores[:,:old_count].log_softmax(-1))).sum(-1)).mean()
  coef=torch.zeros_like(scores);coef[:,:old_count]=qo[:,None]*(pc-qc)/n
 return loss,coef.detach()

def fixtures():
 import torch
 torch.manual_seed(13005)
 for cond in [False,True]:
  s=torch.randn(3,10,dtype=torch.float64,requires_grad=True);t=torch.randn_like(s)
  loss,c=coefficients(s,t,5,cond);g=torch.autograd.grad(loss,s)[0];assert torch.allclose(g,c,atol=1e-12)
  if cond:
   shifted=s.detach().clone();shifted[:,:5]+=3
   assert abs(float(coefficients(shifted,t,5,True)[0]-loss))<1e-12 and bool((c[:,5:]==0).all())
 task=torch.tensor([[0.,1.,0.,1.,0.,1.,0.,1.],[0.]*8]);h=torch.linspace(0,1,16).reshape(2,8);r=task+.5*h;sig=.999+.001*float(r.std(unbiased=False))
 a=(r-r.mean(-1,keepdim=True))/(sig+1e-6);b=(task-task.mean(-1,keepdim=True))/(sig+1e-6);ret=.5*(h-h.mean(-1,keepdim=True))/(sig+1e-6)
 assert torch.allclose(a,b+ret,atol=1e-6) and bool((b[1]==0).all()) and bool((ret[1]!=0).any())
 return {'passed':True,'full_and_conditional_KL_manual_gradient_matches_autograd':True,'conditional_new_class_derivatives_zero':True,'old_mass_shift_unidentified':True,'CTAN_same_rule_separate_PG_verified':True,'new_model_calls':0,'optimizer_updates':0}

def outcome(evals,counts):
 by={r['arm']:r for r in evals};d=by['ACC_OLD_COND'];c=by['ACC_FULL_KL'];b=by['ACC_CTAN'];e=by['CE_OLD_COND']
 def subset(r,ids):return sum(r['correct'][i]/r['denominator'][i] for i in ids)/len(ids)
 gates={'D_minus_C_macro_ge_1pp':d['macro']-c['macro']>=.01,'D_minus_C_tail_ge_2pp':d['tail']-c['tail']>=.02,'D_old_macro_ge_C':subset(d,range(5))>=subset(c,range(5)),'D_new_macro_ge_B_minus1pp':subset(d,range(5,10))>=subset(b,range(5,10))-.01,'D_macro_over_matched_CE_positive':d['macro']>e['macro'],'D_tail_over_matched_CE_positive':d['tail']>e['tail']}
 return {'gates':gates,'performance_qualification_passed':all(gates.values()),'readouts':evals,'old_new':{a:{'old_macro':subset(by[a],range(5)),'new_macro':subset(by[a],range(5,10))} for a in ARMS},'not_original_nine_gates_or_full_continual':True}
