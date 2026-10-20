import numpy as np
import torch

def points(s,y):
 p=s.argmax(1);ce=torch.nn.functional.cross_entropy(s,y,reduction='none');r=s.clone();r[torch.arange(len(y)),y]=-torch.inf;mar=s[torch.arange(len(y)),y]-r.max(1).values
 rows=[]
 for c in range(4):
  ids=y==c;rows.append(dict(count=int(ids.sum()),correct=int((p[ids]==c).sum()),recall=float((p[ids]==c).double().mean()),CE=float(ce[ids].mean()),margin_mean=float(mar[ids].mean()),margin_quantiles=torch.quantile(mar[ids],torch.tensor([0.,.1,.5,.9,1.],dtype=s.dtype)).tolist()))
 return dict(BA=sum(v['recall'] for v in rows)/4,classes=rows)

def contrasts(tab):
 return dict(BA_pp=100*((tab['T_seen']['BA']-tab['T_out']['BA'])-(tab['F_seen']['BA']-tab['F_out']['BA'])),class2_pp=100*((tab['T_seen']['classes'][2]['recall']-tab['T_out']['classes'][2]['recall'])-(tab['F_seen']['classes'][2]['recall']-tab['F_out']['classes'][2]['recall'])))

def evaluate(score,y,groups):
 assert set(score)=={'T_seen','T_out','F_seen','F_out'} and all(torch.isfinite(v).all() for v in score.values())
 tab={k:points(v,y) for k,v in score.items()};ds=contrasts(tab);passed=ds['BA_pp']>=5-1e-12 and ds['class2_pp']>=10-1e-12
 pairs={}
 for family in ['T','F']:
  seen,out=score[family+'_seen'].argmax(1),score[family+'_out'].argmax(1);rows=[]
  for c in range(4):
   ids=y==c;so,oo=seen[ids]==c,out[ids]==c;r=int((oo & ~so).sum());h=int((so & ~oo).sum());assert r-h==int(oo.sum())-int(so.sum());rows.append(dict(class_index=c,rescued_out_vs_seen=r,harmed_out_vs_seen=h))
  pairs[family]=rows
 result=dict(summary=tab,adaptation_reuse_gap=ds,point_gate_passed=bool(passed),paired=pairs,bootstrap_performed=False,method_success=False,development_used=False,test_used=False,future_used=False,conditional_on_two_producers=True,not_independent_external_confirmation=True)
 if not passed:return dict(**result,decision='STOP_REUSE_GAP_POINT_GATE')
 rng=np.random.default_rng(74292);rec={k:np.zeros((10000,4)) for k in score}
 for c in range(4):
  ids=torch.where(y==c)[0].tolist();keys=sorted({groups[i] for i in ids});mem=[[i for i in ids if groups[i]==g] for g in keys];size=np.array([len(z) for z in mem]);draw=rng.integers(0,len(keys),(10000,len(keys)));den=size[draw].sum(1)
  for k,v in score.items():
   ok=(v.argmax(1)==y).numpy();hit=np.array([ok[z].sum() for z in mem]);rec[k][:,c]=hit[draw].sum(1)/den
 delta=100*((rec['T_seen']-rec['T_out'])-(rec['F_seen']-rec['F_out']))
 ci={name:dict(adjusted_CI_pp=np.quantile(z,[.05/4,1-.05/4]).tolist()) for name,z in [('BA',delta.mean(1)),('class2',delta[:,2])]}
 signal=all(v['adjusted_CI_pp'][0]>0 for v in ci.values());result.update(bootstrap_performed=True,bootstrap=dict(seed=74292,draws=10000,family=2),contrasts=ci)
 return dict(**result,decision='CONDITIONAL_ADAPTATION_REUSE_GAP_SIGNAL' if signal else 'STOP_REUSE_GAP_EVIDENCE_INSUFFICIENT')
