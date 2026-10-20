import numpy as np
import torch
ARMS=['F','E','L','R','D']
def points(pred,y):
 rec=[float((pred[y==c]==c).double().mean()) for c in range(4)];return dict(BA=sum(rec)/4,recall=rec,correct=[int(((pred==c)&(y==c)).sum()) for c in range(4)],counts=[int((y==c).sum()) for c in range(4)])
def pairs(base,pred,y):
 rows=[]
 for c in range(4):
  b,p=base[y==c],pred[y==c];bo,po=b==c,p==c;r=int((~bo&po).sum());h=int((bo&~po).sum());rows.append(dict(class_index=c,count=len(b),rescued=r,harmed=h,net_correct=r-h,both_correct=int((bo&po).sum()),both_wrong=int((~bo&~po).sum()),harmed_destination=torch.bincount(p[bo&~po],minlength=4).tolist()))
 gain=25*sum(v['net_correct']/v['count'] for v in rows);assert abs(gain-100*(points(pred,y)['BA']-points(base,y)['BA']))<1e-10;return dict(BA_gain_pp=gain,classes=rows)
def gate(tab):
 if not all(a in tab for a in ARMS):return dict(pass_all=False,missing=True)
 d,f=tab['D'],tab['F'];dr=np.array(d['recall'])-f['recall'];control={a:dict(BA_gain_pp=100*(d['BA']-tab[a]['BA']),class2_gain_pp=100*(d['recall'][2]-tab[a]['recall'][2])) for a in ['E','L','R']};g=dict(BA_vs_F_at_least1=d['BA']-f['BA']>=.01-1e-12,class2_vs_F_at_least5=dr[2]>=.05-1e-12,other_class_cost_within1=min(dr[[0,1,3]])>=-.01-1e-12,BA_all_controls_positive=all(x['BA_gain_pp']>0 for x in control.values()),class2_all_controls_nonnegative=all(x['class2_gain_pp']>=-1e-12 for x in control.values()),class2_vs_E_at_least5=control['E']['class2_gain_pp']>=5-1e-12);g={k:bool(v) for k,v in g.items()};return dict(**g,pass_all=all(g.values()),control_contrasts=control,BA_vs_F_pp=100*(d['BA']-f['BA']),class2_vs_F_pp=100*dr[2],risk_gains_pp=(100*dr[[0,1,3]]).tolist())
def evaluate(pred,y,groups):
 if not all(a in pred for a in ARMS):return dict(decision='UNRESOLVED_MISSING_ENDPOINT',method_success=False)
 tab={a:points(p,y) for a,p in pred.items()};g=gate(tab);paired={a:pairs(pred[a],pred['D'],y) for a in ['F','E','L','R']};result=dict(summary=tab,development_gate=g,paired_D_vs=paired,method_success=False,test_used=False,future_used=False,development_exposed=True,not_independent_confirmation=True)
 if not g['pass_all']:return dict(**result,decision='STOP_DECISION_ANCHOR_DEVELOPMENT',mechanism_bootstrap_performed=False)
 rng=np.random.default_rng(74252);rec={a:np.zeros((10000,4)) for a in ARMS};yy=y.numpy()
 for c in range(4):
  ids=np.where(yy==c)[0];keys=sorted({groups[i] for i in ids});members=[[i for i in ids if groups[i]==k] for k in keys];sizes=np.array([len(v) for v in members]);draw=rng.integers(0,len(keys),(10000,len(keys)));den=sizes[draw].sum(1)
  for a in ARMS:
   ok=pred[a].numpy()==yy;hits=np.array([ok[v].sum() for v in members]);rec[a][:,c]=hits[draw].sum(1)/den
 contrasts={}
 for a in ['F','E','L','R']:
  for metric,ds,point in [('BA',100*(rec['D'].mean(1)-rec[a].mean(1)),100*(tab['D']['BA']-tab[a]['BA'])),('class2',100*(rec['D'][:,2]-rec[a][:,2]),100*(tab['D']['recall'][2]-tab[a]['recall'][2]))]:contrasts[metric+'_vs_'+a]=dict(gain_pp=point,CI95_pp=np.quantile(ds,[.025,.975]).tolist(),CI8_adjusted_pp=np.quantile(ds,[.05/16,1-.05/16]).tolist())
 support=all(v['CI8_adjusted_pp'][0]>0 for v in contrasts.values());return dict(**result,decision='DEVELOPMENT_DIRECTIONAL_PROTECTION_SIGNAL' if support else 'STOP_DIRECTIONAL_MECHANISM_EVIDENCE_INSUFFICIENT',mechanism_bootstrap_performed=True,contrasts=contrasts,bootstrap=dict(seed=74252,draws=10000,family=8),mechanism_signal=bool(support))
def synthetic():
 y=torch.arange(80)%4;groups=['toy'+str(i) for i in range(80)];base=y.clone();base[::3]=(base[::3]+1)%4;pred={a:base.clone() for a in ARMS};pred['D']=y.clone();good=evaluate(pred,y,groups);assert good['development_gate']['pass_all'] and len(good['contrasts'])==8;pred['D']=base.clone();bad=evaluate(pred,y,groups);assert not bad['development_gate']['pass_all'];assert evaluate({},y,groups)['decision']=='UNRESOLVED_MISSING_ENDPOINT';import json;json.dumps([good,bad],allow_nan=False);return dict(eight_contrasts_success_failure_missing_JSON=True,paired_identity=True,point_gates_discrete=True)
