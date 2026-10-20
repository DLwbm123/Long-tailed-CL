"""Independent fixed-rule and cached-selector acceptance; never trains."""
import time
START=time.time()
import os,sys,json,resource,traceback
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());sys.path.insert(0,cfg['parent_code']);sys.path.insert(0,cfg['selector_code'])
from runtime import Data,write,summary
RULES=[('context_P00',0,0),('context_P10',1,0),('context_P01',0,1),('context_P11',1,1)]


def own_inputs(old,b,f,c,counts,tail):
 bp=np.argmax(b,axis=1);fp=np.argmax(f,axis=1);cp=np.argmax(c,axis=1)
 if counts.ndim==1:counts=np.tile(counts,(len(b),1))
 num=np.array([np.log(1+counts[i,fp[i]])-np.log(1+counts[i,bp[i]]) for i in range(len(b))]);bits=np.array([[int(bb in tail),int(ff in tail),int(cc==bb),int(cc==ff)] for bb,ff,cc in zip(bp,fp,cp)],dtype=float)
 return np.column_stack((old[:,:4],num)),np.column_stack((old,bits))


def prob(m,x):
 if m['fallback']:return np.tile([0.,1.,0.],(len(x),1))
 z=np.column_stack(((x-m['mean'])/m['scale'],np.ones(len(x))));s=z@m['coef'];return np.exp(s-logsumexp(s,axis=1)[:,None])


def difference(a,b):
 if isinstance(a,dict):assert a.keys()==b.keys();return max([difference(a[k],b[k]) for k in a] or [0.])
 if isinstance(a,list):assert len(a)==len(b);return max([difference(x,y) for x,y in zip(a,b)] or [0.])
 if isinstance(a,(float,int)):return abs(a-b)
 assert a==b;return 0.


def verify_diag(diags,scores,y,g,fold,order,choices):
 B=scores['df_matched_joint'].argmax(1);F=scores['plain_correction'].argmax(1);C=scores['cls768_ridge'].argmax(1);bc=B==y;fc=F==y;cc=C==y;maxerr=0.
 for row in diags:
  scope=np.ones(len(y),dtype=bool) if row['scope']=='pooled' else fold==int(row['scope'].split('_')[1]);label=row['original_class'];mask=scope if label is None else scope&(y==order.index(label));ids=np.flatnonzero(mask);w=1/(20*np.bincount(y[ids],minlength=20)[y[ids]]) if label is None else np.full(len(ids),1/len(ids));mass=lambda a:float(w@a[ids]);dis=B!=F
  assert row['images']==len(ids) and row['patients']==len(np.unique(g[ids])) and row['disagreement_images']==int((mask&dis).sum()) and row['disagreement_patients']==len(np.unique(g[mask&dis]));checks=dict(oracle=mass(bc|fc),unavoidable_CLS_harm=mass(cc&~bc&~fc),B_CLS_harm=mass(cc&~bc),F_CLS_harm=mass(cc&~fc),rescue=mass(fc&~bc),harm=mass(bc&~fc),both_wrong=mass(~bc&~fc),disagreement_both_wrong=mass(dis&~bc&~fc))
  for k,val in checks.items():maxerr=max(maxerr,abs(row[k]-val))
  for a,select in choices.items():
   ok=np.where(select,fc,bc);rr=fc&~bc;hh=bc&~fc;checks=dict(BA_or_recall=mass(ok),CLS_harm=mass(cc&~ok),accepted_rescue=mass(select&rr),rejected_rescue=mass(~select&rr),accepted_harm=mass(select&hh),rejected_harm=mass(~select&hh),CLS_harm_delta_vs_B=mass(cc.astype(int)*(bc.astype(int)-ok.astype(int))),chosen_F_images=int((select&mask).sum()))
   for k,val in checks.items():maxerr=max(maxerr,abs(row['choices'][a][k]-val))
   assert abs((mass(ok)-mass(fc))-(mass(~select&hh)-mass(~select&rr)))<1e-12
 return maxerr


def verify():
 torch.set_num_threads(cfg['CPU_threads']);out=Path(cfg['output']);old=Path(cfg['selector_run']);parent=Path(cfg['parent_run']);assert json.loads((out/'status.private.json').read_text())['state']=='COMPLETE';assert json.loads((old/'VERIFICATION.json').read_text())['state']=='PASS';data=Data(cfg);y,g,fold=data.y,data.g,data.fold;tail=[cfg['order'].index(c) for c in cfg['frequency_bins']['tail']];base=np.load(parent/'outer_predictions.private.npz');oldp=np.load(old/'predictions.private.npz');saved=np.load(out/'predictions.private.npz');assert np.array_equal(saved['labels'],y) and np.array_equal(saved['groups'],g) and np.array_equal(saved['folds'],fold);scores={a:base[a] for a in ['cls768_ridge','scale_joint','df_matched_joint','plain_correction','utility_correction']};scores.update({a:oldp[a] for a in ['score_selector','support_selector']});arms=['prototype_direct','count_selector',*[a for a,_,_ in RULES]];scores.update({a:np.empty((len(y),20)) for a in arms});choices={a:np.zeros(len(y),dtype=bool) for a in [*arms,'score_selector','support_selector']};maxfeature=0.;maxscore=0.;audits=json.loads((out/'OPTIMIZATION.json').read_text());e0=json.loads((out/'E0_SEAL.json').read_text());seal=json.loads((out/'FIT_SEAL.json').read_text());assert len(audits)==6 and e0['selector_attempts']==0 and max(a['finished_at'] for a in audits)<=seal['frozen_at'] and e0['completed_at']<=min(a['started_at'] for a in audits)
 for outer in range(3):
  ids=np.flatnonzero(fold!=outer);test=np.flatnonzero(fold==outer);assert not np.intersect1d(g[ids],g[test]).size;sig=np.load(old/f'signals_{outer}.private.npz');tr=np.load(out/f'training_{outer}.private.npz');assert np.array_equal(sig['rows'],ids) and np.array_equal(tr['rows'],ids) and np.array_equal(sig['teacher'],tr['teacher']);cnt=np.empty((len(ids),20),dtype=int);C=np.empty((len(ids),20))
  for teacher in [i for i in range(3) if i!=outer]:
   mask=sig['teacher']==teacher;rows=np.flatnonzero(fold==teacher);assert not np.intersect1d(g[ids[mask]],g[rows]).size;counts=np.array([len(set(g[rows[y[rows]==c]].tolist())) for c in range(20)]);cnt[mask]=counts;h=torch.load(old/f'singleton_{teacher}.private.pt',weights_only=False);C[mask]=data.scores(ids[mask],h,'CLS')
  assert np.max(np.abs(C-tr['CLS']))<1e-10;xc,xctx=own_inputs(sig['features'],sig['B'],sig['F'],C,cnt,tail);maxfeature=max(maxfeature,float(np.max(np.abs(xc-tr['count_features']))),float(np.max(np.abs(xctx-tr['context_features']))));d=sig['B'].argmax(1)!=sig['F'].argmax(1);target=np.where((sig['F'].argmax(1)==y[ids])&(sig['B'].argmax(1)!=y[ids]),0,np.where((sig['B'].argmax(1)==y[ids])&(sig['F'].argmax(1)!=y[ids]),1,2));assert np.array_equal(d,sig['disagree']) and np.array_equal(target,sig['target']);ww=1/np.bincount(y[ids],minlength=20)[y[ids]];ww/=ww.sum();assert np.max(np.abs(ww-sig['weights']))<1e-14;m=torch.load(out/f'model_{outer}.private.pt',weights_only=False)
  for a,x in [('count_selector',xc),('context_selector',xctx)]:
   model=m[a];audit=next(z for z in audits if z['outer']==outer and z['arm']==a);assert audit['success']
   if model['fallback']:assert not d.any() or not np.isin(target[d],[0,1]).any();continue
   xx=x[d];w=ww[d];w/=w.sum();mean=np.sum(w[:,None]*xx,axis=0);sd=np.sqrt(np.sum(w[:,None]*(xx-mean)**2,axis=0));sd=np.where(sd>1e-12,sd,1.);assert np.max(np.abs(model['mean']-mean))<1e-10 and np.max(np.abs(model['scale']-sd))<1e-10;z=np.column_stack(((xx-mean)/sd,np.ones(len(xx))));s=z@model['coef'];lp=s-logsumexp(s,axis=1)[:,None];val=float(-w@lp[np.arange(len(xx)),target[d]]+cfg['selector_regularizer']*np.sum(model['coef'][:-1]**2));p=np.exp(lp);p[np.arange(len(xx)),target[d]]-=1;grad=z.T@(w[:,None]*p);grad[:-1]+=2*cfg['selector_regularizer']*model['coef'][:-1];assert abs(val-audit['objective'])<1e-9 and abs(float(np.abs(grad).max())-audit['gradient_max'])<1e-9
  counts=np.array([len(set(g[ids[y[ids]==c]].tolist())) for c in range(20)]);assert np.array_equal(counts,m['outer_counts']);h=torch.load(parent/f'outer_{outer}.private.pt',weights_only=True);proto=torch.load(old/f'selector_{outer}.private.pt',weights_only=False);B=data.scores(test,h,'df_joint')/h['temperature'];F=data.scores(test,dict(h,joint=h['joint']/h['temperature']+h['plain_correction']));C=data.scores(test,h,'CLS');maxscore=max(maxscore,float(np.max(np.abs(B-base['df_matched_joint'][test]/h['temperature']))),float(np.max(np.abs(F-base['plain_correction'][test]))),float(np.max(np.abs(C-base['cls768_ridge'][test]))));olddec=np.load(old/f'decisions_{outer}.private.npz');cos=data.cosine(test,dict(h,prototypes=proto['prototypes'],present=proto['present']));bi=B.argmax(1);fi=F.argmax(1);r=np.arange(len(test));cs=np.partition(B,-2,axis=1);fs=np.partition(F,-2,axis=1);features=np.column_stack([cs[:,-1]-cs[:,-2],fs[:,-1]-fs[:,-2],B[r,bi]-B[r,fi],F[r,fi]-F[r,bi],2*(cos[r,fi]-cos[r,bi])]);maxfeature=max(maxfeature,float(np.max(np.abs(features-olddec['features']))));xc,xctx=own_inputs(features,B,F,C,counts,tail);dec=np.load(out/f'decisions_{outer}.private.npz');maxfeature=max(maxfeature,float(np.max(np.abs(xc-dec['count_features']))),float(np.max(np.abs(xctx-dec['context_features']))));d=bi!=fi;pc=prob(m['count_selector'],xc);local=dict(prototype_direct=d&(features[:,4]>0),count_selector=d&(pc[:,0]>pc[:,1]));p=prob(m['context_selector'],xctx)
  for a,lt,lc in RULES:
   gain=(1+lt*20/7*xctx[:,6]+lc*xctx[:,8])*p[:,0]-(1+lt*20/7*xctx[:,5]+lc*xctx[:,7])*p[:,1];local[a]=d&(gain>0)
  for a,select in local.items():
   assert np.array_equal(select,dec[a]);choices[a][test]=select;scores[a][test]=np.where(select[:,None],F,B);maxscore=max(maxscore,float(np.max(np.abs(scores[a][test]-saved[a][test]))))
  for a in ['score_selector','support_selector']:choices[a][test]=olddec[a]
 diagerr=verify_diag(json.loads((out/'E0_DIAGNOSTICS.json').read_text()),scores,y,g,fold,cfg['order'],{a:choices[a] for a in ['score_selector','support_selector','prototype_direct']});diagerr=max(diagerr,verify_diag(json.loads((out/'DECISION_DIAGNOSTICS.json').read_text()),scores,y,g,fold,cfg['order'],choices));metricerr=difference(summary(scores,y,g,fold,cfg),json.loads((out/'results.json').read_text()));assert maxscore<1e-8 and maxfeature<1e-9 and diagerr<1e-12 and metricerr<1e-12;cost=json.loads((out/'costs.json').read_text());assert cost['selector_attempts']==cost['selector_completed']==6 and cost['new_encodes']==cost['new_ridge_heads']==cost['new_base_correction_heads']==0;size=sum(p.stat().st_size for rr in [out.parent,*map(Path,cfg['prior_roots'])] for p in rr.rglob('*') if p.is_file());assert size<cfg['new_storage_cap_bytes']-100000000;result=dict(state='PASS',new_predictions_recomputed=6*len(y),metric_records=52,E0_records=84,decision_records=84,max_score_error=maxscore,max_feature_error=maxfeature,max_diagnostic_error=diagerr,max_metric_error=metricerr,patient_isolation=True,E0_before_fit=True,old_history_consistent=True,all_objectives_and_gradients_checked=True,all_cost_rules_independently_checked=True,worker_costs=cost,combined_remote_bytes=size,verification_seconds=time.time()-START,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_reserved_bytes=0);write(out/'VERIFICATION.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
 try:verify()
 except BaseException:
  write(Path(cfg['output'])/'VERIFICATION_FAILED.private.json',dict(error=traceback.format_exc(),verification_seconds=time.time()-START,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024));raise
