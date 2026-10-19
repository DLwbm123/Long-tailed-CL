"""Read-only failure qualification: no optimizer, fitted head or outer evaluation."""
import json,os,time
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
from core import weights
from run import write


def main():
 start=time.time();torch.set_num_threads(4);cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);root=out.parent;src=Path(cfg['cache_run'])
 state=json.loads((out/'status.private.json').read_text());terminal=json.loads((root/'terminal.private.json').read_text());audit=json.loads((out/'optimization_0.json').read_text())[0]
 assert state['state']=='FAILED' and terminal['returncode']==1 and audit['success'] is False and audit['status']==1 and audit['iterations']==500
 assert not (out/'FIT_SEAL.json').exists() and not (out/'outer_predictions.private.npz').exists() and not (out/'results.json').exists()
 for name in ['guardian.private.json','launch.private.json']:
  ident=json.loads((root/name).read_text());p=Path(f"/proc/{ident['pid']}/stat");assert not p.exists() or p.read_text().split()[21]!=ident['start_ticks']
 idx=np.load(src/'train_index.private.npz');y=idx['labels'].astype(int);g=idx['groups'];f=idx['folds'];cache=np.load(src/'train768.private.npy',mmap_mode='r');inner=np.load(out/'inner_0.private.npz');ids=np.flatnonzero(f!=0);assert np.array_equal(ids,inner['rows'])
 w=weights(y[ids],g[ids]);w/=w.sum();assert np.max(np.abs(w-inner['weights']))<1e-14
 h=torch.load(src/'fold_head_5_0.private.pt',weights_only=True);fac=h['factors'].numpy();anchor=h['joint'].numpy();ix=list(range(768))+[1536];inner_err=0.;normal_err=0.;scale_errors=[]
 for fold in [1,2]:
  teacher=3-fold;teacherhead=torch.load(out/f'inner_head_{teacher}.private.pt',weights_only=True);teach={k:v.numpy() for k,v in teacherhead.items()};mask=f[ids]==fold;rows=ids[mask];cc=[];jj=[]
  for off in range(0,len(rows),1024):
   ri=rows[off:off+1024];x=np.column_stack([np.array(cache[ri],dtype=np.float64),np.ones(len(ri))]);cc.append(x[:,ix]@teach['CLS']);jj.append((x*teach['factors'])@teach['joint'])
  inner_err=max(inner_err,float(np.max(np.abs(inner['CLS'][mask]-np.concatenate(cc)))),float(np.max(np.abs(inner['joint'][mask]-np.concatenate(jj)))))
 assert inner_err<1e-10
 # Independently derive weighting, temperature and weighted-median keep target.
 cc=inner['CLS'].argmax(1)==y[ids];jj=inner['joint'].argmax(1)==y[ids];center=inner['joint']-inner['joint'].mean(1,keepdims=True);temp=float(np.sqrt((w[:,None]*center**2).sum()/20));wrong=inner['joint'].copy();wrong[np.arange(len(ids)),y[ids]]=-np.inf;margin=(inner['joint'][np.arange(len(ids)),y[ids]]-wrong.max(1))/temp;order=np.argsort(margin[jj],kind='stable');ww=w[jj];cap=float(margin[jj][order][np.searchsorted(ww[order].cumsum(),ww.sum()/2)]);target=np.where(jj,np.minimum(margin,cap),0.)
 assert np.array_equal(cc&~jj,inner['repair']) and np.array_equal(jj,inner['keep']) and abs(temp-float(inner['temperature']))<1e-12 and abs(cap-float(inner['cap']))<1e-12 and np.max(np.abs(target-inner['target']))<1e-12
 # Check persisted teachers and reused outer anchor by independent normal equations; no solves.
 for selected,head in [([fold],torch.load(out/f'inner_head_{fold}.private.pt',weights_only=True)) for fold in range(3)]+[([1,2],h)]:
  rows=np.flatnonzero(np.isin(f,selected));ww=weights(y[rows],g[rows]);hh={k:v.numpy() for k,v in head.items()};diag=np.zeros(1537);rhs={k:np.zeros_like(hh[k]) for k in ['CLS','joint']};residual={k:hh[k].copy() for k in ['CLS','joint']};energy=np.zeros(2)
  for off in range(0,len(rows),1024):
   ri=rows[off:off+1024];x=np.column_stack([np.array(cache[ri],dtype=np.float64),np.ones(len(ri))]);wt=ww[off:off+len(ri)];one=np.eye(20)[y[ri]];z=x*hh['factors'];diag+=(wt[:,None]*z**2).sum(0);m=y[ri]<10;energy[0]+=(wt[m,None]*x[m,:768]**2).sum();energy[1]+=(wt[m,None]*x[m,768:1536]**2).sum()
   for key,v in [('CLS',x[:,ix]),('joint',z)]:rhs[key]+=v.T@(one*wt[:,None]);residual[key]+=v.T@((v@hh[key]-one)*wt[:,None])
  se=abs((energy[0]/energy[1])**.5-float(hh['factors'][768])*2**.5);scale_errors.append(se);assert se<1e-12
  for key in rhs:normal_err=max(normal_err,float(np.abs(residual[key]).max()/max(1.,float(np.abs(rhs[key]).max()))))
  if selected==[1,2]:pre=1/np.sqrt(diag/20+2*cfg['regularizer'])
 assert normal_err<1e-9
 u=np.load(out/'last_candidate_0_plain_correction.private.npy').reshape(1537,20);r=pre[:,None]*u;obj=cfg['regularizer']*float((r*r).sum());initial=0.;grad=2*cfg['regularizer']*r
 for off in range(0,len(ids),1024):
  ri=ids[off:off+1024];z=np.column_stack([np.array(cache[ri],dtype=np.float64),np.ones(len(ri))])*fac;yy=y[ri];wt=w[off:off+len(ri)];base=z@anchor/temp;s=base+z@r;lz=logsumexp(s,axis=1);obj+=float(wt@(lz-s[np.arange(len(ri)),yy]));initial+=float(wt@(logsumexp(base,axis=1)-base[np.arange(len(ri)),yy]));p=np.exp(s-lz[:,None]);p[np.arange(len(ri)),yy]-=1;grad+=z.T@(p*wt[:,None])
 gmax=float(np.abs(grad*pre[:,None]).max());assert abs(obj-audit['final_objective'])<1e-8 and abs(initial-audit['initial_objective'])<1e-8 and abs(gmax-audit['gradient_max'])<1e-7
 # Strong convexity supplies only a conservative training-objective gap bound, not acceptance or accuracy.
 gradient_original_norm=float(np.linalg.norm(grad));gap_bound=gradient_original_norm**2/(4*cfg['regularizer'])
 result=dict(state='PASS_FAILURE_EVIDENCE_ONLY',scientific_comparison_available=False,outer_evaluation_opened=False,model_fit_seal_absent=True,worker_and_guardian_retired=True,failed_outer=0,failed_arm='plain_correction',failure='fixed_iteration_limit',iterations=500,scipy_function_calls=538,total_closure_calls=539,objective_initial=initial,objective_candidate=obj,preconditioned_gradient_max=gmax,original_gradient_norm=gradient_original_norm,strong_convexity_objective_gap_upper_bound=gap_bound,normal_equation_max_error=normal_err,scale_max_error=max(scale_errors),inner_prediction_max_error=inner_err,zero_new_encodes=True,read_only_no_fits_or_optimization=True,new_ridge_completed=7,persisted_new_ridge_heads=3*2,unpersisted_df_head_count=1,correction_attempts=1,accepted_correction_heads=0,utility_attempted=False,wall_seconds=time.time()-start,peak_training_RSS_and_GPU_receipts_available=False)
 write(out/'FAILURE_VERIFICATION.json',result);print(json.dumps(result))


if __name__=='__main__':main()
