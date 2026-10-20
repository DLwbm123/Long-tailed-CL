"""Audit old frozen signals, then six fixed control fits; seal before new evaluation."""
import time
START=time.time()
import json,os,sys,resource,traceback
from pathlib import Path
import numpy as np
import torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());sys.path.insert(0,cfg['parent_code'])
from core import weights,loss,metrics
from solver import solve
from control import transport,shuffled_loss
from run_helpers import write,contrasts
RUNTIME={}


def audit_existing(out,cache,y,g,folds,parent):
 old=np.load(parent/'outer_predictions.private.npz');report=dict(scope='existing_frozen_models_only',new_fits=0,inner=[],outer=[])
 for outer in range(3):
  a=np.load(parent/f'inner_{outer}.private.npz');ids=a['rows'];h=torch.load(parent/f'outer_{outer}.private.pt',weights_only=True);w=a['weights'];allscore={k:np.empty((len(ids),20)) for k in ['scale_joint','plain_correction','utility_correction']}
  for off in range(0,len(ids),2048):
   rows=ids[off:off+2048];x=torch.cat([torch.from_numpy(np.array(cache[rows],copy=True)).double(),torch.ones(len(rows),1,dtype=torch.float64)],1);z=x*h['factors'];allscore['scale_joint'][off:off+len(rows)]=(z@h['joint']).numpy()
   for arm in ['plain_correction','utility_correction']:allscore[arm][off:off+len(rows)]=(z@(h['joint']/h['temperature']+h[arm])).numpy()
  for c,original in enumerate(cfg['order']):
   mask=y[ids]==c;mass=float(w[mask].sum());repair=a['repair']&mask;keep=a['keep']&mask;jonly=mask&(a['joint'].argmax(1)==y[ids])&(a['CLS'].argmax(1)!=y[ids]);train_support=[int(len(np.unique(g[(y==c)&(folds==f)]))) for f in range(3)]
   report['inner'].append(dict(outer=outer,class_id=original,images=int(mask.sum()),patient_support_by_fold=train_support,repair_images=int(repair.sum()),keep_images=int(keep.sum()),repair_weighted_fraction=float(w@repair/mass),keep_weighted_fraction=float(w@keep/mass),joint_only_weighted_fraction=float(w@jonly/mass),temperature=float(a['temperature']),training_fit_on_signals={arm:{kind:float(w[sel]@(score.argmax(1)[sel]==y[ids][sel])/w[sel].sum()) if sel.any() else None for kind,sel in [('repair',repair),('keep',keep),('joint_only',jonly)]} for arm,score in allscore.items()}))
 for original in cfg['audit_classes']:
  c=cfg['order'].index(original);mask=y==c;rec=dict(class_id=original,images=int(mask.sum()),arms={},harmed_transitions={})
  for arm in ['cls768_ridge','scale_joint','df_matched_joint','plain_correction','utility_correction']:
   s=old[arm][mask];yy=y[mask];wrong=s.copy();wrong[np.arange(len(yy)),yy]=-np.inf;margin=s[np.arange(len(yy)),yy]-wrong.max(1);pred=s.argmax(1);rec['arms'][arm]=dict(correct=int((pred==yy).sum()),predicted_class_counts={str(cfg['order'][j]):int((pred==j).sum()) for j in np.unique(pred)},true_vs_max_wrong_margin_quantiles=np.quantile(margin,[0,.25,.5,.75,1]).tolist())
  for arm in ['plain_correction','utility_correction']:
   harmed=mask&(old['scale_joint'].argmax(1)==y)&(old[arm].argmax(1)!=y);dest=old[arm][harmed].argmax(1);rows=np.flatnonzero(harmed);rec['harmed_transitions'][arm]=dict(images=len(rows),destinations={str(cfg['order'][j]):int((dest==j).sum()) for j in np.unique(dest)})
  report['outer'].append(rec)
 write(out/'EXISTING_SIGNAL_AUDIT.json',report)


def main():
 torch.set_num_threads(cfg['CPU_threads']);torch.manual_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;out=Path(cfg['output']);out.mkdir(exist_ok=False);parent=Path(cfg['parent_run']);cost=dict(attempts=1,new_encodes=0,new_ridge_heads=0,correction_attempts=0,correction_completed=0,closure_evaluations=0);RUNTIME.update(cost=cost)
 def status(phase,**kw):
  peaks=dict(RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0);write(out/'resource_peaks.private.json',peaks);write(out/'status.private.json',dict(state='RUNNING',phase=phase,updated=time.time(),costs=cost,**peaks,**kw));assert sum(p.stat().st_size for root in [out.parent,Path(cfg['prior_root'])] for p in root.rglob('*') if p.is_file())<cfg['new_storage_cap_bytes']-100000000
 assert json.loads((parent/'status.private.json').read_text())['state']=='COMPLETE' and json.loads((parent/'VERIFICATION.json').read_text())['state']=='PASS';write(out/'config.private.json',cfg)
 a=np.load(Path(cfg['cache_run'])/'train_index.private.npz');y=a['labels'].astype(int);g=a['groups'];folds=a['folds'];cache=np.load(Path(cfg['cache_run'])/'train768.private.npy',mmap_mode='r');assert cache.shape==(68058,1536)
 status('audit_existing_signals');audit_existing(out,cache,y,g,folds,parent);status('audit_complete');print('EXISTING_AUDIT_COMPLETE',flush=True)
 ci=list(range(768))+[1536];audits=[]
 for outer in range(3):
  ids=np.flatnonzero(folds!=outer);h=torch.load(parent/f'outer_{outer}.private.pt',weights_only=True);inner=np.load(parent/f'inner_{outer}.private.npz');assert np.array_equal(ids,inner['rows']);ys=y[ids];ww=weights(ys,g[ids]);ww/=ww.sum();assert np.max(np.abs(ww-inner['weights']))<1e-14
  status('outer_training_inputs',outer=outer);x=torch.cat([torch.from_numpy(np.array(cache[ids],copy=True)).double(),torch.ones(len(ids),1,dtype=torch.float64)],1);z=x*h['factors'];gram=z.T@(z*torch.from_numpy(ww)[:,None]);del z
  # The CLS block has exactly the same units and residual L2 coefficient as the joint CLS block.
  factorsC=h['factors'][ci];anchorC=h['CLS']/factorsC[:,None];center=inner['CLS']-inner['CLS'].mean(1,keepdims=True);temperatureC=float(np.sqrt(np.sum(ww[:,None]*center**2)/20));assert temperatureC>1e-12
  e,d,m=transport(ys,ww,cfg['seed'],outer);np.savez(out/f'shuffle_{outer}.private.npz',rows=ids,recipients=e,donors=d,mass=m);transport_audit=[]
  for c in range(20):
   mask=ys==c;edge=ys[e]==c;original_r=float(ww[mask]@inner['repair'][mask]);original_k=float(ww[mask]@inner['keep'][mask]);shuffled_r=float(m[edge]@inner['repair'][d[edge]]);shuffled_k=float(m[edge]@inner['keep'][d[edge]]);assert max(abs(original_r-shuffled_r),abs(original_k-shuffled_k))<1e-12
   transport_audit.append(dict(class_id=cfg['order'][c],repair_mass=original_r,shuffled_repair_mass=shuffled_r,keep_mass=original_k,shuffled_keep_mass=shuffled_k,self_assigned_mass=float(m[edge&(e==d)].sum())))
  models=dict(CLS_anchor=anchorC,CLS_factors=factorsC,CLS_temperature=temperatureC,joint_anchor=h['joint'],joint_factors=h['factors'],joint_temperature=h['temperature']);write(out/f'TRANSPORT_{outer}.json',transport_audit)
  for arm in cfg['new_arms']:
   iscls=arm=='cls_plain_correction';gg=gram[ci][:,ci] if iscls else gram;fac=factorsC if iscls else h['factors'];anchor=anchorC if iscls else h['joint'];temp=temperatureC if iscls else h['temperature'];pre=1/torch.sqrt(torch.diag(gg)+2*cfg['regularizer']);proxy=gg*pre[:,None]*pre[None,:]/20+torch.diag(2*cfg['regularizer']*pre**2);data=(x[:,ci]*fac) if iscls else x*fac;zg=data.cuda();base=zg@anchor.cuda()/temp;xp=zg*pre.cuda();del zg,data
   ty=torch.from_numpy(ys).cuda();tw=torch.from_numpy(ww).cuda();repair=torch.from_numpy(inner['repair'].astype(float)).cuda();keep=torch.from_numpy(inner['keep'].astype(float)).cuda();target=torch.from_numpy(inner['target']).cuda();tp=pre.cuda();ee=torch.from_numpy(e).cuda();dd=torch.from_numpy(d).cuda();mm=torch.from_numpy(m).cuda();cost['correction_attempts']+=1;status('fit_control',outer=outer,arm=arm)
   def fun(v):
    cost['closure_evaluations']+=1;status('fit_control',outer=outer,arm=arm);u=torch.tensor(v.reshape(len(pre),20),dtype=torch.float64,device='cuda',requires_grad=True)
    value=loss(u,xp,base,ty,tw,repair,keep,target,tp,False,cfg['regularizer']) if iscls else shuffled_loss(u,xp,base,ty,tw,repair,target,keep,tp,ee,dd,mm,cfg['regularizer']);value.backward();val=float(value.detach());grad=u.grad.cpu().numpy().ravel().copy();assert np.isfinite(val) and np.isfinite(grad).all();return val,grad
   checkpoint_file=out/f'candidate_{outer}_{arm}.private.npy'
   def checkpoint(v):
    tmp=checkpoint_file.with_suffix('.tmp')
    with tmp.open('wb') as f:np.save(f,v)
    tmp.replace(checkpoint_file)
   candidate,info=solve(fun,np.zeros(len(pre)*20),proxy,cfg,checkpoint);info.update(outer=outer,arm=arm);audits.append(info);write(out/'OPTIMIZATION.json',audits);checkpoint(candidate);assert info['success'],'Control solve failed fixed acceptance'
   models[arm]=pre[:,None]*torch.from_numpy(candidate.reshape(len(pre),20));models[arm+'_precondition']=pre;cost['correction_completed']+=1;torch.save(models,out/f'partial_{outer}.private.pt');status('control_frozen',outer=outer,arm=arm);print(json.dumps(info),flush=True)
   del xp,base,ty,tw,repair,keep,target,tp,ee,dd,mm;torch.cuda.empty_cache()
  torch.save(models,out/f'outer_{outer}.private.pt');del x,gram
 assert cost['correction_completed']==6 and cost['new_encodes']==0;write(out/'FIT_SEAL.json',dict(frozen_at=time.time(),outer_models=3,new_control_heads=6,new_encodes=0,original_validation_or_test_access=False))
 # Old outer predictions are already exposed diagnostics; new control metrics remain sealed until here.
 old=np.load(parent/'outer_predictions.private.npz');scores={a:old[a] for a in ['cls768_ridge','scale_joint','df_matched_joint','plain_correction','utility_correction']};scores.update({arm:np.empty((len(y),20)) for arm in cfg['new_arms']})
 for outer in range(3):
  ids=np.flatnonzero(folds==outer);h=torch.load(out/f'outer_{outer}.private.pt',weights_only=True);status('evaluate_sealed_controls',outer=outer)
  for off in range(0,len(ids),2048):
   rows=ids[off:off+2048];x=torch.cat([torch.from_numpy(np.array(cache[rows],copy=True)).double(),torch.ones(len(rows),1,dtype=torch.float64)],1)
   scores['cls_plain_correction'][rows]=((x[:,ci]*h['CLS_factors'])@(h['CLS_anchor']/h['CLS_temperature']+h['cls_plain_correction'])).numpy();scores['mass_shuffled_correction'][rows]=((x*h['joint_factors'])@(h['joint_anchor']/h['joint_temperature']+h['mass_shuffled_correction'])).numpy()
 np.savez(out/'predictions.private.npz',labels=y,groups=g,folds=folds,**{a:scores[a] for a in cfg['new_arms']});results=[]
 for fold in [None,0,1,2]:
  mask=np.ones(len(y),dtype=bool) if fold is None else folds==fold;ss={a:v[mask] for a,v in scores.items()};cs,gs=contrasts(ss,y[mask],g[mask]);results.append(dict(scope='pooled' if fold is None else f'outer_{fold}',metrics={a:metrics(v,y[mask],g[mask],cfg['order'],cfg['frequency_bins']) for a,v in ss.items()},contrasts=cs,groups=gs))
 write(out/'results.json',results);status('final_checks');cost.update(process_wall_seconds=time.time()-START,cumulative_recorded_seconds=cfg['prior_recorded_seconds']+cfg.get('engineering_seconds',0)+time.time()-START,prior_cost_record_incomplete=True,**json.loads((out/'resource_peaks.private.json').read_text()));write(out/'costs.json',cost);write(out/'status.private.json',dict(state='COMPLETE',costs=cost,finished=time.time(),independent_verification='PENDING'));print('COMPLETE',flush=True)


if __name__=='__main__':
 try:main()
 except BaseException:
  out=Path(cfg['output']);out.mkdir(exist_ok=True);peaks=dict(RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0);write(out/'resource_peaks.private.json',peaks);write(out/'failure.private.json',dict(error=traceback.format_exc(),costs=RUNTIME.get('cost',{}),process_wall_seconds=time.time()-START,**peaks));write(out/'status.private.json',dict(state='FAILED',updated=time.time(),costs=RUNTIME.get('cost',{}),**peaks));raise
