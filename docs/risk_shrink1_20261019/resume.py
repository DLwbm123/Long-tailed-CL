"""Authorized continuation from frozen cache and OOF evidence; never encodes images."""
import csv,importlib.util,json,os,resource,sys,time,traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from core import fold_id,weights,center,problem,uniform,change
from conic import constrained
BASE=Path(os.environ['RUN_CONFIG']).resolve().parents[1]
sys.path.insert(0,str(BASE/'case_hierarchical1_20261018'))
from data import read_rows,Images
RUNTIME={}


def write(path,value):
 q=path.with_suffix(path.suffix+'.tmp');q.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');q.replace(path)


def metrics(score,labels,seen,old):
 p=torch.tensor(seen)[score.argmax(1)];counts=[int((labels==c).sum()) for c in seen];correct=[int(((labels==c)&(p==c)).sum()) for c in seen];rec=[a/b for a,b in zip(correct,counts)]
 return dict(macro=sum(rec)/len(seen),micro=float((p==labels).double().mean()),old_macro=sum(rec[:old])/old if old else None,new_macro=sum(rec[old:])/(len(seen)-old),counts=counts,correct=correct,recall=rec,confusion=[[int(((labels==c)&(p==d)).sum()) for d in seen] for c in seen])


def main(cfg):
 torch.manual_seed(cfg['seed']);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 root=Path(cfg['output']);start=time.time()
 history=json.loads((root/'status.private.json').read_text());assert history['state']=='FAILED'
 old_audits=json.loads((root/'TRAINING_AUDIT.json').read_text());assert len(old_audits)==3
 assert all(x['solver']['success'] for x in old_audits[:2]) and not old_audits[2]['solver']['success']
 baseline=float(os.environ['CUMULATIVE_BASE_WALL']);costs=dict(history['costs']);costs['attempts']+=1
 assert costs['new_training_images']==68058 and costs['new_validation_images']==0
 costs.update(continuation_new_training_images=0,continuation_new_validation_images=0,engineering_solver_calls=2,prior_failed_solver_calls=1)
 RUNTIME.update(start=start,costs=costs,baseline=baseline)
 def output_bytes():return sum(p.stat().st_size for directory in [root,root.parent/'solver_deps'] for p in directory.rglob('*') if p.is_file())
 def status(phase,**kw):
  rss=max(history['RSS_peak_bytes'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
  allocated=history['GPU_peak_allocated_bytes'];reserved=history['GPU_peak_reserved_bytes']
  write(root/'status.private.json',dict(state='RUNNING',phase=phase,started=start,updated=time.time(),costs=costs,RSS_peak_bytes=rss,GPU_peak_allocated_bytes=allocated,GPU_peak_reserved_bytes=reserved,cumulative_wall_seconds=baseline+time.time()-start,**kw))
  assert baseline+time.time()-start<cfg['max_wall_seconds'];assert costs['new_training_images']==68058 and costs['new_validation_images']==0
  assert rss<=cfg['host_RSS_budget_bytes'] and reserved<=cfg['gpu_reserved_budget_bytes']
  assert output_bytes()<=cfg['output_budget_bytes']
 status('data_binding')
 train=read_rows(cfg,'train');val=read_rows(cfg,'val');assert len(train)==68058 and len(val)==300
 assert not ({r['group'] for r in train}&{r['group'] for r in val})
 prior=Path(cfg['prior_run']);matched=Path(cfg['matched_run']);joint=Path(cfg['joint_run']);score_run=Path(cfg['score_run'])
 for p in [prior,matched,joint,score_run]:assert json.loads((p/'status.private.json').read_text())['state']=='COMPLETE'
 originals=json.loads((prior/'config.private.json').read_text())
 for key in ['seed','order','task_sizes','manifest','images','weight','author_labels']:assert originals[key]==cfg[key]
 rows=[];bounds=[0]
 for c in cfg['order']:rows.extend(r for r in train if r['label']==c);bounds.append(len(rows))
 assert len(rows)==len(train)
 group_names=sorted({r['group'] for r in rows});group_index={g:i for i,g in enumerate(group_names)}
 groups=np.array([group_index[r['group']] for r in rows],dtype=np.int32);folds=np.array([fold_id(r['group']) for r in rows],dtype=np.int8)
 labels=np.concatenate([np.full(bounds[i+1]-bounds[i],i,dtype=np.int16) for i in range(20)])
 masses=np.concatenate([weights(groups[bounds[i]:bounds[i+1]].tolist()) for i in range(20)])
 old_index=np.load(root/'train_index.private.npz')
 for key,value in dict(labels=labels,groups=groups,folds=folds,weights=masses,bounds=np.array(bounds)).items():assert np.array_equal(old_index[key],value)
 cache=np.load(root/'train768.private.npy',mmap_mode='r');assert cache.shape==(68058,1536) and cache.dtype==np.float32
 # Fold moments remain in RAM; no duplicate full training cache or patch-token cache.
 status('fold_statistics')
 grams=torch.zeros(20,3,1537,1537,dtype=torch.float64);targets=torch.zeros(20,3,1537,dtype=torch.float64);fold_audit=[];parity=[]
 patient_counts=np.zeros((20,3),dtype=int)
 for ci,c in enumerate(cfg['order']):
  lo,hi=bounds[ci:ci+2]
  for fold in range(3):patient_counts[ci,fold]=len(set(groups[lo:hi][folds[lo:hi]==fold].tolist()))
  for lo_batch in range(lo,hi,2048):
   stop=min(lo_batch+2048,hi);x=torch.cat([torch.from_numpy(np.array(cache[lo_batch:stop],copy=True)).double(),torch.ones(stop-lo_batch,1,dtype=torch.float64)],1)
   for fold in range(3):
    mask=torch.from_numpy(folds[lo_batch:stop]==fold);v=x[mask];w=torch.from_numpy(masses[lo_batch:stop][folds[lo_batch:stop]==fold])
    if len(v):grams[ci,fold]+=v.T@(v*w[:,None]);targets[ci,fold]+=(v*w[:,None]).sum(0)
  total=grams[ci].sum(0);old=torch.load(prior/f'class_{c:02d}.private.pt',weights_only=True);pm=torch.load(matched/f'class_{c:02d}.private.pt',weights_only=True);cross=torch.load(joint/f'cross_{c:02d}.private.pt',weights_only=True)['cross'];ix=list(range(768))+[1536];jx=list(range(768,1536))+[1536]
  err=dict(CLS=float((total[ix][:,ix]-old['ridge_xx']).abs().max()),patch=float((total[jx][:,jx]-pm['patch_mean_gram']).abs().max()),cross=float((total[:768,768:1536]-cross).abs().max()))
  assert max(err.values())<cfg['moment_atol'];parity.append(dict(class_id=c,errors=err))
  fold_audit.append(dict(class_id=c,patients=int(patient_counts[ci].sum()),images=hi-lo,fold_patients=patient_counts[ci].tolist(),fold_images=[int((folds[lo:hi]==f).sum()) for f in range(3)]));status('fold_statistics',class_id=c)
 write(root/'FOLD_AUDIT.json',fold_audit);write(root/'CACHE_PARITY.json',parity)
 solutions=[];training_audits=[];ix=torch.tensor(list(range(768))+[1536]);eye_c=torch.eye(769,dtype=torch.float64);eye_j=torch.eye(1537,dtype=torch.float64)
 for stage in range(6):
  if stage<2:
   a=old_audits[stage];training_audits.append(a);solutions.append((a['uniform_alpha'],np.array(a['risk_limited_alpha'])));continue
  k=sum(cfg['task_sizes'][:stage+1]);n=bounds[k];c_oof=np.zeros((n,k));j_oof=np.zeros((n,k));assigned=np.zeros(n,dtype=bool);missing=[];fold_scales=[]
  if stage==2:
   saved=np.load(root/'OOF_2.private.npz');c_oof=saved['CLS'];j_oof=saved['joint'];assigned[:]=True
   missing=old_audits[2]['missing_training_classes'];fold_scales=old_audits[2]['fold_scales']
  for fold in (range(3) if stage>2 else []):
   gs=[];bs=[]
   for ci in range(k):
    nt=patient_counts[ci].sum()-patient_counts[ci,fold]
    if nt==0:gs.append(torch.zeros_like(grams[ci,0]));bs.append(torch.zeros_like(targets[ci,0]));missing.append(dict(fold=fold,class_id=cfg['order'][ci],heldout_images=int(((labels[:n]==ci)&(folds[:n]==fold)).sum())))
    else:
     ratio=patient_counts[ci].sum()/nt;gs.append((grams[ci].sum(0)-grams[ci,fold])*ratio);bs.append((targets[ci].sum(0)-targets[ci,fold])*ratio)
   g=torch.stack(gs).sum(0);b=torch.stack(bs,1);base_g=torch.stack(gs[:10]).sum(0);scale=float(base_g[:768,:768].trace()/base_g[768:1536,768:1536].trace())**.5
   factors=torch.ones(1537,dtype=torch.float64);factors[:768]=2**-.5;factors[768:1536]=scale*2**-.5
   costs['fold_head_attempts']+=1;status('fit_OOF_heads',stage=stage,fold=fold)
   hc=torch.linalg.solve(g[ix][:,ix]+eye_c,b[ix]);costs['fold_head_completed']+=1
   costs['fold_head_attempts']+=1;status('fit_OOF_heads',stage=stage,fold=fold)
   hj=torch.linalg.solve(g*factors[:,None]*factors[None,:]+eye_j,b*factors[:,None]);costs['fold_head_completed']+=1
   torch.save(dict(CLS=hc,joint=hj,factors=factors),root/f'fold_head_{stage}_{fold}.private.pt');fold_scales.append(scale)
   indexes=np.flatnonzero(folds[:n]==fold)
   for off in range(0,len(indexes),2048):
    ids=indexes[off:off+2048];x=torch.cat([torch.from_numpy(np.array(cache[ids],copy=True)).double(),torch.ones(len(ids),1,dtype=torch.float64)],1)
    c_oof[ids]=(x[:,ix]@hc).numpy();j_oof[ids]=((x*factors)@hj).numpy();assigned[ids]=True
  assert assigned.all();c=center(c_oof);d=center(j_oof-c_oof);ys=labels[:n].astype(int);w=masses[:n];q,b,r0=problem(c,d,ys,w);scalar=uniform(q,b)
  qs=[];bs=[];rs=[];metadata=[];skipped=[];correct=c_oof.argmax(1)==ys
  for fold in range(3):
   for ci in range(k):
    mask=(folds[:n]==fold)&(ys==ci)
    for kind,m in [('class',mask),('CLS_correct',mask&correct)]:
     patients=len(set(groups[:n][m].tolist()))
     if not m.any() or (kind=='CLS_correct' and patients<cfg['correct_subset_min_patients']):skipped.append(dict(fold=fold,class_id=cfg['order'][ci],kind=kind,images=int(m.sum()),patients=patients));continue
     qq,bb,rr=problem(c[m],d[m],ys[m],w[m]);qs.append(qq);bs.append(bb);rs.append(rr);metadata.append(dict(fold=fold,class_id=cfg['order'][ci],kind=kind,images=int(m.sum()),patients=patients,baseline_risk=rr))
  qs=np.stack(qs);bs=np.stack(bs);rs=np.array(rs);fixed={cfg['order'].index(m['class_id']) for m in missing}
  if stage==2:
   original=np.load(root/'problem_2.private.npz')
   for value,key in [(q,'Q'),(b,'b'),(qs,'Qs'),(bs,'bs'),(rs,'baselines')]:assert np.max(np.abs(value-original[key]))<1e-14
   q,b,qs,bs,rs=[original[key] for key in ['Q','b','Qs','bs','baselines']];r0=float(original['baseline'])
  else:np.savez(root/f'OOF_{stage}.private.npz',CLS=c_oof,joint=j_oof,labels=ys,folds=folds[:n],groups=groups[:n],weights=w)
  if stage>2:np.savez(root/f'problem_{stage}.private.npz',Q=q,b=b,baseline=r0,Qs=qs,bs=bs,baselines=rs,fixed=np.array(sorted(fixed),dtype=int))
  costs['shrink_solver_calls']+=1;status('solve_shrinkage',stage=stage);solver_start=time.time()
  a,solver=constrained(q,b,qs,bs,rs,fixed,cfg['solver_maxiter'],cfg['solver_ftol'],cfg['feasibility_tolerance']);solver['wall_seconds']=time.time()-solver_start
  limits=np.einsum('i,gij,j->g',a,qs,a)+2*bs@a
  audit=dict(stage=stage,classes=k,uniform_alpha=scalar,risk_limited_alpha=a.tolist(),OOF_CLS_risk=r0,OOF_uniform_risk=r0+change(q,b,np.ones(k)*scalar),OOF_limited_risk=r0+change(q,b,a),solver=solver,constraints=[dict(**m,risk_delta=float(delta),normalized_delta=float(delta/max(m['baseline_risk'],1e-8))) for m,delta in zip(metadata,limits)],skipped_constraints=skipped,missing_training_classes=missing,fixed_zero_classes=[cfg['order'][i] for i in sorted(fixed)],fold_scales=fold_scales)
  training_audits.append(audit);write(root/'TRAINING_AUDIT.json',training_audits)
  assert solver['success'] and solver['max_normalized_violation']<=cfg['feasibility_tolerance'] and np.isfinite(a).all() and a.min()>=-1e-10 and a.max()<=1+1e-10,'Constrained solve failed frozen acceptance'
  solutions.append((scalar,a));print(json.dumps(dict(event='stage_shrinkage_complete',stage=stage,uniform=scalar,limited_min=float(a.min()),limited_max=float(a.max()),solver=solver)),flush=True)
 write(root/'FIT_SEAL.json',dict(continuation=True,previous_failed_solve_retained=True,frozen_at=time.time(),query_opened=False,stages=6,new_training_images=costs['new_training_images'],fold_head_completed=costs['fold_head_completed'],shrink_solver_calls=costs['shrink_solver_calls']))
 # Only now open previously saved development predictions; no new validation encoding.
 results=[];groups_diag=[]
 for stage,(scalar,a) in enumerate(solutions):
  previous=torch.load(score_run/f'stage_{stage}.private.pt',weights_only=True);seen=previous['seen'];y=previous['labels'];k=len(seen);old=sum(cfg['task_sizes'][:stage]);assert seen==cfg['order'][:k]
  scores=dict(previous['scores']);c=center(scores['cls768_ridge'].numpy());d=center(scores['scale_joint'].numpy()-scores['cls768_ridge'].numpy())
  assert np.array_equal(c.argmax(1),scores['cls768_ridge'].argmax(1).numpy()) and np.array_equal(center(c+d).argmax(1),scores['scale_joint'].argmax(1).numpy())
  scores['uniform_shrink']=torch.from_numpy(center(c+d*scalar));scores['risk_limited_shrink']=torch.from_numpy(center(c+d*a))
  rec=dict(stage=stage,seen=seen,metrics={arm:metrics(v,y,seen,old) for arm,v in scores.items()},contrasts={});pred={arm:torch.tensor(seen)[v.argmax(1)] for arm,v in scores.items()}
  for arm in cfg['new_arms']:
   for ref in ['cls768_ridge','scale_joint','df_matched_joint','uniform_shrink']:
    if arm==ref:continue
    p,z=pred[arm],pred[ref];rec['contrasts'][arm+'-'+ref]=dict(delta_BA=rec['metrics'][arm]['macro']-rec['metrics'][ref]['macro'],rescue=int(((p==y)&(z!=y)).sum()),harm=int(((p!=y)&(z==y)).sum()),changed=int((p!=z).sum()))
  bc=pred['cls768_ridge']==y;bp=pred['patchmean768_ridge']==y;masks=dict(both_correct=bc&bp,CLS_only=bc&~bp,patch_only=~bc&bp,both_wrong=~bc&~bp)
  groups_diag.append(dict(stage=stage,groups={name:dict(images=int(mask.sum()),correct={arm:int(((p==y)&mask).sum()) for arm,p in pred.items()}) for name,mask in masks.items()}))
  torch.save(dict(scores=scores,labels=y,seen=seen),root/f'stage_{stage}.private.pt');results.append(rec);status('validation_scoring',stage=stage)
 freq=json.loads((score_run/'FREQUENCY.json').read_text());group_results={}
 for criterion,data in freq.items():group_results[criterion]=dict(bins=data['bins'],BA={arm:{name:sum(m['recall'][cfg['order'].index(c)] for c in members)/len(members) for name,members in data['bins'].items()} for arm,m in results[-1]['metrics'].items()})
 cohorts={arm:dict(base_correct=sum(m['correct'][:10]),later_correct=sum(m['correct'][10:])) for arm,m in results[-1]['metrics'].items()}
 write(root/'results.json',results);write(root/'GROUP_DIAGNOSTICS.json',groups_diag);write(root/'FREQUENCY.json',group_results);write(root/'COHORTS.json',cohorts)
 with (root/'summary.csv').open('w',newline='') as f:
  wr=csv.writer(f,lineterminator='\n');wr.writerow(['stage','arm','BA','old_BA','new_BA'])
  for rec in results:
   for arm,m in rec['metrics'].items():wr.writerow([rec['stage'],arm,m['macro'],m['old_macro'],m['new_macro']])
 costs.update(wall_seconds=baseline+time.time()-start,continuation_wall_seconds=time.time()-start,prior_and_repair_wall_seconds=baseline,GPU_peak_allocated_bytes=history['GPU_peak_allocated_bytes'],GPU_peak_reserved_bytes=history['GPU_peak_reserved_bytes'],RSS_peak_bytes=max(history['RSS_peak_bytes'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),feature_cache_bytes=(root/'train768.private.npy').stat().st_size,historical_creation_excluded=True)
 write(root/'costs.json',costs);status('final_budget_check');costs['output_bytes']=output_bytes();write(root/'costs.json',costs)
 write(root/'status.private.json',dict(state='COMPLETE',started=start,finished=time.time(),costs=costs));print('COMPLETE',flush=True)



if __name__=='__main__':
 cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
 try:main(cfg)
 except Exception:
  root=Path(cfg['output']);root.mkdir(exist_ok=True);p=root/'status.private.json';state=json.loads(p.read_text()) if p.exists() else {};state.update(state='FAILED',updated=time.time(),error=traceback.format_exc(),costs=RUNTIME.get('costs',{}),wall_seconds=RUNTIME.get('baseline',0)+time.time()-RUNTIME.get('start',time.time()),RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(p,state);raise
