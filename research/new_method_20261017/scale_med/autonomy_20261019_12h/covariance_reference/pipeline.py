"""Class statistics retire inputs; fixed covariance references on original caches."""
import json,os,sys,time,resource
from pathlib import Path
import numpy as np
import torch
from core_attribution import weights
from solver import selfcheck,fit_class,fit_pool,predict
from evaluation import evaluate
ARMS=['G','P','C'];KINDS=['NCM','SharedM','RawM','QDA','CorrM']
def write(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else (_ for _ in ()).throw(TypeError(type(v).__name__))));t.replace(p)
def main(cfg):
 start=time.monotonic();out=Path(cfg['output']);ref=Path(cfg['reference']).resolve();cached=Path(cfg['cached_features']).resolve();inherited=Path(cfg['inherited_predictions']).resolve();sealed=False;stage=-1
 torch.set_num_threads(4);assert not torch.cuda.is_initialized();write(out/'SELFCHECK.json',selfcheck())
 def check(phase=None,**kw):
  from space import check_space
  check_space(cfg['window_root'])
  if phase:write(out/'status.json',dict(state='RUNNING',phase=phase,elapsed_seconds=time.monotonic()-start,**kw))
 bc=json.loads(Path(cfg['benchmark_config']).read_text());imgroots=[Path(bc['datasets']['ISIC'][k]).resolve() for k in ['images','development_images']]
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve();assert not any(p.is_relative_to(z) for z in imgroots),'image access forbidden'
  if p.is_relative_to(ref/'inputs'):assert p==ref/'inputs'/f'{stage}.private.pt','retired or future input access'
  if p.parent==cached and p.name.startswith('stage') and p.name.endswith('.features.private.pt'):assert p.name==f'stage{stage}.features.private.pt','retired or future features access'
  if p==cached/'test.features.private.pt' or p==inherited:assert sealed,'test before model seal'
 sys.addaudithook(guard)
 try:guard('open',(str(cached/'test.features.private.pt'),'rb'))
 except AssertionError:pass
 else:raise AssertionError('test access probe failed')
 states={a:{} for a in ARMS};pools={};receipts=[];groups=set();validgroups=set();counts=[];parity={};reconA={};reconB={};traincount=0;classcounts=[]
 for stage in range(3):
  k=stage+4;check('load_arrived_features',stage=stage)
  obj=torch.load(ref/'inputs'/f'{stage}.private.pt',map_location='cpu',weights_only=False);data=torch.load(cached/f'stage{stage}.features.private.pt',map_location='cpu',weights_only=False);tr=~obj['valid'];va=obj['valid'];y=obj['y'];gg=obj['groups'];ty=y[tr];tg=[gg[i] for i in torch.where(tr)[0].tolist()];vg={gg[i] for i in torch.where(va)[0].tolist()}
  assert not set(tg)&vg and not validgroups&set(tg) and not groups&vg;validgroups|=vg;groups|=set(tg)
  assert len(y)==[3236,2254,36][stage] and int(tr.sum())==[2555,1812,28][stage]
  new=sorted(int(c) for c in ty.unique());assert new==([0,1,2,3] if stage==0 else [stage+3]);q=weights(ty,tg).numpy();assert all(abs(q[ty.numpy()==c].sum()-1)<1e-12 for c in new)
  traincount+=len(ty);classcounts +=[int((ty==c).sum()) for c in new];counts.append(dict(stage=stage,training_rows=traincount,class_counts=list(classcounts),new_validation_rows=int(va.sum())))
  stats=torch.load(ref/f'{stage}.statistics.private.pt',map_location='cpu',weights_only=False)
  for a in ARMS:
   assert len(data[a])==len(y) and torch.isfinite(data[a]).all() and (data[a][:,-1]-1).abs().max()<1e-12
   x=data[a][tr,:-1].numpy();d=x.shape[1]
   if stage==0:reconA[a]=np.zeros((d+1,d+1));reconB[a]=np.zeros((d+1,6))
   for c in new:
    assert c not in states[a];mask=ty.numpy()==c;check('fit_class_covariance',arm=a,stage=stage,label=c);state,info=fit_class(x[mask],q[mask]);states[a][c]=state
    info.update(unit='class',arm=a,stage=stage,label=c,known_groups=len({tg[i] for i in np.where(mask)[0]}));receipts.append(info)
    mu=state['mean'];reconA[a][:-1,:-1]+=state['cov']+np.outer(mu,mu);reconA[a][:-1,-1]+=mu;reconA[a][-1,:-1]+=mu;reconA[a][-1,-1]+=1;reconB[a][:-1,c]=mu;reconB[a][-1,c]=1
    from space import check_space
    check_space(cfg['window_root'],3*d*d*8+2**20);torch.save(state,out/f'class_{a}_{c}.private.pt');write(out/'partial.json',dict(units=receipts,counts=counts,parity=parity))
   err=max(float(abs(reconA[a]-stats[a][0].numpy()).max()),float(abs(reconB[a]-stats[a][1].numpy()).max()));assert err<=1e-8;parity[f'{a}.{stage}.statistics']=err
   check('fit_pooled_covariance',arm=a,stage=stage);pool,info=fit_pool([states[a][c] for c in range(k)]);pools[a,stage]=pool;info.update(unit='pooled',arm=a,stage=stage);receipts.append(info);torch.save(pool,out/f'pooled_{a}_{stage}.private.pt');write(out/'partial.json',dict(units=receipts,counts=counts,parity=parity))
  del obj,data,x,stats
  if stage<2:
   retired=cached/f'stage{stage}.features.private.pt';stage_next=stage+1;oldstage=stage;stage=stage_next
   try:guard('open',(str(retired),'rb'))
   except AssertionError:pass
   else:raise AssertionError('retirement guard probe failed')
   stage=oldstage
 assert traincount==4395 and not groups&validgroups
 complete=[a+'_'+label for a in ARMS for label in KINDS];nominal={a:6*(states[a][0]['mean'].size**2+states[a][0]['mean'].size)*8 for a in ARMS}
 seal=dict(source_commit=cfg['source_commit'],class_units=sum(r['unit']=='class' for r in receipts),pooled_units=sum(r['unit']=='pooled' for r in receipts),factor_count=sum(r['factor_count'] for r in receipts),complete_arms=complete,chosen_gamma=[1,1],validation_used_for_selection=False,test_used=False,all_models_sealed_at=time.time(),nominal_CorrM_dense_state_bytes=nominal,original_memory_cap_bytes=19058800,memory_cap_compliant=False,old_examples_reloaded_after_retirement=False)
 assert seal['class_units']==18 and seal['pooled_units']==9 and seal['factor_count']==45;write(out/'MODEL_SEAL.json',seal);sealed=True;stage=3
 testx=torch.load(cached/'test.features.private.pt',map_location='cpu',weights_only=False);prior=torch.load(inherited,map_location='cpu',weights_only=False);ty=prior['y'];gg=prior['groups'];assert len(ty)==586 and len(set(gg))==500 and not set(gg)&(groups|validgroups)
 scores=dict(prior['scores']);invariance={}
 for st in range(3):
  rheads=torch.load(ref/f'{st}.heads.private.pt',map_location='cpu',weights_only=False)
  for a in ARMS:
   old=scores[a+'_ridge',st];s=testx[a]@rheads[a]['0.01'];err=float((s-old).abs().max());assert err<=1e-7 and torch.equal(s.argmax(1),old.argmax(1));parity[f'{a}.{st}.ridge_test_score']=err
   result=predict(testx[a][:,:-1].numpy(),[states[a][c] for c in range(st+4)],pools[a,st])
   for label,v in result.items():
    assert np.isfinite(v).all();scores[a+'_'+label,st]=torch.from_numpy(v)
    if st>0 and label!='SharedM':
     birth=scores[a+'_'+label,0 if st==1 else st-1];err=float((scores[a+'_'+label,st][:,:birth.shape[1]]-birth).abs().max());assert err==0;invariance[f'{a}.{label}.{st}']=err
 torch.save(dict(scores=scores,y=ty,groups=gg),out/'predictions.private.pt');result=evaluate(scores,ty,gg,complete);assert not torch.cuda.is_initialized()
 costs=dict(wall_seconds=time.monotonic()-start,cpu_seconds=sum(resource.getrusage(resource.RUSAGE_SELF)[i] for i in [0,1]),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_initialized=False,feature_extraction_images=0,new_feature_cache_bytes=0,full_class_covariance_over_original_cap=True,old_examples_reloaded_after_retirement=False)
 result.update(status='COMPLETE',source_commit=cfg['source_commit'],units=receipts,counts=counts,parity=parity,score_invariance=invariance,model_seal=seal,costs=costs,historical_test_exposed=True,patient_isolation='UNKNOWN',new_parameter_search_after_base=False)
 write(out/'EVALUATION.json',result);write(out/'aggregate.json',result);write(out/'COST_RECEIPT.json',costs);write(out/'status.json',dict(state='COMPLETE',decision=result['decision']))
