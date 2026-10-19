"""CPU-only known nonlinear kernel baselines on sealed original G/P/C caches."""
import json,os,sys,time,resource
from pathlib import Path
import numpy as np
import torch
from core_attribution import weights
from solver import fit,selfcheck,scales,predict
from evaluation import evaluate,metrics
ARMS=['G','P','C'];KINDS=['Linear','Poly2','RBF'];LAMBDAS=[.0001,.01,1.]
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
  if p.is_relative_to(ref/'inputs'):assert p==ref/'inputs'/f'{stage}.private.pt','future input access'
  if p.parent==cached and p.name.startswith('stage') and p.name.endswith('.features.private.pt'):assert p.name==f'stage{stage}.features.private.pt','future features access'
  if p==cached/'test.features.private.pt' or p==inherited:assert sealed,'test before model seal'
 sys.addaudithook(guard)
 try:guard('open',(str(cached/'test.features.private.pt'),'rb'))
 except AssertionError:pass
 else:raise AssertionError('test access probe failed')
 fits=[];chosen={};heads={};params={};counts=[];parity={};ys=[];groups=[];validgroups=set();xx={a:[] for a in ARMS};qs={}
 for stage in range(3):
  k=stage+4;check('load_cached_features',stage=stage)
  obj=torch.load(ref/'inputs'/f'{stage}.private.pt',map_location='cpu',weights_only=False);data=torch.load(cached/f'stage{stage}.features.private.pt',map_location='cpu',weights_only=False);tr=~obj['valid'];va=obj['valid'];y=obj['y'];gg=obj['groups'];tg=[gg[i] for i in torch.where(tr)[0].tolist()];vg={gg[i] for i in torch.where(va)[0].tolist()}
  assert not set(tg)&vg and not validgroups&set(tg) and not set(groups)&vg;validgroups|=vg
  assert len(y)==[3236,2254,36][stage] and int(tr.sum())==[2555,1812,28][stage]
  ys.append(y[tr]);groups+=tg;yy=torch.cat(ys);q=weights(yy,groups);qs[stage]=q.numpy();assert all(abs(float(q[yy==c].sum())-1)<1e-12 for c in range(k))
  counts.append(dict(stage=stage,training_rows=len(yy),class_counts=[int((yy==c).sum()) for c in range(k)]));stats=torch.load(ref/f'{stage}.statistics.private.pt',map_location='cpu',weights_only=False)
  for a in ARMS:
   assert len(data[a])==len(y) and torch.isfinite(data[a]).all() and (data[a][:,-1]-1).abs().max()<1e-12
   xx[a].append(data[a][tr]);x=torch.cat(xx[a]);hot=torch.nn.functional.one_hot(yy,6).double();err=max(float((x.T@(x*q[:,None])-stats[a][0]).abs().max()),float((x.T@(hot*q[:,None])-stats[a][1]).abs().max()));assert err<=1e-8;parity[f'{a}.{stage}.statistics']=err
   z=x[:,:-1].numpy()
   if stage==0:
    check('fit_base_kernel_scale',arm=a);params[a]=scales(z,q.numpy());write(out/'KERNEL_SCALE_SEAL.json',dict(params=params,test_used=False,initial_training_rows=2555,source_commit=cfg['source_commit']))
   for kind in KINDS:
    name=a+'_'+kind;candidates=[]
    for alpha in LAMBDAS if stage==0 else ([chosen[name]] if chosen[name] is not None else []):
     check('kernel_fit',arm=a,stage=stage,kernel=kind,alpha=alpha)
     head,info=fit(z,yy.numpy(),q.numpy(),k,alpha,kind,params[a],check);info.update(arm=name,stage=stage)
     if stage==0 and info['status']=='CERTIFIED':
      score=torch.from_numpy(predict(head,data[a][va,:-1].numpy(),z,q.numpy()));info['base_validation_BA']=metrics(score,y[va],[gg[i] for i in torch.where(va)[0].tolist()],k)['BA']
     fits.append(info)
     if head is not None:torch.save(head,out/f'head_{name}_{stage}_{alpha}.private.pt')
     candidates.append((alpha,head,info));write(out/'partial.json',dict(fits=fits,chosen=chosen,counts=counts,parity=parity))
    if stage==0:
     chosen[name]=None
     if len(candidates)==3 and all(v[2]['status']=='CERTIFIED' for v in candidates):
      best=max(candidates,key=lambda v:(v[2]['base_validation_BA'],v[0]));chosen[name]=best[0];heads[name,stage]=best[1]
    elif candidates and candidates[0][2]['status']=='CERTIFIED':heads[name,stage]=candidates[0][1]
    write(out/'partial.json',dict(fits=fits,chosen=chosen,counts=counts,parity=parity))
  if stage==0:write(out/'SELECTION_SEAL.json',dict(chosen=chosen,grid=LAMBDAS,params=params,metric='initial_validation_image_BA',validation_rows=681,test_used=False))
 assert len(yy)==4395 and not validgroups&set(groups)
 complete=[a+'_'+r for a in ARMS for r in KINDS if all((a+'_'+r,s) in heads for s in range(3))]
 seal=dict(source_commit=cfg['source_commit'],fit_units=len(fits),expected_units=45,certified_units=sum(i['status']=='CERTIFIED' for i in fits),complete_arms=complete,chosen=chosen,test_used=False,all_models_sealed_at=time.time());write(out/'MODEL_SEAL.json',seal);sealed=True;stage=3
 testx=torch.load(cached/'test.features.private.pt',map_location='cpu',weights_only=False);prior=torch.load(inherited,map_location='cpu',weights_only=False);ty=prior['y'];gg=prior['groups'];assert len(ty)==586 and len(set(gg))==500 and not set(gg)&(set(groups)|validgroups)
 scores=dict(prior['scores']);full={a:torch.cat(xx[a]).numpy() for a in ARMS}
 for st in range(3):
  n=counts[st]['training_rows'];rheads=torch.load(ref/f'{st}.heads.private.pt',map_location='cpu',weights_only=False)
  for a in ARMS:
   refscore=testx[a]@rheads[a]['0.01'];old=scores[a+'_ridge',st];err=float((refscore-old).abs().max());assert err<=1e-7 and torch.equal(refscore.argmax(1),old.argmax(1));parity[f'{a}.{st}.ridge_test_score']=err
   for kind in KINDS:
    name=a+'_'+kind
    if name in complete:scores[name,st]=torch.from_numpy(predict(heads[name,st],testx[a][:,:-1].numpy(),full[a][:n,:-1],qs[st]))
 torch.save(dict(scores=scores,y=ty,groups=gg),out/'predictions.private.pt');result=evaluate(scores,ty,gg,complete);assert not torch.cuda.is_initialized()
 if len(fits)!=45 or any(i['status']!='CERTIFIED' for i in fits):result['decision']='UNRESOLVED_NUMERICAL_NO_SCIENTIFIC_ACCEPTANCE'
 costs=dict(wall_seconds=time.monotonic()-start,cpu_seconds=sum(resource.getrusage(resource.RUSAGE_SELF)[i] for i in [0,1]),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_initialized=False,feature_extraction_images=0,new_feature_cache_bytes=0,privileged_full_history=True)
 result.update(status='COMPLETE',source_commit=cfg['source_commit'],fits=fits,chosen=chosen,kernel_params=params,counts=counts,parity=parity,model_seal=seal,costs=costs,historical_test_exposed=True,patient_isolation='UNKNOWN',new_parameter_search_after_base=False)
 write(out/'EVALUATION.json',result);write(out/'aggregate.json',result);write(out/'COST_RECEIPT.json',costs);write(out/'status.json',dict(state='COMPLETE',decision=result['decision']))
