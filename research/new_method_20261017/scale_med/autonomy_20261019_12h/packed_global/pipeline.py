import json,os,sys,time,resource
from pathlib import Path
import torch,numpy as np
from core_attribution import weights
from solver import PERM,CAP,pack,fit
from evaluation import evaluate,metrics
from space import check_space
ARMS=['G','P','C'];NAMES=[a+'_'+v for a in ARMS for v in ['Full','Scale']]
def write(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else (_ for _ in ()).throw(TypeError(type(v).__name__))));t.replace(p)
def main(cfg):
 start=time.monotonic();out=Path(cfg['output']);ref=Path(cfg['reference']).resolve();cached=Path(cfg['cached_features']).resolve();inherited=Path(cfg['inherited_predictions']).resolve();bc=json.loads(Path(cfg['benchmark_config']).read_text());testroot=Path(bc['output']).resolve();sealed=False;stage=-1
 roots=[Path(bc['datasets']['ISIC'][z]).resolve() for z in ['images','development_images']]
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve();assert not any(p.is_relative_to(z) for z in roots),'raw image forbidden'
  if p.is_relative_to(ref/'inputs'):assert p==ref/'inputs'/f'{stage}.private.pt','retired or future input'
  if p.parent==cached and p.name.startswith('stage') and p.name.endswith('.features.private.pt'):assert p.name==f'stage{stage}.features.private.pt','retired or future mapped input'
  if p==inherited or p==cached/'test.features.private.pt' or p.is_relative_to(testroot):assert sealed,'test before model seal'
 sys.addaudithook(guard)
 for target in [inherited,cached/'test.features.private.pt']:
  try:guard('open',(str(target),'rb'))
  except AssertionError:pass
  else:raise AssertionError('seal probe')
 transforms=torch.load(ref/'transforms.private.pt',weights_only=False);maps={};stats={};heads={};fits=[];parity={};sizes={};counts=[];alpha=None;train_groups=set();val_groups=set()
 def checkpoint(phase,**kw):
  check_space(cfg['window_root']);write(out/'status.json',dict(state='RUNNING',phase=phase,elapsed_seconds=time.monotonic()-start,**kw));write(out/'partial.json',dict(fits=fits,artifact_bytes=sizes,parity=parity,counts=counts,alpha=alpha))
 def raw(g,data,a):
  t=transforms[a];core=((g-t['g_norm'][0])/t['g_norm'][1])[:,PERM]
  if a=='G':return core
  extra=data[a][:,t['d']:-1]*t['out_norm'][1]+t['out_norm'][0][t['d']:];return torch.cat([core,extra],1)
 def mapped(z,name):
  a,v=name.split('_');t=maps[name]
  if v=='Scale':z=torch.cat([z[:,:transforms[a]['d']],z[:,1536:]],1)
  return torch.cat([(z-t['out_norm'][0])/t['out_norm'][1],torch.ones(len(z),1,dtype=torch.float64)],1)
 for stage in range(3):
  k=stage+4;checkpoint('load_stage',stage=stage);obj=torch.load(ref/'inputs'/f'{stage}.private.pt',weights_only=False);data=torch.load(cached/f'stage{stage}.features.private.pt',weights_only=False);tr=~obj['valid'];va=obj['valid'];y=obj['y'];gg=obj['groups'];tg=[gg[i] for i in torch.where(tr)[0].tolist()];vg={gg[i] for i in torch.where(va)[0].tolist()};assert not set(tg)&(vg|val_groups) and not train_groups&vg;train_groups.update(tg);val_groups|=vg;q=weights(y[tr],tg);assert abs(float(q.sum())-(4 if stage==0 else 1))<1e-10
  oldstats=torch.load(ref/f'{stage}.statistics.private.pt',weights_only=False);counts.append(dict(stage=stage,new_training_rows=int(tr.sum()),new_validation_rows=int(va.sum())))
  for a in ARMS:
   z=raw(obj['g'],data,a);t=transforms[a];oldz=torch.cat([z[:,:t['d']],z[:,1536:]],1);oldmap=torch.cat([(oldz-t['out_norm'][0])/t['out_norm'][1],torch.ones(len(z),1,dtype=torch.float64)],1);err=float((oldmap-data[a]).abs().max());assert err<=1e-10;parity[f'{a}.{stage}.original_map']=err
   oldA=oldmap[tr].T@(oldmap[tr]*q[:,None]);oldB=oldmap[tr].T@(torch.nn.functional.one_hot(y[tr],6).double()*q[:,None]);key=a+'_old';oa,ob=stats.get(key,(torch.zeros_like(oldA),torch.zeros_like(oldB)));stats[key]=(oa+oldA,ob+oldB);err=max(float((stats[key][0]-oldstats[a][0]).abs().max()),float((stats[key][1]-oldstats[a][1]).abs().max()));assert err<=1e-8;parity[f'{a}.{stage}.statistics']=err
   if stage==0:
    mean=(z[tr]*q[:,None]).sum(0)/q.sum();scale=torch.sqrt(((z[tr]-mean).square().sum(1)*q).sum()/q.sum());assert scale>0
    for v in ['Full','Scale']:
     name=a+'_'+v;idx=torch.cat([torch.arange(t['d']),torch.arange(1536,z.shape[1])]) if v=='Scale' else torch.arange(z.shape[1]);m={key:value for key,value in t.items() if key!='out_norm'};m.update(out_norm=(mean[idx].clone(),scale.clone()),scale_only=v=='Scale',permutation=PERM.clone());maps[name]=m
   for v in ['Full','Scale']:
    name=a+'_'+v;x=mapped(z,name);newA=x[tr].T@(x[tr]*q[:,None]);newA=(newA+newA.T)/2;newB=x[tr].T@(torch.nn.functional.one_hot(y[tr],6).double()*q[:,None]);oa,ob=stats.get(name,(torch.zeros_like(newA),torch.zeros_like(newB)));stats[name]=(oa+newA,ob+newB)
   del z,oldz,oldmap
  # All initial maps frozen before alpha selection. Only initial G_Full selects common penalty.
  if stage==0:
   write(out/'TRANSFORM_SEAL.json',dict(test_used=False,training_rows=2555,transforms=list(maps),source_commit=cfg['source_commit']))
   z=raw(obj['g'],data,'G');x=mapped(z,'G_Full');candidates=[]
   for al in [.0001,.01,1.]:
    w,err=fit(*stats['G_Full'],k,al);ba=metrics((x[va]@w)[:,:k],y[va],[gg[i] for i in torch.where(va)[0].tolist()],k)['BA'];candidates.append((ba,al,w));fits.append(dict(arm='G_Full',stage=0,alpha=al,status='CERTIFIED',residual=err,base_validation_BA=ba))
   _,alpha,selected=max(candidates,key=lambda v:(v[0],v[1]));heads['G_Full',0]=selected;write(out/'SELECTION_SEAL.json',dict(alpha=alpha,source='initial_G_Full_validation_BA',rows=681,grid=[.0001,.01,1.],tie='larger_alpha',test_used=False));del z,x,candidates
  for name in NAMES:
   checkpoint('solve',stage=stage,arm=name)
   if (name,stage) not in heads:
    w,err=fit(*stats[name],k,alpha);heads[name,stage]=w;fits.append(dict(arm=name,stage=stage,alpha=alpha,status='CERTIFIED',residual=err))
   aa,bb=stats[name];state=dict(mapper=maps[name],A=pack(aa),B=bb.contiguous(),W=heads[name,stage].contiguous(),alpha=alpha,classes=k,source_commit=cfg['source_commit'],permutation_seed=74010)
   fn=out/f'{name}_{stage}.state.private.pt';torch.save(state,fn);size=fn.stat().st_size;assert size<=CAP;sizes[f'{name}.{stage}']=size
   # Roundtrip of complete persistent state, using the same unpacked solve.
   from solver import unpack
   loaded=torch.load(fn,weights_only=False);assert torch.equal(unpack(loaded['A'],len(aa)),aa) and torch.equal(loaded['W'],heads[name,stage])
  checkpoint('stage_sealed',stage=stage);del obj,data,oldstats
 assert len(fits)==20 and len(heads)==18
 write(out/'MODEL_SEAL.json',dict(source_commit=cfg['source_commit'],complete_arms=NAMES,heads=18,fit_units=20,test_used=False,artifact_bytes=sizes,cap=CAP));sealed=True;stage=3
 prior=torch.load(inherited,weights_only=False);testx=torch.load(cached/'test.features.private.pt',weights_only=False);test=torch.load(testroot/'ISIC.test.features.private.pt',weights_only=False);rawg=torch.nn.functional.normalize(test['x'].double(),dim=1);ty=prior['y'];gg=prior['groups'];assert len(ty)==586 and len(set(gg))==500 and not set(gg)&(train_groups|val_groups)
 scores=dict(prior['scores'])
 for a in ARMS:
  z=raw(rawg,testx,a);t=transforms[a];oldz=torch.cat([z[:,:t['d']],z[:,1536:]],1);oldmap=(oldz-t['out_norm'][0])/t['out_norm'][1];err=float((oldmap-testx[a][:,:-1]).abs().max());assert err<=1e-10;parity[a+'.test.original_map']=err
  for v in ['Full','Scale']:
   name=a+'_'+v;x=mapped(z,name)
   for st in range(3):scores[name,st]=(x@heads[name,st])[:,:st+4]
 torch.save(dict(scores=scores,y=ty,groups=gg),out/'predictions.private.pt');result=evaluate(scores,ty,gg,NAMES);assert not torch.cuda.is_initialized();costs=dict(worker_body_wall_seconds=time.monotonic()-start,CPU_seconds=sum(resource.getrusage(resource.RUSAGE_SELF)[i] for i in [0,1]),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_initialized=False,feature_extractions=0)
 result.update(source_commit=cfg['source_commit'],fits=fits,artifact_bytes=sizes,cap=CAP,alpha=alpha,parity=parity,counts=counts,costs=costs,historical_test_exposed=True,patient_isolation='UNKNOWN',packing_is_not_method_novelty=True)
 write(out/'aggregate.json',result);write(out/'COST_RECEIPT.json',costs);write(out/'status.json',dict(state='COMPLETE',decision=result['decision']))
