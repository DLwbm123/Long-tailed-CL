"""Frozen support-only geometry experiment using qualified cached features."""
import os,json,time,math,traceback,resource,sys
from pathlib import Path
import torch
from core import weights,paired
from design import validate,SEEDS,CLASSES,score
from model import Model,ARMS,selfcheck
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);out=root;torch.set_num_threads(4)
counts=dict(fits=0,optimizer_steps=0,backward=0,image_encoder_rows=0,text_encoder_rows=0,query_rows=0)
def write(n,v):
 p=out/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def status(**v):
 write('STATUS.json',v);(root/'CURRENT.json').write_text(json.dumps(dict(backbone=out.name,**v,counts=counts)))
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def interval(diff,y,gs):
 gen=torch.Generator().manual_seed(86422);parts=[]
 for c in CLASSES:
  keys=sorted({g for g,v in zip(gs,y.tolist()) if v==c});values=torch.tensor([float(diff[torch.tensor([v==g for v in gs])].mean()) for g in keys],dtype=torch.double)
  parts.append(values[torch.randint(len(values),(2000,len(values)),generator=gen)].mean(1))
 d=torch.stack(parts).mean(0)
 return dict(lower=float(torch.quantile(d,.025)),upper=float(torch.quantile(d,.975)),replicates=2000,conditional_on_fixed_models=True)
def main():
 global out
 prior=Path(cfg['prior_run']);plan=json.loads((prior/'PLAN.private.json').read_text());validate(plan)
 assert plan==json.loads((root/'PLAN.private.json').read_text())
 rows=plan['rows'];y=torch.tensor([r['label'] for r in rows]);gs=[r['identity_component'] for r in rows]
 eligible=[i for i,r in enumerate(rows) if r['label']!=1];ey=y[eligible];eg=[gs[i] for i in eligible]
 def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(args[0])).resolve()
   assert not any(s in p.name for s in ['HK.E.','HK.Qgate.','responses.private']), 'QUERY_FORBIDDEN'
   assert not p.is_relative_to(Path(cfg['image_root'])), 'IMAGE_ACCESS_FORBIDDEN'
 sys.addaudithook(guard)
 write('SYNTHETIC.json',selfcheck());torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 final={}
 for kind in ['CLIP','BiomedCLIP']:
  out=root/kind;out.mkdir(exist_ok=True)
  feat=load(prior/kind/'support_features.private.pt');assert feat['row_indices']==list(range(3140))
  x,t=feat['x'],feat['text'];assert x.shape==(3140,512) and t.shape==(20,512) and torch.isfinite(x).all() and torch.isfinite(t).all()
  assert float((x.norm(dim=1)-1).abs().max())<1e-5 and float((t.norm(dim=1)-1).abs().max())<1e-5
  write('FEATURE_RECEIPT.json',dict(rows=3140,dimension=512,text_rows=20,row_indices_exact=True,prior_source=cfg['prior_source'],conditional_producer_lineage=True,new_encodes=0))
  t=t.cuda();reports={};allpred=[]
  for seed in SEEDS:
   ref=load(prior/kind/f'{seed}.oof.private.pt');assert ref['indices']==eligible
   probabilities={arm:torch.full((3140,13),float('nan')) for arm in ARMS};folds=[]
   for k,fold in enumerate(plan['folds']):
    tr,te=fold['train'],fold['test'];xx=x[tr].cuda();yy=y[tr].cuda();q=weights(y[tr],[gs[i] for i in tr])
    orders=torch.multinomial(q,600*64,replacement=True,generator=torch.Generator().manual_seed(seed)).reshape(600,64)
    for arm in ARMS:
     model=Model().cuda();assert sum(p.numel() for p in model.parameters())==786432
     opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);trace=[]
     name=f'{seed}.o{k}.{arm}'
     for step,ids in enumerate(orders):
      opt.param_groups[0]['lr']=.001*.5*(1+math.cos(math.pi*step/600));opt.zero_grad(set_to_none=True)
      val,parts=model.objective(xx[ids],t,yy[ids],arm,seed);assert torch.isfinite(val)
      val.backward();counts['backward']+=1
      assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
      gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));opt.step();counts['optimizer_steps']+=1
      assert all(torch.isfinite(p).all() for p in model.parameters())
      if step%100==0 or step==599:
       trace.append(dict(step=step+1,loss=float(val.detach()),gradient_norm=gn,**{k:float(v.detach()) for k,v in parts.items()}));status(phase='TRAIN',fit=name,step=step+1,total=600)
     model.eval()
     with torch.no_grad():
      prob=model.logits(x[te].cuda(),t,arm[0]).softmax(1).float().cpu();v,z=model.representations(xx[:64],t)
      numerics=dict(min_text_radius=float(z.norm(dim=1).min()),max_text_radius=float(z.norm(dim=1).max()),text_aperture_capped=int((z.norm(dim=1)<=.2/(1-1e-5)).sum()))
     probabilities[arm][te]=prob
     folds.append(dict(seed=seed,outer_fold=k,arm=arm,metrics=score(prob,y[te],[gs[i] for i in te]),numerics=numerics))
     torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},train_indices=tr,seed=seed,arm=arm,steps=600),out/(name+'.private.pt'))
     counts['fits']+=1;write(name+'.trace.json',trace);write(f'{seed}.folds.json',folds);del model,opt
    del xx,yy
   probabilities={arm:p[eligible] for arm,p in probabilities.items()};probabilities['reference_B']=ref['predictions']['B'];probabilities['NCM']=ref['predictions']['A']
   mixes={arm:.5*(p+probabilities['NCM']) for arm,p in probabilities.items() if arm in ARMS}
   comparisons=[('Etrue','Eflat'),('Etrue','Eshuffled'),('Htrue','Hflat'),('Htrue','Hshuffled'),('Htrue','Etrue'),('Etrue','reference_B'),('Htrue','reference_B')]
   reports[str(seed)]=dict(primary={arm:score(p,ey,eg) for arm,p in probabilities.items()},secondary_mix={arm:score(p,ey,eg) for arm,p in mixes.items()},folds=folds,
                          paired={a+'-'+b:paired(probabilities[b],probabilities[a],ey) for a,b in comparisons})
   torch.save(dict(indices=eligible,predictions=probabilities),out/f'{seed}.oof.private.pt');allpred.append(probabilities)
  comparisons={}
  for a,b in [('Etrue','Eflat'),('Etrue','Eshuffled'),('Htrue','Hflat'),('Htrue','Hshuffled'),('Htrue','Etrue'),('Etrue','reference_B'),('Htrue','reference_B')]:
   ds=[reports[str(s)]['primary'][a]['group_BA']-reports[str(s)]['primary'][b]['group_BA'] for s in SEEDS]
   diff=torch.stack([(p[a].argmax(1)==ey).double()-(p[b].argmax(1)==ey).double() for p in allpred]).mean(0)
   protection=[]
   for c in CLASSES:
    d=sum(reports[str(s)]['primary'][a]['recall'][c]-reports[str(s)]['primary'][b]['recall'][c] for s in SEEDS)/3;limit=0 if c in [2,12] else -.01
    protection.append(dict(class_id=c,delta=d,limit=limit,passed=d>=limit-1e-12))
   comparisons[a+'-'+b]=dict(seed_delta=ds,mean_delta=sum(ds)/3,CI=interval(diff,ey,eg),protection=protection,protection_pass=all(v['passed'] for v in protection))
  def passes(a,b,threshold):
   z=comparisons[a+'-'+b]
   return z['mean_delta']>=threshold and all(d>0 for d in z['seed_delta']) and z['CI']['lower']>0 and z['protection_pass']
  signals={}
  for g in ['E','H']:
   reference=comparisons[g+'true-reference_B'];signals[g+'_hierarchy']=passes(g+'true',g+'flat',.01) and passes(g+'true',g+'shuffled',.01) and reference['mean_delta']>=0 and reference['protection_pass']
  signals['additional_hyperbolic_value']=signals['H_hierarchy'] and passes('Htrue','Etrue',.005)
  final[kind]=dict(seed_reports=reports,comparisons=comparisons,signals=signals,method_success=False,decision='MECHANISM_CANDIDATE' if any(signals.values()) else 'NO_QUALIFIED_MECHANISM_SIGNAL',patient_independence='UNKNOWN',support_reused_after_prior_analysis=True)
  write('aggregate.json',final[kind]);del x,t,feat,allpred;torch.cuda.empty_cache()
 out=root;assert counts['fits']==144 and counts['optimizer_steps']==86400 and counts['query_rows']==0
 write('aggregate.json',dict(backbones=final,counts=counts,source_commit=cfg['source_commit'],scientific_complete=True,full_HASTEN=False,no_followup_authorized=True));status(phase='COMPLETE_ANALYZE_THEN_PAUSE')
start=time.time();code=1
try:main();code=0
except BaseException as e:
 out=root;write('FAILURE.json',dict(type=type(e).__name__,scientific_complete=False));(out/'FAILURE.private.txt').write_text(traceback.format_exc());status(phase='FAILED')
finally:
 out=root;u=resource.getrusage(resource.RUSAGE_SELF);write('WORKER_RESOURCE.json',dict(wall_seconds=time.time()-start,CPU_seconds=u.ru_utime+u.ru_stime,peak_RSS_bytes=u.ru_maxrss*1024,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0));write('ACTUAL_COUNTS.json',counts);write('WORKER_EXIT.json',dict(exit_code=code))
raise SystemExit(code)
