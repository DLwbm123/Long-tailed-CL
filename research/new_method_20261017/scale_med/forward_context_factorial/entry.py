"""Four fixed forward cells; no loss, backward, fit or parameter update."""
import json, os, resource, sys, time, traceback
from pathlib import Path
import torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output'])
torch.set_num_threads(4)
def write(name,value):
 p=out/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False));t.replace(p)
def delta(a,b):
 d=(a-b).double();return dict(maxabs=float(d.abs().max()),RMS=float(d.square().mean().sqrt()))
def main():
 z=torch.zeros(2,3);a=z+1;b=z+2;c=z+3
 assert delta(z,a)==dict(maxabs=1.,RMS=1.) and torch.equal((c-b)-(a-z),z)
 write('SYNTHETIC.json',dict(pass_all=True,factorial_arithmetic=True))
 sealed=Path(cfg['sealed_root']);sys.path.insert(0,str(sealed))
 from train_base import setup,raw,install_guard
 from data import pixels
 install_guard(cfg)
 old=Path(cfg['old_selection']);allowed={sealed/'prep/PLAN.private.pt',old}
 window=Path(cfg['window_root']).resolve();matched=Path(cfg['matched']).resolve()
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve()
  if p.is_relative_to(window) and not p.is_relative_to(out) and not p.is_relative_to(matched/'pixels_0'):
   assert p in allowed or (p.is_relative_to(sealed) and p.suffix=='.py'),'NONALLOWLIST_ASSET'
 sys.addaudithook(guard)
 plan=torch.load(sealed/'prep/PLAN.private.pt',weights_only=False);assert len(plan['groups'])==2555
 previous=torch.load(old,weights_only=False)
 excluded={plan['groups'][i] for cell in previous for ids in cell['group_rows'] for i in ids};assert len(excluded)==64
 groups=sorted(set(plan['groups'])-excluded);gen=torch.Generator().manual_seed(74320)
 chosen=[groups[j] for j in torch.randperm(len(groups),generator=gen)[:32].tolist()]
 ids=torch.tensor([plan['groups'].index(g) for g in chosen]);assert len(chosen)==len(set(chosen))==32 and not(set(chosen)&excluded)
 torch.save(dict(rows=ids,groups=chosen),out/'selection.private.pt')
 meta=dict(groups=32,excluded_groups=64,disjoint=True,seed=74320,class_counts=torch.bincount(plan['y'][ids],minlength=4).tolist(),fold_counts=torch.bincount(plan['fold'][ids],minlength=2).tolist(),class1_not_studied=True)
 write('PRIVATE_PLAN_SEAL.json',meta)
 x=pixels(dict(cfg,root=str(sealed)));m,params,keys,ckeys,heads,pointwise=setup(cfg)
 versions={k:p._version for k,p in m.named_parameters()};bias={k:params[k].detach().clone() for k in keys};oldheads=[p.detach().clone() for p in heads]
 for k in keys:params[k].requires_grad_(True)
 assert sum(params[k].numel() for k in keys)==102912 and sorted(k for k,p in params.items() if p.requires_grad)==keys
 counts=dict(logical_forward_images=0,forward_calls=0,backward_calls=0,optimizer_steps=0,new_fits=0)
 matrices={};receipts=[]
 for name,batch,grad in [('B32_NG',32,False),('B32_GRAD',32,True),('B1_NG',1,False),('B1_GRAD',1,True)]:
  pieces=[]
  with torch.set_grad_enabled(grad):
   for part in ids.split(batch):
    o=raw(m,pointwise,x[part].cuda())
    g=torch.nn.functional.normalize(torch.nn.functional.normalize(torch.cat([o['pre_logits'],o['pre_logits_few']],1),dim=1).double(),dim=1)
    pieces.append(g.detach().cpu());counts['logical_forward_images']+=len(part);counts['forward_calls']+=1
    del g,o
  matrices[name]=torch.cat(pieces);finite=bool(torch.isfinite(matrices[name]).all())
  receipts.append(dict(cell=name,rows=32,finite=finite,batch_size=batch,gradient_context=grad))
  write('CELL_RECEIPTS.json',receipts);assert finite,'NONFINITE_FEATURES'
 write('FEATURE_SEAL.json',dict(cells=receipts,counts=counts,source_commit=cfg['source_commit']))
 names=list(matrices);pairs={names[i]+'_vs_'+names[j]:delta(matrices[names[i]],matrices[names[j]]) for i in range(4) for j in range(i+1,4)}
 comparisons={k:delta(v,plan['cached_g'][ids]) for k,v in matrices.items()}
 interaction=delta(matrices['B1_GRAD']-matrices['B1_NG'],matrices['B32_GRAD']-matrices['B32_NG'])
 unchanged=dict(bias_exact=all(torch.equal(params[k],v) for k,v in bias.items()),native_heads_exact=all(torch.equal(p,v) for p,v in zip(heads,oldheads)),parameter_versions_exact=all(p._version==versions[k] for k,p in m.named_parameters()))
 # Save observed numbers before asserting acceptance invariants.
 result=dict(decision='ENGINEERING_FACTORIAL_COMPLETE',method_success=False,selection=meta,pairs=pairs,cache_comparisons=comparisons,interaction=interaction,unchanged=unchanged,counts=counts,GGA_unchanged_unresolved=True,no_scientific_risk_or_gradient_conclusion=True)
 write('OBSERVATIONS.json',result)
 assert all(unchanged.values()) and counts['logical_forward_images']==128 and counts['forward_calls']==66
 write('aggregate.json',result);write('status.json',dict(state='COMPLETE'))
start=time.time();code=1
try:main();code=0
except BaseException as error:
 write('WORKER_FAILURE.json',dict(type=type(error).__name__,no_automatic_retry=True));(out/'FAILURE.private.txt').write_text(traceback.format_exc())
finally:
 u=resource.getrusage(resource.RUSAGE_SELF);write('WORKER_RESOURCE_RECEIPT.json',dict(wall_seconds=time.time()-start,CPU_seconds=u.ru_utime+u.ru_stime,peak_RSS_bytes=u.ru_maxrss*1024,GPU_initialized=torch.cuda.is_initialized(),peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0));write('WORKER_EXIT.json',dict(exit_code=code))
raise SystemExit(code)
