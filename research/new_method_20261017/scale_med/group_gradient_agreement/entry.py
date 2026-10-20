"""Zero-update, support-only parameter-gradient diagnostic. No optimizer exists."""
import json, os, resource, sys, time, traceback
from pathlib import Path
import torch

cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output'])
torch.set_num_threads(4)

def write(name,value):
 p=out/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False));t.replace(p)

def metrics(v):
 v=v.double();gram=v@v.T;norm=gram.diag().clamp_min(0).sqrt();n=len(v)
 if not torch.isfinite(gram).all():raise ValueError('NONFINITE_GRAM')
 if (norm<=1e-15).any():return dict(available=False,reason='ZERO_NORM'),gram
 a,b=v[:4].mean(0),v[4:].mean(0)
 if a.norm()<=1e-15 or b.norm()<=1e-15:return dict(available=False,reason='ZERO_HALF_MEAN'),gram
 unit=gram/norm[:,None]/norm[None,:];eig=torch.linalg.eigvalsh(gram).clamp_min(0);mass=eig/eig.sum();pos=mass>0
 return dict(available=True,half_mean_cosine=float(a@b/a.norm()/b.norm()),mean_unit_pair_cosine=float((unit.sum()-n)/(n*(n-1))),mean_to_mean_norm_ratio=float(v.mean(0).norm()/norm.mean()),group_norm_min_median_max=[float(norm.min()),float(norm.median()),float(norm.max())],gram_effective_rank=float(torch.exp(-(mass[pos]*mass[pos].log()).sum()))),gram

def synthetic():
 parallel=torch.ones(8,3);opposite=parallel.clone();opposite[4:]*=-1
 assert abs(metrics(parallel)[0]['half_mean_cosine']-1)<1e-12
 assert abs(metrics(opposite)[0]['half_mean_cosine']+1)<1e-12
 orth=torch.eye(8,dtype=torch.float64);assert abs(metrics(orth)[0]['half_mean_cosine'])<1e-12
 assert not metrics(torch.zeros(8,3))[0]['available']
 write('SYNTHETIC.json',dict(pass_all=True,parallel_antiparallel_orthogonal_zero=True))

def main():
 synthetic();sealed=Path(cfg['sealed_root']);sys.path.insert(0,str(sealed))
 from train_base import setup,raw,install_guard
 from geometry import transform
 from data import pixels
 install_guard(cfg)
 allowed={sealed/'prep/PLAN.private.pt',sealed/'prep/0.teacher.private.pt',sealed/'prep/1.teacher.private.pt'}
 window=Path(cfg['window_root']).resolve();matched=Path(cfg['matched']).resolve()
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve()
  if p.is_relative_to(window) and not p.is_relative_to(out) and not p.is_relative_to(matched/'pixels_0'):
   assert p in allowed or (p.is_relative_to(sealed) and p.suffix=='.py'),'OLD_LEARNED_OR_NONALLOWLIST_ASSET'
 sys.addaudithook(guard)
 plan=torch.load(sealed/'prep/PLAN.private.pt',weights_only=False);assert len(plan['y'])==2555
 selected=[];gen=torch.Generator().manual_seed(74310)
 for k in range(2):
  for c in range(4):
   gs=sorted({g for i,g in enumerate(plan['groups']) if int(plan['fold'][i])==k and int(plan['y'][i])==c});assert len(gs)>=8
   chosen=[gs[j] for j in torch.randperm(len(gs),generator=gen)[:8].tolist()]
   ids=[[i for i,g in enumerate(plan['groups']) if g==key] for key in chosen]
   assert all(all(int(plan['fold'][i])==k and int(plan['y'][i])==c for i in group) for group in ids)
   selected.append(dict(fold=k,class_id=c,group_rows=ids))
 rows=sum(len(ids) for cell in selected for ids in cell['group_rows']);assert rows<=512
 torch.save(selected,out/'selection.private.pt');write('PRIVATE_PLAN_SEAL.json',dict(cells=8,groups=64,training_images=rows,seed=74310,support_only=True))
 x=pixels(dict(cfg,root=str(sealed)));m,params,keys,ckeys,heads,pointwise=setup(cfg)
 versions={k:p._version for k,p in m.named_parameters()};bias={k:params[k].detach().clone() for k in keys};oldheads=[p.detach().clone() for p in heads]
 ps=[params[k] for k in keys]
 for p in ps:p.requires_grad_(True)
 assert sum(p.numel() for p in ps)==102912 and sorted(k for k,p in params.items() if p.requires_grad)==keys
 counts=dict(logical_forward_images=0,forward_calls=0,optimizer_steps=0);private=[];cells=[];preflight=None
 def forward(ids,mp,w):
  counts['logical_forward_images']+=len(ids);counts['forward_calls']+=1
  o=raw(m,pointwise,x[ids].cuda());g=torch.nn.functional.normalize(torch.nn.functional.normalize(torch.cat([o['pre_logits'],o['pre_logits_few']],1),dim=1).double(),dim=1)
  z=transform(g,mp);loss=torch.nn.functional.cross_entropy(z@w[:-1]+w[-1],plan['y'][ids].cuda());assert torch.isfinite(loss)
  gradients=torch.autograd.grad(loss,ps+[z]);assert all(torch.isfinite(v).all() for v in gradients)
  flat=torch.cat([p.detach().flatten().cpu().double() for p in gradients[:-1]])
  return flat,gradients[-1].detach().cpu().double().sum(0),g.detach().cpu(),float(loss.detach())
 for cell in selected:
  k,c=cell['fold'],cell['class_id'];teacher=torch.load(sealed/'prep'/f'{k}.teacher.private.pt',weights_only=False)
  mp={key:tuple(vv.cuda() for vv in value) if isinstance(value,tuple) else value.cuda() for key,value in teacher['mapper'].items()};w=teacher['W'].cuda().detach()
  vectors=[];zvectors=[];losses=[]
  for ids in cell['group_rows']:
   gs=[];zs=[];ce=[];features=[]
   for i in ids:
    gv,zv,feat,loss=forward(torch.tensor([i]),mp,w);gs.append(gv);zs.append(zv);features.append(feat);ce.append(loss)
   mean=torch.stack(gs).mean(0);vectors.append(mean);zvectors.append(torch.stack(zs).mean(0));losses.append(sum(ce)/len(ce))
   if preflight is None:
    feature_error=float((torch.cat(features)-plan['cached_g'][ids]).abs().max());assert feature_error<=1e-10
    assert len(ids)<=32,'FIRST_GROUP_BATCH_ABOVE_32'
    batch,_,_,_=forward(torch.tensor(ids),mp,w);error=float((batch-mean).abs().max());assert torch.allclose(batch,mean,atol=1e-5,rtol=1e-3)
    preflight=dict(G_replay_max_error=feature_error,group_mean_gradient_max_error=error,pass_all=True);write('PREFLIGHT.json',preflight)
  vg,zg=torch.stack(vectors),torch.stack(zvectors);bm,bgram=metrics(vg);zm,zgram=metrics(zg)
  private.append(dict(fold=k,class_id=c,bias_gram=bgram,z_gram=zgram))
  cells.append(dict(fold=k,class_id=c,groups=8,images=sum(map(len,cell['group_rows'])),parameter_gradient=bm,z_gradient=zm,group_mean_CE_min_mean_max=[min(losses),sum(losses)/len(losses),max(losses)]))
  write('status.json',dict(state='RUNNING',completed_cells=len(cells),total_cells=8,forward_counts=counts))
 assert all(torch.equal(params[k],v) for k,v in bias.items()) and all(torch.equal(p,v) for p,v in zip(heads,oldheads))
 assert all(p._version==versions[k] for k,p in m.named_parameters())
 torch.save(private,out/'group_grams.private.pt');write('GRADIENT_SEAL.json',dict(groups=64,cells=8,all_initial_parameters_unchanged=True,query_scoring=False,forward_counts=counts,source_commit=cfg['source_commit']))
 # Classification of the fixed diagnostic is made only after all gradients are sealed.
 target=[v['parameter_gradient'] for v in cells if v['class_id']==1]
 if not all(v['available'] for v in target):decision='UNRESOLVED'
 elif all(v['half_mean_cosine']<=0 for v in target):decision='CLASS1_DIRECTION_DISAGREEMENT_AT_REFERENCE'
 elif all(v['half_mean_cosine']>0 for v in target):decision='POSITIVE_CLASS1_MEAN_ALIGNMENT_AT_REFERENCE'
 else:decision='MIXED_CLASS1_MEAN_ALIGNMENT_AT_REFERENCE'
 write('aggregate.json',dict(decision=decision,method_success=False,cells=cells,groups=64,selected_images=rows,preflight=preflight,all_parameters_unchanged=True,parameter_dimension=102912,forward_counts=counts,development_used=False,future_used=False,test_used=False,new_fits=0))
 write('status.json',dict(state='COMPLETE',decision=decision))

start=time.time();code=1
try:main();code=0
except BaseException as error:
 write('WORKER_FAILURE.json',dict(type=type(error).__name__,no_automatic_retry=True));(out/'FAILURE.private.txt').write_text(traceback.format_exc())
finally:
 u=resource.getrusage(resource.RUSAGE_SELF);write('WORKER_RESOURCE_RECEIPT.json',dict(wall_seconds=time.time()-start,CPU_seconds=u.ru_utime+u.ru_stime,peak_RSS_bytes=u.ru_maxrss*1024,GPU_initialized=torch.cuda.is_initialized(),peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0));write('WORKER_EXIT.json',dict(exit_code=code))
raise SystemExit(code)
