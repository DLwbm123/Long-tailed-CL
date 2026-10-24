"""Fixed TRAIN crossfit semantic-prior mean estimation; no optimizer or neural forward."""
import os,json,ast,csv,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);ref=Path(cfg['reference']);torch.set_num_threads(4);torch.set_num_interop_threads(2)
def write(n,v):(out/n).write_text(json.dumps(v,indent=2,allow_nan=False))
def main():
 records=[];summary={}
 for ds in ['NIH','ISIC','HK']:
  dc=json.loads((ref/'linear64'/ds/'config.private.json').read_text())
  if ds=='NIH':
   tree=ast.parse(Path(dc['base_runner']).read_text());v={a.targets[0].id:ast.literal_eval(a.value) for a in tree.body if isinstance(a,ast.Assign) and isinstance(a.targets[0],ast.Name) and a.targets[0].id in ['ORDER','ORIGINAL_NAMES','LEAF_TEMPLATE']};order=v['ORDER'];n=10;names=[v['ORIGINAL_NAMES'][i] for i in order];template=v['LEAF_TEMPLATE']
  else:order=dc['order'];n=dc['task_sizes'][0];names=[dc['names'][i] for i in order];template=dc['leaf_template']
  with open(Path(dc['manifest'])/'train.csv') as f:rows=list(csv.DictReader(f))
  assert all(r['split']=='train' for r in rows);labels=torch.tensor([order.index(int(r['original_label'])) for r in rows]);bank=torch.load(Path(dc['cache_root'] if ds=='NIH' else dc['shared_cache'])/'stage_0.private.pt',map_location='cpu',weights_only=False)
  assert bank['ids']==[i for i,y in enumerate(labels.tolist()) if y<n]
  x=bank['x'];text=bank['text'];x=F.normalize(x,dim=-1) if ds=='NIH' else x;text=F.normalize(text,dim=-1) if ds=='NIH' else text
  assert torch.allclose(x.norm(dim=1),torch.ones(len(x)),atol=2e-6);assert torch.allclose(text.norm(dim=1),torch.ones(len(text)),atol=2e-6)
  prompts=[template.format(v) for v in names[:n]];t=text[[bank['prompts'].index(p) for p in prompts]].double();x=x.double();y=labels[bank['ids']];counts=torch.bincount(y,minlength=n)
  for fold in [0,1]:
   fit=[];held=[];means=[];variances=[]
   for c in range(n):
    ids=torch.where(y==c)[0];a=ids[fold::2];b=ids[1-fold::2];assert len(a)>0 and len(b)>0 and not set(a.tolist())&set(b.tolist());fit.append(a);held.append(b);m=x[a].mean(0);means.append(m);variances.append(float((x[a]-m).square().sum()/(len(a)-1)) if len(a)>1 else None)
   means=torch.stack(means);candidate=[];weights=[];priors=[]
   for c in range(n):
    prior=t[c]+sum(means[j]-t[j] for j in range(n) if j!=c)/(n-1);var=variances[c]
    if var is None:var=sum(v for j,v in enumerate(variances) if j!=c and v is not None)/sum(j!=c and v is not None for j,v in enumerate(variances))
    u=var/len(fit[c]);w=min(1.,u/(float((means[c]-prior).square().sum())+1e-12));candidate.append((1-w)*means[c]+w*prior);weights.append(w);priors.append(prior)
   candidate=torch.stack(candidate);assert torch.isfinite(candidate).all()
   for c in range(n):
    write('STATUS.json',dict(phase='TRAIN_MEAN_DISCRIMINATION',dataset=ds,fold=fold,class_index=c,time=time.time()));z=x[held[c]];base=int((z@F.normalize(means,dim=1).T).argmax(1).eq(c).sum());new=int((z@F.normalize(candidate,dim=1).T).argmax(1).eq(c).sum());target=z.mean(0)
    records.append(dict(dataset=ds,fold=fold,class_index=c,train_count=int(counts[c]),fit_rows=len(fit[c]),holdout_rows=len(z),tail=bool(counts[c]<250),base_correct=base,semantic_correct=new,weight=weights[c],base_mean_squared_error=float((means[c]-target).square().sum()),semantic_mean_squared_error=float((candidate[c]-target).square().sum()),fit_holdout_disjoint=True,own_class_excluded_from_offset=True))
  summary[ds]=[]
  for fold in [0,1]:
   rr=[r for r in records if r['dataset']==ds and r['fold']==fold];tt=[r for r in rr if r['tail']];assert len(rr)==n and tt
   delta=lambda z:100*sum((r['semantic_correct']-r['base_correct'])/r['holdout_rows'] for r in z)/len(z)
   summary[ds].append(dict(fold=fold,classes=n,tail_classes=len(tt),macro_delta_pp=delta(rr),tail_delta_pp=delta(tt)))
 assert len(records)==54
 passed=all(s['macro_delta_pp']>=1 and s['tail_delta_pp']>=1 for s in summary['NIH']) and all(sum(s[k] for s in summary[ds])/2>=-1 for ds in ['ISIC','HK'] for k in ['macro_delta_pp','tail_delta_pp'])
 write('RESULTS.json',dict(records=records,summary=summary,optimizer_steps=0,model_forwards=0,VAL_reads=0,TEST_reads=0,no_random_draws=True,statistical_estimation_performed=True,not_patient_independent=True));write('FEASIBILITY.json',dict(passed=passed,summary=summary));write('STATUS.json',dict(phase='COMPLETE',time=time.time()))
try:main()
except BaseException:
 (out/'FAILURE.private.txt').write_text(traceback.format_exc());raise
