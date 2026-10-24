"""Fixed TRAIN local visual label-structure diagnostic; no optimizer or neural forward."""
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
   fit=[];held=[];means=[]
   for c in range(n):
    ids=torch.where(y==c)[0];a=ids[fold::2][:64];b=ids[1-fold::2][:64];assert len(a)>0 and len(b)>0 and not set(a.tolist())&set(b.tolist());fit.append(a);held.append(b);means.append(x[a].mean(0))
   means=F.normalize(torch.stack(means),dim=1);support=torch.cat(fit);sy=y[support];sx=x[support]
   for c in range(n):
    write('STATUS.json',dict(phase='VISUAL_LOCAL_STRUCTURE',dataset=ds,fold=fold,class_index=c,time=time.time()));z=x[held[c]];base=int((z@means.T).argmax(1).eq(c).sum());local=int(sy[(z@sx.T).argmax(1)].eq(c).sum())
    records.append(dict(dataset=ds,fold=fold,class_index=c,train_count=int(counts[c]),fit_rows=len(fit[c]),holdout_rows=len(z),tail=bool(counts[c]<250),centroid_correct=base,nearest_neighbor_correct=local,fit_holdout_disjoint=True,same_support_both_readouts=True))
  summary[ds]=[]
  for fold in [0,1]:
   rr=[r for r in records if r['dataset']==ds and r['fold']==fold];tt=[r for r in rr if r['tail']];assert len(rr)==n and tt
   delta=lambda z:100*sum((r['nearest_neighbor_correct']-r['centroid_correct'])/r['holdout_rows'] for r in z)/len(z)
   summary[ds].append(dict(fold=fold,classes=n,tail_classes=len(tt),macro_delta_pp=delta(rr),tail_delta_pp=delta(tt)))
 assert len(records)==54
 passed=all(s['macro_delta_pp']>=0 and s['tail_delta_pp']>=1 for s in summary['NIH']) and all(sum(s[k] for s in summary[ds])/2>=-1 for ds in ['ISIC','HK'] for k in ['macro_delta_pp','tail_delta_pp'])
 write('RESULTS.json',dict(records=records,summary=summary,optimizer_steps=0,model_forwards=0,VAL_reads=0,TEST_reads=0,no_random_draws=True,statistical_estimation_performed=True,not_patient_independent=True,not_inference_candidate=True));write('FEASIBILITY.json',dict(passed=passed,summary=summary,training_support_requires_separate_gradient_evidence=True));write('STATUS.json',dict(phase='COMPLETE',time=time.time()))
try:main()
except BaseException:
 (out/'FAILURE.private.txt').write_text(traceback.format_exc());raise
