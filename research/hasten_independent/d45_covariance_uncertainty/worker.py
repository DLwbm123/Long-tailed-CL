"""Fixed two-fold TRAIN covariance predictive-density probe; no neural model."""
import os,json,ast,csv,time,traceback,math
from pathlib import Path
import torch
import torch.nn.functional as F
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);ref=Path(cfg['reference']);torch.set_num_threads(4);torch.set_num_interop_threads(2)
def write(n,v):(out/n).write_text(json.dumps(v,indent=2,allow_nan=False))
def main():
 records=[];summary={};d=512;eye=torch.eye(d,dtype=torch.double)
 for ds in ['NIH','ISIC','HK']:
  dc=json.loads((ref/'linear64'/ds/'config.private.json').read_text())
  if ds=='NIH':
   tree=ast.parse(Path(dc['base_runner']).read_text());v={a.targets[0].id:ast.literal_eval(a.value) for a in tree.body if isinstance(a,ast.Assign) and isinstance(a.targets[0],ast.Name) and a.targets[0].id=='ORDER'};order=v['ORDER'];n=10
  else:order=dc['order'];n=dc['task_sizes'][0]
  with open(Path(dc['manifest'])/'train.csv') as f:rows=list(csv.DictReader(f))
  assert all(r['split']=='train' for r in rows);labels=torch.tensor([order.index(int(r['original_label'])) for r in rows]);bank=torch.load(Path(dc['cache_root'] if ds=='NIH' else dc['shared_cache'])/'stage_0.private.pt',map_location='cpu',weights_only=False)
  assert bank['ids']==[i for i,y in enumerate(labels.tolist()) if y<n]
  x=bank['x'];x=F.normalize(x,dim=-1) if ds=='NIH' else x;y=labels[bank['ids']];assert torch.allclose(x.norm(dim=1),torch.ones(len(x)),atol=2e-6);x=x.double();counts=torch.bincount(y,minlength=n)
  for fold in [0,1]:
   stats=[];held=[]
   for c in range(n):
    ids=torch.where(y==c)[0];fit=ids[fold::2];test=ids[1-fold::2];assert len(fit)>0 and len(test)>0 and not set(fit.tolist())&set(test.tolist())
    a=x[fit];mean=a.mean(0);center=a-mean;cov=center.T@center/max(len(a)-1,1);stats.append((mean,cov,len(a)));held.append(test)
   pooled=sum(cov for _,cov,_ in stats)/n
   for c,(mean,cov,nfit) in enumerate(stats):
    write('STATUS.json',dict(phase='COVARIANCE_PREDICTIVE_DENSITY',dataset=ds,fold=fold,class_index=c,time=time.time()))
    lam=d/(d+max(nfit-1,0));matrices={'base':cov+1e-5*eye,'shrink':(1-lam)*cov+lam*pooled+1e-5*eye};res={};diff=(x[held[c]]-mean).T
    for name,C in matrices.items():
     L=torch.linalg.cholesky(C);z=torch.linalg.solve_triangular(L,diff,upper=False);quad=z.square().sum(0);ld=2*L.diag().log().sum();nll=.5*(quad+ld+d*math.log(2*math.pi))/d
     assert torch.isfinite(nll).all();res[name]=dict(mean_NLL_per_dim=float(nll.mean()),mahalanobis_per_dim=float(quad.mean()/d),logdet_per_dim=float(ld/d))
    records.append(dict(dataset=ds,fold=fold,class_index=c,train_count=int(counts[c]),fit_rows=nfit,holdout_rows=len(held[c]),tail=bool(counts[c]<250),rank_upper_bound=min(nfit-1,d),lambda_c=lam,**res,delta_NLL_per_dim=res['shrink']['mean_NLL_per_dim']-res['base']['mean_NLL_per_dim'],fit_holdout_disjoint=True))
  summary[ds]=[]
  for fold in [0,1]:
   rr=[r for r in records if r['dataset']==ds and r['fold']==fold];tt=[r for r in rr if r['tail']];assert len(rr)==n and len(tt)>0
   summary[ds].append(dict(fold=fold,classes=n,tail_classes=len(tt),macro_delta_NLL_per_dim=sum(r['delta_NLL_per_dim'] for r in rr)/n,tail_delta_NLL_per_dim=sum(r['delta_NLL_per_dim'] for r in tt)/len(tt)))
  del x,bank,stats,held
 assert len(records)==54
 passed=all(s['tail_delta_NLL_per_dim']<=-.1 for s in summary['NIH']) and all(s['macro_delta_NLL_per_dim']<=.01 for z in summary.values() for s in z)
 write('RESULTS.json',dict(records=records,summary=summary,optimizer_steps=0,model_forwards=0,VAL_reads=0,TEST_reads=0,no_random_draws=True,statistical_estimation_performed=True,not_accuracy_evidence=True));write('FEASIBILITY.json',dict(passed=passed,summary=summary,not_patient_independent=True));write('STATUS.json',dict(phase='COMPLETE',time=time.time()))
try:main()
except BaseException:
 (out/'FAILURE.private.txt').write_text(traceback.format_exc());raise
