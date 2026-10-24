"""Saved weight spectra and saved prediction transitions; no fitting/model forward."""
import os,json
from pathlib import Path
import torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);rows=[]
for ds in ['NIH','ISIC','HK']:
 d=root/ds;dc=json.loads((d/'config.private.json').read_text())
 for seed in [74002,74003,74004]:
  for stage in range(len(dc['stages'])):
   h=torch.load(d/str(seed)/f'head{stage}.private.pt',map_location='cpu',weights_only=False);u=h['state']['U'].double();v=h['state']['V'].double();_,ru=torch.linalg.qr(u,mode='reduced');_,rv=torch.linalg.qr(v,mode='reduced');sv=torch.linalg.svdvals(ru@rv.T);sq=sv.square();p=sq/sq.sum();f=torch.load(d/str(seed)/f'responses{stage}.private.pt',map_location='cpu',weights_only=False);a=f['scores']['shared_metric'].argmax(1);b=f['scores']['raw_ncm'].argmax(1);y=f['labels'];n=len(h['means']);counts=dc['train_counts_ordered'][:n];tail=torch.tensor([counts[i]<250 for i in y.tolist()]);rows.append(dict(dataset=ds,seed=seed,stage=stage,F_norm=float(sv.norm()),operator_norm=float(sv[0]),leading_energy_fraction=float(p[0]),participation_rank=float(1/p.square().sum()),singular_values=sv.tolist(),changed_predictions=int((a!=b).sum()),response_rows=len(y),changed_tail_predictions=int(((a!=b)&tail).sum()),tail_rows=int(tail.sum()),raw_wrong_to_shared_correct=int(((a==y)&(b!=y)).sum()),raw_correct_to_shared_wrong=int(((a!=y)&(b==y)).sum())))
(root/'results'/'MECHANISM_ACCOUNTING.json').write_text(json.dumps(dict(scope='savedweights/savedresponses only; no new inference/optimization/VAL feature reads; descriptive not causal',records=rows),indent=2))
for ds in ['NIH','ISIC','HK']:
 r=[z for z in rows if z['dataset']==ds];print(ds,'allstages_changed',sum(z['changed_predictions'] for z in r),'over',sum(z['response_rows'] for z in r),'tail_changed',sum(z['changed_tail_predictions'] for z in r),'leadingenergy',min(z['leading_energy_fraction'] for z in r),max(z['leading_energy_fraction'] for z in r),'effective_rank',min(z['participation_rank'] for z in r),max(z['participation_rank'] for z in r))
