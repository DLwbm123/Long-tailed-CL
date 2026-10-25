from pathlib import Path
import json,torch
r=Path('/tmp/q1092');out={}
for ds in ['NIH','ISIC','HK']:
 d=r/ds;a=json.loads((d/'aggregate.json').read_text());stages=sorted({v['stage'] for v in a['results']});out[ds]=[]
 for seed in [74002,74003,74004]:
  f=torch.load(d/'responses'/str(seed)/f'responses{stages[-1]}.private.pt',map_location='cpu',weights_only=False);scores=f['scores']['support_lora'];fp=scores.argmax(1);ids=f['ids'];idx={v:i for i,v in enumerate(ids)};fh=torch.load(d/str(seed)/f'head{stages[-1]}.private.pt',map_location='cpu',weights_only=False);rows=[]
  for stage in stages:
   z=torch.load(d/'responses'/str(seed)/f'responses{stage}.private.pt',map_location='cpu',weights_only=False);n=z['scores']['support_lora'].shape[1];ind=torch.tensor([idx[v] for v in z['ids']]);assert torch.equal(z['labels'],f['labels'][ind]);pr=z['scores']['support_lora'].argmax(1);restricted=scores[ind,:n].argmax(1);full=fp[ind];h=torch.load(d/str(seed)/f'head{stage}.private.pt',map_location='cpu',weights_only=False)
   c=[]
   for k in range(n):
    mask=z['labels']==k;aa=pr[mask];bb=restricted[mask];cc=full[mask];yy=z['labels'][mask];c.append(dict(class_index=k,count=int(mask.sum()),A=int((aa==yy).sum()),B=int((bb==yy).sum()),C=int((cc==yy).sum()),AB_rescue=int(((aa!=yy)&(bb==yy)).sum()),AB_harm=int(((aa==yy)&(bb!=yy)).sum()),BC_rescue=int(((bb!=yy)&(cc==yy)).sum()),BC_harm=int(((bb==yy)&(cc!=yy)).sum()),prototype_cos=float((h['prototypes'][k]*fh['prototypes'][k]).sum())))
   rows.append(dict(stage=stage,seen=n,classes=c))
  out[ds].append(dict(seed=seed,rows=rows))
print(json.dumps(out,allow_nan=False))
