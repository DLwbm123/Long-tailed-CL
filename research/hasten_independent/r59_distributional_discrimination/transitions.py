"""Saved-response conditional ablations and old-tail/new-head error accounting."""
import os,json
from pathlib import Path
import torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);records=[]
for ds in ['NIH','ISIC','HK']:
 d=root/ds;dc=json.loads((d/'config.private.json').read_text());stage=len(dc['stages'])-1;counts=dc['train_counts_ordered'];new=set(dc['stages'][stage]);oldtail={c for c,n in enumerate(counts) if n<250 and c not in new};newhead={c for c in new if counts[c]>=250}
 for seed in [74002,74003,74004]:
  r=torch.load(d/str(seed)/f'responses{stage}.private.pt',map_location='cpu',weights_only=False);y=r['labels'];ot=torch.tensor([int(z) in oldtail for z in y]);pred={k:z.argmax(1) for k,z in r['scores'].items()}
  for arm,p in pred.items():
   conf=torch.zeros(len(counts),len(counts),dtype=torch.long);conf.index_put_((y,p),torch.ones(len(y),dtype=torch.long),accumulate=True);into=torch.tensor([int(z) in newhead for z in p]);records.append(dict(dataset=ds,seed=seed,arm=arm,old_tail_classes=sorted(oldtail),new_head_classes=sorted(newhead),old_tail_rows=int(ot.sum()),old_tail_correct=int(((p==y)&ot).sum()),old_tail_predicted_new_head=int((ot&into).sum()),confusion=conf.tolist(),counts=torch.bincount(y,minlength=len(counts)).tolist()))
(root/'results'/'FINAL_OLD_TAIL_NEW_HEAD.json').write_text(json.dumps(dict(scope='Saved final responses only, descriptive competition conditional on candidate set, no new inference/fit or causal decomposition',records=records),indent=2))
print('27finalreadoutconfusions and old-tail/new-head counts saved')
