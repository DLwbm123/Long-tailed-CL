from pathlib import Path
import torch

def load_stage(cfg,stage,fold):
 root=Path(cfg['feature_source_run']);f=torch.load(root/f'stage{stage}.private.pt',map_location='cpu',weights_only=False);h=torch.load(root/f's{stage}f{fold}.private.pt',map_location='cpu',weights_only=False);q=torch.load(root/f'scores_s{stage}f{fold}.private.pt',map_location='cpu',weights_only=False)
 n=sum(map(len,cfg['stages'][:stage+1]));assert h['seen']==n and len(h['prototypes'])==n
 assert q['Q']==h['Q'] and q['ids']==[f['ids'][i] for i in h['Q']] and torch.equal(q['labels'],f['labels'][h['Q']])
 assert set(h['S']).isdisjoint(h['Q']);assert all(int(v['step'])==24*(stage+1) for v in f['source_optimizer_state']['state'].values())
 assert torch.equal(q['arms']['NCM']['prediction'],q['arms']['NCM']['logits'].argmax(1))
 support=torch.bincount(f['labels'][h['S']],minlength=n).double();assert (support>0).all();age=torch.zeros(n,dtype=torch.double)
 for birth,cc in enumerate(cfg['stages'][:stage+1]):age[cc]=stage-birth
 eligible=[c for c in range(n) if not(fold==1 and c==1)];old=sum(map(len,cfg['stages'][:stage]));protected=[c for c in eligible if c<old or cfg['train_counts'][c]<250]
 # Full cached arrays are read-only. Only selected training labels enter a policy objective.
 return dict(q=q,h=h,n=n,support=support,ages=age,eligible=eligible,protected=protected,proto=h['prototypes'].double())

def subset(v,tag):
 ids=set(v['h'][tag+'_ids']) if tag!='full' else set(v['q']['ids']);mask=torch.tensor([i in ids for i in v['q']['ids']],dtype=torch.bool);assert mask.any();return mask
