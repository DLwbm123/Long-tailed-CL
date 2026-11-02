import csv,json,sys
from pathlib import Path
import torch
from core import DT,state,unit

def load_stage(cfg,s):
 root=Path(cfg['source_root']);dc=json.loads((root/'r77_distributional_support/HK/config.private.json').read_text())
 with open(Path(cfg['manifest'])/'train.csv') as f:rows=list(csv.DictReader(f))
 labels=[cfg['order'].index(int(r['original_label'])) for r in rows];ids=[]
 for st in range(s+1):
  desc=torch.load(root/f'r77_distributional_support/arrival_meta/HK/{st}/descriptor.private.pt',map_location='cpu',weights_only=False);ids+=desc['ids']
 sys.path.insert(0,str(root/'r77_distributional_support'));from folds import assign_folds
 fold=assign_folds([labels[i] for i in ids],[rows[i]['identity_component'] for i in ids]);ids=[i for b in [1,0] for i,f in zip(ids,fold) if f==b]
 bank=torch.load(root/f'r77_distributional_support/HK/74002/SFT_SUPPORT{s}.private.pt',map_location='cpu',weights_only=False);head=torch.load(root/f'r67_clip_grpo/SFT/HK/74002/head{s}.private.pt',map_location='cpu',weights_only=False)
 assert bank['labels'].tolist()==[labels[i] for i in ids]
 roles=json.loads((root/'r90_evidence_prompt_program/ROLES.private.json').read_text());C=sum(map(len,cfg['stages'][:s+1]));index={i:k for k,i in enumerate(ids)};birth=[j for j,cc in enumerate(cfg['stages']) for c in cc]
 return {'z':bank['features'].to(DT),'y':bank['labels'],'ids':ids,'index':index,'head':head,'roles':{k:[i for i in v if i in index] for k,v in roles.items()},'C':C,'birth':birth}

def episode(cfg,d,s,idx,classes=None,full=False):
 classes=list(range(d['C'])) if classes is None else classes;cap=[1,2,4,8,None][idx%5];cap=None if full else cap;support=[];supportids=[]
 for c in classes:
  pool=sorted(i for i in d['roles']['S'] if int(d['y'][d['index'][i]])==c);size=len(pool) if cap is None else min(cap,len(pool));off=(74002+100003*s+idx//5)%len(pool) if idx>=5 else 0
  chosen=[pool[(off+j)%len(pool)] for j in range(size)];supportids+=chosen;support.append(d['z'][[d['index'][i] for i in chosen]])
 st,p0=state(support,classes,s,cfg['train_counts'],d['birth']);return st,p0,supportids

def query(d,role,classes=None):
 classes=list(range(d['C'])) if classes is None else classes;ids=[i for i in d['roles'][role] if int(d['y'][d['index'][i]]) in classes];ix=[d['index'][i] for i in ids];mapping={c:j for j,c in enumerate(classes)}
 return d['z'][ix],torch.tensor([mapping[int(d['y'][j])] for j in ix]),ids

def meta(cfg,C,s):
 tail=torch.tensor([n<250 for n in cfg['train_counts'][:C]]);old=sum(map(len,cfg['stages'][:s]));protected=tail|(torch.arange(C)<old);return tail,protected

def reference(cfg,d,s,ids,score0):
 y=d['y'][[d['index'][i] for i in ids]];C=d['C'];ref=torch.stack([(score0.argmax(1)[y==c]==c).to(DT).mean() for c in range(C)])
 if s:
  previous=load_stage(cfg,s-1);old=previous['C'];positions=[j for j,i in enumerate(ids) if i in previous['index']];z=previous['z'][[previous['index'][ids[j]] for j in positions]];yy=y[positions];pred=(20*z@previous['head']['prototypes'].to(DT).T).argmax(1)
  for c in range(old):ref[c]=(pred[yy==c]==c).to(DT).mean()
 return ref
