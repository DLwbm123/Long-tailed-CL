import json,csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from common import write,request_cache

def train_rows(cfg):
 with open(Path(cfg['manifest'])/'train.csv') as f:r=list(csv.DictReader(f))
 assert all(v['split']=='train' for v in r);y=torch.tensor([cfg['order'].index(int(v['original_label'])) for v in r]);assert torch.bincount(y,minlength=len(cfg['order'])).tolist()==cfg['train_counts_ordered'];return r,y

def get_cache(rootcfg,dc,stage=None,preflight=False,val=False):
 root=Path(rootcfg['output']);baseid=dc['job_base'];j=0 if preflight else baseid+(len(dc['stages']) if val else stage);dest=root/'cache'/str(j)
 if (dest/'READY.json').exists():return dest
 rows,labels=train_rows(dc)
 if val:
  seal=json.loads((Path(dc['output'])/'ALL_HEADS_SEALED.json').read_text());assert seal['VAL_reads']==0 and seal['records']==3*len(dc['stages'])
  with open(Path(dc['manifest'])/'val.csv') as f:part=list(csv.DictReader(f))
  assert all(v['split']=='val' for v in part);ids=list(range(len(rows),len(rows)+len(part)));y=[dc['order'].index(int(v['original_label'])) for v in part];ref=torch.load(dc['validation_cache'],map_location='cpu',weights_only=False);assert ref['ids']==ids;x=ref['x'];text=None
 else:
  b=torch.load(Path(dc['train_cache'])/f'stage_{stage}.private.pt',map_location='cpu',weights_only=False);ids=[i for i,v in enumerate(labels.tolist()) if v in dc['stages'][stage]];assert b['ids']==ids;part=[rows[i] for i in ids];y=labels[ids].tolist();x=F.normalize(b['x'],dim=1) if dc['dataset']=='NIH' else b['x'];n=sum(len(v) for v in dc['stages'][:stage+1]);bank=F.normalize(b['text'],dim=1) if dc['dataset']=='NIH' else b['text'];text=bank[[b['prompts'].index(dc['leaf_template'].format(v)) for v in dc['ordered_names'][:n]]]
 if preflight:ids=ids[:2];part=part[:2];y=y[:2];x=x[:2]
 assert torch.isfinite(x).all() and torch.allclose(x.norm(dim=1),torch.ones(len(x)),atol=2e-6);refpath=root/'refs'/f'{j}.pt';refpath.parent.mkdir(exist_ok=True);torch.save(dict(ids=ids,x=x,text=text),refpath)
 job=dict(id=j,dataset=dc['dataset'],role='preflight' if preflight else 'cache',split='val' if val else 'train',stage=stage,TRAIN_sealed_all3seeds=bool(val),rows=[dict(id=i,relative_path=r['relative_path'],label=int(c)) for i,r,c in zip(ids,part,y)])
 return request_cache(rootcfg,job)

class Tokens(Dataset):
 def __init__(self,paths):
  self.index=[];base=[];labels=[];ids=[]
  for path in paths:
   path=Path(path);m=json.loads((path/'META.json').read_text())
   for ch in m['chunks']:
    base.append(torch.from_numpy(np.load(path/ch['base']).copy()));start=ch['start'];n=ch['rows'];labels.extend(m['labels'][start:start+n]);ids.extend(m['ids'][start:start+n]);self.index.extend([(str(path/ch['file']),n,k) for k in range(n)])
  self.base=torch.cat(base);self.labels=torch.tensor(labels,dtype=torch.long);self.ids=ids;assert len(ids)==len(set(ids))==len(self.index)
 def __len__(self):return len(self.index)
 def __getitem__(self,i):
  p,n,k=self.index[i];m=np.memmap(p,dtype=np.float32,mode='r',shape=(n,197,768));x=torch.from_numpy(np.array(m[k],copy=True));m._mmap.close();return x,self.base[i],self.labels[i],i

class OrderSampler:
 def __init__(self,n):self.order=list(range(n))
 def __iter__(self):return iter(self.order)
 def __len__(self):return len(self.order)
