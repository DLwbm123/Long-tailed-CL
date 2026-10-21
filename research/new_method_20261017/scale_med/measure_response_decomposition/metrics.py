import json,os,time,resource
from pathlib import Path
from collections import Counter
import torch

torch.set_num_threads(4)
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def transform(g,mp):return (((g-mp['gn'][0])/mp['gn'][1])[:,mp['ids']]-mp['on'][0])/mp['on'][1]
def same_mapper(a,b):return torch.equal(a['ids'],b['ids']) and all(torch.equal(x,y) for k in ['gn','on'] for x,y in zip(a[k],b[k]))
def hessian(a):
 h=a.clone();h.diagonal()[:-1].add_(.01);return h

def qweights(y,groups):
 n=Counter(groups);nc={c:len({groups[i] for i in torch.where(y==c)[0]}) for c in range(4)}
 return torch.tensor([1/(4*nc[int(c)]*n[g]) for c,g in zip(y,groups)],dtype=torch.float64)
def points(s,y,groups):
 pred=s.argmax(1);correct=[int((pred[y==c]==c).sum()) for c in range(4)];counts=[int((y==c).sum()) for c in range(4)];rec=[k/n for k,n in zip(correct,counts)]
 ce=torch.nn.functional.cross_entropy(s,y,reduction='none');other=s.clone();other[torch.arange(len(y)),y]=-torch.inf;mar=s[torch.arange(len(y)),y]-other.max(1).values;p=qweights(y,groups)
 cm=torch.bincount(y*4+pred,minlength=16).reshape(4,4)
 return dict(BA=sum(rec)/4,correct=correct,counts=counts,recall=rec,group_balanced_BA=float(p@(pred==y).double()),confusion_matrix=cm.tolist(),classes=[dict(class_index=c,CE=float(ce[y==c].mean()),margin_quantiles=torch.quantile(mar[y==c],torch.tensor([0.,.1,.5,.9,1.],dtype=mar.dtype)).tolist()) for c in range(4)])
def pairs(b,p,y):
 bb,pp=b.argmax(1),p.argmax(1);rows=[]
 for c in range(4):
  ids=y==c;bo,po=bb[ids]==c,pp[ids]==c;r,h=int((~bo & po).sum()),int((bo & ~po).sum());assert r-h==int(po.sum())-int(bo.sum())
  rows.append(dict(class_index=c,rescued=r,harmed=h,net=r-h,both_correct=int((bo & po).sum()),both_wrong=int((~bo & ~po).sum()),harmed_destination=torch.bincount(pp[ids][bo & ~po],minlength=4).tolist()))
 assert abs(25*sum(z['net']/int((y==c).sum()) for c,z in enumerate(rows))-100*sum(float((pp[y==c]==c).double().mean()-(bb[y==c]==c).double().mean()) for c in range(4))/4)<1e-10
 return rows
