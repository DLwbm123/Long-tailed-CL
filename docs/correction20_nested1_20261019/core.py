"""Fixed smooth discriminative correction and training-only weighting helpers."""
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F


def weights(y,g):
 w=np.empty(len(y),dtype=np.float64)
 for c in np.unique(y):
  ids=np.flatnonzero(y==c);count=Counter(g[ids].tolist());w[ids]=[1/(len(count)*count[z]) for z in g[ids]]
 return w


def targets(cls,joint,y,w):
 c=cls.argmax(1)==y;j=joint.argmax(1)==y;center=joint-joint.mean(1,keepdims=True)
 temperature=float(np.sqrt(np.sum(w[:,None]*center**2)/(w.sum()*joint.shape[1])))
 assert temperature>1e-12 and j.any()
 other=joint.copy();other[np.arange(len(y)),y]=-np.inf
 margins=(joint[np.arange(len(y)),y]-other.max(1))/temperature
 values=margins[j];ww=w[j];idx=np.argsort(values,kind='stable');cap=float(values[idx][np.searchsorted(np.cumsum(ww[idx]),ww.sum()/2)]);assert cap>0
 return c&~j,j,np.where(j,np.minimum(margins,cap),0.),temperature,cap


def loss(u,x,base,y,w,repair,keep,target,precondition,utility,regularizer):
 score=base+x@u
 ce=F.cross_entropy(score,y,reduction='none')
 value=(w*ce*(1+repair if utility else 1)).sum()
 if utility:
  gap=score.gather(1,y[:,None])-score
  penalty=F.softplus(target[:,None]-gap);penalty=penalty.scatter(1,y[:,None],0.)
  value=value+(w*keep*penalty.sum(1)/(score.shape[1]-1)).sum()
 return value+regularizer*((precondition[:,None]*u)**2).sum()


def spectrum(g):
 ev=torch.linalg.eigvalsh((g+g.T)/2);assert float(ev.min())>=-max(1.,float(ev.abs().max()))*1e-10
 return ev.clamp_min(0)


def matched_alpha(ev,target):
 def df(a):return float((ev/(ev+a)).sum())
 lo,hi=1e-12,1.
 assert df(lo)>target
 while df(hi)>target:hi*=2;assert hi<=1e12
 for _ in range(80):
  m=(lo+hi)/2
  if df(m)>target:lo=m
  else:hi=m
 a=(lo+hi)/2;assert abs(df(a)-target)<1e-7
 return a


def metrics(score,y,g,seen,frequency):
 p=score.argmax(1);w=weights(y,g);correct=p==y
 rec=[float(correct[y==c].mean()) for c in range(20)]
 return dict(BA=sum(rec)/20,PP_accuracy=float(np.dot(w,correct)/w.sum()),correct=int(correct.sum()),images=len(y),patients=len(np.unique(g)),counts=[int((y==c).sum()) for c in range(20)],correct_by_class=[int(correct[y==c].sum()) for c in range(20)],recall=rec,confusion=[[int(((y==c)&(p==d)).sum()) for d in range(20)] for c in range(20)],frequency_BA={name:sum(rec[seen.index(c)] for c in cs)/len(cs) for name,cs in frequency.items()})
