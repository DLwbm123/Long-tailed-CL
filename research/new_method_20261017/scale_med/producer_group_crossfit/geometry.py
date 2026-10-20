import torch
from collections import Counter
PERM=torch.randperm(1536,generator=torch.Generator().manual_seed(74010))
def weights(y,groups):
 counts=Counter(groups); nc={c:len({g for i,g in enumerate(groups) if int(y[i])==c}) for c in range(4)}
 return torch.tensor([1/(nc[int(c)]*counts[g]) for c,g in zip(y,groups)],dtype=torch.float64)
def norm(g,q):
 p=q/q.sum();mu=p@g;return mu,((g-mu).square()*p[:,None]).sum().sqrt().clamp_min(1e-12)
def mapper(g,q):
 gn=norm(g,q);z=(g-gn[0])/gn[1];ids=PERM[:1535];return dict(gn=gn,ids=ids,on=norm(z[:,ids],q))
def transform(g,mp):return (((g-mp['gn'][0])/mp['gn'][1])[:,mp['ids']]-mp['on'][0])/mp['on'][1]
def teacher(g,y,groups):
 q=weights(y,groups);mp=mapper(g,q);z=transform(g,mp);x=torch.cat([z,torch.ones(len(z),1,dtype=g.dtype)],1);p=q/q.sum();a=x.T@(x*p[:,None]);a=(a+a.T)/2;b=x.T@(torch.nn.functional.one_hot(y,6).double()*p[:,None]);h=a.clone();h.diagonal()[:-1].add_(.01);w=torch.linalg.solve(h,b);err=float((h@w-b).abs().max());assert err<=1e-7 and torch.isfinite(w).all();return dict(mapper=mp,W=w[:,:4],A=a,B=b,residual=err)
def split(y,groups):
 gen=torch.Generator().manual_seed(74290);f=torch.full((len(y),),-1,dtype=torch.long)
 assert all(len({int(y[i]) for i,g in enumerate(groups) if g==key})==1 for key in set(groups))
 for c in range(4):
  gs=sorted({groups[i] for i in torch.where(y==c)[0]});gs=[gs[i] for i in torch.randperm(len(gs),generator=gen).tolist()]
  for j,g in enumerate(gs):f[torch.tensor([v==g for v in groups])]=j%2
  assert all(len({groups[i] for i in torch.where((y==c)&(f==k))[0]})>=8 for k in [0,1])
 assert (f>=0).all() and not {groups[i] for i in torch.where(f==0)[0]} & {groups[i] for i in torch.where(f==1)[0]}
 return f
