import torch
from core_depth import PERM

def views(x):
 return [torch.rot90(x,k,(-2,-1)).contiguous() for k in range(4)]+[torch.rot90(x.flip(-1),k,(-2,-1)).contiguous() for k in range(4)]
def normfit(v,q):
 w=q/q.sum();mu=w@v.mean(1);s=((v-mu).square().mean(1)*w[:,None]).sum().sqrt().clamp_min(1e-12);return mu,s
def mapper(v,q,ids=None):
 ids=PERM[:1535] if ids is None else ids;gn=normfit(v,q);z=(v-gn[0])/gn[1];on=normfit(z[:,:,ids],q);return dict(gn=gn,on=on,ids=ids)
def transform(v,m):return (((v-m['gn'][0])/m['gn'][1])[...,m['ids']]-m['on'][0])/m['on'][1]
def stats(v,y,q):
 x=torch.cat([v,torch.ones(*v.shape[:-1],1,dtype=v.dtype)],-1);bar=x.mean(1);am=bar.T@(bar*q[:,None]);delta=x-bar[:,None];c=torch.einsum('nvd,nve,n->de',delta,delta,q)/v.shape[1];c=(c+c.T)/2;c[-1,:]=0;c[:,-1]=0;ai=am.clone();ai.diagonal()[:-1].add_(c.diagonal().sum()/(len(c)-1));b=bar.T@(torch.nn.functional.one_hot(y,6).double()*q[:,None]);return dict(M=(am,b),I=(ai,b),A=(am+c,b)),c,bar

def fit(a,b,k,alpha):
 h=a/k;h=h.clone();h.diagonal()[:-1].add_(alpha);w=torch.linalg.solve(h,b/k);e=float((h@w-b/k).abs().max());assert e<=1e-7 and torch.isfinite(w).all();return w,e

def synthetic():
 x=torch.arange(25).reshape(1,1,5,5);vs=views(x);assert len({tuple(t.flatten().tolist()) for t in vs})==8
 for xx in vs:assert {tuple(t.flatten().tolist()) for t in views(xx)}=={tuple(t.flatten().tolist()) for t in vs}
 gen=torch.Generator().manual_seed(74230);v=torch.randn(24,8,11,generator=gen,dtype=torch.float64);y=torch.arange(24)%4;q=torch.rand(24,generator=gen,dtype=v.dtype)+.1;m=mapper(v,q,torch.arange(10));z=transform(v,m);s,c,bar=stats(z,y,q);xx=torch.cat([z,torch.ones(24,8,1,dtype=v.dtype)],-1).reshape(-1,11);qq=q.repeat_interleave(8)/8;direct=xx.T@(xx*qq[:,None]);err=float((direct-s['A'][0]).abs().max());assert err<=1e-10 and torch.linalg.eigvalsh(c).min()>=-1e-10 and not c[-1].any();assert abs(float((s['I'][0]-s['M'][0]).trace()-c.trace()))<1e-10
 w,_=fit(*s['A'],4,.01);wd,_=fit(direct,s['A'][1],4,.01);assert float((bar@w-bar@wd).abs().max())<1e-10;assert float(((xx@w).reshape(24,8,6).mean(1)-bar@w).abs().max())<1e-10
 repeated=v[:,0:1].expand(-1,8,-1);ss,cc,bb=stats(transform(repeated,mapper(repeated,q,torch.arange(10))),y,q);assert float(cc.abs().max())<1e-25;return dict(group_closure=True,expanded_statistics_error=err,PSD=True,intercept_unpenalized=True,mean_score_identity=True,duplicate_view_zero_covariance=True,equal_trace=True)
