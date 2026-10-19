"""Lossless symmetric storage; dense working solve is temporary RAM."""
import torch,io
CAP=19058800
PERM=torch.randperm(1536,generator=torch.Generator().manual_seed(74010))
def pack(a):return a[torch.triu_indices(len(a),len(a)).unbind()].contiguous()
def unpack(v,n):
 ix=torch.triu_indices(n,n);a=torch.zeros((n,n),dtype=v.dtype);a[ix.unbind()]=v;a[(ix[1],ix[0])]=v;return a
def fit(a,b,k,alpha):
 aa=unpack(pack(a),len(a));h=aa.clone();h.diagonal()[:-1].add_(k*alpha);w=torch.linalg.solve(h,b);err=float((h@w-b).abs().max());assert err<=1e-7 and torch.isfinite(w).all();return w,err

def mapper(g,h,t):
 gg=(g-t['g_norm'][0])/t['g_norm'][1];core=gg[:,PERM];oldcore=core[:,:t['d']]
 if t['arm']=='G':z=core
 else:
  hh=(h-t['h_norm'][0])/t['h_norm'][1];extra=hh@t['U']
  if t['arm']=='C':extra-=oldcore@t['V']
  z=torch.cat([core,extra],1)
 if t.get('scale_only',False):z=torch.cat([z[:,:t['d']],z[:,1536:]],1)
 return torch.cat([(z-t['out_norm'][0])/t['out_norm'][1],torch.ones((len(z),1),dtype=torch.float64)],1)
def selfcheck():
 torch.manual_seed(74158);x=torch.cat([torch.randn(24,7,dtype=torch.float64),torch.ones(24,1,dtype=torch.float64)],1);q=torch.ones(24,dtype=torch.float64)/6;y=torch.arange(24)%4;b=x.T@(torch.nn.functional.one_hot(y,6).double()*q[:,None]);a=x.T@(x*q[:,None]);w,err=fit(a,b,4,.01);h=a.clone();h.diagonal()[:-1].add_(.04);dense=torch.linalg.solve(h,b);assert (a-unpack(pack(a),len(a))).abs().max()<1e-12 and (w-dense).abs().max()<1e-12
 # Actual maximum-sized state with all inference transforms/statistics and one head.
 sizes={}
 for arm in ['G','P','C']:
  n=1537 if arm=='G' else 1665;t=dict(arm=arm,d=1318 if arm=='C' else 1375,g_norm=(torch.zeros(1536,dtype=torch.float64),torch.tensor(1.,dtype=torch.float64)),out_norm=(torch.zeros(n-1,dtype=torch.float64),torch.tensor(1.,dtype=torch.float64)),permutation=PERM,alpha=.01)
  if arm!='G':t.update(h_norm=(torch.zeros(768,dtype=torch.float64),torch.tensor(1.,dtype=torch.float64)),U=torch.zeros(768,128,dtype=torch.float64))
  if arm=='C':t['V']=torch.zeros(1318,128,dtype=torch.float64)
  state=dict(mapper=t,A=torch.zeros(n*(n+1)//2,dtype=torch.float64),B=torch.zeros(n,6,dtype=torch.float64),W=torch.zeros(n,6,dtype=torch.float64));buf=io.BytesIO();torch.save(state,buf);sizes[arm]=buf.tell();assert sizes[arm]<=CAP
 return dict(packed_dense_exact=True,solve_max_residual=err,synthetic_complete_artifact_bytes=sizes,cap=CAP)
