"""Fixed training-only decision-difference space and trace-matched controls."""
import torch
from solver import mapper,transform
from core_attribution import weights

def build(g,y,groups):
 q=weights(y,groups);mp=mapper(g,q);z=transform(g,mp);w=q/q.sum();mu=w@z;c=(z-mu).T@((z-mu)*w[:,None]);c=(c+c.T)/2;e,u=torch.linalg.eigh(c);assert e.min()>=-1e-8;keep=e>e.max()*1e-10;S=u[:,keep];m=S.shape[1];assert m>3;t=float(e.clamp_min(0).sum());xx=torch.cat([z,torch.ones(len(z),1,dtype=z.dtype)],1);a=xx.T@(xx*w[:,None]);a=(a+a.T)/2;a.diagonal()[:-1].add_(.01);rhs=xx.T@(torch.nn.functional.one_hot(y,4).double()*w[:,None]);W=torch.linalg.solve(a,rhs);wc=W[:-1]-W[:-1].mean(1,keepdim=True);left,s,_=torch.linalg.svd(wc,full_matrices=False);rank=int((s>s.max()*1e-10).sum());assert 0<rank<=3
 D=left[:,:rank];err=float((D-S@(S.T@D)).abs().max());assert err<=1e-8;D=torch.linalg.qr(S@(S.T@D),mode='reduced')[0];gen=torch.Generator().manual_seed(74251);R=S@torch.linalg.qr(torch.randn(m,rank,dtype=g.dtype,generator=gen),mode='reduced')[0]
 assert torch.allclose(D.T@D,torch.eye(rank,dtype=g.dtype),atol=1e-9) and torch.allclose(R.T@R,torch.eye(rank,dtype=g.dtype),atol=1e-9)
 return dict(mapper=mp,S=S,D=D,R=R,m=m,r=rank,t=t,teacher_W=W,projection_error=err,random_overlap_singular_values=torch.linalg.svdvals(D.T@R).tolist(),seed=74251)

def penalty(delta,state,arm):
 if arm=='E':return delta.sum()*0
 basis=state['S'] if arm=='L' else state[arm];v=(delta@basis).square().sum(1).mean();return v/state['t'] if arm=='L' else v/(state['t']*state['r']/state['m'])

def device_state(s,device):
 return {k:({n:(v.to(device) if torch.is_tensor(v) else tuple(t.to(device) for t in v)) for n,v in a.items()} if k=='mapper' else a.to(device) if torch.is_tensor(a) else a) for k,a in s.items()}

def synthetic():
 gen=torch.Generator().manual_seed(74251);g=torch.randn(32,9,dtype=torch.float64,generator=gen);g=torch.nn.functional.normalize(g,dim=1);y=torch.arange(32)%4
 # Local indices only for the synthetic geometry, never changing formal PERM/width.
 from solver import normfit
 mp0=mapper
 import protection
 def toy_mapper(x,q):
  gn=normfit(x,q);z=(x-gn[0])/gn[1];return dict(gn=gn,on=normfit(z,q),ids=torch.arange(x.shape[1]))
 protection.mapper=toy_mapper
 try:s=build(g,y,['toy'+str(i) for i in range(32)])
 finally:protection.mapper=mp0
 for arm in ['D','R','L']:
  d=torch.randn(8,9,dtype=g.dtype,generator=gen,requires_grad=True);direction=torch.randn(8,9,dtype=g.dtype,generator=gen);direction/=direction.norm();v=penalty(d,s,arm);grad=torch.autograd.grad(v,d)[0];h=1e-5;fd=(penalty(d+h*direction,s,arm)-penalty(d-h*direction,s,arm))/(2*h);ad=(grad*direction).sum();assert abs(fd-ad)<1e-7
  zero=torch.zeros_like(d,requires_grad=True);gz=torch.autograd.grad(penalty(zero,s,arm),zero)[0];assert torch.equal(gz,torch.zeros_like(gz))
 r,m=s['r'],s['m'];assert abs(float((s['D']@s['D'].T).trace())-r)<1e-9;assert abs(float((s['R']@s['R'].T).trace())-r)<1e-9;assert abs(float((r/m*s['S']@s['S'].T).trace())-r)<1e-9
 shifted=s['teacher_W'][:-1]+torch.ones(9,1,dtype=g.dtype);a=shifted-shifted.mean(1,keepdim=True);orig=s['teacher_W'][:-1]-s['teacher_W'][:-1].mean(1,keepdim=True);assert torch.allclose(a,orig,atol=1e-12)
 return dict(orthogonal_trace_matched=True,quadratic_finite_difference=True,zero_drift_gradient=True,common_logit_shift_invariance=True,synthetic_rank=r,synthetic_support_dimension=m)
