import torch
from core_depth import PERM

def normfit(g,q):
 w=q/q.sum();mu=w@g;scale=((g-mu).square()*w[:,None]).sum().sqrt().clamp_min(1e-12);return mu,scale
def normmap(g,t):return (g-t[0])/t[1]
def mapper(g,q,indices=None):
 ids=PERM[:1535].to(g.device) if indices is None else indices
 gn=normfit(g,q);core=normmap(g,gn)[:,ids];on=normfit(core,q);return dict(gn=gn,on=on,ids=ids)
def transform(g,t):return normmap(normmap(g,t['gn'])[:,t['ids']],t['on'])
def dual(x,y,q,alpha=.01):
 w=q/q.sum();hot=torch.nn.functional.one_hot(y,int(y.max())+1).to(x);mx=w@x;my=w@hot;xx=(x-mx)*w.sqrt()[:,None];yy=(hot-my)*w.sqrt()[:,None];a=xx@xx.T+alpha*torch.eye(len(x),dtype=x.dtype,device=x.device);v=torch.linalg.solve(a,yy);W=xx.T@v;b=my-mx@W;return W,b

def objective(g,y,mode,indices=None):
 # A/B are both differentiable; query pixels never enter the opposite support norm.
 a,b=g[:16],g[16:];ya,yb=y[:16],y[16:];q=torch.ones(16,dtype=g.dtype,device=g.device)/4
 ta,tb=mapper(a,q,indices),mapper(b,q,indices);xa,xb=transform(a,ta),transform(b,tb);wa,ba=dual(xa,ya,q);wb,bb=dual(xb,yb,q)
 if mode=='I':pa,pb=xa@wa+ba,xb@wb+bb
 else:pa,pb=transform(a,tb)@wb+bb,transform(b,ta)@wa+ba
 return (torch.nn.functional.cross_entropy(pa,ya)+torch.nn.functional.cross_entropy(pb,yb))/2

def synthetic():
 gen=torch.Generator().manual_seed(74220);x=torch.randn(16,7,generator=gen,dtype=torch.float64);y=torch.arange(16)%4;q=torch.rand(16,generator=gen,dtype=x.dtype)+.1;alpha=.01;W,b=dual(x,y,q,alpha);z=torch.cat([x,torch.ones(16,1,dtype=x.dtype)],1);w=q/q.sum();aa=z.T@(z*w[:,None]);aa.diagonal()[:-1].add_(alpha);rhs=z.T@(torch.nn.functional.one_hot(y,4)*w[:,None]);dense=torch.linalg.solve(aa,rhs);err=float((z@dense-(x@W+b)).abs().max());assert err<1e-9
 g=torch.randn(32,9,generator=gen,dtype=torch.float64,requires_grad=True);yy=torch.cat([torch.arange(16)//4]*2);direction=torch.randn(32,9,generator=gen,dtype=g.dtype);direction/=direction.norm();e=1e-5;loss=objective(g,yy,'O',torch.arange(8));grad=torch.autograd.grad(loss,g)[0];fd=float((objective(g+e*direction,yy,'O',torch.arange(8))-objective(g-e*direction,yy,'O',torch.arange(8)))/(2*e));ad=float((grad*direction).sum());assert abs(fd-ad)<=1e-7+1e-5*abs(ad)
 return dict(primal_dual_prediction_error=err,directional_gradient_error=abs(fd-ad),support_and_query_gradient_nonzero=bool(grad[:16].norm()>0 and grad[16:].norm()>0))
