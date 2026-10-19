"""Known OvR baselines, fixed objective and accepted-point certificates."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.sparse.linalg import LinearOperator,cg

def binary_weights(y,q,k,variant):
 hot=np.eye(k)[y]
 w=np.broadcast_to(q[:,None]/k,hot.shape).copy() if variant=='S' else q[:,None]*(hot/2+(1-hot)/(2*(k-1)))
 assert np.allclose(w.sum(0),1,atol=1e-12)
 return hot,w

def functions(x,hot,weight,lam):
 def fg(w):
  s=x@w;p=expit(s);f=float((weight*(np.logaddexp(0,s)-hot*s)).sum()+lam/2*(w[:-1]**2).sum());g=x.T@(weight*(p-hot));g[:-1]+=lam*w[:-1];return f,g
 def hv(w,v):
  p=expit(x@w);h=x.T@(weight*p*(1-p)*(x@v));h[:-1]+=lam*v[:-1];return h
 return fg,hv

def fit(x,y,q,k,lam,variant,check):
 tick=time.monotonic();x=np.ascontiguousarray(x);hot,weight=binary_weights(y,q,k,variant);fg,hv=functions(x,hot,weight,lam)
 diagonal=.25*(x*x).T@weight;diagonal[:-1]+=lam;scale=np.sqrt(np.maximum(diagonal,1e-12));shape=scale.shape;calls=0;HVP=0;trace=[]
 def objective(theta):
  nonlocal calls
  check();calls+=1;f,g=fg(theta.reshape(shape)/scale);assert np.isfinite(f) and np.isfinite(g).all();return f,(g/scale).ravel()
 def callback(theta):
  f,g=fg(theta.reshape(shape)/scale);trace.append([f,float(abs(g).max())]);check()
 initial=fg(np.zeros(shape))[0];opt=minimize(objective,np.zeros(scale.size),jac=True,method='L-BFGS-B',callback=callback,options=dict(maxiter=5000,maxfun=20000,maxcor=20,maxls=40,ftol=0,gtol=1e-9))
 w=opt.x.reshape(shape)/scale;f,g=fg(w);newton=[]
 for step in range(10):
  if abs(g).max()<=1e-7:break
  gmax=float(abs(g).max())
  def product(v):
   nonlocal HVP
   HVP+=1;check();return (hv(w,v.reshape(shape)/scale)/scale).ravel()
  d,info=cg(LinearOperator((scale.size,scale.size),matvec=product,dtype=np.float64),(-g/scale).ravel(),rtol=.05,atol=0,maxiter=1000);direction=d.reshape(shape)/scale;res=float(abs(hv(w,direction)+g).max());gd=float((g*direction).sum());log=dict(step=step+1,CG_info=int(info),linear_residual_max=res,raw_gradient_before=gmax)
  if info!=0 or res>.1*gmax or gd>=0:newton.append(log);break
  found=False
  for halves in range(30):
   a=2.**(-halves);candidate=w+a*direction;ff,gg=fg(candidate);calls+=1;check()
   if np.isfinite(ff) and ff<=f+1e-4*a*gd:
    w=candidate;f,g=ff,gg;log.update(step_size=a,raw_gradient_after=float(abs(g).max()));found=True;break
  newton.append(log)
  if not found:break
 status='CONVERGED' if np.isfinite(f) and np.isfinite(w).all() and abs(g).max()<=1e-7 and f<=initial+1e-10 else 'UNRESOLVED'
 return w,dict(status=status,initial_objective=initial,final_objective=f,mean_binary_objective=f/k,gradient_max_abs=float(abs(g).max()),numeric_objective='sum of k binary objectives; equivalent minimizer to their mean',lambda_value=lam,variant=variant,lbfgs_iterations=int(opt.nit),lbfgs_message=str(opt.message),value_gradient_calls=calls,HVP_calls=HVP,newton_log=newton,accepted_trace=trace,wall_seconds=time.monotonic()-tick)

def selfcheck():
 rng=np.random.default_rng(74);x=np.c_[rng.normal(size=(30,5)),np.ones(30)];y=np.arange(30)%3;q=np.ones(30)/10;checks={}
 for variant in ['S','B']:
  hot,w=binary_weights(y,q,3,variant);fg,hv=functions(x,hot,w,.01);beta=rng.normal(size=(6,3))*.1;v=rng.normal(size=beta.shape);f,g=fg(beta);eps=1e-5
  assert abs((fg(beta+eps*v)[0]-fg(beta-eps*v)[0])/(2*eps)-(g*v).sum())<1e-7
  assert np.max(abs((fg(beta+eps*v)[1]-fg(beta-eps*v)[1])/(2*eps)-hv(beta,v)))<1e-7
  b,info=fit(x,y,q,3,.01,variant,lambda:None);assert info['status']=='CONVERGED';checks[variant]=info['gradient_max_abs']
 return dict(state='PASSED',analytic_gradient_HVP=True,binary_mass=True,original_W_gradient_certificates=checks,synthetic_only=True)
