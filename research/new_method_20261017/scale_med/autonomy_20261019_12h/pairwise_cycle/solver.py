"""Known OvR baselines, fixed objective and accepted-point certificates."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.sparse.linalg import LinearOperator,cg

def binary_weights(y,q,k,variant):
 pairs=[(i,j) for i in range(k) for j in range(i+1,k)]
 hot=np.column_stack([(y==i).astype(float) for i,j in pairs])
 weight=np.column_stack([q*((y==i)|(y==j))/2 for i,j in pairs])
 assert np.allclose(weight.sum(0),1,atol=1e-12)
 return hot,weight

def readouts(m,k):
 pairs=[(i,j) for i in range(k) for j in range(i+1,k)]
 potential=np.zeros((len(m),k));soft=potential.copy();hard=potential.copy()
 for c,(i,j) in enumerate(pairs):
  v=m[:,c];p=expit(v);potential[:,i]+=v/k;potential[:,j]-=v/k
  soft[:,i]+=p;soft[:,j]+=1-p;hard[:,i]+=(v>=0);hard[:,j]+=(v<0)
 return dict(Borda=soft,Hodge=potential,Vote=hard)

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
 return w,dict(status=status,initial_objective=initial,final_objective=f,mean_binary_objective=f/shape[1],gradient_max_abs=float(abs(g).max()),numeric_objective='sum of all pair objectives; same minimizers as their mean',lambda_value=lam,variant=variant,lbfgs_iterations=int(opt.nit),lbfgs_message=str(opt.message),value_gradient_calls=calls,HVP_calls=HVP,newton_log=newton,accepted_trace=trace,wall_seconds=time.monotonic()-tick)

def selfcheck():
 rng=np.random.default_rng(74156);x=np.c_[rng.normal(size=(30,5)),np.ones(30)];y=np.arange(30)%3;q=np.ones(30)/10
 hot,w=binary_weights(y,q,3,'PAIR');fg,hv=functions(x,hot,w,.01);beta=rng.normal(size=(6,3))*.1;v=rng.normal(size=beta.shape);eps=1e-5;f,g=fg(beta)
 assert abs((fg(beta+eps*v)[0]-fg(beta-eps*v)[0])/(2*eps)-(g*v).sum())<1e-7
 assert np.max(abs((fg(beta+eps*v)[1]-fg(beta-eps*v)[1])/(2*eps)-hv(beta,v)))<1e-7
 W,info=fit(x,y,q,3,.01,'PAIR',lambda:None);assert info['status']=='CONVERGED'
 for k in [4,5,6]:
  a=rng.normal(size=(100,k));m=np.column_stack([a[:,i]-a[:,j] for i in range(k) for j in range(i+1,k)]);r=readouts(m,k)
  assert np.array_equal(r['Borda'].argmax(1),a.argmax(1)) and np.allclose(r['Hodge'],a-a.mean(1,keepdims=True))
 # Explicit cyclic score field where fixed readouts disagree.
 m=rng.normal(size=(1000,6));r=readouts(m,4);assert np.any(r['Borda'].argmax(1)!=r['Hodge'].argmax(1))
 return dict(state='PASSED',analytic_gradient_HVP=True,pair_mass=True,raw_gradient=info['gradient_max_abs'],potential_Borda_equivalence=True,cyclic_field_readout_can_differ=True,synthetic_only=True)
