import hashlib
from collections import Counter
import numpy as np
from scipy.optimize import minimize


def fold_id(group):
    return int.from_bytes(hashlib.sha256(('risk-shrink-v1:74002:'+str(group)).encode()).digest()[:8],'little')%3


def weights(groups):
    counts=Counter(groups)
    return np.array([1/(len(counts)*counts[g]) for g in groups],dtype=np.float64)


def center(x):return x-x.mean(1,keepdims=True)


def problem(c,d,labels,w):
    k=c.shape[1];y=np.eye(k)[labels];e=c-center(y);mass=w.sum()
    h=np.eye(k)-np.ones((k,k))/k
    q=((d*w[:,None]).T@d)*h/(mass*k)
    b=((e*d)*w[:,None]).sum(0)/(mass*k)
    baseline=float((e*e*w[:,None]).sum()/(mass*k))
    return (q+q.T)/2,b,baseline


def change(q,b,a):return float(a@q@a+2*b@a)


def uniform(q,b):
    den=float(q.sum())
    return float(np.clip(-b.sum()/den,0,1)) if den>1e-14 else 0.


def constrained(q,b,qs,bs,baselines,fixed,maxiter=1000,ftol=1e-10,tol=1e-7):
    k=len(b);scales=np.maximum(baselines,1e-8)
    qs=qs/scales[:,None,None];bs=bs/scales[:,None]
    objective_scale=max(float(np.abs(q).max()),float(np.abs(b).max()),1e-12)
    def fun(a):return change(q,b,a)/objective_scale
    def jac(a):return (2*q@a+2*b)/objective_scale
    def cons(a):return -(np.einsum('i,gij,j->g',a,qs,a)+2*bs@a)
    def cjac(a):return -(2*np.einsum('gij,j->gi',qs,a)+2*bs)
    result=minimize(fun,np.zeros(k),jac=jac,bounds=[(0.,0.) if i in fixed else (0.,1.) for i in range(k)],constraints=[dict(type='ineq',fun=cons,jac=cjac)],method='SLSQP',options=dict(maxiter=maxiter,ftol=ftol,disp=False))
    a=np.asarray(result.x);violation=max(0.,float(-cons(a).min()))
    audit=dict(success=bool(result.success),status=int(result.status),message=str(result.message),iterations=int(result.nit),evaluations=int(result.nfev),max_normalized_violation=violation,objective_delta=change(q,b,a),alpha_min=float(a.min()),alpha_max=float(a.max()))
    return a,audit
