"""Exact full-Gram coordinate preconditioning; scientific objective/tolerances unchanged."""
import time
import numpy as np
import torch
from scipy.optimize import minimize


def solve(fun,initial,hessian_proxy,cfg,checkpoint):
 # U=L^{-T}V is bijective. No clipping, rank truncation, jitter, or new regularizer.
 start=time.time();l=torch.linalg.cholesky(hessian_proxy);lnorm=float(l.abs().sum(1).max());inner_gtol=cfg['gtol']/max(lnorm,1e-300)
 original=np.asarray(initial,dtype=np.float64).reshape(l.shape[0],-1);v0=(l.T@torch.from_numpy(original)).numpy().ravel();history=[];last={};calls=0
 def convert(v):return torch.linalg.solve_triangular(l.T,torch.from_numpy(v.reshape(original.shape)),upper=True).numpy()
 def transformed(v):
  nonlocal calls
  u=convert(v);value,gu=fun(u.ravel());gv=torch.linalg.solve_triangular(l,torch.from_numpy(gu.reshape(original.shape)),upper=False).numpy().ravel();calls+=1;last.update(v=v.copy(),u=u.copy(),value=value,gu=gu.copy());return value,gv
 initial_loss,_=transformed(v0)
 def callback(v):
  assert np.array_equal(v,last['v']);history.append(float(last['value']));checkpoint(last['u'].ravel())
 res=minimize(transformed,v0,jac=True,method='L-BFGS-B',callback=callback,options=dict(maxiter=cfg['maxiter'],maxfun=cfg['maxfun'],maxls=cfg['maxls'],maxcor=cfg['maxcor'],ftol=cfg['ftol'],gtol=inner_gtol))
 u=convert(res.x);final,gu=fun(u.ravel());gmax=float(np.abs(gu).max());relative=abs(history[-2]-history[-1])/max(1.,abs(history[-2]),abs(history[-1])) if len(history)>=2 else None
 accepted=bool(res.success) and np.isfinite(u).all() and final<=initial_loss+1e-9 and (gmax<=cfg['gtol'] or (relative is not None and relative<=cfg['ftol']))
 audit=dict(success=bool(accepted),scipy_success=bool(res.success),status=int(res.status),message=str(res.message),iterations=int(res.nit),function_calls=int(res.nfev),extra_initial_evaluations=1,extra_final_evaluations=1,initial_objective=float(initial_loss),final_objective=float(final),gradient_max=gmax,transformed_gradient_max=float(np.abs(res.jac).max()),mapped_internal_gtol=inner_gtol,original_gtol=cfg['gtol'],ftol=cfg['ftol'],relative_accepted_objective_change=relative,solver='full_Gram_equivalent_coordinates_LBFGSB',wall_seconds=time.time()-start)
 return u.ravel(),audit
