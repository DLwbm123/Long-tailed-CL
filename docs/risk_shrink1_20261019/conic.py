"""Equivalent SOCP form of the original bounded convex quadratic risk constraints."""
import time
import numpy as np
from scipy import sparse
import clarabel


def factor(q):
    values, vectors = np.linalg.eigh((q+q.T)/2)
    assert values.min() >= -1e-12, 'Risk quadratic is not numerically PSD'
    # Only roundoff-negative eigenvalues are zeroed for a PSD square root.
    l = np.sqrt(np.maximum(values, 0))[:, None]*vectors.T
    error = float(np.max(np.abs(l.T@l-q)))
    assert error <= 1e-12
    return l, error


def constrained(q,b,qs,bs,baselines,fixed,maxiter=1000,ftol=1e-10,tol=1e-7):
    start = time.time(); k = len(b)
    scales = np.maximum(baselines,1e-8)
    nq = qs/scales[:,None,None]; nb = bs/scales[:,None]
    objective_scale = max(float(np.abs(q).max()),float(np.abs(b).max()),1e-12)
    matrices = [-np.eye(k),np.eye(k)]; rhs = [np.zeros(k),np.array([0. if i in fixed else 1. for i in range(k)])]
    cones = [clarabel.NonnegativeConeT(k),clarabel.NonnegativeConeT(k)]
    max_error = 0.
    for qq,bb in zip(nq,nb):
        l,error = factor(qq); max_error=max(max_error,error)
        # s=(1+t,2La,1-t), t=-2b'a: SOC iff a'Qa+2b'a<=0.
        matrices.append(np.vstack([2*bb,-2*l,-2*bb]))
        rhs.append(np.r_[1.,np.zeros(k),1.]); cones.append(clarabel.SecondOrderConeT(k+2))
    settings = clarabel.DefaultSettings(); settings.verbose=False; settings.max_iter=maxiter
    settings.max_threads=4
    settings.tol_gap_abs=ftol; settings.tol_gap_rel=ftol; settings.tol_feas=ftol
    settings.tol_infeas_abs=ftol; settings.tol_infeas_rel=ftol
    solver=clarabel.DefaultSolver(sparse.triu(sparse.csc_matrix(2*q/objective_scale)).tocsc(),2*b/objective_scale,sparse.csc_matrix(np.vstack(matrices)),np.concatenate(rhs),cones,settings)
    result=solver.solve(); raw=np.array(result.x)
    # Numerical bound polishing only; original risk constraints are rechecked after it.
    assert raw.min()>=-1e-10 and raw.max()<=1+1e-10
    a=np.clip(raw,0.,1.)
    for i in fixed: assert abs(raw[i])<=1e-10; a[i]=0.
    violation=max(0.,float(np.max(np.einsum('i,gij,j->g',a,nq,a)+2*nb@a)))
    delta=float(a@q@a+2*b@a)
    audit=dict(success=str(result.status)=='Solved',status=str(result.status),message=str(result.status),solver='Clarabel',version=clarabel.__version__,iterations=int(result.iterations),evaluations=None,max_normalized_violation=violation,objective_delta=delta,alpha_min=float(a.min()),alpha_max=float(a.max()),wall_seconds=time.time()-start,objective_scale=objective_scale,PSD_factor_max_error=max_error,primal_residual=float(result.r_prim),dual_residual=float(result.r_dual),absolute_gap=float(abs(result.obj_val-result.obj_val_dual)),relative_gap=float(abs(result.obj_val-result.obj_val_dual)/max(1.,abs(result.obj_val),abs(result.obj_val_dual))),raw_bound_polish_max=float(np.max(np.abs(a-raw))),settings=dict(max_iter=maxiter,tol_gap_abs=ftol,tol_gap_rel=ftol,tol_feas=ftol,acceptance_normalized_violation=tol))
    return a,audit
