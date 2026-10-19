"""Original SLSQP objective/constraints/tolerances, initialized from a verified feasible point."""
import time
import numpy as np
from scipy.optimize import minimize,linprog
from certificate import verify


def initial_from_direction(q,b,qs,bs,baselines,d):
    d=np.maximum(np.asarray(d),0.);assert d.sum()>0;d/=d.sum()
    nq=qs/np.maximum(baselines,1e-8)[:,None,None];nb=bs/np.maximum(baselines,1e-8)[:,None]
    slopes=2*nb@d;curves=np.einsum('i,gij,j->g',d,nq,d)
    assert slopes.max()<0 and 2*b@d<0
    limits=[1/float(d.max())]
    limits += [-s/c for s,c in zip(slopes,curves) if c>0]
    if d@q@d>0:limits.append(float(-(2*b@d)/(d@q@d)))
    a=d*(.5*min(limits))
    assert a.min()>=0 and a.max()<1
    assert (np.einsum('i,gij,j->g',a,nq,a)+2*nb@a).max()<0
    return a


def proof_or_initial(q,b,qs,bs,baselines,fixed):
    from fractions import Fraction as F
    assert not fixed,'Unexpected missing fold training class'
    start=time.time();m,k=bs.shape;t=2*bs/np.maximum(baselines,1e-8)[:,None]
    lp=linprog(np.r_[np.zeros(m),-1.],A_ub=np.c_[-t.T,np.ones(k)],b_ub=np.zeros(k),A_eq=np.array([np.r_[np.ones(m),0.]]),b_eq=[1.],bounds=[(0,None)]*m+[(None,None)],method='highs',options=dict(threads=4))
    diagnostic=dict(LP_calls=1,status=str(lp.message),success=bool(lp.success))
    assert lp.success,'Proof/direction search did not terminate successfully'
    diagnostic['gamma']=float(lp.x[-1]);proof=None;initial=None
    if lp.x[-1]>0:
        w=[F(format(max(0.,float(x)),'.8f')) for x in lp.x[:-1]];total=sum(w)
        assert total>0
        proof=verify(qs,bs,baselines,[x/total for x in w])
    else:
        # Dual variables of the same certificate LP give a tangent direction; no second LP.
        initial=initial_from_direction(q,b,qs,bs,baselines,-lp.ineqlin.marginals)
        diagnostic['initialization']='strictly_feasible_point_from_same_LP_dual'
        diagnostic['initial_alpha_max']=float(initial.max())
    diagnostic['wall_seconds']=time.time()-start
    return proof,initial,diagnostic


def constrained(q,b,qs,bs,baselines,fixed,initial,maxiter=1000,ftol=1e-10,tol=1e-7):
    start=time.time();k=len(b);nq=qs/np.maximum(baselines,1e-8)[:,None,None];nb=bs/np.maximum(baselines,1e-8)[:,None]
    scale=max(float(np.abs(q).max()),float(np.abs(b).max()),1e-12)
    def change(a):return float(a@q@a+2*b@a)
    def con(a):return -(np.einsum('i,gij,j->g',a,nq,a)+2*nb@a)
    initial=np.asarray(initial,dtype=np.float64)
    assert np.isfinite(initial).all() and initial.min()>=0 and initial.max()<=1 and con(initial).min()>0
    result=minimize(lambda a:change(a)/scale,initial,jac=lambda a:(2*q@a+2*b)/scale,bounds=[(0.,0.) if i in fixed else (0.,1.) for i in range(k)],constraints=[dict(type='ineq',fun=con,jac=lambda a:-(2*np.einsum('gij,j->gi',nq,a)+2*nb))],method='SLSQP',options=dict(maxiter=maxiter,ftol=ftol,disp=False))
    a=np.array(result.x);violation=max(0.,float(-con(a).min()))
    audit=dict(success=bool(result.success),status=int(result.status),message=str(result.message),solver='SLSQP_feasible_start',iterations=int(result.nit),evaluations=int(result.nfev),max_normalized_violation=violation,objective_delta=change(a),alpha_min=float(a.min()),alpha_max=float(a.max()),wall_seconds=time.time()-start,objective_scale=scale,settings=dict(maxiter=maxiter,ftol=ftol,acceptance_normalized_violation=tol),initial_max_normalized_constraint_delta=float(-con(initial).min()),initial_alpha_max=float(initial.max()),initial_objective_delta=change(initial))
    return a,audit
