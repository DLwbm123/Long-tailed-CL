"""Exact zero-only acceptance for the unchanged saved quadratic constraints."""
import time
from fractions import Fraction as F
import numpy as np


def verify(qs,bs,baselines,weights):
    start=time.time();m,k=bs.shape;w=[F(x) for x in weights]
    assert len(w)==m and all(x>=0 for x in w) and sum(w)==1
    support=[i for i,x in enumerate(w) if x]
    factors={i:w[i]/F.from_float(max(float(baselines[i]),1e-8)) for i in support}
    v=[sum(2*factors[i]*F.from_float(float(bs[i,j])) for i in support) for j in range(k)]
    gamma=min(v);assert gamma>0
    q=[[sum(factors[i]*F.from_float(float(qs[i,j,l])) for i in support) for l in range(k)] for j in range(k)]
    assert all(q[i][j]==q[j][i] for i in range(k) for j in range(k))
    l=[[F(int(i==j)) for j in range(k)] for i in range(k)];diagonal=[]
    for j in range(k):
        pivot=q[j][j]-sum(l[j][h]**2*diagonal[h] for h in range(j))
        assert pivot>0;diagonal.append(pivot)
        for i in range(j+1,k):l[i][j]=(q[i][j]-sum(l[i][h]*l[j][h]*diagonal[h] for h in range(j)))/pivot
    assert all(sum(l[i][h]*diagonal[h]*l[j][h] for h in range(min(i,j)+1))==q[i][j] for i in range(k) for j in range(k))
    return dict(status='EXACT_ZERO_ONLY',weights_exact=[str(x) for x in w],support=support,gradient_min_exact=str(gamma),gradient_min_float=float(gamma),weighted_quadratic_positive_definite_exact=True,exact_LDL_pivots_float=[float(x) for x in diagonal],wall_seconds=time.time()-start)


def search(qs,bs,baselines,fixed):
    from scipy.optimize import linprog
    start=time.time();m,k=bs.shape
    if fixed:return None,dict(status='fixed_outputs_use_original_solver',LP_calls=0,wall_seconds=time.time()-start)
    t=2*bs/np.maximum(baselines,1e-8)[:,None]
    lp=linprog(np.r_[np.zeros(m),-1.],A_ub=np.c_[-t.T,np.ones(k)],b_ub=np.zeros(k),A_eq=np.array([np.r_[np.ones(m),0.]]),b_eq=[1.],bounds=[(0,None)]*m+[(None,None)],method='highs',options=dict(threads=4))
    diagnostic=dict(status=str(lp.message),LP_calls=1,wall_seconds=time.time()-start)
    if not lp.success or lp.x[-1]<=0:return None,diagnostic
    # Auxiliary witness rounding never changes a problem coefficient.
    w=[F(format(max(0.,float(x)),'.8f')) for x in lp.x[:-1]];total=sum(w)
    if not total:return None,diagnostic
    try:proof=verify(qs,bs,baselines,[x/total for x in w])
    except AssertionError:
        diagnostic['certificate_rejected']=True;return None,diagnostic
    diagnostic['wall_seconds']=time.time()-start
    return proof,diagnostic


def acceptance(proof,k,tol):
    assert proof['status']=='EXACT_ZERO_ONLY'
    return np.zeros(k),dict(success=True,status='CERTIFIED_ZERO',message='Exact rational positive-combination and LDL certificate',solver='ExactZeroCertificate',iterations=0,evaluations=0,max_normalized_violation=0.,objective_delta=0.,alpha_min=0.,alpha_max=0.,wall_seconds=proof['wall_seconds'],acceptance_normalized_violation=tol,certificate=proof)
