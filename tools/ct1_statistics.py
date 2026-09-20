"""Locked CT1 virtual moments, diagonal transport and constrained risk readout."""
import time
import numpy as np


def joint(raw):
    z=np.asarray(raw,dtype=np.float64).reshape(len(raw),-1)
    norm=np.linalg.norm(z,axis=1)
    assert np.isfinite(z).all() and np.all(norm>0),'BLOCKED_FEATURE_NORM'
    return z/norm[:,None]


def empty(d):
    return dict(S=np.zeros((d,d)),mu=np.empty((d,0)),v=np.empty((d,0)),
                e=np.empty((d,0)),n=np.empty(0,dtype=np.int64),arrival=[],space_version=0)


def append(state,z,y,classes,task):
    s={k:(v.copy() if isinstance(v,np.ndarray) else list(v) if isinstance(v,list) else v)
       for k,v in state.items()}
    assert list(classes)==list(range(s['mu'].shape[1],s['mu'].shape[1]+len(classes)))
    for c in classes:
        x=np.asarray(z[y==c],dtype=np.float64);assert len(x)>0
        mu=x.mean(0);v=np.mean((x-mu)**2,axis=0)
        s['S']+=x.T@x/len(x)
        for name,a in [('mu',mu),('v',v),('e',np.zeros_like(mu))]:
            s[name]=np.column_stack([s[name],a])
        s['n']=np.append(s['n'],len(x));s['arrival'].append(task)
    s['space_version']=task
    return s


def fit_map(before,after,y):
    x=np.asarray(before,dtype=np.float64);z=np.asarray(after,dtype=np.float64)
    assert x.shape==z.shape and np.isfinite(x).all() and np.isfinite(z).all()
    classes=np.unique(y);w=np.zeros(len(y))
    for c in classes:w[y==c]=1/(len(classes)*np.sum(y==c))
    mx=w@x;my=w@z;dx=x-mx;dy=z-my
    vx=w@(dx*dx);cov=w@(dx*dy);tau=.001*max(float(vx.mean()),1e-12)
    unclipped=(cov+tau)/(vx+tau);a=np.clip(unclipped,.5,2.);b=my-a*mx
    r=w@((z-x*a-b)**2)
    return a,b,r,dict(a_min=float(a.min()),a_max=float(a.max()),a_mean=float(a.mean()),
                       b_norm=float(np.linalg.norm(b)),clip_fraction=float(np.mean(a!=unclipped)),
                       residual_mean=float(r.mean()),residual_max=float(r.max()),
                       current_class_counts={str(int(c)):int(np.sum(y==c)) for c in classes})


def transport(state,a,b,r,version):
    s={k:(v.copy() if isinstance(v,np.ndarray) else list(v) if isinstance(v,list) else v)
       for k,v in state.items()}
    m=s['mu'].shape[1];du=a*s['mu'].sum(1)
    s['S']=a[:,None]*s['S']*a[None,:]+np.outer(du,b)+np.outer(b,du)+m*np.outer(b,b)
    s['mu']=a[:,None]*s['mu']+b[:,None]
    s['v']=a[:,None]**2*s['v'];s['e']=a[:,None]**2*s['e']+r[:,None]
    s['space_version']=version
    return s


def ridge(state):
    c=len(state['n']);assert c>0
    G=state['S']/c;G=(G+G.T)/2;R=state['mu']/c
    A=G+.001*np.eye(len(G));W=np.linalg.solve(A,R)
    residual=float(np.linalg.norm(A@W-R)/max(np.linalg.norm(R),1e-30))
    assert residual<1e-10 and np.isfinite(W).all(),'BLOCKED_RIDGE_SOLVE'
    ev=np.linalg.eigvalsh(G)
    return W,dict(residual=residual,trace=float(np.trace(G)),
                  Gram_condition=None if ev[0]<=np.finfo(float).eps*max(1.,ev[-1]) else float(ev[-1]/ev[0]),
                  regularized_condition=float((ev[-1]+.001)/(ev[0]+.001)),
                  mean_norms=np.linalg.norm(state['mu'],axis=0).tolist(),
                  diagonal_variance_min=float(state['v'].min()),
                  error_proxy_mean=float(state['e'].mean()))


def psd_factor(q):
    q=(q+q.T)/2;values,vectors=np.linalg.eigh(q)
    assert values.min()>=-1e-10*max(1.,abs(values).max()),'BLOCKED_RISK_NONPSD'
    # Eigenvalues only clipped within roundoff; no adaptive jitter.
    return np.sqrt(np.maximum(values,0))[:,None]*vectors.T


def risk(state,W0,eta=.1):
    import cvxpy as cp
    import clarabel
    c=len(state['n']);d=len(W0);assert W0.shape==(d,c)
    v=state['v'];shrink=10/(state['n']+10)
    vt=(1-shrink)*v+shrink*v.mean(1,keepdims=True)+1e-8/d
    u=vt/np.maximum(state['n'],1)+state['e']
    A=state['S']/c+.001*np.eye(d);H=W0.T@A@W0
    B=cp.Variable((c,c));xi=cp.Variable(c,nonneg=True);I=np.eye(c)
    objective=cp.sum_squares(psd_factor(H)@(B-I))+(eta/c)*cp.sum(xi)+1e-10*cp.sum_squares(B-I)
    factors=[(psd_factor(W0.T@(vt[:,j,None]*W0)),psd_factor(W0.T@(u[:,j,None]*W0))) for j in range(c)]
    constraints=[]
    for j in range(c):
        mean=state['mu'][:,j]@W0;fv,fu=factors[j]
        for k in range(c):
            if j==k:continue
            b=B[:,j]-B[:,k]
            constraints.append(.1+cp.norm(fv@b)+cp.norm(fu@b)-mean@b<=xi[j])
    problem=cp.Problem(cp.Minimize(objective),constraints);assert problem.is_dcp()
    started=time.monotonic()
    problem.solve(solver='CLARABEL',max_iter=200,time_limit=120.,
                  tol_gap_abs=1e-8,tol_gap_rel=1e-8,tol_feas=1e-8,verbose=False)
    record=dict(status=problem.status,seconds=time.monotonic()-started,
                cvxpy=cp.__version__,clarabel=clarabel.__version__,iterations=problem.solver_stats.num_iters,
                eta=eta,v_tilde_mean_by_class=vt.mean(0).tolist(),u_mean_by_class=u.mean(0).tolist(),
                objective=float(problem.value) if problem.value is not None and np.isfinite(problem.value) else None)
    if problem.status!='optimal':return None,record
    bv=np.asarray(B.value);xv=np.asarray(xi.value);W=W0@bv
    violations=[];slacks=[]
    for j in range(c):
        for k in range(c):
            if j==k:continue
            q=W[:,j]-W[:,k];lhs=float(state['mu'][:,j]@q)
            rhs=.1+np.sqrt(np.sum(vt[:,j]*q*q))+np.sqrt(np.sum(u[:,j]*q*q))-xv[j]
            violations.append(max(0.,rhs-lhs)/max(1.,abs(lhs),abs(rhs)))
            slacks.append(lhs-rhs)
    violation=max(max(violations),max(0.,float(-xv.min())))
    record.update(normalized_max_constraint_violation=violation,xi=xv.tolist(),
                  active_constraints=int(np.sum(np.asarray(slacks)<1e-6)))
    # Retrieve the native Clarabel termination residuals without substituting solver tolerances.
    solver=problem._solver_cache.get('CLARABEL')
    if solver is not None:
        info=solver.get_info();record.update(primal_residual=float(info.res_primal),dual_residual=float(info.res_dual))
    if violation>1e-6 or not np.isfinite(W).all():
        record['status']='BLOCKED_RISK_CONSTRAINT';return None,record
    return dict(W=W,B=bv,xi=xv),record


def selfcheck(include_solver=True):
    rng=np.random.default_rng(45001);d=9;xs=[rng.normal(size=(n,d)) for n in (7,4,9)]
    X=np.concatenate(xs);y=np.repeat(np.arange(3),[len(x) for x in xs])
    s=append(empty(d),X,y,range(3),1);W,audit=ridge(s)
    weights=np.concatenate([np.full(len(x),1/len(x)/3) for x in xs])
    G=X.T@(weights[:,None]*X);R=X.T@(weights[:,None]*np.eye(3)[y])
    np.testing.assert_allclose(W,np.linalg.solve(G+.001*np.eye(d),R),atol=1e-11)
    a=rng.uniform(.6,1.8,d);b=rng.normal(size=d);r=np.zeros(d)
    transported=transport(s,a,b,r,2);direct=append(empty(d),X*a+b,y,range(3),2)
    for k in ('S','mu','v'):np.testing.assert_allclose(transported[k],direct[k],atol=1e-12)
    aa,bb,rr,_=fit_map(X,X,y)
    np.testing.assert_allclose(aa,1,atol=1e-12);np.testing.assert_allclose(bb,0,atol=1e-12)
    np.testing.assert_allclose(rr,0,atol=1e-12)
    identity=transport(s,aa,bb,rr,2)
    for k in ('S','mu','v'):np.testing.assert_allclose(identity[k],s[k],atol=1e-12)
    B=rng.normal(size=(3,3));D=W@(B-np.eye(3));A=G+.001*np.eye(d)
    loss=lambda Q: np.sum(weights[:,None]*(X@Q-np.eye(3)[y])**2)+.001*np.sum(Q*Q)
    difference=loss(W@B)-loss(W);formula=float(np.trace(D.T@A@D))
    np.testing.assert_allclose(difference,formula,atol=1e-11)
    record=dict(status='PASS',aggregate_affine_exact=True,identity_exact=True,
                class_balanced_objective_exact=True,cross_Gram_retained=True,
                subspace_objective_difference_exact=True)
    if include_solver:
        zero,zrecord=risk(s,W,eta=0);assert zero is not None,zrecord
        np.testing.assert_allclose(zero['W'],W,atol=1e-5,rtol=1e-5)
        small,small_record=risk(s,W);assert small is not None,small_record
        tx=rng.normal(size=(230,48));ty=np.repeat(np.arange(23),10)
        big=append(empty(48),tx,ty,range(23),1);bw,_=ridge(big)
        br,brecord=risk(big,bw);assert br is not None,brecord
        record.update(eta_zero_recovers_W0=zrecord,toy_risk=small_record,max23_toy=brecord)
    return record
