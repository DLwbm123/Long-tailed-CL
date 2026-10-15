"""Synthetic SPD counterexamples and constrained-envelope checks, without torch or data IO."""
import itertools
import json
import time
import numpy as np

VERSION='pointwise_training_margin_CPU_design_v1'


def solve(H,b,A,t):
    """Tiny exhaustive active-set oracle, deliberately unsuitable for production scale."""
    if not len(t):
        w=np.linalg.solve(H,b);return w,np.empty(0),.5*w@H@w-b@w
    best=None
    for bits in itertools.product((False,True),repeat=len(t)):
        take=np.flatnonzero(bits);w=np.linalg.solve(H,b);lam=np.zeros(len(t))
        if len(take):
            active=A[take];U=np.linalg.solve(H,active.T);S=active@U
            dual=np.linalg.lstsq(S,t[take]-active@w,rcond=None)[0]
            if np.max(np.abs(S@dual-(t[take]-active@w)))>1e-9:continue
            w=w+U@dual;lam[take]=dual
        if np.min(A@w-t)<-1e-9 or np.min(lam)<-1e-9:continue
        value=.5*w@H@w-b@w
        if best is None or value<best[0]:best=(value,w,lam)
    if best is None:raise ValueError('No feasible active set; retain failure')
    value,w,lam=best
    assert np.linalg.norm(H@w-b-A.T@lam)<1e-8 and np.max(np.abs(lam*(A@w-t)))<1e-8
    return w,lam,value


def system(old,x):
    x=np.atleast_2d(x);G=.5*np.outer(old,old)+.5*x.T@x/len(x)+.001*np.eye(2)
    B=.5*np.column_stack([old,x.mean(0)])
    return np.kron(np.eye(2),G),B.reshape(-1,order='F')


def constraints(x):return np.hstack([-np.atleast_2d(x),np.atleast_2d(x)])


def run():
    start=time.monotonic();old=np.array([.5,.02]);M=np.diag([2.8,1.]);moved=old@M;x=np.array([1.,0.])
    H0,b0=system(old,x);ref=np.linalg.solve(H0,b0);H,b=system(moved,x);proposal=np.linalg.solve(H,b)
    A=constraints(x);target=A@ref;before=float((A@ref)[0]);after=float((A@proposal)[0])
    assert before>0 and after<0
    # The moment formula is exact, although the new class switches to wrong.
    Q=np.outer(old,old);assert np.allclose(M.T@Q@M,np.outer(moved,moved))
    W,lam,val=solve(H,b,A,target);assert lam[0]>0 and np.min(A@W-target)>-1e-9
    centered=lambda w:(w.reshape(2,2,order='F')-w.reshape(2,2,order='F').mean(1,keepdims=True))
    functional_delta=float(np.linalg.norm(centered(W)-centered(ref)))
    assert functional_delta>1e-3
    epsilon=1e-6
    def blocks(theta):
        z=np.array([1.,theta]);hh,bb=system(moved,z);aa=constraints(z);return hh,bb,aa,aa@ref
    def optimized(theta):
        hh,bb,aa,tt=blocks(theta);return solve(hh,bb,aa,tt)[2]
    hp,bp,ap,tp=blocks(epsilon);hm,bm,am,tm=blocks(-epsilon)
    optimized_derivative=(optimized(epsilon)-optimized(-epsilon))/(2*epsilon)
    naive=(.5*W@(hp-hm)@W-(bp-bm)@W)/(2*epsilon)
    lagrangian=naive-float(lam@((ap-am)@W-(tp-tm)))/(2*epsilon)
    assert abs(optimized_derivative-lagrangian)<1e-6 and abs(optimized_derivative-naive)>1e-3
    # Mean-only protection misses one of two observed points.
    points=np.array([[1.,1.],[1.,-1.]])
    reference=np.array([1.,0.]);changed=np.array([1.,.2])
    assert np.allclose(points.mean(0)@reference,points.mean(0)@changed) and np.min(points@(changed-reference))<0
    xx=np.array([[1.,.03],[1.,-.03]]);h0,b0=system(old,xx);reference2=np.linalg.solve(h0,b0);hh,bb=system(moved,xx);aa=constraints(xx)
    constrained,dual,_=solve(hh,bb,aa,aa@reference2)
    two_delta=float(np.linalg.norm(centered(constrained)-centered(reference2)))
    # Disable is exact and class-common gauge cannot create a functional effect.
    off=np.linalg.solve(H,b);empty=np.empty((0,4));disabled,_,_=solve(H,b,empty,np.empty(0))
    assert np.array_equal(disabled,off)
    gauge=np.tile(np.array([.2,-.1]),2);assert np.allclose(A@gauge,0)
    P=np.eye(4)[[2,3,0,1]];permuted,_,_=solve(P@H@P.T,P@b,A@P.T,target);assert np.allclose(P.T@permuted,W)
    T=np.array([[0.,-1.],[1.,0.]]);C=np.kron(np.eye(2),T);rotated,_,_=solve(C@H@C.T,C@b,A@C.T,target);assert np.allclose(C.T@rotated,W)
    return dict(status='CPU_DESIGN_PASS_NOT_PRODUCTION_READY',version=VERSION,synthetic_only=True,GPU_processes=0,medical_images_read=0,
        medical_checkpoint_reads=0,val_reads=0,test_accessed=False,optimizer_updates=0,old_moment_pushforward_exact=True,
        unchanged_new_statistics_margin_before=before,unchanged_new_statistics_margin_after=after,
        training_point_guard_margin=float((A@W)[0]),guard_active_multiplier=float(lam[0]),noncommon_head_delta= functional_delta,
        objective_value_derivative=float(optimized_derivative),naive_detached_head_derivative=float(naive),Lagrangian_derivative=float(lagrangian),
        Lagrangian_finite_difference_error=float(abs(optimized_derivative-lagrangian)),naive_envelope_error=float(abs(optimized_derivative-naive)),
        mean_guard_counterexample=True,two_point_noncommon_delta=two_delta,two_point_primal_slack_min=float(np.min(aa@constrained-aa@reference2)),
        class_permutation_and_orthogonal_coordinates_checked=True,public_logit_gauge_excluded=True,disabled_exact_checked=True,
        full_native_operator_checked=False,highdimensional_CPU_checked=False,production_integrated=False,GPU_ready=False,
        independent_old_risk_identified=False,new_val_accuracy_guaranteed=False,CPU_pipeline_seconds=time.monotonic()-start)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
