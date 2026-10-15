"""CPU design: a detached fixed-frame head projection with every constraint present."""
import math
import time
import numpy as np
import torch
from scipy.optimize import minimize
import native_pointwise_guard_design as native


def validate_snapshot(s):
    if (s.get('event') not in ('epoch_init','boundary_final') or s.get('scope')!='fit'
        or s.get('encoder_eval') is not True or s.get('frame')!=s.get('support_frame')
        or s.get('encoder_version')!=s.get('support_encoder_version')
        or not s.get('frame') or not s.get('encoder_version')):
        raise ValueError('Stale/non-fit/non-snapshot guard request')


def actions(x,y,k):
    scale=1/math.sqrt(len(x));mask=~torch.nn.functional.one_hot(y,k).bool()
    def forward(w):return (native.contrasts(x,y,w)*scale)[mask]
    def adjoint(lam):
        z=x.new_zeros((len(x),k));z[mask]=lam
        v=-z;v.scatter_add_(1,y[:,None],z.sum(1,keepdim=True))
        return (x.T@v)*scale
    return forward,adjoint,scale,mask


def solve(op,rhs,inverse,unconstrained,reference,x,y,old_count,snapshot,enabled=True,
          maxiter=2000,time_limit=840.):
    validate_snapshot(snapshot)
    if x.device.type!='cpu' or x.dtype!=torch.float64 or x.ndim!=2 or len(x)==0 or y.dtype!=torch.long or y.shape!=(len(x),):
        raise ValueError('CPU float64 current-fit inputs required')
    k=reference.shape[1]
    if reference.shape!=unconstrained.shape or x.shape[1]!=len(reference) or not 0<=old_count<k:
        raise ValueError('Guard head dimensions differ')
    if bool(((y<old_count)|(y>=k)).any()) or not all(torch.isfinite(v).all() for v in (x,reference,unconstrained)):
        raise ValueError('Invalid guard labels or nonfinite input')
    if not enabled:return unconstrained.clone(),dict(status='DISABLED_EXACT',new_head_solves=0),None
    begin=time.monotonic();A,At,scale,mask=actions(x,y,k);d=A(reference-unconstrained)
    m=len(d);trace=[];calls=0;iterations=0;last=None;max_residual=0.
    class TimeCap(Exception):pass
    def evaluate(values):
        nonlocal calls,last,max_residual
        if time.monotonic()-begin>time_limit:raise TimeCap()
        lam=torch.from_numpy(values).to(dtype=x.dtype)
        force=At(lam);correction,audit=inverse(force)
        max_residual=max(max_residual,audit['relative_residual'])
        value=.5*(force*correction).sum()-(d*lam).sum()
        gradient=A(correction)-d;calls+=1
        last=dict(lam=lam.clone(),force=force,correction=correction,value=float(value),gradient=gradient,
                  inverse_residual=audit['relative_residual'])
        return float(value),gradient.numpy().copy()
    def callback(values):
        nonlocal iterations
        iterations+=1
        if iterations==1 or iterations%25==0:
            trace.append(dict(iteration=iterations,evaluations=calls,dual_objective=last['value'],
                minimum_raw_slack=float(last['gradient'].min()/scale),
                positive_duals=int((last['lam']*scale>1e-9).sum()),solver_seconds=time.monotonic()-begin))
    native_feasible=float((-d/scale).min())>=-2e-8
    if native_feasible:
        values=np.zeros(m,dtype=np.float64);evaluate(values);termination='NATIVE_FEASIBLE'
    else:
        try:
            result=minimize(evaluate,np.zeros(m,dtype=np.float64),jac=True,method='L-BFGS-B',
                bounds=[(0.,None)]*m,callback=callback,
                options=dict(maxiter=maxiter,maxfun=10000,maxcor=10,ftol=1e-20,gtol=1e-12,maxls=40))
            evaluate(result.x);termination=str(result.message);iterations=int(result.nit)
        except TimeCap:termination='CPU_SOFT_TIME_CAP'
    if last is None:raise ValueError('No dual evaluation completed')
    lam=last['lam'];w=unconstrained+last['correction'];raw=native.contrasts(x,y,w-reference)
    off=native.objective(op,rhs,reference);proposal=native.objective(op,rhs,unconstrained)
    value=native.objective(op,rhs,w);dual=proposal-last['value']
    force=last['force'];stationarity=float((op(w)-rhs-force).norm()/(1+rhs.norm()+force.norm()))
    comp=float((lam*last['gradient']).abs().max());gap=float(value-dual)
    feasible=float(raw.min())>=-2e-8
    certified=(feasible and stationarity<=1e-8 and comp<=1e-7 and abs(gap)<=1e-7*(1+abs(float(value))))
    delta=(w-reference)-(w-reference).mean(1,keepdim=True)
    available=float(off-proposal);gain=float(off-value)
    audit=dict(status='CERTIFIED' if certified else 'UNRESOLVED',termination=termination,
        constraint_count=m,all_constraints_present_from_start=True,active_atom_cap=None,
        cached_inverse_directions=0,dual_variables=m,positive_duals=int((lam*scale>1e-9).sum()),
        iterations=iterations,H_inverse_calls=calls,minimum_raw_slack=float(raw.min()),
        stationarity=stationarity,complementarity=comp,primal_dual_gap=gap,
        objective_OFF=float(off),objective_unconstrained=float(proposal),objective_candidate=float(value),
        dual_lower_bound=float(dual),available_gain=available,
        certified_gain=gain if certified else None,
        retained_gain=gain/available if certified and available>1e-14 else None,
        discriminative_delta=float(delta.norm()) if certified else None,
        identity=(float(delta.norm())<=1e-10) if certified else None,
        dual_vector_bytes=lam.numel()*lam.element_size(),LBFGS_history_pairs=10,
        max_inverse_relative_residual=max_residual,trace=trace,
        solver_seconds=time.monotonic()-begin,encoder_or_statistics_mutated=False,
        protection_event=snapshot['event'],scope='fit',no_continuous_training_or_val_guarantee=True)
    cert=dict(force=force,reference=reference,head=w,snapshot=dict(snapshot),certificate=audit) if certified else None
    return w,audit,cert


def verify_stationarity(op,rhs,certificate,snapshot):
    validate_snapshot(snapshot)
    if certificate['snapshot']!=snapshot:raise ValueError('Checkpoint snapshot differs')
    w=certificate['head'];force=certificate['force']
    value=float((op(w)-rhs-force).norm()/(1+rhs.norm()+force.norm()))
    if value>1e-8 or certificate['certificate']['status']!='CERTIFIED':raise ValueError('Guard head certificate differs')
    return value
