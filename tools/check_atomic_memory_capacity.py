"""Finite-atom counterexample, not a medical experiment or a compression optimizer."""
import math
import os
import time
import torch
import torch.nn.functional as F

DT=torch.float64


def logcosh(x):
    return torch.logaddexp(x,-x)-math.log(2.)


def reference_loss(q,b):
    # Exact finite reference: uniform on the 2d points +/- coordinate axes.
    return float((F.softplus(b*q)+F.softplus(-b*q)).mean()/2)


def witness(nodes,b):
    # Identity adapter still normalizes each latent node, as in the frozen query family.
    nodes=F.normalize(nodes,dim=1,eps=1e-8)
    m,d=nodes.shape;gram=nodes@nodes.T/m
    values,vectors=torch.linalg.eigh(gram);lam=float(values[-1]);rank=int((values>1e-12).sum())
    q=F.normalize(nodes.T@vectors[:,-1],dim=0) if lam>1e-12 else F.one_hot(torch.tensor(0),d).to(DT)
    ref=reference_loss(q,b)
    plus=float(F.softplus(b*(nodes@q)).mean());minus=float(F.softplus(-b*(nodes@q)).mean())
    tau=float(nodes.square().sum()/m);c=float(logcosh(torch.tensor(b/2,dtype=DT)))
    lower=c*lam-b*b/(8*d);universal=c*tau/m-b*b/(8*d)
    signed_pair=(plus+minus)/2-ref
    assert abs(float(q.norm())-1)<1e-12
    assert lam+1e-12>=tau/m and signed_pair+1e-12>=lower>=universal-1e-12
    assert ref<=math.log(2.)+b*b/(8*d)+1e-12
    dense=torch.ones(d,dtype=DT)/math.sqrt(d)
    perpendicular=F.one_hot(torch.tensor(m),d).to(DT)
    def pair_error(direction):
        r=reference_loss(direction,b)
        return max(abs(float(F.softplus(sign*b*(nodes@direction)).mean())-r) for sign in (-1,1))
    return dict(node_count=m,dimension=d,reference_support_count=2*d,
        trace=tau,rank=rank,largest_second_moment_eigenvalue=lam,
        mean_norm=float(nodes.mean(0).norm()),min_radius=float(nodes.norm(dim=1).min()),
        max_radius=float(nodes.norm(dim=1).max()),reference_loss=ref,memory_plus_loss=plus,memory_minus_loss=minus,
        witness_signed_pair_gap=signed_pair,witness_max_abs_error=max(abs(plus-ref),abs(minus-ref)),
        spectrum_lower_bound=lower,trace_only_lower_bound=universal,
        trace_bound_nontrivial=universal>0,deterministic_dense_query_error=pair_error(dense),
        orthogonal_query_error=pair_error(perpendicular),logical_node_bytes=nodes.numel()*8,
        query_norm=2.,identity_adapter=True)


def run_scale(cfg,lock,journal):
    start=time.monotonic();d=cfg['dimension'];m=lock['nodes_per_class'];b=lock['beta']*lock['query_norm']
    assert d>m and m==16
    orthogonal=torch.zeros(m,d,dtype=DT);orthogonal[:,:m]=torch.eye(m,dtype=DT)
    simplex=F.normalize(orthogonal-orthogonal.mean(0),dim=1)
    antipodal=torch.cat([orthogonal[:m//2],-orthogonal[:m//2]],0)
    zero=torch.zeros_like(orthogonal)
    rows=[]
    for name,nodes in [('orthogonal',orthogonal),('simplex',simplex),('antipodal',antipodal),('zero_trace_control',zero)]:
        journal(dict(phase='exact_finite_reference',mode=name,completed_modes=len(rows)))
        a=witness(nodes,b);a['mode']=name
        if name!='zero_trace_control':assert abs(a['trace']-1)<1e-12
        if name in ('simplex','antipodal'):assert a['mean_norm']<1e-12
        rows.append(a)
    return dict(status='COMPLETED',case=cfg,rows=rows,all_inequalities_verified=True,
        optimizer_updates=0,reference_is_medical=False,CPU_pipeline_seconds=time.monotonic()-start)


def unit_checks(lock):
    b=lock['beta']*lock['query_norm'];t=torch.linspace(-1,1,10001,dtype=DT)
    h=logcosh(b*t/2);c=logcosh(torch.tensor(b/2,dtype=DT))
    lo=c*t.square();hi=b*b*t.square()/8
    assert float((lo-h).max())<1e-12 and float((h-hi).max())<1e-12
    # Check the algebraic pair identity independently of the bound.
    err=float(((F.softplus(b*t)+F.softplus(-b*t))/2-math.log(2.)-h).abs().max())
    assert err<1e-12
    return dict(status='PASS',paired_softplus_identity_error=err,
        lower_inequality_max_violation=float((lo-h).max()),upper_inequality_max_violation=float((h-hi).max()),
        grid_points=len(t),proof_is_analytic_grid_only_arithmetic_check=True,optimizer_updates=0)


def run(cfg,lock,journal):
    torch.set_num_threads(lock['threads']);torch.set_default_dtype(DT)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    a=unit_checks(lock) if cfg['kind']=='unit' else run_scale(cfg,lock,journal)
    a.update(id=cfg['id'],GPU_initialized=torch.cuda.is_initialized(),medical_images=0,
        medical_checkpoints=0,val_accessed=False,test_accessed=False,formal_updates=0)
    assert not a['GPU_initialized']
    return a
