"""Deterministic equal-moment, equal-support-count nonlinear-loss counterexample."""
import math
import os
import time
import torch
import torch.nn.functional as F
from check_progressive_decision_memory import DT,losses
from check_block_spherical_memory import decode


def supports(d,mean_length):
    assert d>=5 and mean_length in (0.,.6)
    # Five regular-simplex vertices in rank four; rotate one vertex onto the first axis.
    q=torch.linalg.qr(torch.cat((torch.eye(4,dtype=DT),-torch.ones(1,4,dtype=DT)),0)).Q
    s=q*math.sqrt(5/4);e=torch.eye(4,dtype=DT);u=s[0]-e[0];h=e-2*u[:,None]*u[None,:]/u.square().sum();s=s@h
    bisector=F.normalize(s[0]+s[1],dim=0);u=e[0]-bisector;h=e-2*u[:,None]*u[None,:]/u.square().sum();t=s@h
    amplitude=math.sqrt(1-mean_length**2);mu=torch.zeros(1,d,dtype=DT);mu[0,0]=mean_length
    a=mu.repeat(5,1);b=a.clone();a[:,1:5]=amplitude*s;b[:,1:5]=amplitude*t
    basis=torch.zeros(1,d,15,dtype=DT);basis[0,1:5,:4]=e
    ev=torch.zeros(1,15,dtype=DT);ev[0,:4]=amplitude**2/4;tail=torch.zeros(1,dtype=DT);trace=ev.sum(1)
    queries=torch.zeros(4,d,dtype=DT);queries[0,1]=1;queries[1,1:5]=bisector;queries[2,4]=1;queries[3,0]=1
    return a,b,(mu,basis,ev,tail,trace),queries


def audit(d,mean_length):
    a,b,distribution,queries=supports(d,mean_length);p=[torch.zeros(d,8,dtype=DT),torch.zeros(d,8,dtype=DT),2*queries]
    # All support lies in the five displayed coordinates; their moments are the full moments.
    small_a=a[:,:5];small_b=b[:,:5]
    mean_error=float((small_a.mean(0)-small_b.mean(0)).abs().max());second_error=float((small_a.T@small_a/5-small_b.T@small_b/5).abs().max());unit_error=max(float((z.norm(dim=1)-1).abs().max()) for z in (a,b))
    assert max(mean_error,second_error,unit_error)<1e-12
    fa=losses(a,p).mean(0);fb=losses(b,p).mean(0);amp=math.sqrt(1-mean_length**2)
    # Exact projection values: A=(1,-1/4,-1/4,-1/4,-1/4), B=(sqrt(3/8),sqrt(3/8),-1/sqrt(6),-1/sqrt(6),-1/sqrt(6)).
    soft=lambda z:float(F.softplus(torch.tensor(z,dtype=DT)))
    analytic_a=(soft(16*amp)+4*soft(-4*amp))/5
    analytic_b=(2*soft(16*amp*math.sqrt(3/8))+3*soft(-16*amp/math.sqrt(6)))/5
    formula_error=max(abs(float(fa[0])-analytic_a),abs(float(fb[0])-analytic_b));assert formula_error<1e-12
    assert abs(analytic_a-analytic_b)>.1
    rows=[dict(mean_length=mean_length,query=name,reference_A_loss=float(fa[j]),reference_B_loss=float(fb[j]),absolute_loss_gap=float(abs(fa[j]-fb[j])),common_prediction_error_lower_bound=float(abs(fa[j]-fb[j])/2)) for j,name in enumerate(('vertex','bisector','orthogonal','mean_axis'))]
    batches=[]
    for mode in ('gaussian','block_spherical'):
        for draw in range(8):
            z=decode(*distribution,319+draw,mode,16)[0];pred=losses(z,p).mean(0)
            ea=(pred-fa).abs();eb=(pred-fb).abs();assert bool((torch.maximum(ea,eb)+1e-12>=(fa-fb).abs()/2).all())
            for j,row in enumerate(rows):batches.append(dict(mean_length=mean_length,mode=mode,batch=draw,query=row['query'],prediction=float(pred[j]),error_A=float(ea[j]),error_B=float(eb[j]),max_error=float(max(ea[j],eb[j])),lower_bound=row['common_prediction_error_lower_bound']))
    return rows,batches,dict(mean_length=mean_length,support_count_A=5,support_count_B=5,intrinsic_rank=4,mean_error=mean_error,second_moment_error=second_error,unit_norm_error=unit_error,analytic_formula_error=formula_error,shared_representation=True)


def run(cfg,lock,journal):
    begun=time.monotonic();torch.set_num_threads(lock['threads']);torch.set_default_dtype(DT);assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    if cfg['kind']=='unit':
        rows,_,check=audit(8,.6)
        a,b,dist,q=supports(8,.6);mid=(rows[0]['reference_A_loss']+rows[0]['reference_B_loss'])/2
        assert abs(max(abs(mid-rows[0]['reference_A_loss']),abs(mid-rows[0]['reference_B_loss']))-rows[0]['common_prediction_error_lower_bound'])<1e-12
        result=dict(status='PASS',triangle_bound_midpoint_sharp=True,checks=check,formula_rows=rows,optimizer_updates=0)
    else:
        rows=[];batches=[];checks=[]
        for mu in (0.,.6):
            journal(dict(phase='equal_moment_counterexample',mean_length=mu));a,b,c=audit(cfg['dimension'],mu);rows.extend(a);batches.extend(b);checks.append(c)
        result=dict(status='COMPLETED',case=cfg,rows=rows,batches=batches,checks=checks,optimizer_updates=0,query_optimization_updates=0,shared_storage_scalars=16*cfg['dimension']+16,points_per_use=16,example_is_not_previous_fixture=True)
    result.update(id=cfg['id'],CPU_pipeline_seconds=time.monotonic()-begun,GPU_initialized=torch.cuda.is_initialized(),medical_images=0,medical_checkpoints=0,val_accessed=False,test_accessed=False,formal_updates=0);assert not result['GPU_initialized'];return result
