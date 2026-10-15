"""Current-support moment centers plus fresh isotropic residuals; synthetic CPU only."""
import itertools
import os
import time
import torch
import torch.nn.functional as F
from check_progressive_decision_memory import (
    DT, fixture, family, losses, grouped, memory_losses, moments_objective,
    project_ball, constrain_family,
)


def decode(nodes,rho,seed):
    g=torch.Generator().manual_seed(seed)
    noise=(2*torch.randint(2,nodes.shape,generator=g)-1).to(DT)/nodes.shape[-1]**.5
    raw=nodes+rho.sqrt()[:,None,None]*noise
    assert bool((raw.norm(dim=2)>1e-12).all()), 'Undefined decoded unit direction'
    z=F.normalize(raw,dim=2)
    return torch.where((rho==0)[:,None,None],nodes,z)


def unit_checks():
    n=torch.tensor([[[.3,.1,0.],[0.,.2,.4]]],dtype=DT)
    w=torch.zeros(1,2,dtype=DT);rho=torch.tensor([.4],dtype=DT)
    signs=torch.tensor(list(itertools.product((-1.,1.),repeat=3)),dtype=DT)/3**.5
    raw=n[:,:,None,:]+rho.sqrt()[:,None,None,None]*signs[None,None,:,:]
    assert float((raw.mean(2)-n).abs().max())<1e-12
    trace_error=float((raw.square().sum(3).mean(2)-(n.square().sum(2)+rho[:,None])).abs().max())
    assert trace_error<1e-12
    z=F.normalize(raw,dim=3)
    p=[q.requires_grad_() for q in family(3,2,2,263)]
    exact=losses(z.reshape(-1,3),p).mean()
    grad=torch.autograd.grad(exact,p[0])[0]
    # Exhaust all shared sign draws. Linearity also gives this expectation for independent node draws.
    samples=[];grads=[]
    for j in range(8):
        value=memory_losses(z[:,:,j],w,p).mean();samples.append(value)
        grads.append(torch.autograd.grad(value,p[0])[0])
    err=float((torch.stack(samples).mean()-exact).abs())
    gerr=float((torch.stack(grads).mean(0)-grad).abs().max())
    assert max(err,gerr)<1e-12
    direction=torch.ones_like(p[0]);direction/=direction.norm();eps=1e-6
    pp=[q.detach().clone() for q in p];pm=[q.detach().clone() for q in p]
    pp[0]+=eps*direction;pm[0]-=eps*direction
    fd=abs(float((losses(z.reshape(-1,3),pp).mean()-losses(z.reshape(-1,3),pm).mean())/(2*eps)-(grad*direction).sum()))
    assert fd<1e-7 and torch.equal(decode(n,torch.zeros_like(rho),264),n)
    assert torch.equal(decode(n,rho,264),decode(n,rho,264)) and not torch.equal(decode(n,rho,264),decode(n,rho,265))
    return dict(status='PASS',raw_trace_identity_error=trace_error,enumerated_loss_expectation_error=err,
        enumerated_gradient_expectation_error=gerr,gradient_fd_error=fd,zero_residual_exact=True,
        post_normalization_mean_shift=float((z.mean(2)-n).norm()),
        generated_distribution_expectation_not_true_distribution_unbiased=True,optimizer_updates=0)


def run_scale(cfg,lock,journal):
    begun=time.monotonic();x,y,counts=fixture(cfg);c,d=cfg['classes'],cfg['dimension'];m=lock['nodes_per_class']
    means=torch.stack([x[y==j].mean(0) for j in range(c)])
    second=torch.stack([x[y==j].T@x[y==j]/counts[j] for j in range(c)])
    g=torch.Generator().manual_seed(cfg['seed']+1)
    init=torch.stack([x[y==j][torch.randint(counts[j],(m,),generator=g)] for j in range(c)])
    zero=torch.zeros(c,m,dtype=DT);n=init.clone().requires_grad_();w=zero.clone().requires_grad_()
    opt=torch.optim.Adam([n,w],lr=lock['memory_lr']);construction_start=time.monotonic()
    for step in range(lock['outer_steps']):
        opt.zero_grad();moments_objective(n,w,means,second).backward();opt.step()
        with torch.no_grad():n.copy_(project_ball(n));w.clamp_(-8,8)
        if step%8==0:journal(dict(phase='shared_moment_construction',step=step+1))
    construction_seconds=time.monotonic()-construction_start
    n=n.detach();w=w.detach();frozen_n=n.clone();frozen_w=w.clone()
    assert bool((n.norm(dim=2)>1e-12).all())
    fixed={'random':(init,zero),'moment':(n,w),'normalized_moment':(F.normalize(n,dim=2),w)}
    target_trace=second.diagonal(dim1=1,dim2=2).sum(1)
    memory_trace=(n.square().sum(2)*w.softmax(1)).sum(1)
    rho=(target_trace-memory_trace).clamp_min(0)
    q=torch.einsum('cm,cmd,cme->cde',w.softmax(1),n,n)
    delta=second-q;delta_trace=delta.diagonal(dim1=1,dim2=2).sum(1)
    residual_error=delta.square().sum((1,2))-2*rho*delta_trace/d+rho.square()/d
    del q,delta
    params=[family(d,lock['adapter_rank'],lock['queries'],cfg['seed']+100+i) for i in range(4)]
    evaluation_start=time.monotonic()
    for i in range(2):
        p=[a.clone().requires_grad_() for a in params[i]];adv=torch.optim.Adam(p,lr=lock['query_lr'])
        for step in range(lock['evaluation_adversary_steps']):
            adv.zero_grad();target=grouped(losses(x,p),y,c)
            fresh=decode(n,rho,cfg['seed']+5000+100*i+step)
            errors=[(target-memory_losses(a,b,p)).square().mean() for a,b in fixed.values()]
            errors.append((target-memory_losses(fresh,w,p)).square().mean())
            (-torch.stack(errors).mean()).backward();adv.step();constrain_family(p)
        params.append([a.detach() for a in p])
    rows=[];batches=[];predictions=[]
    def summary(mode,pred):
        e=(targets-pred).abs()
        return dict(mode=mode,mean_abs_error=float(e.mean()),max_abs_error=float(e.max()),
            random_max=float(e[:,:32].max()),stress_max=float(e[:,32:].max()),
            per_class_mean=[float(z) for z in e.mean(1)],per_class_max=[float(z) for z in e.max(1).values])
    with torch.no_grad():
        targets=torch.cat([grouped(losses(x,p),y,c) for p in params],1)
        for mode,(a,b) in fixed.items():rows.append(summary(mode,torch.cat([memory_losses(a,b,p) for p in params],1)))
        for draw in range(lock['audit_batches']):
            journal(dict(phase='frozen_query_fresh_batch_audit',completed_batches=draw))
            fresh=decode(n,rho,cfg['seed']+7000+draw)
            pred=torch.cat([memory_losses(fresh,w,p) for p in params],1);predictions.append(pred)
            a=summary('residual_distribution',pred);a.update(batch=draw,
                weighted_trace=[float(z) for z in (fresh.square().sum(2)*w.softmax(1)).sum(1)],
                mean_sq_error=[float(z) for z in ((fresh*w.softmax(1)[:,:,None]).sum(1)-means).square().sum(1)])
            batches.append(a)
    baseline={r['mode']:r for r in rows}
    passed=all(a['max_abs_error']<=.9*baseline[b]['max_abs_error'] and a['mean_abs_error']<=baseline[b]['mean_abs_error']
               for a in batches for b in ('moment','random'))
    # This average is a diagnostic only; the deployment and primary gate each use one 16-node draw.
    averaged=summary('eight_batch_average_DIAGNOSTIC_ONLY',torch.stack(predictions).mean(0))
    assert torch.equal(n,frozen_n) and torch.equal(w,frozen_w)
    return dict(status='COMPLETED',case=cfg,synthetic_counts=counts,rows=rows,batches=batches,
        averaged_prediction_diagnostic=averaged,finite_query_screen_pass=passed,
        every_batch_pass_required=True,centers_weights_unchanged=True,optimizer_steps=lock['outer_steps'],
        construction_wall_seconds=construction_seconds,evaluation_wall_seconds=time.monotonic()-evaluation_start,
        residual_trace=[float(z) for z in rho],raw_memory_trace=[float(z) for z in memory_trace],
        raw_isotropic_covariance_residual_fro_sq=[float(z) for z in residual_error],
        residual_PSD_not_verified=True,base_memory_bytes=8*(n.numel()+w.numel()),
        candidate_memory_bytes=8*(n.numel()+w.numel()+rho.numel()),
        fresh_batch_bytes=8*n.numel(),query_parameters_fixed_before_audit=True,
        generated_law_not_true_distribution=True,CPU_pipeline_seconds=time.monotonic()-begun)


def run(cfg,lock,journal):
    torch.set_num_threads(lock['threads']);torch.set_default_dtype(DT)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    a=unit_checks() if cfg['kind']=='unit' else run_scale(cfg,lock,journal)
    a.update(id=cfg['id'],GPU_initialized=torch.cuda.is_initialized(),medical_images=0,
        medical_checkpoints=0,val_accessed=False,test_accessed=False,formal_updates=0)
    assert not a['GPU_initialized']
    return a
