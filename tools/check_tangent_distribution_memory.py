"""Mean-preserving symmetric tangent noise; synthetic CPU-only falsification."""
import itertools
import os
import time
import torch
import torch.nn.functional as F
from check_progressive_decision_memory import (
    DT, fixture, family, losses, grouped, memory_losses, moments_objective,
    project_ball, constrain_family,
)
from check_antithetic_residual_memory import pair_centers, decode_antithetic


def tangent_from_noise(nodes,noise):
    norm_sq=nodes.square().sum(2,keepdim=True)
    assert bool(torch.isfinite(nodes).all()) and bool((norm_sq<=1+1e-12).all()), 'Center outside unit ball'
    residual=(1-norm_sq).clamp_min(0)
    projected=noise-(noise*nodes).sum(2,keepdim=True)/norm_sq.clamp_min(1e-300)*nodes
    length=projected.norm(dim=2,keepdim=True)
    assert bool(((length>1e-12)|(residual==0)).all()), 'Degenerate tangent direction'
    unit=projected/length.clamp_min(1e-300)
    z=nodes+residual.sqrt()*unit
    return torch.where(residual==0,nodes,z)


def decode_tangent(nodes,seed,pairs):
    g=torch.Generator().manual_seed(seed)
    noise=(2*torch.randint(2,nodes.shape,generator=g)-1).to(DT)/nodes.shape[-1]**.5
    for j in range(nodes.shape[0]):noise[j,pairs[j,:,1]]=-noise[j,pairs[j,:,0]]
    return tangent_from_noise(nodes,noise)


def unit_checks():
    n=torch.tensor([[[.3,.1,0.],[0.,.2,.4]]],dtype=DT);w=torch.tensor([[.2,-.3]],dtype=DT)
    pairs=pair_centers(n);signs=torch.tensor(list(itertools.product((-1.,1.),repeat=3)),dtype=DT)/3**.5
    samples=torch.stack([tangent_from_noise(n,torch.stack((e,-e))[None]) for e in signs])
    mean_error=float((samples.mean(0)-n).abs().max());norm_error=float((samples.norm(dim=3)-1).abs().max())
    assert max(mean_error,norm_error)<1e-12
    g=torch.Generator().manual_seed(299);noise=torch.randn(n.shape,generator=g,dtype=DT)
    odd_error=float((tangent_from_noise(n,noise)+tangent_from_noise(n,-noise)-2*n).abs().max());assert odd_error<1e-12
    zero=torch.zeros_like(n);unit=torch.tensor([[[1.,0.,0.],[0.,1.,0.]]],dtype=DT)
    assert torch.equal(tangent_from_noise(unit,noise),unit)
    assert float((tangent_from_noise(zero,noise)-F.normalize(noise,dim=2)).abs().max())<1e-12
    rejected=0
    for bad_nodes,bad_noise in ((torch.tensor([[[.5,0.,0.]]],dtype=DT),torch.tensor([[[1.,0.,0.]]],dtype=DT)),(2*unit,noise)):
        try:tangent_from_noise(bad_nodes,bad_noise)
        except AssertionError:rejected+=1
    assert rejected==2
    assert torch.equal(decode_tangent(n,263,pairs),decode_tangent(n,263,pairs))
    assert not torch.equal(decode_tangent(n,263,pairs),decode_tangent(n,264,pairs))
    p=[q.requires_grad_() for q in family(3,2,2,263)]
    objective=torch.stack([memory_losses(z,w,p).mean() for z in samples]).mean()
    grad=torch.autograd.grad(objective,p[0])[0];direction=torch.ones_like(p[0]);direction/=direction.norm();eps=1e-6
    def value(first):return torch.stack([memory_losses(z,w,[first,*p[1:]]).mean() for z in samples]).mean()
    fd=abs(float((value(p[0]+eps*direction)-value(p[0]-eps*direction))/(2*eps)-(grad*direction).sum()));assert fd<1e-7
    # Exact paired enumeration and independent-node marginals have the same expected loss.
    marginal=torch.stack([memory_losses(tangent_from_noise(n,e[None,None].expand_as(n)),w,p).mean() for e in signs]).mean()
    marginal_error=float((marginal-objective.detach()).abs());assert marginal_error<1e-12
    return dict(status='PASS',enumerated_mean_error=mean_error,unit_norm_error=norm_error,symmetric_noise_mean_identity_error=odd_error,
        gradient_fd_error=fd,paired_marginal_loss_error=marginal_error,unit_center_exact=True,zero_center_normalized_noise=True,
        illegal_domain_rejections=rejected,true_class_mean_not_guaranteed=True,true_covariance_not_guaranteed=True,
        Rademacher_tangent_law_not_rotation_invariant=True,optimizer_updates=0)


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
    pair_started=time.monotonic();pairs=pair_centers(n);pairing_seconds=time.monotonic()-pair_started
    params=[family(d,lock['adapter_rank'],lock['queries'],cfg['seed']+100+i) for i in range(4)]
    evaluation_start=time.monotonic()
    for i in range(2):
        p=[a.clone().requires_grad_() for a in params[i]];adv=torch.optim.Adam(p,lr=lock['query_lr'])
        for step in range(lock['evaluation_adversary_steps']):
            adv.zero_grad();target=grouped(losses(x,p),y,c)
            fresh=decode_antithetic(n,rho,cfg['seed']+5000+100*i+step,pairs)
            errors=[(target-memory_losses(a,b,p)).square().mean() for a,b in fixed.values()]
            errors.append((target-memory_losses(fresh,w,p)).square().mean())
            paired=decode_tangent(n,cfg['seed']+5000+100*i+step,pairs)
            errors.append((target-memory_losses(paired,w,p)).square().mean())
            (-torch.stack(errors).mean()).backward();adv.step();constrain_family(p)
        params.append([a.detach() for a in p])
    rows=[];batches=[];predictions={'isotropic_paired':[],'tangent_paired':[]}
    def summary(mode,pred):
        e=(targets-pred).abs()
        return dict(mode=mode,mean_abs_error=float(e.mean()),max_abs_error=float(e.max()),
            random_max=float(e[:,:32].max()),stress_max=float(e[:,32:].max()),
            per_class_mean=[float(z) for z in e.mean(1)],per_class_max=[float(z) for z in e.max(1).values])
    with torch.no_grad():
        targets=torch.cat([grouped(losses(x,p),y,c) for p in params],1)
        for mode,(a,b) in fixed.items():rows.append(summary(mode,torch.cat([memory_losses(a,b,p) for p in params],1)))
        for draw in range(lock['audit_batches']):
            journal(dict(phase='frozen_query_paired_batch_audit',completed_batches=draw))
            for mode in predictions:
                fresh=(decode_antithetic(n,rho,cfg['seed']+7000+draw,pairs) if mode=='isotropic_paired'
                       else decode_tangent(n,cfg['seed']+7000+draw,pairs))
                pred=torch.cat([memory_losses(fresh,w,p) for p in params],1);predictions[mode].append(pred)
                a=summary(mode,pred);a.update(batch=draw,
                    weighted_trace=[float(z) for z in (fresh.square().sum(2)*w.softmax(1)).sum(1)],
                    mean_sq_error=[float(z) for z in ((fresh*w.softmax(1)[:,:,None]).sum(1)-means).square().sum(1)])
                batches.append(a)
    assert all(sum(a['mode']==mode for a in batches)==lock['audit_batches'] for mode in predictions)
    baseline={r['mode']:r for r in rows}
    passed=all(a['max_abs_error']<=.9*baseline[b]['max_abs_error'] and a['mean_abs_error']<=baseline[b]['mean_abs_error']
               for a in batches if a['mode']=='tangent_paired' for b in ('moment','random'))
    # This average is a diagnostic only; the deployment and primary gate each use one 16-node draw.
    averaged=[summary(mode+'_eight_batch_average_DIAGNOSTIC_ONLY',torch.stack(pred).mean(0)) for mode,pred in predictions.items()]
    variance={mode:dict(mean=float(torch.stack(pred).var(0,unbiased=True).mean()),max=float(torch.stack(pred).var(0,unbiased=True).max()),per_class_mean=[float(z) for z in torch.stack(pred).var(0,unbiased=True).mean(1)]) for mode,pred in predictions.items()}
    assert torch.equal(n,frozen_n) and torch.equal(w,frozen_w)
    return dict(status='COMPLETED',case=cfg,synthetic_counts=counts,rows=rows,batches=batches,
        averaged_prediction_diagnostic=averaged,prediction_variance_diagnostic=variance,finite_query_screen_pass=passed,
        pairing_seconds=pairing_seconds,pair_cache_bytes=pairs.numel()*pairs.element_size(),
        pair_mean_sq_distance=float((n.gather(1,pairs[:,:,0,None].expand(-1,-1,d))-n.gather(1,pairs[:,:,1,None].expand(-1,-1,d))).square().sum(2).mean()),
        every_batch_pass_required=True,centers_weights_unchanged=True,optimizer_steps=lock['outer_steps'],
        construction_wall_seconds=construction_seconds,evaluation_wall_seconds=time.monotonic()-evaluation_start,
        residual_trace=[float(z) for z in rho],raw_memory_trace=[float(z) for z in memory_trace],
        raw_isotropic_covariance_residual_fro_sq=[float(z) for z in residual_error],
        residual_PSD_not_verified=True,base_memory_bytes=8*(n.numel()+w.numel()),
        candidate_memory_bytes=8*(n.numel()+w.numel()),
        isotropic_control_memory_bytes=8*(n.numel()+w.numel()+rho.numel()),
        expected_candidate_mean_sq_error=[float(z) for z in ((n*w.softmax(1)[:,:,None]).sum(1)-means).square().sum(1)],
        exact_true_covariance_claim=False,
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
