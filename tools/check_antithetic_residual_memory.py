"""Paired residual draws preserve each center's marginal law; synthetic CPU only."""
import itertools
import os
import time
import torch
import torch.nn.functional as F
from check_progressive_decision_memory import (
    DT, fixture, family, losses, grouped, memory_losses, moments_objective,
    project_ball, constrain_family,
)
from check_residual_distribution_memory import decode, unit_checks as residual_unit_checks


def pair_centers(nodes):
    # The frozen centers alone determine pairing; no query or evaluation labels enter.
    c,m,d=nodes.shape;assert m%2==0
    result=[]
    for row in nodes:
        distances=(row[:,None]-row[None,:]).square().sum(2)
        remaining=list(range(m));pairs=[]
        while remaining:
            a,b=min(((a,b) for a in remaining for b in remaining if a<b),key=lambda q:(float(distances[q[0],q[1]]),q[0],q[1]))
            pairs.append((a,b));remaining.remove(a);remaining.remove(b)
        result.append(pairs)
    return torch.tensor(result,dtype=torch.long)


def decode_antithetic(nodes,rho,seed,pairs):
    g=torch.Generator().manual_seed(seed)
    noise=(2*torch.randint(2,nodes.shape,generator=g)-1).to(DT)/nodes.shape[-1]**.5
    for j in range(nodes.shape[0]):
        noise[j,pairs[j,:,1]]=-noise[j,pairs[j,:,0]]
    raw=nodes+rho.sqrt()[:,None,None]*noise
    assert bool((raw.norm(dim=2)>1e-12).all()), 'Undefined decoded unit direction'
    return torch.where((rho==0)[:,None,None],nodes,F.normalize(raw,dim=2))


def unit_checks():
    inherited=residual_unit_checks()
    n=torch.tensor([[[.3,.1,0.],[0.,.2,.4]]],dtype=DT)
    rho=torch.tensor([.4],dtype=DT);w=torch.tensor([[.2,-.3]],dtype=DT)
    pairs=pair_centers(n);assert pairs.tolist()==[[[0,1]]]
    assert torch.equal(decode_antithetic(n,torch.zeros_like(rho),264,pairs),n)
    assert torch.equal(decode_antithetic(n,rho,264,pairs),decode_antithetic(n,rho,264,pairs))
    assert not torch.equal(decode_antithetic(n,rho,264,pairs),decode_antithetic(n,rho,265,pairs))
    signs=torch.tensor(list(itertools.product((-1.,1.),repeat=3)),dtype=DT)/3**.5
    p=[q.requires_grad_() for q in family(3,2,2,263)]
    marginal=[];paired=[]
    for e in signs:
        marginal.append(memory_losses(F.normalize(n+rho.sqrt()[:,None,None]*e,dim=2),w,p).mean())
        paired.append(memory_losses(F.normalize(n+rho.sqrt()[:,None,None]*torch.stack((e,-e))[None],dim=2),w,p).mean())
    v1=torch.stack(marginal).mean();v2=torch.stack(paired).mean()
    ge=float((torch.autograd.grad(v1,p[0],retain_graph=True)[0]-torch.autograd.grad(v2,p[0],retain_graph=True)[0]).abs().max())
    le=float((v1-v2).abs());assert max(le,ge)<1e-12
    direction=torch.ones_like(p[0]);direction/=direction.norm();eps=1e-6
    def objective(first):
        pp=[first,*p[1:]]
        return torch.stack([memory_losses(F.normalize(n+rho.sqrt()[:,None,None]*torch.stack((e,-e))[None],dim=2),w,pp).mean() for e in signs]).mean()
    grad=torch.autograd.grad(v2,p[0])[0]
    fd=abs(float((objective(p[0]+eps*direction)-objective(p[0]-eps*direction))/(2*eps)-(grad*direction).sum()));assert fd<1e-7
    # Linear-sign counterexamples: equal responses benefit, opposing responses worsen.
    independent=torch.tensor([(a-b)/2 for a,b in itertools.product((-1.,1.),repeat=2)],dtype=DT).var(unbiased=False)
    opposing=torch.tensor([-1.,1.],dtype=DT).var(unbiased=False)
    assert float(independent)==.5 and float(opposing)==1.
    test=torch.tensor([[[0.,0.],[.1,0.],[3.,0.],[3.2,0.]]],dtype=DT)
    assert pair_centers(test).tolist()==[[[0,1],[2,3]]]
    assert sorted(pair_centers(test).flatten().tolist())==list(range(4))
    return dict(status='PASS',inherited_residual_checks=inherited,paired_loss_expectation_error=le,
        paired_gradient_expectation_error=ge,paired_gradient_fd_error=fd,zero_residual_exact=True,
        equal_linear_response_variance_independent=.5,equal_linear_response_variance_paired=0.,
        opposing_linear_response_variance_independent=float(independent),opposing_linear_response_variance_paired=float(opposing),
        universal_variance_reduction=False,same_marginal_law=True,optimizer_updates=0)


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
            fresh=decode(n,rho,cfg['seed']+5000+100*i+step)
            errors=[(target-memory_losses(a,b,p)).square().mean() for a,b in fixed.values()]
            errors.append((target-memory_losses(fresh,w,p)).square().mean())
            paired=decode_antithetic(n,rho,cfg['seed']+5000+100*i+step,pairs)
            errors.append((target-memory_losses(paired,w,p)).square().mean())
            (-torch.stack(errors).mean()).backward();adv.step();constrain_family(p)
        params.append([a.detach() for a in p])
    rows=[];batches=[];predictions={'independent_residual':[],'antithetic_residual':[]}
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
                fresh=(decode(n,rho,cfg['seed']+7000+draw) if mode=='independent_residual'
                       else decode_antithetic(n,rho,cfg['seed']+7000+draw,pairs))
                pred=torch.cat([memory_losses(fresh,w,p) for p in params],1);predictions[mode].append(pred)
                a=summary(mode,pred);a.update(batch=draw,
                    weighted_trace=[float(z) for z in (fresh.square().sum(2)*w.softmax(1)).sum(1)],
                    mean_sq_error=[float(z) for z in ((fresh*w.softmax(1)[:,:,None]).sum(1)-means).square().sum(1)])
                batches.append(a)
    assert all(sum(a['mode']==mode for a in batches)==lock['audit_batches'] for mode in predictions)
    baseline={r['mode']:r for r in rows}
    passed=all(a['max_abs_error']<=.9*baseline[b]['max_abs_error'] and a['mean_abs_error']<=baseline[b]['mean_abs_error']
               for a in batches if a['mode']=='antithetic_residual' for b in ('moment','random'))
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
