"""Fit moments of the generated spherical law, rather than its centers alone."""
import os
import time
import torch
import torch.nn.functional as F
from check_progressive_decision_memory import DT,fixture,family,losses,grouped,memory_losses,moments_objective,project_ball,constrain_family
from check_antithetic_residual_memory import pair_centers
from check_tangent_distribution_memory import tangent_from_noise


def generated_moments(nodes,logits):
    d=nodes.shape[-1];assert d>=2
    sq=nodes.square().sum(2)
    assert bool(torch.isfinite(nodes).all()) and bool((sq>1e-24).all()) and bool((sq<=1+1e-12).all()), 'Undefined Gaussian latitude center'
    u=nodes/sq.sqrt()[:,:,None];p=logits.softmax(1);t=(1-sq).clamp_min(0)/(d-1)
    mu=(nodes*p[:,:,None]).sum(1)
    q=torch.einsum('cm,cmd,cme->cde',p*(sq-t),u,u)
    q=q+(p*t).sum(1)[:,None,None]*torch.eye(d,dtype=DT)[None]
    return mu,q


def generated_objective(nodes,logits,means,second):
    mu,q=generated_moments(nodes,logits)
    return (mu-means).square().sum(1).mean()+(q-second).square().sum((1,2)).mean()


def decode_gaussian(nodes,seed,pairs):
    # Gaussian projection gives a uniform direction in each center's tangent space.
    assert bool((nodes.square().sum(2)>1e-24).all()), 'Zero-center extension not part of this protocol'
    g=torch.Generator().manual_seed(seed);noise=torch.randn(nodes.shape,generator=g,dtype=DT)
    for j in range(nodes.shape[0]):noise[j,pairs[j,:,1]]=-noise[j,pairs[j,:,0]]
    return tangent_from_noise(nodes,noise)


def unit_checks():
    n=torch.tensor([[[.3,.1,0.],[0.,.2,.4]]],dtype=DT);w=torch.tensor([[.2,-.3]],dtype=DT)
    mu,q=generated_moments(n,w);mus=[];qs=[]
    # +/- orthonormal tangent directions integrate first and second moments exactly,
    # not arbitrary nonlinear losses of the Gaussian latitude distribution.
    for v in n[0]:
        basis=torch.linalg.qr(v[:,None],mode='complete').Q[:,1:]
        z=v[None]+(1-v.square().sum()).sqrt()*torch.cat((basis.T,-basis.T),0)
        mus.append(z.mean(0));qs.append(z.T@z/len(z))
    weights=w.softmax(1)[0];merr=float((mu[0]-(torch.stack(mus)*weights[:,None]).sum(0)).abs().max())
    qerr=float((q[0]-(torch.stack(qs)*weights[:,None,None]).sum(0)).abs().max());assert max(merr,qerr)<1e-12
    trace=float((q.diagonal(dim1=1,dim2=2).sum(1)-1).abs().max());mineig=float(torch.linalg.eigvalsh(q).min());assert trace<1e-12 and mineig>=-1e-12
    target_mu=torch.zeros_like(mu);target_q=torch.eye(3,dtype=DT)[None]/3
    nn=n.clone().requires_grad_();ww=w.clone().requires_grad_();value=generated_objective(nn,ww,target_mu,target_q)
    gn,gw=torch.autograd.grad(value,(nn,ww));dn=torch.ones_like(n);dn/=dn.norm();dw=torch.tensor([[.4,-.4]],dtype=DT);eps=1e-6
    fd=abs(float((generated_objective(n+eps*dn,w+eps*dw,target_mu,target_q)-generated_objective(n-eps*dn,w-eps*dw,target_mu,target_q))/(2*eps)-(gn*dn).sum()-(gw*dw).sum()));assert fd<1e-7
    pairs=pair_centers(n);z=decode_gaussian(n,303,pairs);norm=float((z.norm(dim=2)-1).abs().max());assert norm<1e-12
    assert torch.equal(z,decode_gaussian(n,303,pairs)) and not torch.equal(z,decode_gaussian(n,304,pairs))
    unit=torch.tensor([[[1.,0.,0.],[0.,1.,0.]]],dtype=DT);assert torch.equal(decode_gaussian(unit,303,pairs),unit)
    rejected=0
    for bad in (torch.zeros_like(n),2*unit):
        try:generated_moments(bad,w)
        except AssertionError:rejected+=1
    assert rejected==2
    permutation=torch.tensor([1,0]);mu2,q2=generated_moments(n[:,permutation],w[:,permutation]);assert max(float((mu-mu2).abs().max()),float((q-q2).abs().max()))<1e-12
    p=[a.requires_grad_() for a in family(3,2,2,303)];val=memory_losses(z,w,p).mean();gp=torch.autograd.grad(val,p[0])[0];direction=torch.ones_like(p[0]);direction/=direction.norm()
    def f(first):return memory_losses(z,w,[first,*p[1:]]).mean()
    qfd=abs(float((f(p[0]+eps*direction)-f(p[0]-eps*direction))/(2*eps)-(gp*direction).sum()));assert qfd<1e-7
    return dict(status='PASS',moment_cubature_mean_error=merr,moment_cubature_second_error=qerr,analytic_trace_error=trace,analytic_min_eigenvalue=mineig,generated_objective_fd_error=fd,sampled_query_fd_error=qfd,generated_norm_error=norm,unit_center_exact=True,illegal_domain_rejections=rejected,cubature_not_nonlinear_loss_integral=True,optimizer_updates=0)


def run_scale(cfg,lock,journal):
    begun=time.monotonic();x,y,counts=fixture(cfg);c,d=cfg['classes'],cfg['dimension'];m=lock['nodes_per_class']
    means=torch.stack([x[y==j].mean(0) for j in range(c)]);second=torch.stack([x[y==j].T@x[y==j]/counts[j] for j in range(c)])
    g=torch.Generator().manual_seed(cfg['seed']+1);init=torch.stack([x[y==j][torch.randint(counts[j],(m,),generator=g)] for j in range(c)])
    zero=torch.zeros(c,m,dtype=DT);fits={};fit_diagnostics=[];construction_start=time.monotonic()
    for mode,objective in (('center_moment',moments_objective),('generated_moment',generated_objective)):
        n=init.clone().requires_grad_();w=zero.clone().requires_grad_();opt=torch.optim.Adam([n,w],lr=lock['memory_lr']);trace=[]
        initial=float(objective(n,w,means,second).detach())
        for step in range(lock['outer_steps']):
            opt.zero_grad();objective(n,w,means,second).backward();opt.step()
            with torch.no_grad():n.copy_(project_ball(n));w.clamp_(-8,8)
            if step%8==0:trace.append(dict(step=step+1,objective=float(objective(n,w,means,second).detach())));journal(dict(phase=mode,step=step+1,objective_trace=trace))
        fits[mode]=(n.detach(),w.detach());fit_diagnostics.append(dict(mode=mode,initial_objective=initial,final_objective=float(objective(n,w,means,second).detach()),trace=trace,steps=lock['outer_steps']))
    construction_seconds=time.monotonic()-construction_start
    n0,w0=fits['center_moment'];n1,w1=fits['generated_moment'];frozen=[a.clone() for a in (n0,w0,n1,w1)]
    assert bool((n0.norm(dim=2)>1e-12).all())
    fixed={'random':(init,zero),'moment':(n0,w0),'normalized_moment':(F.normalize(n0,dim=2),w0)}
    laws={'posthoc_gaussian':(n0,w0),'generated_moment':(n1,w1)}
    pair_start=time.monotonic();pairs={mode:pair_centers(n) for mode,(n,w) in laws.items()};pair_seconds=time.monotonic()-pair_start
    geometry=[]
    with torch.no_grad():
        for mode,(n,w) in laws.items():
            mu,q=generated_moments(n,w);p=w.softmax(1)
            geometry.append(dict(mode=mode,mean_sq_error=[float(a) for a in (mu-means).square().sum(1)],second_fro_sq_error=[float(a) for a in (q-second).square().sum((1,2))],generated_trace=[float(a) for a in q.diagonal(dim1=1,dim2=2).sum(1)],center_trace=[float(a) for a in (n.square().sum(2)*p).sum(1)],effective_nodes=[float(a) for a in 1/p.square().sum(1)]))
    params=[family(d,lock['adapter_rank'],lock['queries'],cfg['seed']+100+i) for i in range(4)];evaluation_start=time.monotonic()
    for i in range(2):
        p=[a.clone().requires_grad_() for a in params[i]];adv=torch.optim.Adam(p,lr=lock['query_lr'])
        for step in range(lock['evaluation_adversary_steps']):
            adv.zero_grad();target=grouped(losses(x,p),y,c)
            errors=[(target-memory_losses(a,b,p)).square().mean() for a,b in fixed.values()]
            for mode,(n,w) in laws.items():errors.append((target-memory_losses(decode_gaussian(n,cfg['seed']+5000+100*i+step,pairs[mode]),w,p)).square().mean())
            (-torch.stack(errors).mean()).backward();adv.step();constrain_family(p)
        params.append([a.detach() for a in p])
    rows=[];batches=[];predictions={mode:[] for mode in laws}
    def summary(mode,pred):
        e=(targets-pred).abs()
        return dict(mode=mode,mean_abs_error=float(e.mean()),max_abs_error=float(e.max()),random_max=float(e[:,:32].max()),stress_max=float(e[:,32:].max()),per_class_mean=[float(z) for z in e.mean(1)],per_class_max=[float(z) for z in e.max(1).values])
    with torch.no_grad():
        targets=torch.cat([grouped(losses(x,p),y,c) for p in params],1)
        for mode,(n,w) in fixed.items():rows.append(summary(mode,torch.cat([memory_losses(n,w,p) for p in params],1)))
        for draw in range(lock['audit_batches']):
            journal(dict(phase='frozen_query_batch_audit',completed_batches=draw))
            for mode,(n,w) in laws.items():
                fresh=decode_gaussian(n,cfg['seed']+7000+draw,pairs[mode]);pred=torch.cat([memory_losses(fresh,w,p) for p in params],1);predictions[mode].append(pred)
                a=summary(mode,pred);a.update(batch=draw,weighted_trace=[float(z) for z in (fresh.square().sum(2)*w.softmax(1)).sum(1)],mean_sq_error=[float(z) for z in ((fresh*w.softmax(1)[:,:,None]).sum(1)-means).square().sum(1)]);batches.append(a)
    assert all(sum(a['mode']==mode for a in batches)==lock['audit_batches'] for mode in laws)
    baseline={r['mode']:r for r in rows};passed=all(a['max_abs_error']<=.9*baseline[b]['max_abs_error'] and a['mean_abs_error']<=baseline[b]['mean_abs_error'] for a in batches if a['mode']=='generated_moment' for b in ('moment','random'))
    averaged=[summary(mode+'_eight_batch_average_DIAGNOSTIC_ONLY',torch.stack(pred).mean(0)) for mode,pred in predictions.items()]
    variance={mode:dict(mean=float(torch.stack(pred).var(0,unbiased=True).mean()),max=float(torch.stack(pred).var(0,unbiased=True).max()),per_class_mean=[float(z) for z in torch.stack(pred).var(0,unbiased=True).mean(1)]) for mode,pred in predictions.items()}
    assert all(torch.equal(a,b) for a,b in zip((n0,w0,n1,w1),frozen))
    return dict(status='COMPLETED',case=cfg,synthetic_counts=counts,rows=rows,batches=batches,averaged_prediction_diagnostic=averaged,prediction_variance_diagnostic=variance,finite_query_screen_pass=passed,
        fit_diagnostics=fit_diagnostics,analytic_geometry=geometry,every_batch_pass_required=True,centers_weights_unchanged_during_evaluation=True,per_arm_memory_updates=lock['outer_steps'],case_total_memory_updates=2*lock['outer_steps'],construction_wall_seconds=construction_seconds,pairing_seconds=pair_seconds,pair_cache_bytes_per_method=8*c*m,base_memory_bytes=8*(n0.numel()+w0.numel()),candidate_memory_bytes=8*(n1.numel()+w1.numel()),fresh_batch_bytes=8*n1.numel(),evaluation_wall_seconds=time.monotonic()-evaluation_start,query_parameters_fixed_before_audit=True,generated_law_not_true_distribution=True,CPU_pipeline_seconds=time.monotonic()-begun)


def run(cfg,lock,journal):
    torch.set_num_threads(lock['threads']);torch.set_default_dtype(DT);assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    a=unit_checks() if cfg['kind']=='unit' else run_scale(cfg,lock,journal)
    a.update(id=cfg['id'],GPU_initialized=torch.cuda.is_initialized(),medical_images=0,medical_checkpoints=0,val_accessed=False,test_accessed=False,formal_updates=0)
    assert not a['GPU_initialized'];return a
