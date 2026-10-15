"""Hold raw spectral moments fixed while replacing Gaussian block radii by fixed radii."""
import os
import time
import torch
import torch.nn.functional as F
from check_progressive_decision_memory import DT,fixture,family,losses,grouped,memory_losses,moments_objective,project_ball,constrain_family
from check_spectral_distribution_memory import spectral_fit, raw_decode


def bounded_raw(mu,basis,values,tail,noise):
    out=[];d=noise.shape[-1]
    for j in range(len(mu)):
        active=basis[j].square().sum(0)>.5;v=basis[j][:,active];ev=values[j,active];k=v.shape[1]
        g=noise[j];p=g@v;residual=torch.zeros_like(g)
        if k and bool((ev>0).any()):
            norm=p.norm(dim=1,keepdim=True);assert bool((norm>1e-12).all()), 'Degenerate retained projection'
            residual+=((p/norm)*(k**.5)*ev.sqrt())@v.T
        if float(tail[j])>0:
            perp=g-p@v.T;norm=perp.norm(dim=1,keepdim=True)
            assert bool((norm>1e-12).all()), 'Degenerate tail projection'
            residual+=perp/norm*((d-k)*tail[j]).sqrt()
        out.append(mu[j]+residual)
    return torch.stack(out)


def decode(mu,basis,values,tail,trace,seed,mode,points):
    g=torch.Generator().manual_seed(seed);noise=torch.randn(len(mu),points,mu.shape[1],generator=g,dtype=DT);noise[:,1::2]=-noise[:,0::2]
    if mode=='gaussian':raw=raw_decode(mu,basis,values,tail,trace,noise,'spectral')
    else:
        assert mode=='block_spherical';raw=bounded_raw(mu,basis,values,tail,noise)
    assert bool((raw.norm(dim=2)>1e-12).all()), 'Undefined normalized draw'
    return F.normalize(raw,dim=2)


def unit_checks():
    eye=torch.eye(4,dtype=DT);mu=torch.tensor([[.2,0,0,0]],dtype=DT);basis=eye[:,1:3][None];ev=torch.tensor([[.3,.1]],dtype=DT);tail=torch.tensor([.05],dtype=DT);trace=ev.sum(1)+2*tail
    # Independent signed coordinate cubature on each block checks only raw first/second moments.
    probes=torch.stack([s*eye[i]+t*eye[j] for i in (1,2) for j in (0,3) for s in (-1,1) for t in (-1,1)])[None]
    raw=bounded_raw(mu,basis,ev,tail,probes);z=raw[0]-mu[0];cov=tail[0]*eye+(basis[0]*(ev[0]-tail[0])[None])@basis[0].T
    me=float((raw.mean(1)-mu).abs().max());ce=float((z.T@z/len(z)-cov).abs().max());assert max(me,ce)<1e-12
    symmetry=float((bounded_raw(mu,basis,ev,tail,-probes)+raw-2*mu[:,None]).abs().max());assert symmetry<1e-12
    # Rank-one equal-weight unit two-point law is recovered; no claim for general class laws.
    m=torch.tensor([[.6,0,0,0]],dtype=DT);b=eye[:,1:2][None];e=torch.tensor([[.64]],dtype=DT);t=torch.zeros(1,dtype=DT)
    two=decode(m,b,e,t,e.sum(1),311,'block_spherical',16)
    assert float((two.mean(1)-m).abs().max())<1e-12 and float((two[:,:,1].abs()-.8).abs().max())<1e-12
    zero=decode(eye[:1],basis,ev*0,tail*0,trace*0,311,'block_spherical',16)
    assert torch.equal(zero,eye[0][None,None].repeat(1,16,1))
    sample=decode(mu,basis,ev,tail,trace,311,'block_spherical',16)
    assert torch.equal(sample,decode(mu,basis,ev,tail,trace,311,'block_spherical',16)) and not torch.equal(sample,decode(mu,basis,ev,tail,trace,312,'block_spherical',16))
    rejects=0
    for bad in (torch.zeros(1,2,4,dtype=DT),eye[1][None,None].repeat(1,2,1)):
        try:bounded_raw(mu,basis,ev,tail,bad)
        except AssertionError:rejects+=1
    assert rejects==2
    p=[a.requires_grad_() for a in family(4,2,2,311)];w=torch.zeros(1,16,dtype=DT);val=memory_losses(sample,w,p).mean();grad=torch.autograd.grad(val,p[0])[0];direction=torch.ones_like(p[0]);direction/=direction.norm();eps=1e-6
    def f(first):return memory_losses(sample,w,[first,*p[1:]]).mean()
    fd=abs(float((f(p[0]+eps*direction)-f(p[0]-eps*direction))/(2*eps)-(grad*direction).sum()));assert fd<1e-7
    return dict(status='PASS',raw_mean_cubature_error=me,raw_covariance_cubature_error=ce,symmetry_error=symmetry,query_gradient_fd_error=fd,rank_one_two_point_exact=True,zero_covariance_exact=True,degenerate_noise_rejections=rejects,normalized_moments_not_exact=True,optimizer_updates=0)


def run_scale(cfg,lock,journal):
    begun=time.monotonic();x,y,counts=fixture(cfg);c,d=cfg['classes'],cfg['dimension'];m=lock['nodes_per_class']
    means=torch.stack([x[y==j].mean(0) for j in range(c)]);second=torch.stack([x[y==j].T@x[y==j]/counts[j] for j in range(c)])
    g=torch.Generator().manual_seed(cfg['seed']+1);init=torch.stack([x[y==j][torch.randint(counts[j],(m,),generator=g)] for j in range(c)])
    zero=torch.zeros(c,m,dtype=DT);n=init.clone().requires_grad_();w=zero.clone().requires_grad_();opt=torch.optim.Adam([n,w],lr=lock['memory_lr']);fit_start=time.monotonic()
    initial=float(moments_objective(n,w,means,second).detach())
    for step in range(lock['outer_steps']):
        opt.zero_grad();moments_objective(n,w,means,second).backward();opt.step()
        with torch.no_grad():n.copy_(project_ball(n));w.clamp_(-8,8)
        if step%8==0:journal(dict(phase='original_moment_baseline',step=step+1))
    baseline_seconds=time.monotonic()-fit_start;final=float(moments_objective(n,w,means,second).detach());n=n.detach();w=w.detach()
    journal(dict(phase='spectral_construction'));fit_start=time.monotonic();mu,basis,values,tail,trace,geometry=spectral_fit(x,y,c,m);spectral_seconds=time.monotonic()-fit_start
    fixed={'random':(init,zero),'moment':(n,w)};models=('gaussian','block_spherical');frozen=[a.clone() for a in (n,w,mu,basis,values,tail,trace)]
    params=[family(d,lock['adapter_rank'],lock['queries'],cfg['seed']+100+i) for i in range(4)];eval_start=time.monotonic()
    for i in range(2):
        p=[a.clone().requires_grad_() for a in params[i]];adv=torch.optim.Adam(p,lr=lock['query_lr'])
        for step in range(lock['evaluation_adversary_steps']):
            adv.zero_grad();target=grouped(losses(x,p),y,c);errors=[(target-memory_losses(a,b,p)).square().mean() for a,b in fixed.values()]
            for mode in models:errors.append((target-memory_losses(decode(mu,basis,values,tail,trace,cfg['seed']+5000+100*i+step,mode,m),zero,p)).square().mean())
            (-torch.stack(errors).mean()).backward();adv.step();constrain_family(p)
        params.append([a.detach() for a in p])
    rows=[];batches=[];predictions={mode:[] for mode in models}
    def summary(mode,pred):
        e=(targets-pred).abs();return dict(mode=mode,mean_abs_error=float(e.mean()),max_abs_error=float(e.max()),random_max=float(e[:,:32].max()),stress_max=float(e[:,32:].max()),per_class_mean=[float(z) for z in e.mean(1)],per_class_max=[float(z) for z in e.max(1).values])
    with torch.no_grad():
        targets=torch.cat([grouped(losses(x,p),y,c) for p in params],1)
        for mode,(a,b) in fixed.items():rows.append(summary(mode,torch.cat([memory_losses(a,b,p) for p in params],1)))
        for draw in range(lock['audit_batches']):
            journal(dict(phase='frozen_query_fresh_batches',completed_batches=draw))
            for mode in models:
                z=decode(mu,basis,values,tail,trace,cfg['seed']+7000+draw,mode,m);pred=torch.cat([memory_losses(z,zero,p) for p in params],1);predictions[mode].append(pred)
                a=summary(mode,pred);a.update(batch=draw,weighted_trace=[float(t) for t in z.square().sum(2).mean(1)],mean_sq_error=[float(t) for t in (z.mean(1)-means).square().sum(1)]);batches.append(a)
    assert all(sum(b['mode']==mode for b in batches)==8 for mode in models)
    base={a['mode']:a for a in rows};passed=all(a['max_abs_error']<=.9*base[b]['max_abs_error'] and a['mean_abs_error']<=base[b]['mean_abs_error'] for a in batches if a['mode']=='block_spherical' for b in ('moment','random'))
    avg=[summary(mode+'_eight_batch_average_DIAGNOSTIC_ONLY',torch.stack(pred).mean(0)) for mode,pred in predictions.items()]
    variance={mode:dict(mean=float(torch.stack(pred).var(0,unbiased=True).mean()),max=float(torch.stack(pred).var(0,unbiased=True).max()),per_class_mean=[float(a) for a in torch.stack(pred).var(0,unbiased=True).mean(1)]) for mode,pred in predictions.items()}
    assert all(torch.equal(a,b) for a,b in zip((n,w,mu,basis,values,tail,trace),frozen))
    candidate_bytes=8*(mu.numel()+basis.numel()+values.numel()+tail.numel());base_bytes=8*(n.numel()+w.numel());assert candidate_bytes==base_bytes
    return dict(status='COMPLETED',case=cfg,synthetic_counts=counts,rows=rows,batches=batches,averaged_prediction_diagnostic=avg,prediction_variance_diagnostic=variance,finite_query_screen_pass=passed,analytic_geometry=geometry,
        baseline_initial_objective=initial,baseline_final_objective=final,baseline_optimizer_steps=lock['outer_steps'],candidate_optimizer_steps=0,baseline_construction_seconds=baseline_seconds,spectral_construction_seconds=spectral_seconds,evaluation_seconds=time.monotonic()-eval_start,
        candidate_memory_bytes=candidate_bytes,base_memory_bytes=base_bytes,gaussian_memory_bytes=candidate_bytes,raw_covariance_shared_between_generators=True,fresh_batch_bytes=8*c*m*d,retained_slots=m-1,raw_mean_covariance_not_normalized_moments=True,memory_unchanged_during_evaluation=True,CPU_pipeline_seconds=time.monotonic()-begun)


def run(cfg,lock,journal):
    torch.set_num_threads(lock['threads']);torch.set_default_dtype(DT);assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    a=unit_checks() if cfg['kind']=='unit' else run_scale(cfg,lock,journal)
    a.update(id=cfg['id'],GPU_initialized=torch.cuda.is_initialized(),medical_images=0,medical_checkpoints=0,val_accessed=False,test_accessed=False,formal_updates=0);assert not a['GPU_initialized'];return a
