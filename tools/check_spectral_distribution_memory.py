"""Reallocate the same scalar storage to a mean, principal covariance and isotropic tail."""
import os
import time
import torch
import torch.nn.functional as F
from check_progressive_decision_memory import DT,fixture,family,losses,grouped,memory_losses,moments_objective,project_ball,constrain_family


def spectral_fit(x,y,c,slots):
    d=x.shape[1];means=[];bases=[];values=[];tails=[];traces=[];diagnostics=[]
    for j in range(c):
        a=x[y==j];mu=a.mean(0);z=a-mu;cov=z.T@z/len(a);cov=(cov+cov.T)/2
        ev,u=torch.linalg.eigh(cov)
        assert float(ev.min())>=-1e-10 and bool(torch.isfinite(ev).all()), 'Invalid covariance spectrum'
        ev=ev.clamp_min(0);k=min(slots-1,d-1,len(a)-1)
        selected=ev[-k:] if k else ev[:0];basis=u[:,-k:] if k else u[:,:0]
        tail=((cov.trace()-selected.sum()).clamp_min(0)/(d-k));trace=cov.trace()
        pad_u=torch.zeros(d,slots-1,dtype=DT);pad_v=torch.zeros(slots-1,dtype=DT)
        if k:pad_u[:,:k]=basis;pad_v[:k]=selected
        reconstructed=tail*torch.eye(d,dtype=DT)+(basis*(selected-tail)[None])@basis.T
        eig_error=(ev[:d-k]-tail).square().sum();fro_error=(reconstructed-cov).square().sum()
        assert abs(float(eig_error-fro_error))<1e-9
        means.append(mu);bases.append(pad_u);values.append(pad_v);tails.append(tail);traces.append(trace)
        diagnostics.append(dict(class_index=j,retained_directions=k,covariance_trace=float(trace),retained_trace=float(selected.sum()),tail_trace=float(tail*(d-k)),raw_covariance_fro_sq_error=float(fro_error),isotropic_covariance_fro_sq_error=float((cov-trace/d*torch.eye(d,dtype=DT)).square().sum()),raw_second_trace=float(mu.square().sum()+reconstructed.trace()),negative_eigenvalue_clipping_only_numerical=True))
    return torch.stack(means),torch.stack(bases),torch.stack(values),torch.stack(tails),torch.stack(traces),diagnostics


def raw_decode(mu,basis,values,tail,trace,noise,mode):
    if mode=='spectral':
        projected=torch.einsum('cmd,cdk->cmk',noise,basis)
        residual=noise*tail.sqrt()[:,None,None]
        residual+=torch.einsum('cmk,cdk->cmd',projected*(values.sqrt()-tail.sqrt()[:,None])[:,None,:],basis)
    else:
        assert mode=='isotropic';residual=noise*(trace/noise.shape[-1]).sqrt()[:,None,None]
    return mu[:,None]+residual


def decode(mu,basis,values,tail,trace,seed,mode,points):
    g=torch.Generator().manual_seed(seed);noise=torch.randn(len(mu),points,mu.shape[1],generator=g,dtype=DT)
    noise[:,1::2]=-noise[:,0::2]
    raw=raw_decode(mu,basis,values,tail,trace,noise,mode)
    assert bool((raw.norm(dim=2)>1e-12).all()), 'Undefined normalized Gaussian draw'
    return F.normalize(raw,dim=2)


def unit_checks():
    eye=torch.eye(4,dtype=DT);x=torch.stack((eye[0],eye[0],-eye[0],eye[1],-eye[1],eye[2],-eye[2],eye[3]));y=torch.zeros(len(x),dtype=torch.long)
    mu,basis,values,tail,trace,diag=spectral_fit(x,y,1,2)
    cov=tail[0]*eye+(basis[0]*(values[0]-tail[0])[None])@basis[0].T
    cubature=torch.cat((2*eye,-2*eye),0)[None]
    raw=raw_decode(mu,basis,values,tail,trace,cubature,'spectral');center=raw-mu[:,None]
    mean_error=float((raw.mean(1)-mu).abs().max());cov_error=float((center[0].T@center[0]/8-cov).abs().max())
    assert max(mean_error,cov_error)<1e-12 and float(torch.linalg.eigvalsh(cov).min())>=-1e-12
    assert abs(float(cov.trace()-trace[0]))<1e-12
    a=spectral_fit(x,y,1,4);assert a[-1][0]['raw_covariance_fro_sq_error']<1e-24
    same=eye[0][None].repeat(3,1);b=spectral_fit(same,torch.zeros(3,dtype=torch.long),1,4)
    assert torch.equal(decode(*b[:5],303,'spectral',16),same[0][None,None].repeat(1,16,1))
    z=decode(mu,basis,values,tail,trace,303,'spectral',16)
    assert torch.equal(z,decode(mu,basis,values,tail,trace,303,'spectral',16)) and not torch.equal(z,decode(mu,basis,values,tail,trace,304,'spectral',16))
    permutation=torch.arange(len(x)-1,-1,-1);a2=spectral_fit(x[permutation],y,1,2)
    cov2=a2[3][0]*eye+(a2[1][0]*(a2[2][0]-a2[3][0])[None])@a2[1][0].T
    assert float((cov-cov2).abs().max())<1e-12
    p=[a.requires_grad_() for a in family(4,2,2,307)];w=torch.zeros(1,16,dtype=DT);value=memory_losses(z,w,p).mean();grad=torch.autograd.grad(value,p[0])[0]
    direction=torch.ones_like(p[0]);direction/=direction.norm();eps=1e-6
    def f(first):return memory_losses(z,w,[first,*p[1:]]).mean()
    fd=abs(float((f(p[0]+eps*direction)-f(p[0]-eps*direction))/(2*eps)-(grad*direction).sum()));assert fd<1e-7
    return dict(status='PASS',raw_mean_cubature_error=mean_error,raw_covariance_cubature_error=cov_error,query_gradient_fd_error=fd,raw_trace_preserved=True,full_rank_minus_one_covariance_exact=True,zero_covariance_exact=True,law_covariance_permutation_pass=True,normalized_moments_not_exact=True,optimizer_updates=0)


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
    fixed={'random':(init,zero),'moment':(n,w)};models=('isotropic','spectral');frozen=[a.clone() for a in (n,w,mu,basis,values,tail,trace)]
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
    base={a['mode']:a for a in rows};passed=all(a['max_abs_error']<=.9*base[b]['max_abs_error'] and a['mean_abs_error']<=base[b]['mean_abs_error'] for a in batches if a['mode']=='spectral' for b in ('moment','random'))
    avg=[summary(mode+'_eight_batch_average_DIAGNOSTIC_ONLY',torch.stack(pred).mean(0)) for mode,pred in predictions.items()]
    variance={mode:dict(mean=float(torch.stack(pred).var(0,unbiased=True).mean()),max=float(torch.stack(pred).var(0,unbiased=True).max()),per_class_mean=[float(a) for a in torch.stack(pred).var(0,unbiased=True).mean(1)]) for mode,pred in predictions.items()}
    assert all(torch.equal(a,b) for a,b in zip((n,w,mu,basis,values,tail,trace),frozen))
    candidate_bytes=8*(mu.numel()+basis.numel()+values.numel()+tail.numel());base_bytes=8*(n.numel()+w.numel());assert candidate_bytes==base_bytes
    return dict(status='COMPLETED',case=cfg,synthetic_counts=counts,rows=rows,batches=batches,averaged_prediction_diagnostic=avg,prediction_variance_diagnostic=variance,finite_query_screen_pass=passed,analytic_geometry=geometry,
        baseline_initial_objective=initial,baseline_final_objective=final,baseline_optimizer_steps=lock['outer_steps'],candidate_optimizer_steps=0,baseline_construction_seconds=baseline_seconds,spectral_construction_seconds=spectral_seconds,evaluation_seconds=time.monotonic()-eval_start,
        candidate_memory_bytes=candidate_bytes,base_memory_bytes=base_bytes,isotropic_memory_bytes=8*(mu.numel()+trace.numel()),fresh_batch_bytes=8*c*m*d,retained_slots=m-1,raw_mean_covariance_not_normalized_moments=True,memory_unchanged_during_evaluation=True,CPU_pipeline_seconds=time.monotonic()-begun)


def run(cfg,lock,journal):
    torch.set_num_threads(lock['threads']);torch.set_default_dtype(DT);assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    a=unit_checks() if cfg['kind']=='unit' else run_scale(cfg,lock,journal)
    a.update(id=cfg['id'],GPU_initialized=torch.cuda.is_initialized(),medical_images=0,medical_checkpoints=0,val_accessed=False,test_accessed=False,formal_updates=0);assert not a['GPU_initialized'];return a
