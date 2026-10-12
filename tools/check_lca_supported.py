"""Synthetic CPU checks of a design-only module; no production data or workers."""
import copy
import io
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
import core1_competition as competition
import module_additions as modules
import prototype_coherent as native
import lca_supported as lca
from ml_statistics import empty, add, ridge


def rejected(call):
    try:
        call()
    except (ValueError, ArithmeticError):
        return
    raise AssertionError('Invalid input accepted')


def design_check(p):
    if (p['status']!='DESIGN_ONLY_NOT_GPU_AUTHORIZED' or p['GPU_launch_authorized']
        or p['formal_updates']!=0 or p['test_accessed'] or p['future_gpus']!=[0,1]
        or p['seed']!=74002 or p['epochs']!=2 or p['batch_size']!=64
        or p['stage_updates']!={'ISIC':[100,72,296],'HK':[98,4,2,20,18,34]}
        or p['full_future_new_updates']!=3220 or p['full_future_new_trajectories']!=10
        or p['relative_curvature_cap']!=.1 or p['affine_map_ridge']!=.001):
        raise ValueError('Design scope or proposed budget mismatch')


def run():
    start = time.monotonic(); torch.set_num_threads(1); torch.manual_seed(74002)
    assert not torch.cuda.is_initialized()
    x = torch.randn(30, 5, dtype=torch.float64)
    y = torch.arange(3).repeat_interleave(10)
    bank = native.append(native.empty(5), x, y, torch.zeros(30, dtype=torch.long), x.new_full((30,), .1))
    saved = copy.deepcopy(bank)
    weights = x.new_full((30,), 1/30)
    shift = x.new_tensor([.1, -.05, .01, .03, -.02])
    B = .03 * torch.randn(5, 5, dtype=x.dtype)
    residual = (x-x.mean(0)) @ B
    after = x + shift + residual
    old_mu, old_Q = bank['mu'][:2], bank['Q'][:2]
    U, estimate = lca.estimate(x, after, weights, shift, old_mu, old_Q)
    C = (x-x.mean(0)).T @ (x-x.mean(0)) / len(x)
    fitted = torch.linalg.solve(C + .001*torch.eye(5,dtype=x.dtype), C@B)
    oracle = torch.stack([((x[y==c]-x.mean(0))@fitted).T @ ((x[y==c]-x.mean(0))@fitted) / 10 for c in (0,1)]).mean(0)
    assert torch.allclose(U, oracle, atol=1e-12, rtol=1e-10)
    zero, _ = lca.estimate(x, x+shift, weights, shift, old_mu, old_Q)
    assert zero.norm() < 1e-28
    no_old, _ = lca.estimate(x, after, weights, shift, old_mu[:0], old_Q[:0])
    assert not no_old.any()
    # Current fit support is one-dimensional; unsupported old variation is reported, not measured.
    restricted = x.clone(); restricted[:, 1:] = 0
    _, restricted_audit = lca.estimate(restricted, restricted, weights, x.new_zeros(5), old_mu, old_Q)
    assert restricted_audit['current_support_rank']==1
    assert all(0 < value < 1 for value in restricted_audit['old_supported_energy_fractions'])
    rejected(lambda: lca.estimate(x, after, -weights, shift, old_mu, old_Q))
    rejected(lambda: lca.estimate(x, after, weights, shift, old_mu, -old_Q))
    rejected(lambda: lca.psd(torch.full((5,5),float('nan'),dtype=x.dtype)))
    # A shared local-relaxation direction must be removed without rescaling the retained risk.
    R0 = torch.eye(5,dtype=x.dtype); RL = R0.clone(); RL[0,0] = .5
    filtered, dedup = lca.remove_local_prior_span(U,R0,RL)
    P = R0.clone(); P[0,0] = 0
    assert torch.allclose(filtered, P@U@P, atol=1e-12,rtol=1e-10)
    assert filtered.trace() <= U.trace()+1e-12 and dedup['overlap_leakage']<1e-10
    again, _ = lca.remove_local_prior_span(filtered,R0,RL)
    assert torch.allclose(filtered,again,atol=1e-12,rtol=1e-10)
    unchanged, _ = lca.remove_local_prior_span(U,R0,R0)
    assert torch.allclose(unchanged,U,atol=1e-12)
    all_removed, _ = lca.remove_local_prior_span(R0,R0,.5*R0)
    assert not all_removed.any()
    maximum=0.; curvature_bounds=[]
    for arm in ('base','local_transport'):
        local = torch.randn(3,4,5,dtype=x.dtype) if arm=='local_transport' else None
        ref,R,_ = modules.head(bank,arm,'ISIC',[0,1,2],local)
        original,_ = competition.bank_head(bank,competition.competition(bank,ref),regularizer=R)
        pairs=competition.competition(bank,ref)
        q=bank['Q']/3; mu=bank['mu']/3; mass=x.new_full((3,),1/3)
        base= q.sum(0)+.001*R
        S,audit=lca.bounded_penalty(filtered,base,raw_risk=U)
        raw_S,_=lca.bounded_penalty(U,base)
        assert S.trace()<=raw_S.trace()+1e-12
        assert torch.linalg.eigvalsh(.1*base-S).min() > -1e-12
        curvature_bounds.append(audit['actual_relative_bound'])
        M=lca.old_margin_metric(3,2,x)
        assert not M[:,2].any() and torch.allclose(M@x.new_ones(3),x.new_zeros(3))
        rejected(lambda:lca.old_margin_metric(3,4,x))
        reduced,_=lca.solve(q,mu,mass,pairs,R,S*0,M)
        assert torch.equal(reduced,original)
        no_margin,_=lca.solve(q,mu,mass,pairs,R,S,lca.old_margin_metric(3,1,x))
        assert torch.equal(no_margin,original)
        for beta in (0.,.5):
            solved,a=lca.solve(q,mu,mass,pairs,R,S,M,beta=beta)
            def objective(w):
                return (.5*(x@w-F.one_hot(y,3).to(x)).square().sum(1).mean()
                    +.0005*(w*(R@w)).sum()
                    +beta*competition.pair_loss(native.empty(5),x,y,weights,w,pairs)
                    +lca.loss(w,S,M))
            z=torch.zeros_like(ref,requires_grad=True)
            H=torch.autograd.functional.hessian(objective,z).reshape(15,15)
            rhs=-torch.autograd.grad(objective(z),z)[0].flatten()
            oracle_w=torch.linalg.solve(H,rhs).reshape_as(ref)
            assert torch.allclose(solved,oracle_w,atol=1e-8,rtol=1e-8)
            value=solved.detach().requires_grad_()
            assert torch.autograd.grad(objective(value),value)[0].norm()<1e-8
            g=torch.autograd.grad(lca.loss(value,S,M),value)[0]
            assert not g[:,2].any()
            if not beta:
                plain,_=competition.solve(q,mu,mass,pairs,beta=0,regularizer=R)
                assert torch.allclose(solved[:,2],plain[:,2],atol=1e-10)
            maximum=max(maximum,a['relative_residual'])
        old=dict(mu=bank['mu'][:2],Q=bank['Q'][:2],n=bank['n'][:2],components=[c for c in bank['components'] if c['label']<2])
        bx=x[20:]; by=y[20:]; alpha=x.new_full((10,),1/30)
        bq,bmu,bmass=competition.moments(old,bx,by,alpha,3)
        inv,cross=native.proximal_base(old,R,3)
        initial=native.proximal_head(bx,F.one_hot(by,3).to(x),alpha,inv,cross,ref)
        fast,fa=lca.solve(bq,bmu,bmass,pairs,R,S,M,previous=ref,proximal=.01,
            inverse=inv,x=bx,alpha=alpha,native=initial)
        direct,da=lca.solve(bq,bmu,bmass,pairs,R,S,M,previous=ref,proximal=.01)
        assert torch.allclose(fast,direct,atol=1e-8,rtol=1e-8)
        maximum=max(maximum,fa['relative_residual'],da['relative_residual'])
        # Frozen S/M/pairs: envelope gradient through features matches a freshly solved objective.
        def optimized(features):
            nq=torch.stack([features[y==c].T@features[y==c]/30 for c in range(3)])
            nm=torch.stack([features[y==c].sum(0)/30 for c in range(3)])
            w,_=lca.solve(nq.detach(),nm.detach(),mass,pairs,R,S,M);w=w.detach()
            return (.5*(features@w-F.one_hot(y,3).to(x)).square().sum(1).mean()
                +.0005*(w*(R@w)).sum()+.5*competition.pair_loss(native.empty(5),features,y,weights,w,pairs)+lca.loss(w,S,M))
        var=x.clone().requires_grad_();grad=torch.autograd.grad(optimized(var),var)[0][2,1]
        plus=x.clone();minus=x.clone();plus[2,1]+=1e-5;minus[2,1]-=1e-5
        assert torch.allclose(grad,(optimized(plus)-optimized(minus))/2e-5,atol=2e-6,rtol=2e-5)
        buffer=io.BytesIO();torch.save(lca.checkpoint(S,M),buffer);buffer.seek(0)
        state=torch.load(buffer,map_location='cpu',weights_only=False)
        rs,rm=lca.restore(state,5,3,x)
        rebuilt,_=lca.solve(q,mu,mass,pairs,R,rs,rm)
        assert torch.allclose(rebuilt,solved,atol=1e-10,rtol=1e-10)
        rejected(lambda:lca.restore(dict(state,version='other'),5,3,x))
    # Binary: old labels only, no inter-label competition/centering and no added bias penalty.
    labels=np.array([[i%2,(i//2)%2,1] for i in range(30)],dtype=float)
    mask=np.ones_like(labels,dtype=bool);mask[:,2]=False;labels[:,2]=999
    binary=add(empty(5,3),x.numpy(),labels,mask);binary_saved=copy.deepcopy(binary)
    prior=torch.eye(6,dtype=x.dtype)
    penalties=[lca.binary_prior(prior,S,j==0).numpy() for j in range(3)]
    bw,ba=ridge(binary,penalty=lambda j:penalties[j]);plain,_=ridge(binary)
    assert np.array_equal(bw[:,1:],plain[:,1:]) and not bw[:,2].any()
    assert penalties[0][-1,-1]==1 and not penalties[0][-1,:-1].any()
    for k in binary:assert np.array_equal(binary[k],binary_saved[k])
    X=np.c_[x.numpy(),np.ones(len(x))]
    A=sum(X[labels[:,0]==s].T@X[labels[:,0]==s]/(2*np.sum(labels[:,0]==s)) for s in (0,1))+.001*penalties[0]
    b=sum(X[labels[:,0]==s].sum(0)*(2*s-1)/(2*np.sum(labels[:,0]==s)) for s in (0,1))
    assert np.allclose(bw[:,0],np.linalg.solve(A,b),atol=1e-10)
    for k in ('mu','Q'):assert torch.equal(bank[k],saved[k])
    assert bank['n']==saved['n']
    protocol=json.loads((Path(__file__).resolve().parents[1]/'docs/lca_supported_design_20261012/PROTOCOL_DESIGN.json').read_text())
    design_check(protocol)
    for key,value in [('GPU_launch_authorized',True),('future_gpus',[0,2]),('full_future_new_updates',3221),('test_accessed',True)]:
        rejected(lambda:design_check(dict(protocol,**{key:value})))
    # Real global feature width, synthetic moments only; no ViT or pretrained weights.
    dim=1536; classes=3
    fq=torch.eye(dim,dtype=x.dtype).repeat(classes,1,1)/classes
    fm=torch.randn(classes,dim,dtype=x.dtype)/dim
    fp=(torch.ones(classes,classes,dtype=x.dtype)-torch.eye(classes,dtype=x.dtype))/(classes-1)
    direction=torch.randn(dim,1,dtype=x.dtype)/dim
    fs=direction@direction.T
    full,full_a=lca.solve(fq,fm,x.new_full((classes,),1/classes),fp,
        torch.eye(dim,dtype=x.dtype),fs,lca.old_margin_metric(classes,2,x))
    assert full.shape==(1536,3) and torch.isfinite(full).all()
    maximum=max(maximum,full_a['relative_residual'])
    # HK output expansion and exact zero force on all ten newly introduced columns.
    hq=torch.eye(5,dtype=x.dtype).repeat(23,1,1)/23;hm=torch.randn(23,5,dtype=x.dtype)/23
    hp=(torch.ones(23,23,dtype=x.dtype)-torch.eye(23,dtype=x.dtype))/22
    hw,ha=lca.solve(hq,hm,x.new_full((23,),1/23),hp,torch.eye(5,dtype=x.dtype),S,
        lca.old_margin_metric(23,13,x))
    assert hw.shape==(5,23) and ha['explicit_new_column_force_norm']==0
    maximum=max(maximum,ha['relative_residual'])
    assert not torch.cuda.is_initialized()
    return dict(status='PASS',cpu_pipeline_seconds=time.monotonic()-start,gpu_work=False,
        medical_images_or_labels_read=False,test_accessed=False,formal_optimizer_updates=0,
        synthetic_optimizer_updates=0,old_drift_proxy_moment_oracle=True,
        unsupported_old_energy_reported=True,pure_translation_zero=True,no_old_task_zero=True,
        local_relaxation_overlap_removed=True,no_energy_renormalization=True,
        curvature_relative_bounds=curvature_bounds,native_zero_reduction_exact=True,
        competition_hessian_oracle=True,proximal_Woodbury_operator_consistent=True,
        frozen_prior_envelope_gradient=True,new_column_direct_force_zero=True,
        new_column_solution_unchanged_only_when_competition_zero=True,
        new_recall_guarantee=False,checkpoint_roundtrip=True,wrong_version_rejected=True,
        binary_signed_rhs_sign_counts_unknown_and_bias_preserved=True,
        existing_statistics_unchanged=True,max_solve_residual=maximum,
        global_dimension_1536_synthetic_solve=True,HK_23_output_old_new_partition=True,
        design_budget_and_no_launch_scope_checked=True,
        production_worker_integration=False,real_GPU_preflights=0)


if __name__=='__main__':print(json.dumps(run(),indent=2,allow_nan=False))
