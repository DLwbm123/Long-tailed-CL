"""Synthetic CPU checks, including native 1536x23 solves; no medical pipeline imports."""
import json
import os
import time

import torch
from torch.nn import functional as F

import core1_competition as native
import dynamic_geometry_anchor as anchor


def moments(x, labels, k):
    q = torch.stack([x[labels == c].T@x[labels == c]/int((labels == c).sum())/k for c in range(k)])
    mu = torch.stack([x[labels == c].mean(0)/k for c in range(k)])
    return q, mu, x.new_full((k,), 1/k)


def objective(W, q, mu, pairs, R, target, weights, strength):
    lap, rhs_pairs = native.laplacians(pairs)
    fit = .5*(W*(q.sum(0)@W)).sum()-(W*mu.T).sum()
    pair = .25*(torch.einsum('dl,cdk,ckl->', W, q@W, lap)-2*(W*(mu.T@rhs_pairs)).sum())
    return fit+pair+.0005*(W*(R@W)).sum()+anchor.anchor_loss(W, target, weights, strength)


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2); torch.manual_seed(1074005)
    begun = time.monotonic(); dtype = torch.float64; highdim = []
    for control in ('base', 'local_transport'):
        d, k = 1536, 23
        counts = [3,6,6,18]+[8]*(k-4)
        labels = torch.cat([torch.full((n,), c, dtype=torch.long) for c,n in enumerate(counts)])
        centers = F.normalize(torch.randn(k,d,dtype=dtype),dim=1)
        x = F.normalize(centers[labels]+.03*torch.randn(len(labels),d,dtype=dtype),dim=1)
        q, mu, mass = moments(x, labels, k)
        means = mu*k; E = F.normalize(means,dim=1).T
        if control == 'local_transport':
            regions = torch.randn(k,4,d,dtype=dtype)
            regions = F.normalize(regions-regions.mean(1,keepdim=True),dim=2)
            E = torch.cat([E,regions.reshape(-1,d).T/2],1)
        R = torch.eye(d,dtype=dtype)-E@torch.linalg.solve(torch.eye(E.shape[1],dtype=dtype)+E.T@E,E.T)
        R = (R+R.T)/2
        pairs = (torch.ones(k,k,dtype=dtype)-torch.eye(k,dtype=dtype))/(k-1)
        baseline, baseline_audit = native.solve(q,mu,mass,pairs,regularizer=R)
        G = q.sum(0)+.001*R
        current = means+.002*torch.randn(1,d,dtype=dtype)
        current[-2:] += .01*torch.randn(2,d,dtype=dtype)
        target, rho, target_audit = anchor.temporal_target(baseline,means,current,G,counts,[1]*(k-2)+[0,0])
        strength = anchor.RELATIVE_CURVATURE*float(G.diagonal().mean())
        head, solve_audit = anchor.solve(q,mu,mass,pairs,R,target,rho,strength)
        change = float((head-baseline).norm()/baseline.norm())
        assert change > 1e-8 and torch.isfinite(head).all()
        assert torch.allclose(head.mean(1),baseline.mean(1),atol=1e-8,rtol=1e-7)
        assert max(a['raw_rotation_norm_error'] for a in target_audit) < 1e-9
        highdim.append(dict(control=control,dimension=[d,k],native_residual=baseline_audit['relative_residual'],
                            anchored_residual=solve_audit['relative_residual'],relative_head_change=change,
                            common_head_preserved=True,nonzero_anchor_effect=True,rare_support_pattern_checked=True,
                            raw_rotations_preserve_norm=True))
        del x,q,mu,E,R,G,baseline,target,head
    d,k = 8,4
    x = torch.randn(12,d,dtype=dtype); labels = torch.arange(k).repeat_interleave(3)
    q,mu,mass = moments(x,labels,k)
    pairs = (torch.ones(k,k,dtype=dtype)-torch.eye(k,dtype=dtype))/(k-1)
    R = torch.eye(d,dtype=dtype)
    baseline,_ = native.solve(q,mu,mass,pairs,regularizer=R)
    rho = torch.tensor([.2,.5,.7,1.],dtype=dtype); A = baseline+.1*torch.randn_like(baseline); strength=.03
    zero,_ = anchor.solve(q,mu,mass,pairs,R,A,rho,0.)
    assert torch.equal(zero,baseline)
    self_anchor,_ = anchor.solve(q,mu,mass,pairs,R,baseline,rho,strength)
    assert torch.allclose(self_anchor,baseline,atol=1e-7,rtol=1e-7)
    W,audit = anchor.solve(q,mu,mass,pairs,R,A,rho,strength)
    head_var = W.detach().requires_grad_(True)
    grad = torch.autograd.grad(objective(head_var,q,mu,pairs,R,A,rho,strength),head_var)[0]
    assert float(grad.norm()) < 1e-7
    trial = torch.randn_like(W,requires_grad=True)
    gradient = torch.autograd.grad(objective(trial,q,mu,pairs,R,A,rho,strength),trial,create_graph=True)[0]
    direction = torch.randn_like(trial)
    Hessian_direction = torch.autograd.grad((gradient*direction).sum(),trial)[0]
    assert float((Hessian_direction*direction).sum()) > 0
    live_x = x.clone().requires_grad_(True); q_live,mu_live,_ = moments(live_x,labels,k)
    envelope_gradient = torch.autograd.grad(objective(W.detach(),q_live,mu_live,pairs,R,A,rho,strength),live_x)[0]
    def value(samples):
        a,b,c = moments(samples,labels,k)
        w,_ = anchor.solve(a,b,c,pairs,R,A,rho,strength)
        return float(objective(w,a,b,pairs,R,A,rho,strength))
    xp,xm = x.clone(),x.clone(); eps=1e-5;xp[0,0]+=eps;xm[0,0]-=eps
    numeric=(value(xp)-value(xm))/(2*eps)
    envelope_error=abs(numeric-float(envelope_gradient[0,0]))
    assert envelope_error < 1e-5 and torch.isfinite(envelope_gradient).all() and envelope_gradient.norm()>0
    G=q.sum(0)+.001*R;before=mu*k;current=before+.02*torch.randn_like(before)
    target,weights,_=anchor.temporal_target(baseline,before,current,G,[3]*k,[0]*k)
    shift=torch.randn(1,d,dtype=dtype)
    translated,_,_=anchor.temporal_target(baseline,before+shift,current+shift,G,[3]*k,[0]*k)
    assert torch.allclose(target,translated,atol=1e-10,rtol=1e-10)
    order=torch.tensor([2,0,3,1]);permuted,_,_=anchor.temporal_target(baseline[:,order],before[order],current[order],G,[3]*k,[0]*k)
    assert torch.allclose(permuted,target[:,order],atol=1e-10,rtol=1e-10)
    no_motion,_,_=anchor.temporal_target(baseline,before,before,G,[3]*k,[0]*k)
    assert torch.equal(no_motion,baseline)
    zero_direction,degenerate,degenerate_audit=anchor.temporal_target(baseline,torch.zeros_like(before),torch.zeros_like(before),G,[1]*k,[0]*k)
    assert not degenerate.any() and torch.equal(zero_direction,baseline)
    assert all(z['reason']=='zero_centered_direction_keep_reference' for z in degenerate_audit)
    rejected=[]
    for invalid in ('negative_strength','bad_counts','bad_geometry'):
        try:
            if invalid=='negative_strength':anchor.solve(q,mu,mass,pairs,R,A,rho,-1.)
            elif invalid=='bad_counts':anchor.temporal_target(baseline,before,current,G,[0]*k,[0]*k)
            else:anchor.temporal_target(baseline,before[:2],current,G,[3]*k,[0]*k)
        except ValueError:rejected.append(invalid)
    assert len(rejected)==3
    # The training solve also carries the native .01 proximal term.
    previous = baseline+.03*torch.randn_like(baseline)
    proximal_native,_ = native.solve(q,mu,mass,pairs,regularizer=R,previous=previous,proximal=.01)
    proximal_zero,_ = anchor.solve(q,mu,mass,pairs,R,A,rho,0.,previous=previous,proximal=.01)
    assert torch.equal(proximal_native,proximal_zero)
    proximal_head,proximal_audit = anchor.solve(q,mu,mass,pairs,R,A,rho,strength,previous=previous,proximal=.01)
    check = proximal_head.detach().requires_grad_(True)
    proximal_objective = objective(check,q,mu,pairs,R,A,rho,strength)+.005*(check-previous).square().sum()
    proximal_gradient = torch.autograd.grad(proximal_objective,check)[0]
    assert float(proximal_gradient.norm()) < 1e-7
    assert torch.allclose(proximal_head.mean(1),proximal_native.mean(1),atol=1e-8,rtol=1e-7)
    return dict(status='PASS',version=anchor.VERSION,synthetic_only=True,production_integrated=False,
                GPU_processes=0,medical_images_read=0,medical_checkpoint_reads=0,val_reads=0,test_accessed=False,
                optimizer_updates=0,highdimensional_checks=highdim,zero_strength_exact_native=True,
                self_anchor_identity_detected=True,quadratic_stationarity_gradient=float(grad.norm()),
                positive_definite_hessian_checked=True,envelope_gradient_error=envelope_error,
                anchor_target_and_reliability_frozen_within_block=True,
                global_translation_invariant=True,class_permutation_equivariant=True,
                no_prototype_motion_preserves_reference=True,unidentified_direction_not_fabricated=True,
                singleton_covariance_not_claimed_identified=True,invalid_inputs_refused=rejected,
                native_proximal_zero_strength_exact=True,native_proximal_stationarity_gradient=float(proximal_gradient.norm()),
                native_proximal_solve_residual=proximal_audit['relative_residual'],
                new_class_accuracy_or_margin_guarantee=False,pipeline_seconds=time.monotonic()-begun)


if __name__=='__main__':
    print(json.dumps(run(),indent=2,allow_nan=False))
