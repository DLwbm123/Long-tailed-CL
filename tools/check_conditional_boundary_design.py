"""Synthetic arithmetic checks and explicit scientific counterexamples; no data reads."""
import copy
import io
import json
import os
import time

import torch
from torch.nn import functional as F

import conditional_boundary_design as design
import core1_competition as native


def moments(x,y,k):
    q=torch.stack([x[y==c].T@x[y==c]/int((y==c).sum())/k for c in range(k)])
    mu=torch.stack([x[y==c].mean(0)/k for c in range(k)])
    return q,mu,x.new_full((k,),1/k)


def objective(w,q,mu,pairs,R,G,graph,strength):
    lap,targets=native.laplacians(pairs)
    return (.5*(w*(q.sum(0)@w)).sum()-(w*mu.T).sum()
            +.25*(torch.einsum('dl,cdk,ckl->',w,q@w,lap)-2*(w*(mu.T@targets)).sum())
            +.0005*(w*(R@w)).sum()+design.penalty(w,G,graph,strength))


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2);torch.manual_seed(1074005)
    start=time.monotonic();dtype=torch.float64;high=[]
    for control in ('base','local_transport'):
        d,k=1536,23;counts=[3,6,6,18]+[8]*19
        y=torch.cat([torch.full((n,),c,dtype=torch.long) for c,n in enumerate(counts)])
        centers=F.normalize(torch.randn(k,d,dtype=dtype),dim=1)
        x=F.normalize(centers[y]+.03*torch.randn(len(y),d,dtype=dtype),dim=1)
        q,mu,mass=moments(x,y,k);R=torch.eye(d,dtype=dtype)
        if control=='local_transport':
            E=F.normalize(torch.randn(d,4*k,dtype=dtype),dim=0)/2
            R-=E@torch.linalg.solve(torch.eye(4*k,dtype=dtype)+E.T@E,E.T);R=(R+R.T)/2
        pairs=(torch.ones(k,k,dtype=dtype)-torch.eye(k,dtype=dtype))/(k-1)
        base,ba=native.solve(q,mu,mass,pairs,regularizer=R);G=q.sum(0)+.001*R
        ref=base.clone();ref[:,[0,k-1]]=ref[:,[k-1,0]]
        current=y>=k-2;g=design.evidence(x[current],y[current],ref,G,[k-2,k-1])
        assert g['L'].any() and all(r['label']>=k-2 for r in g['rows'])
        w,wa=design.solve(q,mu,mass,pairs,R,G,g,.1)
        zero,za=design.solve(q,mu,mass,pairs,R,G,g,0.)
        assert torch.equal(zero,base)
        assert torch.allclose(w.mean(1),base.mean(1),atol=1e-8,rtol=1e-7)
        change=float((w-base).norm()/base.norm());assert change>1e-8
        v=w.detach().requires_grad_(True)
        stationarity=float(torch.autograd.grad(objective(v,q,mu,pairs,R,G,g,.1),v)[0].norm())
        assert stationarity<1e-7
        high.append(dict(control=control,dimension=[d,k],native_residual=ba['relative_residual'],
                         graph_residual=wa['relative_residual'],stationarity=stationarity,
                         relative_head_change=change,zero_strength_exact=True,common_head_preserved=True,
                         rare_support_fixture=True,old_old_edges_invented=False))
        del q,mu,mass,R,G,base,ref,w,zero,v,g,x
    d,k=6,4;x=torch.randn(24,d,dtype=dtype);y=torch.arange(k).repeat_interleave(6)
    q,mu,mass=moments(x,y,k);R=torch.eye(d,dtype=dtype);G=q.sum(0)+.001*R
    pairs=(torch.ones(k,k,dtype=dtype)-torch.eye(k,dtype=dtype))/(k-1)
    ref=torch.randn(d,k,dtype=dtype);mask=y>=2
    g=design.evidence(x[mask],y[mask],ref,G,[2,3]);assert g['L'].any()
    assert max(abs(r['target_margin']-1) for r in g['rows'] if r['identified'])<1e-9
    shift=torch.randn(d,1,dtype=dtype)
    shifted=design.evidence(x[mask],y[mask],ref+shift,G,[2,3])
    assert torch.allclose(g['L'],shifted['L'],atol=1e-12) and torch.allclose(g['T'],shifted['T'],atol=1e-10)
    order=torch.tensor([2,0,3,1]);inverse=torch.argsort(order)
    perm=design.evidence(x[mask],inverse[y[mask]],ref[:,order],G,sorted(inverse[torch.tensor([2,3])].tolist()))
    assert torch.allclose(perm['L'],g['L'][order][:,order],atol=1e-12)
    assert torch.allclose(perm['T'],g['T'][:,order],atol=1e-10)
    w,a=design.solve(q,mu,mass,pairs,R,G,g,.1)
    trial=torch.randn_like(w,requires_grad=True);direction=torch.randn_like(w)
    grad=torch.autograd.grad(objective(trial,q,mu,pairs,R,G,g,.1),trial,create_graph=True)[0]
    hv=torch.autograd.grad((grad*direction).sum(),trial)[0]
    assert float((hv*direction).sum())>0
    live=x.clone().requires_grad_(True);ql,ml,_=moments(live,y,k)
    derivative=torch.autograd.grad(objective(w.detach(),ql,ml,pairs,R,G,g,.1),live)[0][0,0]
    def value(z):
        a,b,c=moments(z,y,k);head,_=design.solve(a,b,c,pairs,R,G,g,.1)
        return float(objective(head,a,b,pairs,R,G,g,.1))
    eps=1e-5;xp=x.clone();xm=x.clone();xp[0,0]+=eps;xm[0,0]-=eps
    error=abs((value(xp)-value(xm))/(2*eps)-float(derivative));assert error<1e-6
    previous=ref.clone();pw,pa=design.solve(q,mu,mass,pairs,R,G,g,.1,previous=previous,proximal=.01)
    check=pw.detach().requires_grad_(True)
    pg=torch.autograd.grad(objective(check,q,mu,pairs,R,G,g,.1)+.005*(check-previous).square().sum(),check)[0]
    assert float(pg.norm())<1e-7
    buffer=io.BytesIO();torch.save(design.checkpoint(g),buffer);buffer.seek(0)
    saved=torch.load(buffer,weights_only=False);restored=design.restore(saved,w)
    wr,_=design.solve(q,mu,mass,pairs,R,G,restored,.1)
    assert torch.equal(w,wr)
    rejected=[]
    for bad in ('old_labels','negative_strength','invalid_graph','version'):
        try:
            if bad=='old_labels':design.evidence(x,y,ref,G,[2,3])
            elif bad=='negative_strength':design.solve(q,mu,mass,pairs,R,G,g,-.1)
            elif bad=='invalid_graph':
                broken=copy.deepcopy(g);broken['L'][0,0]-=10;design.restore(broken,w)
            else:
                broken=copy.deepcopy(g);broken['version']='wrong';design.restore(broken,w)
        except ValueError:rejected.append(bad)
    assert len(rejected)==4
    # Exact indistinguishable saved n/mu/Q, but different old damage for one fixed move.
    root3=3**.5
    a=torch.tensor([-1.,-1.,1.,1.],dtype=dtype)
    b=torch.tensor([-root3,1/root3,1/root3,1/root3],dtype=dtype)
    assert torch.allclose(a.mean(),b.mean(),atol=1e-12) and torch.allclose(a.square().mean(),b.square().mean(),atol=1e-12)
    damage_a=int(((a>0)&(a-.7<0)).sum());damage_b=int(((b>0)&(b-.7<0)).sum())
    assert (damage_a,damage_b)==(0,3)
    rescued_a=int(((a<0)&(a+.7>0)).sum());rescued_b=int(((-b<0)&(-b+.7>0)).sum())
    assert (rescued_a,rescued_b)==(0,3)
    correct=torch.tensor([.2,2.],dtype=dtype);moved=torch.tensor([-.1,2.3],dtype=dtype)
    assert torch.allclose(correct.mean(),moved.mean()) and int((moved<0).sum())==1
    return dict(status='CPU_ARITHMETIC_PASS_DESIGN_NOT_READY',version=design.VERSION,
                synthetic_only=True,production_integrated=False,GPU_ready=False,GPU_processes=0,
                medical_images_read=0,medical_checkpoint_reads=0,val_reads=0,test_accessed=False,
                optimizer_updates=0,highdimensional_checks=high,PSD_and_stationarity=True,
                finite_difference_error=error,native_proximal_gradient=float(pg.norm()),
                common_logit_shift_invariant=True,class_permutation_equivariant=True,
                checkpoint_roundtrip_exact=True,invalid_inputs_refused=rejected,
                observed_error_target_margin=1.,old_old_error_identified=False,
                scientific_counterexamples=dict(same_old_n_mu_Q_different_damage=[damage_a,damage_b],
                    same_old_n_mu_Q_different_rescue=[rescued_a,rescued_b],
                    preserved_correct_mean_margin_individual_damage=1),
                rejected_claims=['old risk identified from first two moments','mean guard ensures no individual damage',
                                 'current-label errors establish old-old correction'],
                next_GPU_experiment_admitted=False,pipeline_seconds=time.monotonic()-start)


if __name__=='__main__':
    print(json.dumps(run(),allow_nan=False),flush=True)
