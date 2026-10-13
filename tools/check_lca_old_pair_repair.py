"""Synthetic CPU math/compatibility checks; no medical loader or experiment entry."""
import copy
import io
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
import core1_competition as competition
import module_additions as modules
import lca_old_pair_repair as repair


def rejected(call):
    try:call()
    except (ValueError, ArithmeticError, torch.linalg.LinAlgError):return
    raise AssertionError('Invalid input accepted')


def fixture(d=7, k=5):
    ns = [3, 5, 7, 9, 11][:k]
    x = torch.randn(sum(ns), d, dtype=torch.float64)
    y = torch.arange(k).repeat_interleave(torch.tensor(ns))
    x += F.one_hot(y % d, d)*.3
    mu = torch.stack([x[y==c].mean(0) for c in range(k)])
    Q = torch.stack([x[y==c].T @ x[y==c]/ns[c] for c in range(k)])
    bank = dict(mu=mu, Q=Q, n=ns,
        components=[dict(label=c, center=mu[c], mass=x.new_tensor(1.), count=ns[c], radius=x.new_tensor(1.)) for c in range(k)])
    return x, y, bank


def margin(scores, labels):
    other = scores.masked_fill(F.one_hot(labels, scores.shape[1]).bool(), -torch.inf).max(1).values
    return scores.gather(1, labels[:, None]).flatten()-other


def run():
    started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(74002)
    assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
    x,y,bank=fixture();saved=copy.deepcopy(bank);k=5;old=3
    maximum=0.;checks=[]
    for arm in ('base','local_transport'):
        local=torch.randn(k,4,7,dtype=x.dtype) if arm=='local_transport' else None
        initial,R,_=modules.head(bank,arm,'HK',list(range(k)),local)
        pairs=competition.competition(bank,initial)
        W,solved=competition.bank_head(bank,pairs,regularizer=R)
        delta,a=repair.fit(bank,pairs,R,W,old)
        assert delta.norm()>0 and not delta[:,old:].any()
        assert a['pair_deficit_after']<a['pair_deficit_before']
        assert a['effective_curvature']<=1+1e-10 and a['relative_native_energy_step']<=.1+1e-10
        B,V,H,L,omega,edges=repair.geometry(bank,pairs,R,old)
        alpha=x.new_tensor([1/(k*bank['n'][int(c)]) for c in y])
        def explicit(z):
            w=W+B @ z.reshape(B.shape[1],V.shape[1]) @ V.T
            square=.5*(alpha*(x@w-F.one_hot(y,k)).square().sum(1)).sum()
            raw=(x@w).gather(1,y[:,None])-x@w-1
            pair=.25*(alpha*(pairs[y]*raw.square()).sum(1)).sum()
            return square+pair+.0005*(w*(R@w)).sum()
        zeros=torch.zeros(len(H),dtype=x.dtype,requires_grad=True)
        oracle=torch.autograd.functional.hessian(explicit,zeros)
        assert torch.allclose(H,oracle,atol=1e-11,rtol=1e-10)
        assert torch.autograd.grad(explicit(zeros),zeros)[0].norm()<1e-8
        residual=x.new_tensor([max(0.,1-float(bank['mu'][c]@(W[:,c]-W[:,j]))) for c,j in edges])
        z,qa=repair.quadratic_step(H,L,omega,residual,1e6)
        scale=qa['curvature_scale']
        dense=torch.linalg.solve(H+scale*(L.T@(omega[:,None]*L)),scale*(L.T@(omega*residual)))
        assert torch.allclose(z,dense,atol=1e-9,rtol=1e-9)
        plain_free,_=repair.quadratic_step(H,L,omega,residual,1e6,normalized=False)
        assert not torch.allclose(z,plain_free,atol=1e-9,rtol=1e-9)
        constrained,ca=repair.quadratic_step(H,L,omega,residual,1e-8)
        assert ca['trust_active'] and abs(float(constrained@H@constrained)/1e-8-1)<1e-8
        reduced,ra=repair.quadratic_step(H,L,omega,residual*0,1.)
        assert not reduced.any()
        for n in (0,1):
            zero,za=repair.fit(bank,pairs,R,W,n)
            assert not zero.any() and za['exact_identity']
        perm=torch.tensor([2,0,1,4,3]);pb=dict(bank,mu=bank['mu'][perm],Q=bank['Q'][perm],n=[bank['n'][i] for i in perm])
        pd,_=repair.fit(pb,pairs[perm][:,perm],R,W[:,perm],old)
        assert torch.allclose(pd,delta[:,perm],atol=1e-8,rtol=1e-8)
        raw,rawaudit=repair.fit(bank,pairs,R,W,old,normalized=False)
        assert torch.isfinite(raw).all()
        if a['trust_active'] and rawaudit['trust_active']:
            assert torch.allclose(raw,delta,atol=1e-9,rtol=1e-9)
        maximum=max(maximum,solved['relative_residual'],a['relative_residual'],qa['relative_residual'],ca['relative_residual'])
        rejected(lambda:repair.fit(bank,pairs,R,W+.1,old))
        rejected(lambda:repair.fit(bank,pairs,R,W,k+1))
        checks.append(dict(control=arm,head_delta=float(delta.norm()),pair_deficit_before=a['pair_deficit_before'],
            pair_deficit_after=a['pair_deficit_after'],trust_active=a['trust_active'],native_energy_step=a['relative_native_energy_step'],
            plain_scale_head_delta=float(raw.norm()),curvature_normalized=a['effective_curvature'],
            normalized_plain_relative_difference=float((raw-delta).norm()/delta.norm()),
            free_normalization_changes_solution=True,binding_trust_can_remove_normalization_difference=True))
    assert torch.equal(bank['mu'],saved['mu']) and torch.equal(bank['Q'],saved['Q']) and bank['n']==saved['n']
    for c,s in zip(bank['components'],saved['components']):
        assert c.keys()==s.keys() and all(torch.equal(c[key],s[key]) if torch.is_tensor(c[key]) else c[key]==s[key] for key in c)
    # Every possible new truth label, across random inputs, ties, and exact-zero corrections.
    counts={}
    for dtype in (torch.float32,torch.float64):
        base=torch.randn(1000,23,dtype=dtype);change=torch.randn_like(base)*.7;change[:,21:]=0
        base[:4]=0;change[:2]=0
        guarded=repair.guard(base,change,21)
        assert torch.equal(guarded[:,21:],base[:,21:])
        assert torch.equal(guarded[:,:21].max(1).values,base[:,:21].max(1).values)
        assert torch.equal(base.argmax(1)<21,guarded.argmax(1)<21)
        assert torch.equal(repair.guard(base,change*0,21),base)
        for label in (21,22):
            labels=torch.full((len(base),),label,dtype=torch.long)
            assert torch.equal(margin(base,labels),margin(guarded,labels))
            assert torch.equal(base.argmax(1)==label,guarded.argmax(1)==label)
        expected=(base+change)[:,:21].argmax(1)
        assert torch.equal(guarded[:,:21].argmax(1),expected)
        counts[str(dtype)]=dict(new_margin_changed=0,new_correctness_changed=0,old_new_group_changed=0)
        bad=change.clone();bad[:,22]=.1
        rejected(lambda:repair.guard(base,bad,21))
        rejected(lambda:repair.guard(base*float('nan'),change,21))
    base=x.new_tensor([[.8,.9,.1],[.8,.7,1.]])
    change=x.new_tensor([[.2,-.2,0.],[.5,-.5,0.]])
    guarded=repair.guard(base,change,2)
    assert base.argmax(1).tolist()==[1,2] and (base+change).argmax(1).tolist()==[0,0]
    assert guarded.argmax(1).tolist()==[0,2]
    assert not torch.equal(base.softmax(1)[:,2],guarded.softmax(1)[:,2])
    # If the first example's truth were old class 1, the same change would be harmful.
    assert guarded[0].argmax()!=1
    meta=dict(dataset='HK',control='local_transport',control_source='synthetic',order=list(range(k)),seen=list(range(k)),seed=74002,task=2,source='synthetic')
    state=repair.checkpoint(delta,old,meta);buffer=io.BytesIO();torch.save(state,buffer);buffer.seek(0)
    restored=repair.restore(torch.load(buffer,weights_only=True),meta,old,delta.shape)
    assert torch.equal(restored,delta.float())
    rejected(lambda:repair.restore(dict(state,version='wrong'),meta,old,delta.shape))
    rejected(lambda:repair.restore(state,dict(meta,source='other'),old,delta.shape))
    rejected(lambda:repair.restore(state,meta,old-1,delta.shape))
    # Actual width and HK output count; analytic native solution, identity shared second moment.
    d,k,old=1536,23,21;mu=torch.randn(k,d,dtype=torch.float64)*.001;I=torch.eye(d,dtype=mu.dtype)
    bank=dict(mu=mu,Q=I.expand(k,d,d),n=[20]*k)
    pairs=(torch.ones(k,k,dtype=mu.dtype)-torch.eye(k,dtype=mu.dtype))/(k-1)
    lap,targets=competition.laplacians(pairs)
    rhs=mu.T/k+.5*(mu.T/k)@targets
    output=1.001*torch.eye(k,dtype=mu.dtype)+.5*lap.sum(0)/k
    W=torch.linalg.solve(output,rhs.T).T
    large,largeaudit=repair.fit(bank,pairs,I,W,old)
    assert large.shape==(1536,23) and not large[:,old:].any() and largeaudit['coefficients']<=400
    assert largeaudit['pair_deficit_after']<largeaudit['pair_deficit_before']
    maximum=max(maximum,largeaudit['native_residual'],largeaudit['relative_residual'])
    assert not torch.cuda.is_initialized()
    return dict(status='PASS',version=repair.VERSION,cpu_pipeline_seconds=time.monotonic()-started,
        medical_images_opened=0,medical_checkpoints_opened=0,GPU_processes=0,optimizer_updates=0,test_accessed=False,
        expanded_sample_native_hessian_oracle=True,dense_quadratic_oracle=True,trust_KKT_and_radius=True,
        exact_zero_and_first_task_identity=True,old_class_permutation_equivariance=True,
        base_and_local_transport_compatible=True,bank_unchanged=True,
        exact_new_margin_and_group_guards=counts,synthetic_old_correction_and_raw_new_harm_example=True,
        old_class_harm_not_guaranteed_absent=True,sidecar_roundtrip_and_wrong_scope_rejected=True,
        new_softmax_probabilities_not_guaranteed=True,
        actual_width_1536_HK23=True,reduced_coefficients=largeaudit['coefficients'],
        maximum_relative_solve_residual=maximum,small_fixture_controls=checks,
        large_fixture=dict(head_delta=largeaudit['delta_head_norm'],trust_active=largeaudit['trust_active'],
            pair_deficit_before=largeaudit['pair_deficit_before'],pair_deficit_after=largeaudit['pair_deficit_after']))


if __name__=='__main__':
    config=json.loads(Path(os.environ['Q166_CPU_CONFIG']).read_text())
    design=json.loads(Path(config['design']).read_text());repair.validate_design(design)
    for key,value in [('GPU_launch_authorized',True),('optimizer_updates',1),('relative_pair_curvature',2.),('test_accessed',True),('training_feedback',True)]:
        rejected(lambda:repair.validate_design(dict(design,**{key:value})))
    result=run();result['scope_rejection']=True
    Path(config['receipt']).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
