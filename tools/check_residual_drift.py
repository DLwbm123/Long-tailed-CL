"""CPU-only synthetic integration checks; no data, pretrained model or GPU."""
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import core1_competition as competition
import module_additions as modules
import prototype_coherent as native
import residual_drift as drift
from check_rfmid_protocol import TinyRegions, check, permission_fixture, report_fixture
from ml_core import binary_weights
from ml_statistics import add, empty, observation_weights, ridge, translation
from rfmid_local import penalty
from run_multilabel import Trainer


def rejected(call):
    try:
        call()
    except (ValueError, ArithmeticError):
        return
    raise AssertionError('Malformed input accepted')


def run():
    started = time.monotonic()
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    torch.manual_seed(74002)
    x = torch.randn(18, 5, dtype=torch.float64)
    y = torch.arange(3).repeat_interleave(6)
    sample_weights = x.new_full((18,), 1/6)
    bank = native.append(native.empty(5), x, y, torch.zeros(18, dtype=torch.long), sample_weights)
    delta = .1*torch.randn_like(x)
    shift = delta.mean(0)
    U, audit = drift.estimate(x, x+delta, x.new_full((18,), 1/18), shift)
    e = delta - shift
    assert torch.allclose(U, e.T@e/18, atol=1e-14, rtol=1e-12)
    assert torch.linalg.eigvalsh(U).min() >= -1e-12
    moved, _ = drift.estimate(x, x+shift, x.new_ones(18), shift)
    assert moved.abs().max() < 1e-28
    for bad in (-1., float('nan'), float('inf'), True): rejected(lambda: drift.strength(bad))
    rejected(lambda: drift.config_weight(dict(residual_drift_weight=.1,module='base')))
    assert drift.config_weight(dict(module='base')) == 0
    for bad in (-torch.eye(5, dtype=x.dtype), U+torch.triu(torch.ones_like(U),1), torch.full_like(U,float('nan'))):
        rejected(lambda: drift.add_to_prior(torch.eye(5,dtype=x.dtype),bad,.1))
    rejected(lambda: drift.estimate(x,x+delta,x.new_full((18,),-1.),shift))
    rejected(lambda: drift.estimate(x,x+delta,x.new_zeros(18),shift))
    rejected(lambda: drift.estimate(x,x[:2],x.new_ones(18),shift))
    maximum = 0.
    for arm in ('base','local_transport'):
        centers = torch.randn(3,4,5,dtype=x.dtype) if arm=='local_transport' else None
        w0,R0,_ = modules.head(bank,arm,'ISIC',[0,1,2],centers)
        wz,Rz,_ = modules.head(bank,arm,'ISIC',[0,1,2],centers,drift=U,drift_weight=0.)
        assert torch.equal(w0,wz) and torch.equal(R0,Rz)
        w,R,_ = modules.head(bank,arm,'ISIC',[0,1,2],centers,drift=U,drift_weight=.1)
        assert torch.allclose(R,R0+100*U,atol=1e-12,rtol=1e-12)
        assert torch.allclose(w,torch.linalg.solve(bank['Q'].sum(0)/3+.001*R,bank['mu'].T/3))
        pairs = competition.competition(bank,w)
        for beta in (0.,.5):
            solved,solve = competition.bank_head(bank,pairs,beta=beta,regularizer=R)
            def objective(v):
                t=torch.nn.functional.one_hot(y,3).to(x)
                return (.5*(x@v-t).square().sum(1).mean()+.0005*(v*(R0@v)).sum()
                    +.05*(v*(U@v)).sum()+beta*competition.pair_loss(native.empty(5),x,y,x.new_full((18,),1/18),v,pairs))
            z=torch.zeros_like(w,requires_grad=True)
            H=torch.autograd.functional.hessian(objective,z).reshape(15,15)
            b=-torch.autograd.grad(objective(z),z)[0].flatten()
            oracle=torch.linalg.solve(H,b).reshape_as(w)
            assert torch.allclose(solved,oracle,atol=1e-8,rtol=1e-8)
            optimum=solved.detach().requires_grad_()
            assert torch.autograd.grad(objective(optimum),optimum)[0].norm() < 1e-8
            maximum=max(maximum,solve['relative_residual'])
        old=dict(mu=bank['mu'][:1],Q=bank['Q'][:1],n=[6],components=[c for c in bank['components'] if c['label']==0])
        q,mu,mass=competition.moments(old,x[6:],y[6:],x.new_full((12,),1/18),3)
        inv,cross=native.proximal_base(old,R,3)
        start=native.proximal_head(x[6:],torch.nn.functional.one_hot(y[6:],3).to(x),x.new_full((12,),1/18),inv,cross,w)
        fast,a=competition.solve(q,mu,mass,pairs,previous=w,proximal=.01,inverse=inv,
            x=x[6:],alpha=x.new_full((12,),1/18),native=start,regularizer=R)
        direct,b=competition.solve(q,mu,mass,pairs,previous=w,proximal=.01,regularizer=R)
        assert torch.allclose(fast,direct,atol=1e-8,rtol=1e-8)
        v=torch.randn_like(w)
        assert torch.allclose((v*(U@v)).sum(),(e@v).square().sum(1).mean(),atol=1e-12)
        # Envelope gradient holds for U frozen at the epoch boundary.
        def optimized(features):
            new=native.append(native.empty(5),features,y,torch.zeros(18,dtype=torch.long),sample_weights)
            h,_=competition.bank_head(new,pairs,regularizer=R);h=h.detach()
            t=torch.nn.functional.one_hot(y,3).to(x)
            return (.5*(features@h-t).square().sum(1).mean()+.0005*(h*(R@h)).sum()
                +.5*competition.pair_loss(native.empty(5),features,y,x.new_full((18,),1/18),h,pairs))
        var=x.clone().requires_grad_();g=torch.autograd.grad(optimized(var),var)[0][2,1]
        xp=x.clone();xm=x.clone();xp[2,1]+=1e-5;xm[2,1]-=1e-5
        assert torch.allclose(g,(optimized(xp)-optimized(xm))/2e-5,atol=2e-6,rtol=2e-5)
        maximum=max(maximum,a['relative_residual'],b['relative_residual'])
        buffer=io.BytesIO()
        torch.save(dict(bank=bank,local=centers,drift=U,pairs=pairs,head=solved.float()),buffer)
        buffer.seek(0);state=torch.load(buffer,map_location='cpu',weights_only=False)
        _,saved_R,_=modules.head(state['bank'],arm,'ISIC',[0,1,2],state['local'],
            drift=state['drift'],drift_weight=.1)
        rebuilt,_=competition.bank_head(state['bank'],state['pairs'],regularizer=saved_R)
        assert torch.equal(rebuilt.float(),state['head'])
    # A masked binary Hessian oracle, including unchanged regularized bias.
    by=np.array([[i%2,(i//2)%2,1] for i in range(18)],dtype=np.float64)
    mask=np.ones_like(by,dtype=bool);mask[:,2]=False;by[:,2]=999
    weights=binary_weights(torch.from_numpy(by),torch.from_numpy(mask)).numpy()
    rows=observation_weights(by,mask,weights)
    shift_np,_=translation(x.numpy(),(x+delta).numpy(),by,mask,weights)
    Ub,_=drift.estimate(x,x+delta,torch.from_numpy(rows),torch.from_numpy(shift_np))
    by2=by.copy();by2[:,2]=-999
    assert np.array_equal(rows,observation_weights(by2,mask,weights))
    stat=add(empty(5,3),x.numpy(),by,mask)
    Rb=drift.add_to_prior(torch.eye(6,dtype=x.dtype),Ub,.1,bias=True).numpy()
    assert Rb[-1,-1]==1 and np.all(Rb[-1,:-1]==0) and np.all(Rb[:-1,-1]==0)
    W,a=ridge(stat,penalty=lambda _:Rb)
    X=torch.cat((x,torch.ones(18,1,dtype=x.dtype)),1)
    for j in (0,1):
        target=torch.tensor(2*by[:,j]-1,dtype=x.dtype)
        def binary_objective(v):
            loss=sum(.25*(X[by[:,j]==s]@v-target[by[:,j]==s]).square().mean() for s in (0,1))
            return loss+.0005*v.square().sum()+.05*(v[:-1]*(Ub@v[:-1])).sum()
        v=torch.zeros(6,dtype=x.dtype,requires_grad=True)
        H=torch.autograd.functional.hessian(binary_objective,v)
        rhs=-torch.autograd.grad(binary_objective(v),v)[0]
        assert np.allclose(W[:,j],torch.linalg.solve(H,rhs).numpy(),atol=1e-10,rtol=1e-10)
    assert not W[:,2].any() and a['residuals'][2] is None
    # Compare disabled integration to the committed pre-change native trainer.
    spec=importlib.util.spec_from_file_location('previous_native',os.environ['Q150_NATIVE'])
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    torch.manual_seed(74002);initial=TinyRegions()
    trainers=[Trainer(copy.deepcopy(initial),3),original.Trainer(copy.deepcopy(initial),3),
              Trainer(copy.deepcopy(initial),3,drift_weight=.1),
              Trainer(copy.deepcopy(initial),3,local_transport=True,drift_weight=.1),
              Trainer(copy.deepcopy(initial),3,local_transport=True),
              original.Trainer(copy.deepcopy(initial),3,local_transport=True)]
    for task in range(1,6):
        tx=torch.randn(12,5,3);ty=torch.tensor([[i%2,(i//2)%2,i%3==0] for i in range(12)],dtype=torch.float32)
        tm=torch.ones_like(ty,dtype=torch.bool)
        if task==1:tm[:,2]=False
        ty=torch.where(tm,ty,0.)
        loader=DataLoader(TensorDataset(tx,ty,tm),batch_size=6)
        if task in (1,2):
            for t in trainers[2:4]:
                probe=Trainer(copy.deepcopy(t.encoder),3,local_transport=t.local is not None,drift_weight=.1)
                if t.head is not None: probe.load_state_dict(copy.deepcopy(t.state_dict()))
                saved=copy.deepcopy(probe.encoder.state_dict());steps=probe.steps
                result=probe.fit_task(loader,loader,2,preflight=True)
                assert result['optimizer_updates']==0 and probe.steps==steps and result['gradient_norm']>0
                assert all(torch.equal(value,probe.encoder.state_dict()[k]) for k,value in saved.items())
        result=[t.fit_task(loader,loader,2) for t in trainers]
        assert all(t.steps==4*task for t in trainers)
        assert torch.equal(trainers[0].head,trainers[1].head)
        assert all(torch.equal(value,trainers[1].encoder.state_dict()[k]) for k,value in trainers[0].encoder.state_dict().items())
        assert all(np.array_equal(trainers[0].bank[k],trainers[1].bank[k]) for k in trainers[0].bank)
        assert torch.equal(trainers[4].head,trainers[5].head)
        assert all(torch.equal(value,trainers[5].encoder.state_dict()[k]) for k,value in trainers[4].encoder.state_dict().items())
        assert all(np.array_equal(trainers[4].local[k],trainers[5].local[k]) for k in trainers[4].local)
        assert all(np.array_equal(t.bank['n'],trainers[0].bank['n']) for t in trainers[2:])
        for t,r in zip(trainers[2:4],result[2:4]):
            assert r['residual_drift']['residual_trace']>0 and not torch.from_numpy(t.drift).requires_grad
            R=drift.add_to_prior(torch.zeros(7,7,dtype=torch.float64),torch.from_numpy(t.drift),.1,bias=True).numpy()
            expected,_=ridge(t.bank,penalty=lambda j: (np.eye(7) if t.local is None else penalty(t.local,j))+R)
            assert np.array_equal(r['W_final'],expected)
            restored=Trainer(copy.deepcopy(initial),3,local_transport=t.local is not None,drift_weight=.1)
            restored.load_state_dict(copy.deepcopy(t.state_dict()))
            assert torch.equal(restored.head,t.head) and np.array_equal(restored.drift,t.drift)
            rejected(lambda: Trainer(copy.deepcopy(initial),3,local_transport=t.local is not None).load_state_dict(t.state_dict()))
            maximum=max(maximum,*(v for v in r['final_head']['residuals'] if v is not None))
    assert not torch.cuda.is_initialized()
    restored_native=Trainer(copy.deepcopy(initial),3)
    restored_native.load_state_dict(copy.deepcopy(trainers[1].state_dict()))
    assert torch.equal(restored_native.head,trainers[1].head)
    regression=dict(multilabel=check(),permission=permission_fixture(),report=report_fixture())
    competition.self_check()
    return dict(status='PASS',cpu_seconds=time.monotonic()-started,gpu_work=False,
        medical_images_or_labels_read=False,test_accessed=False,formal_adapter_updates=0,
        synthetic_cpu_optimizer_updates_per_arm=20,lambda_zero_native_trajectory_exact=True,
        pure_translation_zero=True,weighted_residual_identity=True,invalid_inputs_rejected=True,
        competition_hessian_oracle=True,proximal_woodbury_consistent=True,frozen_U_envelope_gradient=True,
        binary_hessian_oracle=True,bias_penalty_unchanged=True,unknown_mask_unchanged=True,
        shared_observed_pair_weights=True,multilabel_checkpoint_roundtrip=True,
        competition_checkpoint_refit_exact=True,legacy_checkpoint_load_exact=True,
        checkpoint_weight_mismatch_rejected=True,cpu_zero_optimizer_fixtures_pass=True,
        max_solve_residual=maximum,regressions=regression)


if __name__=='__main__': print(json.dumps(run(),indent=2,allow_nan=False))
