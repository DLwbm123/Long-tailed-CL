"""CPU oracles for budget actions, quadratic solves, policy and branch restore."""
import copy
import json
import time
import torch
from torch import nn
import pcrl_control as control
import core1_competition as competition
import prototype_coherent as native
from next1_support import rng_state, restore_rng


def run():
    started=time.process_time();torch.set_num_threads(2);torch.manual_seed(31)
    x=torch.randn(24,5,dtype=torch.float64);y=torch.arange(4).repeat_interleave(6)
    bank=native.append(native.empty(5),x,y,torch.arange(8).repeat_interleave(3),x.new_full((24,),1/6))
    w,_=native.head(bank,0);base=competition.competition(bank,w)
    groups=control.masks(4,2,'cpu');mass=(groups*base).sum((1,2));outputs=[]
    for action in range(7):
        a=control.allocation(base,2,action);outputs.append(a)
        assert torch.all(a>=0) and torch.equal(a.diag(),torch.zeros(4,dtype=x.dtype))
        assert torch.allclose(a.sum(),base.sum())
        assert torch.all((groups*a).sum((1,2))>=.5*mass-1e-12)
        if action==0:assert torch.equal(a,base)
        target=torch.nn.functional.one_hot(y,4)
        def objective(v):
            return .5*(x@v-target).square().sum(1).mean()+.0005*v.square().sum()+.5*competition.pair_loss(native.empty(5),x,y,x.new_full((24,),1/24),v,a)
        z=torch.zeros((5,4),dtype=x.dtype,requires_grad=True)
        h=torch.autograd.functional.hessian(objective,z).reshape(20,20)
        b=-torch.autograd.grad(objective(z),z)[0].flatten()
        oracle=torch.linalg.solve(h,b).reshape(5,4);actual,_=competition.bank_head(bank,a)
        assert torch.allclose(actual,oracle,atol=1e-8,rtol=1e-8)
    assert all(not torch.equal(outputs[0],a) for a in outputs[1:])
    # Verify each variance used by the old-class proxy against explicit samples.
    old=dict(mu=bank['mu'][:2],Q=bank['Q'][:2],n=bank['n'][:2],components=[])
    for c in range(2):
        d=w[:,c:c+1]-w;m=old['mu'][c]@d
        variance=((old['Q'][c]@d)*d).sum(0)-m.square()
        assert torch.allclose(variance,(x[y==c]@d).var(0,unbiased=False),atol=1e-10)
    reference=control.risks(old,w,x[12:],y[12:]);assert control.reward(reference,reference,2,[0,3])==0.
    assert control.reward(reference*.9,reference,2,[0,3])>0
    assert control.reward(reference*1.1,reference,2,[0,3])<0
    f=control.state(bank,w,base,reference,2,[0,3],0.,0.)
    p=control.Policy(31);valid=torch.ones(7,dtype=torch.bool);actions,prior=p.propose(f,valid)
    assert p.update(f,valid,actions,[0.,0.,0.,0.],prior)==0
    p.update(f,valid,torch.tensor([0,1,2,3]),[0.,1.,-.5,.3],prior)
    assert torch.isfinite(p.theta).all() and torch.isclose(p.probabilities(f,valid).sum(),torch.tensor(1.,dtype=x.dtype))
    # A warmed Adam optimizer and mutable buffer must both round-trip.
    model=nn.Linear(5,4).double();model.register_buffer('counter',torch.zeros(1));opt=torch.optim.AdamW(model.parameters(),lr=.01)
    def step():
        opt.zero_grad();loss=(model(x)-torch.randn(24,4,dtype=x.dtype)).square().mean()
        loss.backward();opt.step();model.counter.add_(1);return loss.detach().clone()
    step();snapshot=control.Snapshot(model,opt)
    losses1=[step(),step()];state1=copy.deepcopy(model.state_dict());r1=torch.rand(3)
    snapshot.restore(opt);snapshot.verify(opt)
    losses2=[step(),step()];state2=model.state_dict();r2=torch.rand(3)
    assert all(torch.equal(a,b) for a,b in zip(losses1,losses2)) and torch.equal(r1,r2)
    assert all(torch.equal(state1[k],state2[k]) for k in state1)
    snapshot.restore(opt);snapshot.verify(opt)
    return dict(status='PASS',cpu_core_seconds=time.process_time()-started,
        checks=['seven bounded distinct actions','action zero exact identity','explicit non-row-normalized Hessian oracle',
                'margin variance versus explicit samples','reward sign and zero identity','finite clipped policy update',
                'warmed Adam parameters buffers and RNG exact two-step replay'],
        scientific_model_optimizer_updates=0,toy_cpu_optimizer_updates=5,test_accessed=False)


if __name__=='__main__':print(json.dumps(run(),indent=2))
