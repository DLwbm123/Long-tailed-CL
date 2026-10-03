"""Strong FD baseline with paired sampled and exact detached rewards."""
import torch
from nb_rl_a1_core import objective as base_objective, selfcheck as estimator_check
CONDITIONS={m:dict(fd=10.,lr=3e-4,exact=m=='E',reward_coefficient=0. if m=='A' else .5) for m in ('A','G','R','E')}

def objective(z,target,W,y,weights,method,task,sigma,generator):
    assert method in CONDITIONS
    source='S' if task==1 else 'H' if method=='A' else method
    _,terms,coefficients,next_sigma,diag,actions=base_objective(z,target,W,y,weights,source,task,sigma,generator)
    coefficients.update(FD=10.,PG=.5 if task>1 and method in ('G','R') else 0.,Exact=.5 if task>1 and method=='E' else 0.)
    return sum(coefficients[k]*v for k,v in terms.items()),terms,coefficients,next_sigma,diag,actions

def selfcheck():
    checked=estimator_check();g=torch.Generator().manual_seed(101001)
    z=torch.randn(5,7,generator=g,dtype=torch.float64,requires_grad=True)
    target=torch.randn(5,7,generator=g,dtype=torch.float64);W=torch.randn(7,4,generator=g,dtype=torch.float64)
    y=torch.tensor([0,1,0,1,1]);weights=torch.tensor([.5,2.,1.,1.])
    start=g.get_state();first=[objective(z,target,W,y,weights,m,1,.5,g) for m in CONDITIONS]
    assert torch.equal(start,g.get_state())
    for v in first:
        torch.testing.assert_close(v[0],first[0][0],rtol=0,atol=0)
        assert v[3]==.5 and v[5] is None
    for m in CONDITIONS:
        before=g.get_state();a=objective(z,target,W,y,weights,m,2,.5,g);g.set_state(before)
        b=base_objective(z,target,W,y,weights,'H' if m=='A' else m,2,.5,g)
        expected=b[0] if m=='A' else b[0]+9*b[1]['FD']+.4*(b[1]['Exact'] if m=='E' else b[1]['PG'])
        torch.testing.assert_close(a[0],expected)
        if m in ('G','R'): assert torch.equal(a[5],b[5])
    return dict(status='PASS',identical_CE_Task1=True,Task1_policy_rng_unchanged=True,FD10_all=True,reward_coefficient=.5,estimator=checked)
