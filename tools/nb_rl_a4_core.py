"""One intermediate learning rate; objective remains the A3 FD10 objective."""
import torch
from nb_rl_a1_core import objective as base_objective

CONDITIONS = {
    'A': dict(fd=10., lr=3e-4, exact=False),
    'M': dict(fd=10., lr=2e-4, exact=False),
}


def objective(z, target, W, y, weights, method, task, sigma, generator):
    assert method in CONDITIONS
    return base_objective(z,target,W,y,weights,'H',task,sigma,generator)


def selfcheck():
    from nb_rl_a2_core import objective as old_objective
    g=torch.Generator().manual_seed(94001)
    z=torch.randn(5,7,generator=g,dtype=torch.float64,requires_grad=True)
    target=torch.randn(5,7,generator=g,dtype=torch.float64)
    W=torch.randn(7,4,generator=g,dtype=torch.float64)
    y=torch.tensor([0,1,0,1,1]);weights=torch.tensor([.5,2.,1.,1.])
    for task in (1,2):
        a=objective(z,target,W,y,weights,'A',task,.5,g)
        m=objective(z,target,W,y,weights,'M',task,.5,g)
        old=old_objective(z,target,W,y,weights,'A',task,.5,g)
        torch.testing.assert_close(a[0],old[0],rtol=0,atol=0)
        torch.testing.assert_close(a[0],m[0],rtol=0,atol=0)
        for key in a[1]:torch.testing.assert_close(a[1][key],m[1][key],rtol=0,atol=0)
        assert a[3]==m[3]==.5 and a[5] is m[5] is None
    assert CONDITIONS['M']['lr']==2e-4 and CONDITIONS['A']['lr']==3e-4
    return dict(status='PASS',same_objective_as_A3=True,only_learning_rate_differs=True,no_rewards=True,CPU_only=True)
