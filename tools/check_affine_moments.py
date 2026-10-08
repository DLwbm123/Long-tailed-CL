"""Explicit-sample and augmented least-squares oracles for affine drift."""
import time
import torch
import affine_moments as affine
import fdrl_control as ctl
import prototype_coherent as method


def run():
    started=time.process_time();torch.set_num_threads(2)
    generator=torch.Generator().manual_seed(93217)
    def normal(*shape):return torch.randn(*shape,generator=generator,dtype=torch.float64)
    for n,d in ((6,9),(35,7)):
        x=normal(n,d);y=torch.arange(n)%2
        actual=torch.eye(d)+.15*normal(d,d);offset=.1*normal(d)
        target=x @ actual.T+offset+.01*normal(n,d)
        mapping=affine.fit(x,target,y)
        weights=torch.tensor([1/(2*int((y==k).sum())) for k in y],dtype=x.dtype)
        design=torch.cat([x,torch.ones(n,1,dtype=x.dtype)],1)*weights.sqrt()[:,None]
        penalty=torch.cat([.001**.5*torch.eye(d,dtype=x.dtype),torch.zeros(d,1,dtype=x.dtype)],1)
        rhs=torch.cat([target*weights.sqrt()[:,None],.001**.5*torch.eye(d,dtype=x.dtype)])
        expected=torch.linalg.lstsq(torch.cat([design,penalty]),rhs).solution
        assert torch.allclose(mapping['matrix'],expected[:-1].T,atol=1e-10,rtol=1e-10)
        assert torch.allclose(mapping['offset'],expected[-1],atol=1e-10,rtol=1e-10)
        repeated=y==0
        duplicate=affine.fit(torch.cat([x,x[repeated]]),torch.cat([target,target[repeated]]),torch.cat([y,y[repeated]]))
        assert torch.allclose(mapping['matrix'],duplicate['matrix'],atol=1e-10,rtol=1e-10)
        identity=affine.fit(x,x,y)
        assert torch.equal(identity['matrix'],torch.eye(d,dtype=x.dtype)) and not identity['offset'].any()
    d=7;x=normal(45,d) @ normal(d,d);y=torch.arange(45)%3
    bank=ctl.meta_append(method.empty(d),x,y)
    bank['components']=[dict(center=x[y==0].mean(0),radius=1.,label=0,mass=1.,count=15)]
    matrix=torch.eye(d)+.3*normal(d,d);offset=.2*normal(d)
    mapping=dict(matrix=matrix,offset=offset)
    moved=affine.transport(bank,mapping);z=x @ matrix.T+offset
    expected=ctl.meta_append(method.empty(d),z,y)
    assert torch.allclose(moved['mu'],expected['mu'],atol=1e-10,rtol=1e-10)
    assert torch.allclose(moved['Q'],expected['Q'],atol=1e-10,rtol=1e-10)
    assert torch.allclose(moved['components'][0]['center'],z[y==0].mean(0))
    w=normal(d,5);explicit=.5*(z @ w-torch.nn.functional.one_hot(y,5)).square().sum(1)
    empirical=torch.stack([explicit[y==c].mean() for c in range(3)])
    assert torch.allclose(.5*method.old_square_losses(moved,w),empirical,atol=1e-10,rtol=1e-10)
    shift=.2*normal(d);mapping=dict(matrix=torch.eye(d,dtype=x.dtype),offset=shift)
    translated=affine.transport(bank,mapping);original=method.translate(bank,shift)
    assert torch.allclose(translated['mu'],original['mu']) and torch.allclose(translated['Q'],original['Q'])
    empty=affine.transport(method.empty(d),mapping)
    assert empty['mu'].shape==(0,d) and empty['Q'].shape==(0,d,d)
    for invalid in (0.,float('nan')):
        try:affine.fit(x,x,y,invalid)
        except ValueError:pass
        else:raise AssertionError('Invalid regularization admitted')
    return dict(status='PASS',checks=['primal and dual fitting versus augmented least squares',
        'class balancing invariant to within-class replication','identity and translation reduction',
        'non-diagonal moments versus transformed explicit samples','historical risk versus per-sample loss',
        'component center, empty memory and invalid input guards'],
        scientific_optimizer_updates=0,cpu_core_seconds=time.process_time()-started)


if __name__=='__main__':
    import json
    print(json.dumps(run(),indent=2))
