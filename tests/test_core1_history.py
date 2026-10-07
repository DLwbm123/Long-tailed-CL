"""Independent sample oracle for historical-only competition."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import torch
import core1_competition as pc
import prototype_coherent as native


def test_historical_only_competition():
    torch.manual_seed(812);torch.set_num_threads(2)
    x=torch.randn(18,5,dtype=torch.float64);y=torch.arange(3).repeat_interleave(6)
    weights=x.new_full((18,),1/6);group=torch.arange(6).repeat_interleave(3)
    bank=native.append(native.empty(5),x,y,group,weights)
    reference,_=native.head(bank,0);targets=torch.nn.functional.one_hot(y,3).double()
    for mode in ('prototype','uniform'):
        original=pc.competition(bank,reference,mode)
        for count in (0,1,3):
            a=pc.competition(bank,reference,mode,historical_count=count)
            assert torch.equal(a[:count],original[:count]) and torch.count_nonzero(a[count:])==0
            assert torch.equal(a.sum(1).round(),torch.tensor([1.]*count+[0.]*(3-count),dtype=x.dtype))
    a=pc.competition(bank,reference,historical_count=1)
    old=dict(n=[6],mu=bank['mu'][:1],Q=bank['Q'][:1])
    alpha=x.new_full((12,),1/18)
    for mean_only in (False,True):
        def explicit(w):
            z=x[:6].mean(0,keepdim=True) if mean_only else x[:6]
            scores=z @ w;extra=(a[0]*(scores[:,0,None]-scores-1).square()).sum(1).mean()/3
            return .5*(x @ w-targets).square().sum(1).mean()+.0005*w.square().sum()+.25*extra
        w,info=pc.bank_head(bank,a,mean_only=mean_only)
        zero=torch.zeros_like(w,requires_grad=True)
        h=torch.autograd.functional.hessian(explicit,zero).reshape(15,15)
        rhs=-torch.autograd.grad(explicit(zero),zero)[0].flatten()
        assert torch.allclose(w,torch.linalg.solve(h,rhs).reshape(5,3),atol=1e-8,rtol=1e-8),info
        compressed=.5*(x @ w-targets).square().sum(1).mean()+.0005*w.square().sum()
        compressed+=.5*pc.pair_loss(old,x[6:],y[6:],alpha,w,a,mean_only)
        assert torch.allclose(compressed,explicit(w),atol=1e-10)
        q,mu,mass=pc.moments(old,x[6:],y[6:],alpha,3)
        inv,cross=native.proximal_base(old,torch.eye(5,dtype=x.dtype),3)
        nw=native.proximal_head(x[6:],targets[6:],alpha,inv,cross,reference)
        fast,_=pc.solve(q,mu,mass,a,mean_only=mean_only,previous=reference,proximal=.01,inverse=inv,x=x[6:],alpha=alpha,native=nw)
        direct,_=pc.solve(q,mu,mass,a,mean_only=mean_only,previous=reference,proximal=.01)
        assert torch.allclose(fast,direct,atol=1e-8,rtol=1e-8)
        current=x[6:].clone().requires_grad_()
        pair=pc.pair_loss(old,current,y[6:],alpha,w.detach(),a,mean_only)
        assert torch.count_nonzero(torch.autograd.grad(pair,current)[0])==0
        def optimized(z):
            q,mu,mass=pc.moments(old,z,y[6:],alpha,3)
            head,_=pc.solve(q,mu,mass,a,mean_only=mean_only);head=head.detach()
            return .5*(alpha*(z @ head-targets[6:]).square().sum(1)).sum()+.5*native.old_square_losses(old,head).sum()/3+.0005*head.square().sum()+.5*pc.pair_loss(old,z,y[6:],alpha,head,a,mean_only)
        g=torch.autograd.grad(optimized(current),current)[0][2,1]
        plus=current.detach().clone();minus=plus.clone();plus[2,1]+=1e-5;minus[2,1]-=1e-5
        assert torch.allclose(g,(optimized(plus)-optimized(minus))/2e-5,atol=2e-6,rtol=2e-5)
    zero=pc.competition(bank,reference,historical_count=0)
    no_history,_=pc.bank_head(bank,zero)
    assert torch.allclose(no_history,reference,atol=1e-10)
    print('PASS: historical-only explicit Hessian, full/mean moments, active rows, no-history reduction, Woodbury, zero direct pair gradient, envelope gradient')


if __name__=='__main__':
    pc.self_check()
    test_historical_only_competition()
