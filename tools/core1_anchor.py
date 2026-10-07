"""Revision 1: preserve historical score differences under the locked shift proxy."""
import torch
import core1_competition as base

competition = base.competition
moments = base.moments


def target_moment(q, mu, mass, a, mean_only, reference, bias, old_count):
    if reference.shape != mu.T.shape or bias.shape != (len(a),) or not 0 <= old_count <= len(a):
        raise ValueError('Invalid historical anchor shape')
    if not torch.isfinite(reference).all() or not torch.isfinite(bias).all():
        raise ValueError('Nonfinite historical anchor')
    lap, targets = base.laplacians(a)
    qp = base.pair_moments(q, mu, mass, mean_only)
    linear = mu[old_count:].T @ targets[old_count:]
    if old_count:
        linear = linear+torch.einsum('cdk,ckl->dl', qp[:old_count] @ reference, lap[:old_count])
        linear = linear+torch.einsum('cd,ckl,k->dl', mu[:old_count], lap[:old_count], bias)
    return linear


def solve(q, mu, mass, a, beta=.5, mean_only=False, previous=None, proximal=0.,
          inverse=None, x=None, alpha=None, native=None, *, reference, bias, old_count):
    linear=target_moment(q,mu,mass,a,mean_only,reference,bias,old_count)
    return base.solve(q,mu,mass,a,beta,mean_only,previous,proximal,inverse,x,alpha,native,pair_linear=linear)


def bank_head(bank, a, beta=.5, mean_only=False, *, reference, bias, old_count):
    k=len(bank['n']);mass=bank['mu'].new_full((k,),1/k)
    return solve(bank['Q']/k,bank['mu']/k,mass,a,beta,mean_only,
                 reference=reference,bias=bias,old_count=old_count)


def pair_loss(old,x,y,alpha,w,a,mean_only=False,*,reference,bias):
    empty=dict(n=[],mu=old['mu'][:0],Q=old['Q'][:0])
    value=base.pair_loss(empty,x,y,alpha,w,a,mean_only)
    if len(old['n']):
        delta=w-reference;lap,_=base.laplacians(a);lap=lap[:len(old['n'])]
        q=old['Q'] if not mean_only else old['mu'][:,:,None]*old['mu'][:,None,:]
        quadratic=torch.einsum('dl,cdk,ckl->c',delta,q @ delta,lap)
        cross=torch.einsum('ck,ckl,l->c',old['mu'] @ delta,lap,bias)
        constant=torch.einsum('k,ckl,l->c',bias,lap,bias)
        value=value+.5*(quadratic-2*cross+constant).sum()/w.shape[1]
    return value


def self_check():
    import prototype_coherent as native
    torch.manual_seed(771);torch.set_num_threads(2)
    x=torch.randn(18,5,dtype=torch.float64);y=torch.arange(3).repeat_interleave(6)
    group=torch.arange(6).repeat_interleave(3);weights=x.new_full((18,),1/6)
    bank=native.append(native.empty(5),x,y,group,weights)
    ref,_=native.head(bank,0);a=competition(bank,ref)
    teacher=torch.randn_like(ref);teacher[:,1:]=0;shift=torch.randn(5,dtype=x.dtype)*.1;bias=-shift @ teacher
    old=dict(n=[6],mu=bank['mu'][:1],Q=bank['Q'][:1])
    target=torch.nn.functional.one_hot(y,3)
    for mean_only in (False,True):
        def explicit(w,xx=x):
            value=.5*(xx @ w-target).square().sum(1).mean()+.0005*w.square().sum()
            extra=w.new_zeros(())
            for c in range(3):
                z=xx[y==c]
                if mean_only:z=z.mean(0,keepdim=True)
                s=z @ w
                if c==0:
                    t=z @ teacher+bias;err=(s[:,c,None]-s)-(t[:,c,None]-t)
                else:err=s[:,c,None]-s-1
                extra+=(a[c]*err.square()).sum(1).mean()/3
            return value+.25*extra
        w,info=bank_head(bank,a,mean_only=mean_only,reference=teacher,bias=bias,old_count=1)
        zero=torch.zeros_like(w,requires_grad=True)
        h=torch.autograd.functional.hessian(explicit,zero).reshape(15,15)
        b=-torch.autograd.grad(explicit(zero),zero)[0].flatten()
        exact=torch.linalg.solve(h,b).reshape(5,3)
        assert torch.allclose(w,exact,atol=1e-8,rtol=1e-8),info
        compressed=.5*(x @ w-target).square().sum(1).mean()+.0005*w.square().sum()
        compressed+=.5*pair_loss(old,x[6:],y[6:],x.new_full((12,),1/18),w,a,mean_only,reference=teacher,bias=bias)
        assert torch.allclose(compressed,explicit(w),atol=1e-10)
        # The affine bias restores exactly the teacher's pre-translation scores.
        original=x[:6]-shift
        assert torch.allclose(x[:6] @ teacher+bias,original @ teacher,atol=1e-12)
        # Mixed historical/current minibatch and Woodbury path use the same objective.
        alpha=x.new_full((12,),1/18);q,mu,mass=moments(old,x[6:],y[6:],alpha,3)
        inv,cross=native.proximal_base(old,torch.eye(5,dtype=x.dtype),3)
        nw=native.proximal_head(x[6:],target[6:].double(),alpha,inv,cross,ref)
        fast,_=solve(q,mu,mass,a,mean_only=mean_only,previous=ref,proximal=.01,inverse=inv,
            x=x[6:],alpha=alpha,native=nw,reference=teacher,bias=bias,old_count=1)
        direct,_=solve(q,mu,mass,a,mean_only=mean_only,previous=ref,proximal=.01,
            reference=teacher,bias=bias,old_count=1)
        assert torch.allclose(fast,direct,atol=1e-8,rtol=1e-8)
        # Differentiate current features only, with a fixed historical proxy/anchor.
        def optimal(current):
            q,mu,mass=moments(old,current,y[6:],alpha,3)
            head,_=solve(q,mu,mass,a,mean_only=mean_only,reference=teacher,bias=bias,old_count=1)
            head=head.detach()
            value=.5*(alpha*(current @ head-target[6:]).square().sum(1)).sum()
            value+=.5*native.old_square_losses(old,head).sum()/3+.0005*head.square().sum()
            return value+.5*pair_loss(old,current,y[6:],alpha,head,a,mean_only,reference=teacher,bias=bias)
        current=x[6:].clone().requires_grad_();g=torch.autograd.grad(optimal(current),current)[0][1,2]
        plus=current.detach().clone();minus=plus.clone();plus[1,2]+=1e-5;minus[1,2]-=1e-5
        finite=(optimal(plus)-optimal(minus))/2e-5
        assert torch.allclose(g,finite,atol=2e-6,rtol=2e-5)
    print('PASS: anchored affine targets, explicit-sample Hessian, full/mean moment loss, Woodbury, conditional envelope gradient')


if __name__=='__main__':self_check()
