"""Anchored discriminative normals on fixed features; no visual optimization."""
import torch
import torch.nn.functional as F

def mass(y,n,balanced):
    count=torch.bincount(y,minlength=n).double()
    assert (count>0).all()
    w=1/count[y] if balanced else torch.full((len(y),),n/len(y),dtype=torch.double)
    assert abs(float(w.sum())-n)<1e-10
    return w

def objective(v,z,y,base,w):
    return (F.cross_entropy(base+z@v[:,:-1].T+v[:,-1],y,reduction='none')*w).sum()+.5*v.square().sum()

def fit(z,y,proto,balanced):
    base=(20*z.float()@proto.float().T).double();z=z.double();n=len(proto)
    w=mass(y,n,balanced);v=torch.zeros(n,z.shape[1]+1,dtype=torch.double,requires_grad=True)
    opt=torch.optim.LBFGS([v],lr=1,max_iter=300,max_eval=600,history_size=10,
                         tolerance_grad=1e-7,tolerance_change=1e-12,line_search_fn='strong_wolfe')
    evaluations=0
    def closure():
        nonlocal evaluations
        evaluations+=1;opt.zero_grad();loss=objective(v,z,y,base,w)
        assert torch.isfinite(loss);loss.backward();assert torch.isfinite(v.grad).all()
        return loss
    opt.step(closure)
    loss=closure();gradient=float(v.grad.abs().max());assert gradient<=1e-5,('HEAD_NONCONVERGENCE',gradient)
    state=opt.state[v]
    return dict(residual=v.detach(),optimizer_state=opt.state_dict(),
                iterations=int(state['n_iter']),closure_evaluations=evaluations,
                final_objective=float(loss.detach()),max_abs_gradient=gradient,
                data_mass=float(w.sum()),balanced=balanced)

def logits(z,proto,residual=None):
    out=(20*z.float()@proto.float().T).double()
    if residual is not None:out=out+z.double()@residual[:,:-1].T+residual[:,-1]
    assert torch.isfinite(out).all()
    return out
