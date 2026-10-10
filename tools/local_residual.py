"""Class-balanced local/global moments for an analytical residual readout."""
import math
import torch


def projection(dim):
    generator=torch.Generator().manual_seed(130)
    return torch.randn(dim,64,generator=generator,dtype=torch.float64)/math.sqrt(64)


def project(parts,matrix):
    if parts.ndim!=3 or parts.shape[1]!=4 or parts.shape[2]!=matrix.shape[0]:
        raise ValueError('Expected four local regions and matching projection')
    return (parts @ matrix).flatten(1)/2


def empty(global_dim,local_dim,device):
    return dict(mu_g=torch.empty(0,global_dim,dtype=torch.float64,device=device),
        mu_l=torch.empty(0,local_dim,dtype=torch.float64,device=device),
        ll=torch.empty(0,local_dim,local_dim,dtype=torch.float64,device=device),
        lg=torch.empty(0,local_dim,global_dim,dtype=torch.float64,device=device),n=[])


def translate(bank,global_shift,local_shift):
    g,l=bank['mu_g'],bank['mu_l'];a,b=local_shift,global_shift
    if b.shape!=g.shape[1:] or a.shape!=l.shape[1:] or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Invalid joint moment translation')
    return dict(mu_g=g+b,mu_l=l+a,n=list(bank['n']),
        ll=bank['ll']+l[:,:,None]*a[None,None,:]+a[None,:,None]*l[:,None,:]+torch.outer(a,a)[None],
        lg=bank['lg']+l[:,:,None]*b[None,None,:]+a[None,:,None]*g[:,None,:]+torch.outer(a,b)[None])


def append(bank,global_x,local_x,labels,weights):
    classes=sorted(labels.unique().tolist());old=len(bank['n'])
    if classes!=list(range(old,old+len(classes))) or not len(classes):
        raise ValueError('Only contiguous new classes can enter local statistics')
    if not all(torch.isfinite(x).all() for x in (global_x,local_x,weights)) or (weights<=0).any():
        raise ValueError('Invalid local statistics input')
    values={k:[] for k in ('mu_g','mu_l','ll','lg')};counts=[]
    for c in classes:
        mask=labels==c;g,l,w=global_x[mask],local_x[mask],weights[mask]
        if not torch.isclose(w.sum(),w.new_tensor(1.),atol=1e-8,rtol=1e-8):
            raise ValueError('Each local statistics class needs unit mass')
        values['mu_g'].append((w[:,None]*g).sum(0));values['mu_l'].append((w[:,None]*l).sum(0))
        values['ll'].append(l.T @ (w[:,None]*l));values['lg'].append(l.T @ (w[:,None]*g));counts.append(len(g))
    return {**{k:torch.cat([bank[k],torch.stack(v)]) for k,v in values.items()},'n':bank['n']+counts}


def head(bank,global_head,residual):
    k=len(bank['n'])
    if not k or global_head.shape[1]!=k:raise ValueError('Head and moment classes differ')
    q=bank['ll'].sum(0)/k;cross=bank['lg'].sum(0)/k;target=bank['mu_l'].T/k
    right=target-cross @ global_head if residual else target
    system=(q+q.T)/2+.001*torch.eye(len(q),device=q.device,dtype=q.dtype)
    value=torch.linalg.solve(system,right)
    relative=float((system @ value-right).norm()/right.norm().clamp_min(1e-12))
    if not torch.isfinite(value).all() or relative>1e-8:raise ValueError('Invalid analytical local head')
    return value,dict(residual_target=residual,local_dim=len(q),local_head_norm=float(value.norm()),
        relative_residual=relative,cross_prediction_norm=float((cross @ global_head).norm()))


def conditioned_head(global_bank,local_bank,W):
    """Remove the local component predicted by existing global class scores."""
    k=len(local_bank['n'])
    if not k or global_bank['n']!=local_bank['n'] or W.shape[1]!=k:
        raise ValueError('Global/local statistics and head classes differ')
    if not torch.allclose(global_bank['mu'],local_bank['mu_g'],atol=1e-10,rtol=1e-10):
        raise ValueError('Global/local means differ')
    if not all(torch.isfinite(v).all() for v in (global_bank['Q'],W,local_bank['ll'],local_bank['lg'],local_bank['mu_l'])):
        raise ValueError('Nonfinite conditional statistics')
    V=W.T @ global_bank['Q'].mean(0) @ W
    D=local_bank['lg'].mean(0) @ W
    system_v=(V+V.T)/2+.001*torch.eye(k,device=W.device,dtype=W.dtype)
    A=torch.linalg.solve(system_v,D.T)
    U=local_bank['ll'].mean(0)-D @ A-A.T @ D.T+A.T @ V @ A
    target=(local_bank['mu_l'].T-A.T @ (global_bank['mu'] @ W).T)/k
    right=target-D+A.T @ V
    system_u=(U+U.T)/2+.001*torch.eye(len(U),device=W.device,dtype=W.dtype)
    B=torch.linalg.solve(system_u,right)
    residuals=[float((system_v @ A-D.T).norm()/D.norm().clamp_min(1e-12)),
               float((system_u @ B-right).norm()/right.norm().clamp_min(1e-12))]
    if not torch.isfinite(A).all() or not torch.isfinite(B).all() or max(residuals)>1e-8:
        raise ValueError('Invalid conditional residual solve')
    # shortcut: ridge conditioning only approximates orthogonality; retain its measured scalar error.
    return A,B,dict(local_dim=len(U),conditioning_dim=k,local_head_norm=float(B.norm()),
        conditioning_norm=float(A.norm()),relative_residual=max(residuals),
        remaining_score_cross_norm=float((D.T-V @ A).norm()),
        raw_score_cross_norm=float(D.norm()))


def conditioned_self_check():
    torch.manual_seed(131);torch.set_num_threads(2)
    g=torch.randn(17,6,dtype=torch.float64);z=torch.randn(17,4,dtype=g.dtype)
    y=torch.cat([torch.zeros(4),torch.ones(6),torch.full((7,),2)]).long()
    w=torch.cat([torch.full((n,),1/n,dtype=g.dtype) for n in (4,6,7)])
    local=append(empty(6,4,'cpu'),g,z,y,w)
    global_bank=dict(mu=local['mu_g'],Q=torch.stack([g[y==c].T @ (w[y==c,None]*g[y==c]) for c in range(3)]),n=local['n'])
    W=torch.randn(6,3,dtype=g.dtype);A,B,audit=conditioned_head(global_bank,local,W)
    v=g @ W;mass=w/3;A_direct=torch.linalg.solve(v.T @ (mass[:,None]*v)+.001*torch.eye(3,dtype=g.dtype),v.T @ (mass[:,None]*z))
    u=z-v @ A_direct;target=torch.nn.functional.one_hot(y,3).double()-v
    B_direct=torch.linalg.solve(u.T @ (mass[:,None]*u)+.001*torch.eye(4,dtype=g.dtype),u.T @ (mass[:,None]*target))
    assert torch.allclose(A,A_direct,atol=1e-11,rtol=1e-11) and torch.allclose(B,B_direct,atol=1e-11,rtol=1e-11)
    labels=torch.nn.functional.one_hot(y,3).double()
    perfect=append(empty(3,4,'cpu'),labels,z,y,w)
    perfect_global=dict(mu=perfect['mu_g'],Q=torch.diag_embed(torch.eye(3,dtype=g.dtype)),n=perfect['n'])
    _,zero,_=conditioned_head(perfect_global,perfect,torch.eye(3,dtype=g.dtype));assert zero.abs().max()<1e-10
    assert audit['relative_residual']<=1e-8
    print('PASS: score-conditioned weighted ridge matches samples; perfect global correction zero')


def risk_coefficients(global_bank,local_bank,W,A,B):
    V=W.T @ global_bank['Q'] @ W
    D=local_bank['lg'] @ W
    U=local_bank['ll']-D @ A-A.T @ D.transpose(1,2)+A.T @ V @ A
    a=torch.einsum('lj,clm,mj->cj',B,U,B)
    b=(D.transpose(1,2) @ B-V @ A @ B).diagonal(dim1=1,dim2=2).clone()
    mean_h=(local_bank['mu_l']-local_bank['mu_g'] @ W @ A) @ B
    ids=torch.arange(len(b),device=b.device);b[ids,ids]-=mean_h[ids,ids]
    if not torch.isfinite(a).all() or not torch.isfinite(b).all() or a.min() < -1e-10:
        raise ValueError('Invalid per-class square-risk coefficients')
    return a.clamp_min(0),b


def guarded_scale(a,b):
    """Convex column shrinkage with every seen class no worse in stored square risk."""
    import numpy as np
    from scipy.optimize import minimize
    aa,bb=a.cpu().numpy(),b.cpu().numpy();ma,mb=aa.mean(0),bb.mean(0)
    def delta(x):return aa @ (x*x)+2*bb @ x
    result=minimize(lambda x:float(ma @ (x*x)+2*mb @ x),np.zeros(len(ma)),
        jac=lambda x:2*ma*x+2*mb,bounds=[(0.,1.)]*len(ma),
        constraints=[dict(type='ineq',fun=lambda x:-delta(x),jac=lambda x:-2*(aa*x+bb))],
        method='SLSQP',options=dict(maxiter=1000,ftol=1e-12))
    if (not result.success or not np.isfinite(result.x).all() or result.x.min() < -1e-10
            or result.x.max()>1+1e-10 or delta(result.x).max()>1e-8 or result.fun>1e-8):
        raise ValueError('Risk guard solve failed: '+str(result.message))
    x=result.x
    return torch.as_tensor(x,device=a.device,dtype=a.dtype),dict(slsqp_success=True,slsqp_iterations=int(result.nit),
        guard_scale_min=float(x.min()),guard_scale_max=float(x.max()),guard_scale_mean=float(x.mean()),
        guard_active_columns_n=int((x<1-1e-8).sum()),guard_zero_columns_n=int((x<=1e-8).sum()),
        max_class_square_delta_before=float(delta(np.ones(len(x))).max()),
        max_class_square_delta_after=float(delta(x).max()),
        average_square_delta_before=float(delta(np.ones(len(x))).mean()),average_square_delta_after=float(delta(x).mean()))


def risk_guard(global_bank,local_bank,W,A,B):
    a,b=risk_coefficients(global_bank,local_bank,W,A,B)
    mean_a,mean_b=a.mean(0),b.mean(0)
    control=torch.where(mean_a>1e-20,(-mean_b/mean_a.clamp_min(1e-20)).clamp(0,1),torch.ones_like(mean_a))
    deviation=float((control-1).abs().max())
    if deviation>1e-8:raise ValueError('Reused unguarded control is not the same unconstrained fit')
    scale,audit=guarded_scale(a,b);audit['unconstrained_scale_max_deviation']=deviation
    return scale,audit


def guard_self_check():
    torch.manual_seed(132);torch.set_num_threads(2)
    g=torch.randn(17,6,dtype=torch.float64);z=torch.randn(17,4,dtype=g.dtype)
    y=torch.cat([torch.zeros(4),torch.ones(6),torch.full((7,),2)]).long()
    w=torch.cat([torch.full((n,),1/n,dtype=g.dtype) for n in (4,6,7)])
    local=append(empty(6,4,'cpu'),g,z,y,w)
    global_bank=dict(mu=local['mu_g'],Q=torch.stack([g[y==c].T @ (w[y==c,None]*g[y==c]) for c in range(3)]),n=local['n'])
    W=torch.randn(6,3,dtype=g.dtype);A,B,_=conditioned_head(global_bank,local,W)
    a,b=risk_coefficients(global_bank,local,W,A,B);v=g @ W;h=(z-v @ A) @ B
    target=torch.nn.functional.one_hot(y,3).double()
    direct_a=torch.stack([(w[y==c,None]*h[y==c].square()).sum(0) for c in range(3)])
    direct_b=torch.stack([(w[y==c,None]*(v[y==c]-target[y==c])*h[y==c]).sum(0) for c in range(3)])
    assert torch.allclose(a,direct_a,atol=1e-11,rtol=1e-11) and torch.allclose(b,direct_b,atol=1e-11,rtol=1e-11)
    scale,audit=risk_guard(global_bank,local,W,A,B)
    assert (a @ scale.square()+2*b @ scale).max()<=1e-8
    aa=torch.ones(2,2,dtype=g.dtype);bb=torch.tensor([[-1.,1.],[.1,-.2]],dtype=g.dtype)
    unsafe=(-bb.mean(0)/aa.mean(0)).clamp(0,1)
    assert (aa @ unsafe.square()+2*bb @ unsafe).max()>0
    safe,_=guarded_scale(aa,bb);assert safe.sum()>1e-6 and (aa @ safe.square()+2*bb @ safe).max()<=1e-8
    zero,_=guarded_scale(aa,torch.ones_like(bb));assert zero.abs().max()<1e-8
    print('PASS: class-risk polynomial equals samples, harm constrained, zero-only case retained')


def self_check():
    torch.manual_seed(130);torch.set_num_threads(2)
    g=torch.randn(15,6,dtype=torch.float64);l=torch.randn(15,4,dtype=g.dtype)
    y=torch.arange(3).repeat_interleave(5);w=torch.full((15,),.2,dtype=g.dtype)
    bank=append(empty(6,4,'cpu'),g,l,y,w)
    a=torch.randn(4,dtype=g.dtype);b=torch.randn(6,dtype=g.dtype)
    moved=translate(bank,b,a);direct=append(empty(6,4,'cpu'),g+b,l+a,y,w)
    assert all(torch.allclose(moved[k],direct[k],atol=1e-12,rtol=1e-12) for k in ('mu_g','mu_l','ll','lg'))
    first=append(empty(6,4,'cpu'),g[:5],l[:5],y[:5],w[:5])
    sequential=append(translate(first,b,a),g[5:]+b,l[5:]+a,y[5:],w[5:])
    assert all(torch.allclose(sequential[k],direct[k],atol=1e-12,rtol=1e-12) for k in ('mu_g','mu_l','ll','lg'))
    W=torch.randn(6,3,dtype=g.dtype);target=torch.nn.functional.one_hot(y,3).double();mass=w/3
    A=l.T @ (mass[:,None]*l)+.001*torch.eye(4,dtype=g.dtype)
    for residual in (False,True):
        B,audit=head(bank,W,residual);rhs=l.T @ (mass[:,None]*(target-g @ W if residual else target))
        assert torch.allclose(B,torch.linalg.solve(A,rhs),atol=1e-11,rtol=1e-11)
        assert audit['relative_residual']<1e-8
    # A label signal already fully explained globally must receive zero residual correction.
    perfect=append(empty(3,4,'cpu'),target,l,y,w);B,_=head(perfect,torch.eye(3,dtype=g.dtype),True)
    assert B.abs().max()<1e-10
    state=torch.random.get_rng_state();P=projection(6);assert torch.equal(state,torch.random.get_rng_state())
    parts=torch.randn(2,4,6,dtype=g.dtype);delta=torch.randn(4,6,dtype=g.dtype)
    assert torch.allclose(project(parts+delta,P),project(parts,P)+project(delta[None],P))
    assert project(parts,P).shape==(2,256)
    print('PASS: weighted residual/direct ridge, joint affine transport, zero redundant correction, isolated projection RNG')


if __name__=='__main__':
    self_check()
    conditioned_self_check()
    guard_self_check()
