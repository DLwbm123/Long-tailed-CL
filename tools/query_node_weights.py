"""One post-update query direction; fixed nodes, class mass and retention weights."""
import math
import numpy as np
import torch
from check_reserved_competition_memory import clone_model, scores, training_batch
from check_fullslot_retention_memory import centered, train


def rollout(teacher, current, memory, q, a, steps=48, graph=False):
    old_n=len(memory);seen=old_n+len(current)
    changed={c:(z,(p.log()+a[c]).softmax(0)) for c,(z,p) in memory.items()}
    x,y,p=training_batch(current,changed)
    _,_,base_p=training_batch(current,memory)
    old=y<old_n;params=clone_model(teacher,True)
    with torch.no_grad():
        ts=scores(x[old],teacher,q)[:,:old_n]
        target=centered(ts);correct=ts.argmax(1)==y[old]
    for _ in range(steps):
        ss=scores(x,params,q)[:,:seen]
        ce=-(ss.log_softmax(1)[torch.arange(len(y)),y]*p).sum()
        retention=(((centered(ss[old,:old_n])-target).square().mean(1)*correct)*base_p[old]*seen/old_n).sum()
        grad=torch.autograd.grad(ce+retention,params,create_graph=graph)
        params=tuple(v-g/16 for v,g in zip(params,grad))
        if not graph:params=clone_model(params,True)
    return params


def query_loss(model,q,x,y,classes,seen):
    ss=scores(x,model,q)[:,:seen]
    losses=-ss.log_softmax(1)[torch.arange(len(y)),y]
    return torch.stack([losses[y==c].mean() for c in classes]).mean()


def project_direction(go,gn,old_n,seen):
    gs=[g.detach().reshape(-1).numpy() for g in (go,gn)]
    g=(old_n*gs[0]+(seen-old_n)*gs[1])/seen
    norm=np.linalg.norm(g)
    if norm<=1e-14:return np.zeros_like(g),dict(reason='ZERO_QUERY_GRADIENT',gradient_norm=float(norm))
    v=-g/norm
    normals=np.array([z/np.linalg.norm(z) for z in gs if np.linalg.norm(z)>1e-14])
    candidates=[np.zeros_like(v)]
    def feasible(d):return len(normals)==0 or np.max(normals@d)<=1e-10
    if feasible(v):candidates.append(v)
    for n in normals:
        d=v-n*np.dot(n,v)
        if feasible(d):candidates.append(d)
    if len(normals)==2:
        lam=np.linalg.pinv(normals@normals.T,rcond=1e-12)@(normals@v)
        d=v-normals.T@lam
        if np.min(lam)>=-1e-10 and feasible(d):candidates.append(d)
    d=min(candidates,key=lambda z:float(np.sum((z-v)**2)))
    if np.linalg.norm(d)<1e-12:d=np.zeros_like(d)
    d=d*min(1.,math.log(1.25)/max(float(np.max(np.abs(d))),1e-30))
    assert feasible(d) and np.max(np.abs(d))<=math.log(1.25)+1e-12
    return d,dict(reason='PROJECTED_ONCE' if np.any(d) else 'ZERO_FEASIBLE_DIRECTION',
        gradient_norm=float(norm),old_gradient_norm=float(np.linalg.norm(gs[0])),
        new_gradient_norm=float(np.linalg.norm(gs[1])),old_directional_derivative=float(gs[0]@d),
        new_directional_derivative=float(gs[1]@d),logit_change_max=float(np.max(np.abs(d))),
        direction_norm=float(np.linalg.norm(d)))


def select(base,can):
    delta={g:can[g]-base[g] for g in ('all','old','new')}
    feasible=all(delta[g]>=-.01-1e-12 for g in ('old','new'))
    use=feasible and delta['all']>1e-12
    return dict(selected='candidate' if use else 'baseline',delta=delta,feasible=feasible,
                reason='FEASIBLE_POSITIVE_GAIN' if use else 'NO_FEASIBLE_POSITIVE_GAIN')


def unit():
    from check_reserved_competition_memory import DT,initial_model,frame
    gen=torch.Generator().manual_seed(74002)
    x=torch.nn.functional.normalize(torch.randn(20,8,generator=gen,dtype=DT),dim=1)
    current={2:x[8:12]};memory={0:(x[:4],torch.tensor([.1,.2,.3,.4],dtype=DT)),1:(x[4:8],torch.ones(4,dtype=DT)/4)}
    teacher=initial_model(8,74002);q=frame(8,3,74002)
    a=torch.zeros(2,4,dtype=DT,requires_grad=True)
    model=rollout(teacher,current,memory,q,a,graph=True)
    original,_=train(teacher,current,memory,q,[0,1,2])
    error=max(float((v-w).detach().abs().max()) for v,w in zip(model,original))
    assert error<1e-12
    y=torch.tensor([0,0,1,1,2,2,0,1])
    loss=query_loss(model,q,x[12:],y,[0,1,2],3)
    g=torch.autograd.grad(loss,a)[0]
    v=torch.randn(a.shape,generator=gen,dtype=DT);v=v/v.norm();eps=1e-5
    def f(b):return float(query_loss(rollout(teacher,current,memory,q,b),q,x[12:],y,[0,1,2],3).detach())
    fd=(f(a.detach()+eps*v)-f(a.detach()-eps*v))/(2*eps)
    error_fd=abs(fd-float((g*v).sum()));assert error_fd<1e-6
    for go,gn in [(torch.tensor([1.,0.],dtype=DT),torch.tensor([0.,1.],dtype=DT)),
                  (torch.tensor([1.,0.],dtype=DT),torch.tensor([-1.,0.],dtype=DT)),
                  (torch.zeros(2,dtype=DT),torch.zeros(2,dtype=DT))]:
        d,info=project_direction(go,gn,1,2)
        assert float(go.numpy()@d)<=1e-10 and float(gn.numpy()@d)<=1e-10
    z=dict(all=0.,old=0.,new=0.)
    assert select(z,z)['selected']=='baseline'
    assert select(z,dict(all=.01,old=.02,new=-.02))['selected']=='baseline'
    assert select(z,dict(all=.01,old=-.005,new=.02))['selected']=='candidate'
    return dict(status='PASS',baseline_48_step_max_error=error,hypergradient_finite_difference_error=error_fd,
                engineering_forward_updates=192,projection_and_selector_checked=True)
