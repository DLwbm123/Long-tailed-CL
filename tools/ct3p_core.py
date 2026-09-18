"""Fixed CT3-P translation and differentiable pointwise FD; no data access."""
from contextlib import contextmanager
import copy
import numpy as np
import torch
from ct1_statistics import empty,append,transport,ridge


def translation(before,after,y):
    classes=np.unique(y);w=np.zeros(len(y))
    for c in classes:w[y==c]=1/(len(classes)*np.sum(y==c))
    delta=after-before;b=w@delta;r=w@((delta-b)**2)
    return np.ones(before.shape[1]),b,r,dict(b_norm=float(np.linalg.norm(b)),residual_mean=float(r.mean()))


@contextmanager
def pointwise(net,learner):
    flags=[(m,m.training) for m in net.modules()]
    pools=[(q,q.batchwise_prompt) for q in (net.backbone.pool,net.backbone.pool_few)]
    attrs=[(m,hasattr(m,'adapt_list'),getattr(m,'adapt_list',None)) for m in net.modules() if hasattr(m,'shared')]
    rng=learner._capture_rng_state()
    try:
        net.eval()
        for q,_ in pools:q.batchwise_prompt=False
        yield
    finally:
        for m,f in flags:m.training=f
        for q,f in pools:q.batchwise_prompt=f
        for m,existed,value in attrs:
            if existed:m.adapt_list=value
            elif hasattr(m,'adapt_list'):del m.adapt_list
        learner._restore_rng_state(rng)


def features(net,x,learner):
    with pointwise(net,learner):
        o=net(x,train=False,weight=None)
        return torch.nn.functional.normalize(torch.cat([o['pre_logits'],o['pre_logits_few']],1),dim=1)


def fd(student,teacher,x,learner):
    with torch.no_grad():target=features(teacher,x,learner)
    z=features(student,x,learner)
    distances=(z-target).square().sum(1)
    return distances.mean(),distances.detach()


def math_check():
    rng=np.random.default_rng(47001);X=rng.normal(size=(27,12));y=np.repeat(np.arange(6),[3,4,5,4,6,5])
    state=append(empty(12),X[y<2],y[y<2],range(2),1);virtual=X[y<2].copy();labels=y[y<2].copy()
    for task in (2,3):
        cur=(y>=2*(task-1))&(y<2*task);before=X[cur];after=before+rng.normal(size=before.shape)*.2
        a,b,r,_=translation(before,after,y[cur]);prior=copy.deepcopy(state)
        out=transport(state,a,b,r,task)
        np.testing.assert_allclose(out['mu'][:,0]-out['mu'][:,1],state['mu'][:,0]-state['mu'][:,1],atol=1e-12)
        np.testing.assert_array_equal(out['v'],state['v'])
        state=append(out,after,y[cur],range(2*(task-1),2*task),task)
        virtual=np.concatenate([virtual+b,after]);labels=np.concatenate([labels,y[cur]])
        direct=append(empty(12),virtual,labels,range(2*task),task)
        for k in ('S','mu','v'):np.testing.assert_allclose(state[k],direct[k],atol=1e-11)
        W,_=ridge(state);np.testing.assert_allclose(W,ridge(direct)[0],atol=1e-10)
        identity=transport(prior,np.ones(12),np.zeros(12),np.zeros(12),task)
        for k in ('S','mu','v'):np.testing.assert_array_equal(identity[k],prior[k])
    return dict(status='PASS',full_second_moment=True,identity=True,cross_Gram=True,immutable_three_task_virtual_sample_equivalence=True,
                old_pairwise_means_and_centered_covariance_preserved=True)
