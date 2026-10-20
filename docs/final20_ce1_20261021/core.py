"""One fixed curvature-matched CE objective in original ridge score units."""
import numpy as np
from scipy.special import logsumexp


def objective(w,x,y,mass,alpha):
    k=w.shape[1];z=k*(x@w);loss=logsumexp(z,axis=1)-z[np.arange(len(y)),y]
    value=float(mass@loss)/k+alpha*.5*float((w*w).sum())
    prob=np.exp(z-logsumexp(z,axis=1)[:,None]);prob[np.arange(len(y)),y]-=1
    grad=x.T@(prob*mass[:,None])+alpha*w
    return value,grad


def self_check():
    rng=np.random.default_rng(74002);x=rng.normal(size=(7,3));y=np.arange(7)%4;mass=np.ones(7)/7;alpha=.7
    w=rng.normal(size=(3,4))*.1;direction=rng.normal(size=w.shape);epsilon=1e-6
    value,g=objective(w,x,y,mass,alpha)
    finite=(objective(w+epsilon*direction,x,y,mass,alpha)[0]-objective(w-epsilon*direction,x,y,mass,alpha)[0])/(2*epsilon)
    assert abs(finite-float((g*direction).sum()))<1e-7
    zero=np.zeros_like(w);_,gz=objective(zero,x,y,mass,alpha)
    target=np.eye(4)[y]-.25
    assert np.max(np.abs(gz+x.T@(target*mass[:,None])))<1e-12
    direction-=direction.mean(1,keepdims=True)
    hv=(objective(epsilon*direction,x,y,mass,alpha)[1]-objective(-epsilon*direction,x,y,mass,alpha)[1])/(2*epsilon)
    expected=x.T@((x@direction)*mass[:,None])+alpha*direction
    assert np.max(np.abs(hv-expected))<1e-7
    return True
