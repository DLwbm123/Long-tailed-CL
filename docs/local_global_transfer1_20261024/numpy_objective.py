"""Independent NumPy paired criterion and projection head, no Torch autograd."""
import numpy as np
from scipy.special import erf


def redundancy(first, second):
    n,d=first.shape[0],first.shape[-1]
    value=25*np.mean((first-second)**2)
    for x in (first,second):
        centered=x-x.mean(axis=0,keepdims=True)
        cov=np.einsum('n...d,n...e->...de',centered,centered)/(n-1)
        diagonal=np.diagonal(cov,axis1=-2,axis2=-1)
        value+=12.5*np.maximum(0,1-np.sqrt(diagonal+1e-4)).mean()
        value+=((cov**2).sum(axis=(-2,-1))-(diagonal**2).sum(axis=-1)).mean()/d
    return float(value)


def project(value,state,name):
    x=value@np.asarray(state[name+'.0.weight']).T+np.asarray(state[name+'.0.bias'])
    x=.5*x*(1+erf(x/np.sqrt(2)))
    return x@np.asarray(state[name+'.2.weight']).T+np.asarray(state[name+'.2.bias'])


def objective(tokens,labels,receipt,state,head,arm):
    tokens=np.asarray(tokens,dtype=np.float64);n=len(labels)
    descriptor=np.concatenate((tokens[:,0],tokens[:,1:].mean(axis=1)),axis=-1)
    if arm=='BASE_SUPERVISED':
        descriptor/=np.maximum(np.linalg.norm(descriptor,axis=1,keepdims=True),1e-12)
        head=np.asarray(head);head=head/np.maximum(np.linalg.norm(head,axis=1,keepdims=True),1e-12)
        logits=25*descriptor@head.T;logits-=logits.max(axis=1,keepdims=True)
        target=np.concatenate((labels,labels))
        return float(np.mean(np.log(np.exp(logits).sum(axis=1))-logits[np.arange(2*n),target]))
    global_values=project(descriptor,state,'global_map')
    value=redundancy(global_values[:n],global_values[n:])
    if arm=='GLOBAL_REDUNDANCY':return value
    local_values=project(tokens[:,1:],state,'local_map')
    first=local_values[:n,receipt['first_cells']];second=local_values[n:,receipt['second_cells']]
    return .5*(value+redundancy(first,second))
