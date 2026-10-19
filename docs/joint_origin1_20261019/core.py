import torch


def surrogate_transform(mapping,scale):
    d=mapping.shape[1]
    t=torch.zeros(d+1,2*d+1,dtype=mapping.dtype)
    t[:d,:d]=torch.eye(d,dtype=mapping.dtype)/2**.5
    t[:,d:2*d]=mapping*scale/2**.5;t[-1,-1]=1
    return t


def same_block(gram,target,cross,alpha):
    penalty=torch.full((len(gram),),2*alpha,dtype=gram.dtype);penalty[-1]=alpha
    q=gram+torch.diag(penalty)
    return torch.linalg.solve(q,target),torch.linalg.solve(q,cross)
