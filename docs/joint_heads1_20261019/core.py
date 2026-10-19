"""Joint sufficient statistics and exact duplicate-feature ridge control."""
import torch


def joint_gram(cls, patch, cross):
    d=len(cls)-1
    g=torch.zeros(2*d+1,2*d+1,dtype=cls.dtype,device=cls.device)
    g[:d,:d]=cls[:-1,:-1];g[d:2*d,d:2*d]=patch[:-1,:-1]
    g[:d,d:2*d]=cross;g[d:2*d,:d]=cross.T
    g[:d,-1]=cls[:-1,-1];g[-1,:d]=cls[-1,:-1]
    g[d:2*d,-1]=patch[:-1,-1];g[-1,d:2*d]=patch[-1,:-1]
    g[-1,-1]=cls[-1,-1]
    return g


def duplicate_head(gram,target):
    # Two unscaled CLS blocks with unit ridge reduce to half feature penalty.
    penalty=torch.ones(len(gram),dtype=gram.dtype,device=gram.device)*.5
    penalty[-1]=1
    return torch.linalg.solve(gram+torch.diag(penalty),target)
