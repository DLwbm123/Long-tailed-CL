"""Training-only scale and ridge effective-degrees-of-freedom controls."""
import torch


def spectrum(gram):
    values=torch.linalg.eigvalsh((gram+gram.T)/2)
    tolerance=max(float(values.abs().max()),1.)*1e-10
    assert float(values.min())>=-tolerance,'Substantial non-PSD Gram'
    return values.clamp_min(0)


def degrees(values,alpha):
    return float((values/(values+alpha)).sum())


def match_alpha(values,target):
    low,high=1e-12,1.
    assert degrees(values,low)>target
    while degrees(values,high)>target:
        high*=2
        assert high<=1e12
    for _ in range(80):
        mid=(low+high)/2
        if degrees(values,mid)>target:low=mid
        else:high=mid
    alpha=(low+high)/2
    assert abs(degrees(values,alpha)-target)<1e-7
    return alpha


def scaled(gram,target,patch_scale):
    d=(len(gram)-1)//2
    factors=torch.ones(len(gram),dtype=gram.dtype,device=gram.device)
    factors[:d]=2**-.5;factors[d:2*d]=patch_scale*2**-.5
    return gram*factors[:,None]*factors[None,:],target*factors[:,None],factors
