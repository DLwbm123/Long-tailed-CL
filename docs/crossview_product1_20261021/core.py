"""Fixed cross-view products and base-only affine centering; no learned gates."""
import torch


def raw_dictionary(cls, patch):
    assert cls.dtype==patch.dtype==torch.float64 and cls.shape==patch.shape
    d=cls.shape[1];assert d%2==0
    perm=(torch.arange(d,device=cls.device)+d//2)%d
    ones=torch.ones(len(cls),1,dtype=cls.dtype,device=cls.device)
    return torch.cat([cls,patch,cls*patch,cls*patch[:,perm],ones],1)


def base_means(grams):
    assert grams
    d=(len(grams[0])-1)//4
    assert len(grams[0])==4*d+1
    mass=sum(float(g[-1,-1]) for g in grams)
    assert abs(mass-len(grams))<1e-8
    mean=sum(g[:2*d,-1] for g in grams)/mass
    return mean[:d],mean[d:]


def indices(d, arm, device=None):
    assert d%2==0 and arm in ['cross_aligned','cross_shifted']
    perm=torch.arange(d,device=device)
    if arm=='cross_shifted':perm=(perm+d//2)%d
    product=2*d if arm=='cross_aligned' else 3*d
    ids=torch.tensor([*range(2*d),*range(product,product+d),4*d],device=device)
    return ids,perm


def center_gram(raw_gram, mu_c, mu_p, arm):
    """Sparse right/left congruence, with source rows kept intact during each side."""
    d=len(mu_c);ids,perm=indices(d,arm,raw_gram.device)
    g=raw_gram[ids][:,ids];mp=mu_p[perm];constant=mu_c*mp
    right=g.clone()
    right[:,2*d:3*d]=g[:,2*d:3*d]-g[:,:d]*mp-g[:,d+perm]*mu_c+g[:,-1:]*constant
    out=right.clone()
    out[2*d:3*d,:]=right[2*d:3*d,:]-mp[:,None]*right[:d,:]-mu_c[:,None]*right[d+perm,:]+constant[:,None]*right[-1:,:]
    return out


def center_targets(raw_targets, mu_c, mu_p, arm):
    d=len(mu_c);ids,perm=indices(d,arm,raw_targets.device)
    target=raw_targets[ids].clone();original=target.clone();mp=mu_p[perm]
    target[2*d:3*d]=original[2*d:3*d]-mp[:,None]*original[:d]-mu_c[:,None]*original[d+perm]+(mu_c*mp)[:,None]*original[-1:]
    return target


def factors(base_gram):
    d=(len(base_gram)-1)//3
    ec=float(base_gram[:d,:d].trace());ep=float(base_gram[d:2*d,d:2*d].trace());eh=float(base_gram[2*d:3*d,2*d:3*d].trace())
    assert ec>0 and ep>0 and eh>0, 'Zero-energy cross-view branch'
    out=torch.ones(3*d+1,dtype=base_gram.dtype,device=base_gram.device)
    out[:d]=2**-.5;out[d:2*d]=(ec/ep)**.5*2**-.5;out[2*d:3*d]=(ec/eh)**.5
    return out


def self_check():
    gen=torch.Generator().manual_seed(74002)
    c=torch.randn(19,8,generator=gen,dtype=torch.float64)
    p=torch.randn(19,8,generator=gen,dtype=torch.float64)
    mc=c[:10].mean(0);mp=p[:10].mean(0);raw=raw_dictionary(c,p)
    for arm in ['cross_aligned','cross_shifted']:
        _,perm=indices(8,arm)
        h=(c-mc)*(p[:,perm]-mp[perm]);direct=torch.cat([c,p,h,torch.ones(19,1,dtype=torch.float64)],1)
        g=center_gram(raw.T@raw,mc,mp,arm)
        assert torch.allclose(g,direct.T@direct,atol=1e-11,rtol=1e-12)
        target=center_targets(raw.T@torch.eye(19,dtype=torch.float64),mc,mp,arm)
        assert torch.allclose(target,direct.T,atol=1e-12,rtol=1e-12)
        assert torch.allclose(g,g.T,atol=1e-11,rtol=1e-12)
    zero=raw_dictionary(torch.ones(4,8,dtype=torch.float64),torch.ones(4,8,dtype=torch.float64))
    zg=center_gram(zero.T@zero,torch.ones(8,dtype=torch.float64),torch.ones(8,dtype=torch.float64),'cross_aligned')
    assert float(zg[16:24,16:24].abs().max())==0
    try:factors(zg)
    except AssertionError:pass
    else:raise AssertionError('Zero interaction energy must stop, not create tunable floor')
    return True
