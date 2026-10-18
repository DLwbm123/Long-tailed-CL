"""Fixed directional moments and class/group-balanced sufficient statistics."""
import torch
ARMS={'G0':['g'],'G1':['g','h'],'M':['g','h','m'],'N':['g','h','m','q'],'S':['g','h','m','q','v'],'K':['g','h','m','q','k']}
ALPHAS=[1e-4,.01,1.]

def directions(d,k,seed):
    x=torch.randn(d,k,generator=torch.Generator().manual_seed(seed),dtype=torch.float64)
    return x/x.norm(dim=0)

def local_moments(p,r):
    p=p.double();m=p.mean(1);z=p@r;q=(m@r).square();v=z.square().mean(1)-q
    assert v.min()>-1e-8
    return m,q,v.clamp_min(0)

def weights(y,groups):
    from collections import Counter
    counts=Counter(groups);byclass={int(c):len({groups[i] for i in torch.where(y==c)[0].tolist()}) for c in y.unique()}
    return torch.tensor([1/(byclass[int(c)]*counts[g]) for c,g in zip(y,groups)],dtype=torch.float64)

def holdout(y,groups,seed=74006):
    rng=torch.Generator().manual_seed(seed);valid=set();missing=[]
    for c in y.unique().tolist():
        gs=sorted({groups[i] for i in torch.where(y==c)[0].tolist()})
        if len(gs)==1:missing.append(c);continue
        ids=torch.randperm(len(gs),generator=rng)[:max(1,int(.2*len(gs)))].tolist();valid.update(gs[i] for i in ids)
    mask=torch.tensor([g in valid for g in groups]);assert mask.any() and (~mask).any()
    return mask,missing

def norm_fit(x,w):
    w=w/w.sum();mu=(w[:,None]*x).sum(0);scale=((x-mu).square().sum(1)*w).sum().sqrt().clamp_min(1e-12)
    return mu,scale

def normalized(raw,norm,rk):
    z={k:(x-norm[k][0])/norm[k][1] for k,x in raw.items()}
    k=torch.cat([z[t] for t in ['g','h','m']],1)@rk
    if 'k' in norm:z['k']=(k.square()-norm['k'][0])/norm['k'][1]
    else:z['k']=k.square()
    return z

def features(z,arm):
    x=torch.cat([z[k] for k in ARMS[arm]],1);return torch.cat([x,torch.ones(len(x),1,dtype=x.dtype)],1)

def stats(x,y,groups,classes):
    w=weights(y,groups);hot=torch.nn.functional.one_hot(y,classes).double()
    return x.T@(x*w[:,None]),x.T@(hot*w[:,None])

def solve(xx,xy,alpha,classes,device='cuda'):
    a=(xx/classes).to(device);b=(xy/classes).to(device);pen=torch.full((len(a),),alpha,dtype=a.dtype,device=device);pen[-1]=0
    a.diagonal().add_(pen);return torch.linalg.solve(a,b).cpu()

def ba(s,y):
    p=s.argmax(1);return sum(float((p[y==c]==c).double().mean()) for c in y.unique())/len(y.unique())

def selfcheck():
    torch.manual_seed(9);p=torch.randn(12,5,4,dtype=torch.float64);r=directions(4,7,8);m,q,v=local_moments(p,r)
    assert torch.allclose(q+v,(p@r).square().mean(1),atol=1e-12)
    assert torch.equal(r,directions(4,7,8))
    y=torch.tensor([0]*6+[1]*6);g=[str(i//2) for i in range(12)];mask,missing=holdout(y,g)
    assert not {g[i] for i in torch.where(mask)[0]}&{g[i] for i in torch.where(~mask)[0]}
    x=torch.cat([m,torch.ones(12,1)],1);a,b=stats(x,y,g,2);a0,b0=stats(x[:6],y[:6],g[:6],2);a1,b1=stats(x[6:],y[6:],g[6:],2)
    assert torch.allclose(solve(a,b,.01,2,'cpu'),solve(a0+a1,b0+b1,.01,2,'cpu'),atol=1e-10)
    print('SELFCHECK_PASS',flush=True)
if __name__=='__main__':selfcheck()
