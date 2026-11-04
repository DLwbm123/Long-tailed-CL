"""Deterministic structural plans; no optimizer or query-guided construction."""
import itertools, math
import torch
import torch.nn.functional as F

def unit(x):
    return F.normalize(x,dim=-1,eps=1e-8)

def key(x):
    return tuple(x.detach().cpu().tolist())

def bank(z, components, cap=20):
    order=sorted(range(len(z)),key=lambda i:key(z[i]))
    unique=[]; seen=set()
    for i in order:
        if components[i] not in seen:
            unique.append(i);seen.add(components[i])
    z=z[unique]
    if len(z)<=cap:return z
    centroid=unit(z.mean(0)); first=int((z@centroid).argmax())
    selected=[first]
    while len(selected)<cap:
        distance=torch.cdist(z,z[selected]).square().min(1).values
        distance[selected]=-1
        selected.append(int(distance.argmax()))
    return z[selected]

def splits(z):
    if len(z)<4:return []
    pairs=list(itertools.combinations(range(len(z)),2))
    pairs.sort(key=lambda ij:(-float((z[ij[0]]-z[ij[1]]).square().sum()),key(z[ij[0]]),key(z[ij[1]])))
    out=[];seen=set()
    for i,j in pairs:
        side=((z-z[i]).square().sum(1)>(z-z[j]).square().sum(1))
        groups=[tuple(torch.where(~side)[0].tolist()),tuple(torch.where(side)[0].tolist())]
        if min(map(len,groups))<2:continue
        partition=tuple(sorted(groups))
        if partition in seen:continue
        seen.add(partition)
        out.append(torch.stack([unit(z[list(g)].mean(0)) for g in groups]))
        if len(out)==2:break
    return out

def blocks(base,candidates):
    eligible=[i for i,v in enumerate(candidates) if len(v)==2]
    out=[]
    while len(eligible)>=2 and len(out)<4:
        pairs=list(itertools.combinations(eligible,2))
        pairs.sort(key=lambda ij:(-float(base[ij[0]]@base[ij[1]]),tuple(sorted((key(base[ij[0]]),key(base[ij[1]]))))))
        pair=pairs[0];out.append(pair)
        eligible=[c for c in eligible if c not in pair]
    return out

def scores(q,base,candidates,pair):
    raw=(20*q@base.T).double()
    columns={}
    for c in pair:
        columns[c]=[raw[:,c]]+[(torch.logsumexp((20*q@p.T).double(),dim=1)-math.log(2)) for p in candidates[c]]
    result=[]
    for actions in itertools.product(range(3),repeat=2):
        v=raw.clone()
        for c,a in zip(pair,actions):v[:,c]=columns[c][a]
        result.append(v)
    return torch.stack(result)

def metrics(s,y,tail,old):
    pred=s.argmax(-1);C=s.shape[-1];den=torch.bincount(y,minlength=C).double()
    assert (den>0).all()
    correct=torch.stack([torch.bincount(y[p==y],minlength=C) for p in pred]).double()
    rec=correct/den
    macro=rec.mean(1);t=rec[:,tail].mean(1);o=rec[:,old].mean(1) if old.any() else macro*0
    reward=macro-macro[0]-torch.relu(o[0]-o)-torch.relu(t[0]-t)
    ref=pred[0];good=ref==y
    harm=((pred!=y)&good).sum(1);rescue=((pred==y)&~good).sum(1)
    tiny=torch.tensor([1,2,12],device=y.device)
    harmtiny=((pred!=y)&good&torch.isin(y,tiny)).sum(1)
    return dict(macro=macro,tail=t,old=o,reward=reward,harm=harm,rescue=rescue,tiny_harm=harmtiny,changes=(pred!=ref).sum(1)),pred

def select(reward):
    plans=list(itertools.product(range(3),repeat=2))
    return min(range(9),key=lambda j:(-float(reward[j]),sum(a!=0 for a in plans[j]),j))

def selfcheck():
    torch.manual_seed(74002)
    z=unit(torch.randn(24,8));comp=list(range(24));b=bank(z,comp);assert len(b)==20
    perm=torch.randperm(len(z));assert torch.equal(b,bank(z[perm],[comp[i] for i in perm]))
    assert splits(z[:1])==[] and splits(z[:3])==[]
    sp=splits(b);assert len(sp)==2 and all(p.shape==(2,8) for p in sp)
    q=unit(torch.randn(12,8));base=unit(torch.randn(3,8));v=scores(q,base,[sp,sp,[]],(0,1))
    assert torch.equal(v[0],(20*q@base.T).double())
    assert torch.allclose(torch.logsumexp(v[0,:,0,None].expand(-1,2),1)-math.log(2),v[0,:,0],atol=1e-12)
    y=torch.arange(12)%3;m,p=metrics(v,y,torch.tensor([True,False,False]),torch.tensor([False,False,False]))
    assert float(m['reward'][0])==0 and int(m['harm'][0])==0
    assert select(torch.zeros(9))==0
    # Hard reward is invariant to confidence changes that preserve every argmax.
    w=v[0:1]*1.001;m2,_=metrics(w,y,torch.tensor([True,False,False]),torch.tensor([False,False,False]))
    assert float(m2['reward'][0])==0
    return {'passed':True,'checks':['component unique bank','permutation invariance','tiny split refusal','two valid partitions','exact KEEP','LME normalization','hard reward identity','tie NOOP'],'encoder':0,'optimizer':0,'F_VAL_TEST':0}

if __name__=='__main__':print(selfcheck())
