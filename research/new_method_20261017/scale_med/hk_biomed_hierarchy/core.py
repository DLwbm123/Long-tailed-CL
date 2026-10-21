"""Small hierarchy-information ablation; not a HASTEN reproduction."""
import random
from collections import Counter
import torch
from torch import nn
from torch.nn import functional as F

NAMES=['dyed resection margins','hemorroids','ileum','ulcerative colitis grade 3','ulcerative colitis grade 2','bbps 0 1','bbps 2 3','retroflex stomach','retroflex rectum','oesophagitis b d','normal z line','oesophagitis a','ulcerative colitis grade 1 2']
PARENTS=['an anatomical landmark','a pathological finding','bowel preparation quality','a therapeutic intervention','hemorrhoids','ulcerative colitis','esophagitis']
TYPE=[3,1,0,1,1,2,2,0,0,1,0,1,1]
SUB=[-1,0,-1,1,1,-1,-1,-1,-1,2,-1,2,1]
SEEDS=[74510,74511,74512]

def permutation(seed):
    rng=random.Random(seed);order=[]
    for t in [1,0,2,3]:
        ids=[i for i,x in enumerate(TYPE) if x==t];rng.shuffle(ids);order.extend(ids)
    p=[0]*13
    for j,i in enumerate(order):p[i]=order[(j+6)%13]
    assert sorted(p)==list(range(13)) and all(TYPE[p[i]]!=TYPE[i] for i in range(13))
    return p

def weights(y,groups,K=13):
    sizes=Counter(groups);nc=[len({groups[i] for i in torch.where(y==c)[0].tolist()}) for c in range(K)]
    assert all(nc)
    by={}
    for g,c in zip(groups,y.tolist()):assert by.setdefault(g,c)==c
    return torch.tensor([1/(K*nc[c]*sizes[g]) for c,g in zip(y.tolist(),groups)],dtype=torch.double)

def metrics(s,y,groups):
    assert torch.isfinite(s).all() and s.shape==(len(y),13)
    pred=s.argmax(1);n=torch.bincount(y,minlength=13);assert (n>0).all()
    ok=torch.bincount(y[pred==y],minlength=13);r=ok.double()/n
    return dict(BA=float(r.mean()),group_BA=float(weights(y,groups)@(pred==y).double()),correct=ok.tolist(),counts=n.tolist(),recall=r.tolist(),confusion=torch.bincount(y*13+pred,minlength=169).reshape(13,13).tolist())

def paired(a,b,y):
    aa=a.argmax(1)==y;bb=b.argmax(1)==y
    return [dict(class_id=c,rescued=int((~aa&bb&(y==c)).sum()),harmed=int((aa&~bb&(y==c)).sum())) for c in range(13)]

class Adapter(nn.Module):
    def __init__(self,d=512):
        super().__init__();self.visual=nn.Parameter(torch.eye(d));self.text=nn.Parameter(torch.eye(d))
    def logits(self,x,text):
        return 10*F.normalize(x@self.visual.T,dim=1)@F.normalize(text@self.text.T,dim=1).T
    def penalty(self):
        eye=torch.eye(self.visual.shape[0],device=self.visual.device)
        return (self.visual-eye).square().mean()+(self.text-eye).square().mean()

def loss(model,x,y,text,arm,perm):
    s=model.logits(x,text);leaf=F.cross_entropy(s[:,:13],y)
    mapping=torch.tensor(perm,device=y.device);target=mapping[y]
    ty=torch.tensor(TYPE,device=y.device)[target];su=torch.tensor(SUB,device=y.device)[target]
    parent=F.cross_entropy(s[:,13:17],ty)
    mask=su>=0
    sub=F.cross_entropy(s[mask,17:20],su[mask],reduction='sum')/len(y) if mask.any() else s[:,17:20].sum()*0
    h=parent+sub
    return leaf+(0 if arm=='flat' else .5)*h+.001*model.penalty(),leaf,h

def selfcheck():
    for seed in SEEDS:
        p=permutation(seed)
        assert Counter(TYPE[i] for i in p)==Counter(TYPE)
        assert Counter(SUB[i] for i in p)==Counter(SUB)
    torch.manual_seed(7);m=Adapter(8);x=F.normalize(torch.randn(13,8),dim=1);t=F.normalize(torch.randn(20,8),dim=1);y=torch.arange(13)
    a,_,_=loss(m,x,y,t,'true',list(range(13)));b,_,_=loss(m,x,y,t,'shuffled',list(range(13)));assert torch.equal(a,b)
    a.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    flat,_,_=loss(m,x,y,t,'flat',list(range(13)));t2=t.clone();t2[13:]=torch.randn_like(t2[13:]);other,_,_=loss(m,x,y,t2,'flat',list(range(13)));assert torch.equal(flat,other)
    assert abs(float(weights(y,list(range(13))).sum())-1)<1e-12
    return dict(shuffle_all_top_level_changed=True,branch_sizes_preserved=True,identity_shuffle_loss_exact=True,flat_parent_independence=True,finite_gradients=True)
