"""Charged role/core flags, protected query globals, and support-only anchors."""
import hashlib,importlib
from pathlib import Path
import numpy as np
import torch
from .supply import birth_roles

mem=importlib.import_module('r110_ecv_opportunity.base.memory')
legacy=importlib.import_module('r110_ecv_opportunity.base.legacy_manager')
qchoice=importlib.import_module('r111_risk_qualification.readiness').query_free_choice
Bank=mem.Bank
DTYPE=np.dtype([('feature','<f4',(512,)),('parent','<i8'),('label','<i2'),('view','u1'),('flag','u1'),('padding','u1',(4,))])
STATS=np.dtype([('count','<u4'),('birth','<i2'),('birth_total','<u2')])
assert DTYPE.itemsize==2064 and STATS.itemsize*23==184


class Evidence:
    def __init__(self,bank,flags,stats):
        self.bank=bank;self.flags=flags.to(device=bank.z.device,dtype=torch.uint8);self.stats=np.array(stats,dtype=STATS,copy=True)
        assert len(self.flags)==len(bank) and len(self.stats)==23
        assert set(self.flags.tolist())<={0,1,2,4}
        assert torch.equal(self.flags&3,(self.flags[bank.gidx]&3)[bank.parent_index]),'MIXED_ROLE_PARENT'
    @property
    def C(self):return int((self.stats['birth']>=0).sum())
    def role(self,role):return self.bank.take(torch.where((self.flags&3)==role)[0])
    def core_keys(self):return (self.bank.pid*8+self.bank.view)[self.flags==4]
    def with_support(self,support):
        support.assert_subset(self.role(0));q=self.bank.take(torch.where((self.flags&3)>0)[0]);out=mem.join(support,q)
        ix=torch.searchsorted(self.bank.pid*8+self.bank.view,out.pid*8+out.view)
        result=Evidence(out,self.flags[ix],self.stats);result.validate();return result
    def validate(self):
        b=self.bank;C=self.C
        assert len(b)<=256 and len(b)*2064+184<=528568,'PERSISTENT_BYTE_LIMIT'
        assert set(self.role(0).y.tolist())==set(range(C))
        assert len(self.core_keys())==C and bool((b.view[self.flags==4]==0).all())
        for role in [1,2]:
            q=self.role(role);assert len(q)==C and set(q.y.tolist())==set(range(C)) and bool((q.view==0).all())
        assert not set(self.role(0).parents.tolist())&set(self.role(1).parents.tolist())
        assert not set(self.role(0).parents.tolist())&set(self.role(2).parents.tolist())
        return self
    def serialize(self,path):
        self.validate();p=Path(path);b=self.bank;a=np.zeros(len(b),dtype=DTYPE)
        for name,attr in [('feature','z'),('parent','pid'),('label','y'),('view','view')]:a[name]=getattr(b,attr).cpu().numpy()
        a['flag']=self.flags.cpu().numpy();a.tofile(p);self.stats.tofile(str(p)+'.stats')
        size=p.stat().st_size+Path(str(p)+'.stats').stat().st_size;assert size==len(b)*2064+184<=528568
        return size
    @classmethod
    def restore(cls,path,device):
        a=np.fromfile(path,dtype=DTYPE);s=np.fromfile(str(path)+'.stats',dtype=STATS);assert len(s)==23
        b=Bank(torch.tensor(a['feature'].copy(),device=device),torch.tensor(a['parent'].copy(),device=device),torch.tensor(a['label'].copy(),device=device),torch.tensor(a['view'].copy(),device=device))
        return cls(b,torch.tensor(a['flag'].copy(),device=device),s).validate()


def born(full,rows,stage):
    ids=full.parents.tolist();roles=birth_roles(ids,rows);chosen={};keep=[];flags=[]
    for c in sorted(set(full.y.tolist())):
        ci=full.parents[full.y[full.gidx]==c].tolist()
        for role in ['S','FIT','AUDIT']:
            available=[i for i in ci if roles[i]==role];assert available,'BIRTH_ROLE_MISSING'
            chosen[c,role]=min(available,key=lambda i:hashlib.sha256(('r112-evidence-v1:'+rows[i]['identity_component']).encode()).digest())
    for j,(pid,c,view) in enumerate(zip(full.pid.tolist(),full.y.tolist(),full.view.tolist())):
        role=roles[pid]
        if role=='S':keep.append(j);flags.append(4 if pid==chosen[c,'S'] and view==0 else 0)
        elif pid==chosen[c,role] and view==0:keep.append(j);flags.append(1 if role=='FIT' else 2)
    stats=np.zeros(23,dtype=STATS);stats['birth']=-1
    for c in sorted(set(full.y.tolist())):
        ci=full.parents[full.y[full.gidx]==c].tolist();assert len(ci)<2**16,'BIRTH_COUNT_REPRESENTATION'
        stats[c]=(sum(roles[i]=='S' for i in ci),stage,len(ci))
    return Evidence(full.take(torch.tensor(keep,device=full.z.device)),torch.tensor(flags,device=full.z.device),stats)


def join(*items):
    items=[x for x in items if x is not None];banks=[x.bank for x in items];keys=torch.cat([b.pid*8+b.view for b in banks]);flags=torch.cat([x.flags for x in items])[keys.argsort()]
    stats=np.zeros(23,dtype=STATS);stats['birth']=-1
    for e in items:
        mask=e.stats['birth']>=0
        old=(stats['birth']>=0)&mask
        assert not old.any(),'REPEATED_CLASS_PACKET'
        stats[mask]=e.stats[mask]
    return Evidence(mem.join(*banks),flags,stats)


class Choice(legacy.Choice):
    def __init__(self,pool,C,capacity,core):
        self.__dict__.update(qchoice(pool,C,capacity).__dict__)
        keys=pool.pid*8+pool.view;ix=torch.searchsorted(keys,core);assert bool((ix<len(keys)).all()) and torch.equal(keys[ix],core)
        self.cover=torch.cat([self.cover,ix[~torch.isin(ix,self.cover)]])
        self.optional=pool.gidx[~torch.isin(pool.gidx,self.cover)]
        self.capacity_hist=torch.bincount(self.localcap[pool.parent_index[self.optional]],minlength=6)
        self.covercap=int(self.localcap[pool.parent_index[self.cover]].sum());D=len(self.cover)
        ng=torch.arange(D,min(self.B,len(pool.gidx))+1,device=pool.z.device)
        self.ng=ng[legacy.top_capacity(self.capacity_hist.expand(len(ng),-1),ng-D)+self.covercap>=self.B-ng]
        assert len(self.ng),'PINNED_SUPPORT_INFEASIBLE'
    def estimate(self,score):
        g=score[self.optional].sort(descending=True).values.cumsum(0);g=torch.cat([g.new_zeros(1),g])
        loc=score[self.pool.view>0].sort(descending=True).values.cumsum(0);loc=torch.cat([loc.new_zeros(1),loc])
        return score[self.cover].sum()+g[self.ng-len(self.cover)]+loc[(self.B-self.ng).clamp(max=len(loc)-1)]


def anchors(evidence,seed):
    S=evidence.role(0);C=evidence.C;B=256-len(evidence.role(1))-len(evidence.role(2))
    choice=Choice(S,C,B,evidence.core_keys())
    z=choice.construct('RANDOM',generator=torch.Generator(device=S.z.device).manual_seed(seed))[0]
    b=choice.construct('BALANCED')[0]
    for bank in [z,b]:
        assert set(evidence.core_keys().tolist())<=set((bank.pid*8+bank.view).tolist())
        evidence.with_support(bank)
    return z,b,{'S_capacity':B,'FIT_tokens':C,'AUDIT_tokens':C,'mandatory_support_globals_union':len(choice.cover),'S_pool_tokens':len(S),'measurement_induced_pressure':len(S)>B and len(S)<=256}


def compressed(evidence,seed):
    S=evidence.role(0);C=evidence.C
    if len(S)<=256-2*C:return evidence.with_support(S)
    choice=Choice(S,C,256-2*C,evidence.core_keys())
    zero=choice.construct('RANDOM',generator=torch.Generator(device=S.z.device).manual_seed(seed))[0]
    return evidence.with_support(zero)


def classes(evidence,selected,mapping=None,refresh_counts=False):
    b=evidence.bank;ix=torch.where(torch.isin(b.y,torch.tensor(selected,device=b.z.device)))[0];bank=b.take(ix)
    if mapping is not None:bank=Bank(bank.z,bank.pid,torch.tensor(mapping,device=b.z.device)[bank.y],bank.view)
    stats=np.zeros(23,dtype=STATS);stats['birth']=-1
    support=evidence.role(0) if refresh_counts else None
    for c in selected:
        j=c if mapping is None else mapping[c];stats[j]=evidence.stats[c]
        if refresh_counts:stats[j]['count']=int((support.y[support.gidx]==c).sum())
    return Evidence(bank,evidence.flags[ix],stats)


def subset(evidence,bank):
    bank.assert_subset(evidence.bank);ix=torch.searchsorted(evidence.bank.pid*8+evidence.bank.view,bank.pid*8+bank.view)
    return Evidence(bank,evidence.flags[ix],evidence.stats)
