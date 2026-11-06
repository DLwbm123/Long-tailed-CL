"""Actual-byte feature memory and subset-enforced access; no hidden old-view lookup."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
TOKEN_BYTES=2064
STATS_BYTES=23*8
BUDGET_BYTES=256*TOKEN_BYTES+STATS_BYTES
DTYPE=np.dtype([('feature','<f4',(512,)),('parent','<i8'),('label','<i2'),('view','u1'),('padding','u1',(5,))])
assert DTYPE.itemsize==TOKEN_BYTES

class Bank:
    def __init__(self,z,pid,y,view):
        self.z=z.detach();self.pid=pid.long();self.y=y.long();self.view=view.long();assert self.z.shape==(len(pid),512)
        key=self.pid*8+self.view;order=key.argsort();self.z=self.z[order];self.pid=self.pid[order];self.y=self.y[order];self.view=self.view[order]
        assert len(key.unique())==len(key) and torch.isfinite(self.z).all() and ((self.view>=0)&(self.view<6)).all()
        g=self.view==0;self.gidx=torch.where(g)[0];self.parents=self.pid[g];assert len(self.parents)==len(self.pid.unique())
        self.parent_index=torch.searchsorted(self.parents,self.pid);assert torch.equal(self.parents[self.parent_index],self.pid)
        assert torch.equal(self.y,self.y[g][self.parent_index])
    def __len__(self):return len(self.pid)
    def take(self,ix):
        return Bank(self.z[ix],self.pid[ix],self.y[ix],self.view[ix])
    def parents_only(self,ids):return self.take(torch.isin(self.pid,ids))
    def bags(self):
        local=self.view>0;total=torch.zeros(len(self.parents),512,device=self.z.device);den=torch.zeros(len(self.parents),device=self.z.device)
        total.index_add_(0,self.parent_index[local],self.z[local]);den.index_add_(0,self.parent_index[local],torch.ones(int(local.sum()),device=self.z.device))
        x=torch.cat([self.z[self.gidx],total/den.clamp_min(1)[:,None],(den>0).float()[:,None]],-1)
        return x.detach(),self.y[self.gidx],self.parents
    def assert_subset(self,pool):
        key=self.pid*8+self.view;source=pool.pid*8+pool.view;ix=torch.searchsorted(source,key);assert (ix<len(source)).all() and torch.equal(source[ix],key)
        assert torch.equal(self.z,pool.z[ix]) and torch.equal(self.y,pool.y[ix]);return True
    def serialize(self,path,counts,birth,capacity=256):
        assert len(self)<=capacity and len(self)*TOKEN_BYTES+STATS_BYTES<=capacity*TOKEN_BYTES+STATS_BYTES
        a=np.zeros(len(self),dtype=DTYPE);a['feature']=self.z.cpu().numpy();a['parent']=self.pid.cpu().numpy();a['label']=self.y.cpu().numpy();a['view']=self.view.cpu().numpy();a.tofile(path)
        st=np.zeros(23,dtype=[('count','<u4'),('birth','<i2'),('padding','<u2')]);st['birth']=-1;st['count'][:len(counts)]=np.asarray(counts);st['birth'][:len(birth)]=np.asarray(birth);st.tofile(str(path)+'.stats')
        assert Path(path).stat().st_size+Path(str(path)+'.stats').stat().st_size==len(self)*TOKEN_BYTES+STATS_BYTES
        restore=np.fromfile(path,dtype=DTYPE);assert np.array_equal(restore['feature'],a['feature']) and np.array_equal(restore['parent'],a['parent']) and np.array_equal(restore['view'],a['view'])
        return len(self)*TOKEN_BYTES+STATS_BYTES

def join(*banks):
    banks=[b for b in banks if b is not None]
    return Bank(*[torch.cat([getattr(b,k) for b in banks]) for k in ['z','pid','y','view']])

def from_views(z,ids,labels):
    return Bank(z.flatten(0,1),ids.repeat_interleave(6),labels.repeat_interleave(6),torch.arange(6,device=z.device).repeat(len(ids)))

class Bags:
    def __init__(self,bank,C):
        self.x,self.y,self.pid=bank.bags();order=self.y.argsort();self.x=self.x[order];self.y=self.y[order];self.pid=self.pid[order];self.nc=torch.bincount(self.y,minlength=C);assert (self.nc>0).all();self.offset=self.nc.cumsum(0)-self.nc;self.C=C
    def sample(self,n,g):
        cls=torch.randint(self.C,(n,),device=self.x.device,generator=g);u=torch.rand(n,device=self.x.device,generator=g);ix=self.offset[cls]+(u*self.nc[cls]).long();return self.x[ix],self.y[ix]
