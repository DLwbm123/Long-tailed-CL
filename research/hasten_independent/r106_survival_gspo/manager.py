"""Canonical global-then-local sets, exact legal masks and shared survival-value input."""
import torch
from torch import nn
import torch.nn.functional as F
from memory import Bank

class ValueNet(nn.Module):
    def __init__(self):super().__init__();self.encoder=nn.Sequential(nn.Linear(528,32),nn.Tanh());self.head=nn.Linear(32,1)
    def forward(self,x):return self.head(self.encoder(x)).squeeze(-1)
class Policy(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(43,32),nn.Tanh(),nn.Linear(32,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x).squeeze(-1)

@torch.no_grad()
def raw_features(pool,m,observed,birth,stage,B):
    x,yy,_=pool.bags();scores=m(x).double();p=scores.softmax(-1);true=p.gather(1,yy[:,None]).squeeze(1);other=scores.clone();other.scatter_(1,yy[:,None],-1e6);margin=(scores.gather(1,yy[:,None]).squeeze(1)-other.max(-1).values)/40
    if m.C==1:margin.zero_()
    entropy=-(p*p.clamp_min(1e-30).log()).sum(-1)/max(1.,__import__('math').log(m.C));proto=F.normalize(m.head[:m.C],dim=-1);relation=proto@proto.T;relation.fill_diagonal_(-2);comp=relation.max(-1).values
    if m.C==1:comp.zero_()
    n=torch.tensor(observed,device=pool.z.device,dtype=torch.float);born=torch.tensor(birth,device=pool.z.device);pn=torch.bincount(yy,minlength=m.C).float();parent=pool.parent_index;cl=pool.y
    scal=torch.stack([torch.log1p(n[cl])/9,torch.log1p(pn[cl])/9,(stage-born[cl]).float()/4,margin[parent].float(),entropy[parent].float(),comp[cl],F.cosine_similarity(pool.z,pool.z[pool.gidx[parent]],dim=-1),torch.full_like(cl,B/max(1,len(pool)),dtype=torch.float),(born[cl]==stage).float(),true[parent].float()],-1)
    out=torch.cat([pool.z,F.one_hot(pool.view,6).float(),scal],-1);assert out.shape==(len(pool),528) and torch.isfinite(out).all();return out.detach()

def top_capacity(hist,k):
    left=torch.broadcast_to(torch.as_tensor(k,device=hist.device),hist.shape[:-1]).clone();out=torch.zeros_like(left)
    for q in range(5,-1,-1):take=torch.minimum(left,hist[...,q]);out+=q*take;left-=take
    return out

class Choice:
    def __init__(self,pool,m,observed,birth,stage,capacity,value):
        self.pool=pool;self.B=min(capacity,len(pool));self.C=m.C;self.raw=raw_features(pool,m,observed,birth,stage,self.B)
        with torch.no_grad():self.emb=value.encoder(self.raw).detach();self.val=value.head(self.emb).squeeze(-1).detach()
        g=pool.gidx;gy=pool.y[g];cent=torch.stack([pool.z[g[gy==c]].mean(0) for c in range(self.C)]);self.cent=F.normalize(cent,dim=-1)
        cov=[]
        for c in range(self.C):ix=g[gy==c];cov.append(ix[(pool.z[ix]@self.cent[c]).argmax()])
        self.cover=torch.stack(cov);self.localcap=torch.bincount(pool.parent_index[pool.view>0],minlength=len(g));self.optional=g[~torch.isin(g,self.cover)]
        caps=self.localcap[pool.parent_index[self.optional]];h=F.one_hot(caps,6).long();self.suffix=torch.cat([h.flip(0).cumsum(0).flip(0),torch.zeros(1,6,device=g.device,dtype=torch.long)])
        self.covercap=int(self.localcap[pool.parent_index[self.cover]].sum());ng=torch.arange(self.C,min(self.B,len(g))+1,device=g.device);need=ng-self.C
        mask=top_capacity(self.suffix[0].expand(len(ng),-1),need)+self.covercap>=self.B-ng;self.ng=ng[mask];assert len(self.ng)>0
    def state(self,ix,selected,ng,phase,fraction=None):
        p=self.pool;gs=torch.bincount(p.y[selected[p.view[selected]==0]],minlength=self.C).float();ls=torch.bincount(p.y[selected[p.view[selected]>0]],minlength=self.C).float();ps=torch.bincount(p.parent_index[selected[p.view[selected]>0]],minlength=len(p.parents)).float()
        if ix is None:
            n=len(self.ng);base=self.emb[p.gidx].mean(0).expand(n,-1);v=self.estimate(self.val).div(self.B);dy=torch.zeros(n,10,device=p.z.device);dy[:,3]=1;dy[:,4]=self.ng/self.B;dy[:,5]=len(self.cover)/self.B;dy[:,6]=1;dy[:,9]=self.ng/self.B
        else:
            n=len(ix);base=self.emb[ix];v=self.val[ix];cl=p.y[ix];dy=torch.stack([gs[cl]/self.B,ls[cl]/self.B,ps[p.parent_index[ix]]/5,torch.full((n,),(self.B-len(selected))/self.B,device=p.z.device),torch.full((n,),ng/self.B,device=p.z.device),torch.full((n,),float((p.view[selected]==0).sum())/self.B,device=p.z.device),torch.zeros(n,device=p.z.device),torch.full((n,),float(phase==1),device=p.z.device),torch.full((n,),float(phase==2),device=p.z.device),torch.zeros(n,device=p.z.device)],-1)
        return torch.cat([base,v[:,None],dy],-1).detach()
    def estimate(self,score):
        # Greedy additive value upper bound ignores parent coupling here; actual construction enforces it.
        g=score[self.optional].sort(descending=True).values.cumsum(0);g=torch.cat([torch.zeros(1,device=g.device),g]);loc=score[self.pool.view>0].sort(descending=True).values.cumsum(0);loc=torch.cat([torch.zeros(1,device=g.device),loc]);k=(self.B-self.ng).clamp(max=len(loc)-1)
        return score[self.cover].sum()+g[self.ng-self.C]+loc[k]
    def construct(self,mode,actor=None,generator=None,record=False):
        p=self.pool;selected=self.cover.clone();trace=[]
        def decide(x,score,keys=None):
            if mode=='POLICY' or mode=='RANDOM':
                with torch.no_grad():lp=(actor(x).double() if mode=='POLICY' else torch.zeros(len(x),device=x.device,dtype=torch.double)).log_softmax(0);j=int(torch.multinomial(lp.exp(),1,generator=generator))
                if record:trace.append({'x':x,'old_logp':lp.detach(),'choice':j,'keys':keys.detach() if keys is not None else None})
            else:j=int(score.argmax())
            return j
        if mode=='GLOBAL':j=int((self.ng-self.B).abs().argmin())
        elif mode=='BALANCED':j=int((self.ng-max(self.C,self.B//2)).abs().argmin())
        elif mode=='UNCERTAINTY':
            score=(1-self.raw[:,-1])*(self.raw[:,-10]*9).exp().clamp_min(1).rsqrt()*(2-self.raw[:,-4]);j=decide(self.state(None,selected,0,0),self.estimate(score),-self.ng-1)
        else:j=decide(self.state(None,selected,0,0),self.estimate(self.val),-self.ng-1)
        ng=int(self.ng[j]);last=-1;cap=self.covercap
        while int((p.view[selected]==0).sum())<ng:
            pos=torch.arange(last+1,len(self.optional),device=p.z.device);ix=self.optional[pos];left=ng-int((p.view[selected]==0).sum())-1
            valid=(len(self.optional)-pos-1>=left)&(cap+self.localcap[p.parent_index[ix]]+top_capacity(self.suffix[pos+1],left)>=self.B-ng);pos=pos[valid];ix=ix[valid];assert len(ix)
            x=self.state(ix,selected,ng,1)
            if mode in ['GLOBAL','BALANCED']:
                gy=p.y[selected[p.view[selected]==0]];n=torch.bincount(gy,minlength=self.C).float();score=-100*n[p.y[ix]]+(p.z[ix]*self.cent[p.y[ix]]).sum(-1)
            elif mode=='UNCERTAINTY':score=(1-self.raw[ix,-1])*(self.raw[ix,-10]*9).exp().clamp_min(1).rsqrt()/(1+torch.bincount(p.y[selected],minlength=self.C)[p.y[ix]])
            else:score=self.val[ix]
            k=decide(x,score,p.pid[ix]*8+p.view[ix]);last=int(pos[k]);chosen=ix[k:k+1];cap+=int(self.localcap[p.parent_index[chosen]]);selected=torch.cat([selected,chosen])
        allowed=(p.view>0)&torch.isin(p.parent_index,p.parent_index[selected]);local=torch.where(allowed)[0];last=-1
        while len(selected)<self.B:
            ix=local[local>last];remaining=self.B-len(selected);ix=ix[:len(ix)-remaining+1];assert len(ix);x=self.state(ix,selected,ng,2)
            if mode in ['GLOBAL','BALANCED']:
                counts=torch.bincount(p.y[selected],minlength=self.C).float();pc=torch.bincount(p.parent_index[selected[p.view[selected]>0]],minlength=len(p.parents)).float();score=-100*counts[p.y[ix]]-pc[p.parent_index[ix]]-.0001*p.view[ix]
            elif mode=='UNCERTAINTY':score=(1-self.raw[ix,-1])*(2-self.raw[ix,-4])/(1+torch.bincount(p.y[selected],minlength=self.C)[p.y[ix]])
            else:score=self.val[ix]
            k=decide(x,score,p.pid[ix]*8+p.view[ix]);last=int(ix[k]);selected=torch.cat([selected,ix[k:k+1]])
        memory=p.take(selected);memory.assert_subset(p);assert len(memory)==self.B and len(memory.gidx)==ng;return memory,trace,{'global_tokens':ng,'local_tokens':self.B-ng,'canonical_tokens':(memory.pid*8+memory.view).cpu().tolist()}

def replacement(choice,memory,generator):
    p=choice.pool;keys=p.pid*8+p.view;selected=torch.searchsorted(keys,memory.pid*8+memory.view);optional=selected[~torch.isin(selected,choice.cover)];order=optional[torch.randperm(len(optional),device=p.z.device,generator=generator)]
    for item in order:
        rest=selected[selected!=item]
        if p.view[item]==0 and ((p.pid[rest]==p.pid[item])&(p.view[rest]>0)).any():continue
        available=torch.where(~torch.isin(torch.arange(len(p),device=p.z.device),selected))[0];global_parents=p.pid[rest[p.view[rest]==0]];valid=(p.view[available]==0)|torch.isin(p.pid[available],global_parents);available=available[valid]
        if not len(available):continue
        alt=available[int(torch.randint(len(available),(1,),device=p.z.device,generator=generator))];out=p.take(torch.cat([rest,alt.reshape(1)]));out.assert_subset(p);assert len(out)==len(memory);return out,int(item),int(alt)
    raise AssertionError('NO_LEGAL_EQUAL_BYTE_COUNTERFACTUAL')
