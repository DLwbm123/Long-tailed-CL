"""Shared strong online classifier; gradients only in aggregation and cosine head."""
import copy
import torch
from torch import nn
import torch.nn.functional as F
from memory import Bags

class Learner(nn.Module):
    def __init__(self,seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);self.V=nn.Linear(1025,16,bias=False);self.U=nn.Linear(16,512,bias=False);nn.init.zeros_(self.U.weight);self.head=nn.Parameter(F.normalize(torch.randn(23,512),dim=-1))
        self.C=0
    def forward(self,x):
        z=F.normalize(x[:,:512]+self.U(F.gelu(self.V(x))),dim=-1);return 20*z@F.normalize(self.head[:self.C],dim=-1).T
    @torch.no_grad()
    def arrive(self,bank,C):
        assert not set(bank.pid.tolist())&getattr(self,'forbidden',set()),'SHADOW_QUERY_IN_PROTOTYPE'
        x,y,_=bank.bags()
        for c in range(self.C,C):
            mask=y==c;assert mask.any();self.head[c].copy_(F.normalize(x[mask,:512].mean(0),dim=0))
        self.C=C

def frozen(m):
    x=copy.deepcopy(m);x.eval().requires_grad_(False);return x

def optimizer(m):return torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001)

def learn(m,opt,bank,steps,seed,counter,teacher=None,batch=64,kind='formal'):
    assert not set(bank.pid.tolist())&getattr(m,'forbidden',set()),'SHADOW_QUERY_IN_GRADIENT_OR_REPLAY'
    data=Bags(bank,m.C);g=torch.Generator(device='cuda').manual_seed(seed)
    for _ in range(steps):
        x,y=data.sample(batch,g);opt.zero_grad(set_to_none=True);logits=m(x);loss=F.cross_entropy(logits,y)
        if teacher is not None and teacher.C:
            mask=y<teacher.C
            if mask.any():
                with torch.no_grad():target=(teacher(x[mask])/2).softmax(-1)
                loss=loss+2*F.kl_div((logits[mask,:teacher.C]/2).log_softmax(-1),target,reduction='batchmean')
        assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm);opt.step();counter[kind+'_classifier_optimizer']=counter.get(kind+'_classifier_optimizer',0)+1;counter[kind+'_query_update_rows']=counter.get(kind+'_query_update_rows',0)+batch

@torch.no_grad()
def quality(m,query,counter=None,kind=None):
    x,y,_=query.bags();mask=y<m.C;x=x[mask];y=y[mask];p=m(x).double().softmax(-1);good=p.argmax(-1)==y;brier=(p-F.one_hot(y,m.C)).square().sum(-1);q=1-brier/2
    if counter is not None:counter[kind+'_reward_query_rows']=counter.get(kind+'_reward_query_rows',0)+len(y)
    ba=torch.stack([good[y==c].double().mean() for c in range(m.C)]);qc=torch.stack([q[y==c].mean() for c in range(m.C)]);assert torch.isfinite(ba).all() and torch.isfinite(qc).all();return float(.5*ba.mean()+.5*qc.mean()),qc.cpu()

def clone_pair(m,opt,seed):
    other=copy.deepcopy(m);oo=optimizer(other);oo.load_state_dict(copy.deepcopy(opt.state_dict()));return other,oo
