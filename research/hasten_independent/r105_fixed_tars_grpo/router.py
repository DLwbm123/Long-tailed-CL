"""Permutation-independent action features and 672-parameter selector."""
import math
import torch
from torch import nn
import torch.nn.functional as F

Q=[.8,.1,.1]
BASE_ARMS=['SUP','RAW_EXACT','GROUP_EXACT','GRPO']
ARMS=BASE_ARMS+['TARS_'+a for a in BASE_ARMS]
class Router(nn.Module):
    def __init__(self):
        super().__init__();self.first=nn.Linear(19,32);self.last=nn.Linear(32,1,bias=False);nn.init.zeros_(self.last.weight)
    def forward(self,x):return self.last(torch.tanh(self.first(x))).squeeze(-1)+torch.tensor(Q,device=x.device).log()

@torch.no_grad()
def features(scores,z,nc,birth,stage):
    p=scores.double().softmax(-1);B,A,C=p.shape;pred=p.argmax(-1);top=p.topk(2,-1).values;teacher=p[:,0];tc=pred[:,0];N=nc.sum();nc=nc.double();birth=birth.to(scores.device)
    value=[]
    for a in range(3):
        pa=p[:,a];ca=pred[:,a];mix=(pa+teacher)/2;js=.5*((pa*(pa.log()-mix.log())).sum(-1)+(teacher*(teacher.log()-mix.log())).sum(-1))
        v=torch.stack([top[:,a,0],top[:,a,0]-top[:,a,1],-(pa*pa.log()).sum(-1)/math.log(C),pa.gather(1,tc[:,None]).squeeze(1),teacher.gather(1,ca[:,None]).squeeze(1),js/math.log(2),(ca==tc).double(),nc[ca]/N,torch.log1p(nc[ca])/torch.log1p(N),(birth[ca]==stage).double(),(stage-birth[ca]).double()/4,torch.log1p(nc[tc])/torch.log1p(N),(birth[tc]==stage).double(),(stage-birth[tc]).double()/4,torch.full_like(tc,[1,2,5][a],dtype=torch.double)/5,torch.ones(B,device=z.device) if a==0 else F.cosine_similarity(z[:,0],z[:,a],dim=-1).double()],-1)
        value.append(torch.cat([v,F.one_hot(torch.full((B,),a,device=z.device),3)],-1))
    x=torch.stack(value,1).float();assert x.shape==(B,3,19) and torch.isfinite(x).all();return x.detach()
