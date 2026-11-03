"""Frozen strong LoRA teacher, exact zero-gate initialization, RLAP residual modules."""
import math,torch
from torch import nn
from model import Adapter,unit

class LoRAQKV(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base;d=base.in_features;self.qA=nn.Linear(d,8,bias=False);self.qB=nn.Linear(8,d,bias=False);self.vA=nn.Linear(d,8,bias=False);self.vB=nn.Linear(8,d,bias=False)
 def forward(self,x):
  q,k,v=self.base(x).chunk(3,-1);return torch.cat([q+self.qB(self.qA(x)),k,v+self.vB(self.vA(x))],-1)

def load_teacher(base,head):
 blocks=base.visual.trunk.blocks
 for b in blocks[10:]:
  if not isinstance(b.attn.qkv,LoRAQKV):b.attn.qkv=LoRAQKV(b.attn.qkv).to(next(base.parameters()).device)
 expected={}
 for i,b in enumerate(blocks[10:]):
  for name in ['qA','qB','vA','vB']:expected[f'blocks.{i}.attn.qkv.{name}.weight']=getattr(b.attn.qkv,name).weight
 assert set(expected)==set(head['delta']) and sum(x.numel() for x in expected.values())==49152
 with torch.no_grad():
  for name,p in expected.items():assert p.shape==head['delta'][name].shape;p.copy_(head['delta'][name].to(p.device));assert torch.equal(p.cpu(),head['delta'][name])
 base.eval().requires_grad_(False)

class StrongAdapter(Adapter):
 def __init__(self):
  super().__init__('JOINT');self.gate=nn.Sequential(nn.Linear(517,64),nn.Tanh(),nn.Linear(64,1));self.zero_gate()
 def zero_gate(self):nn.init.zeros_(self.gate[-1].weight);nn.init.zeros_(self.gate[-1].bias)
 def context(self,proto,raw,support,births,stage):
  scalars=torch.stack([raw.float().log1p()/math.log(6822),support.float().log1p()/math.log(6822),(stage-torch.tensor(births,device=raw.device)).float()/5,torch.tensor([b==stage for b in births],device=raw.device).float(),(support>1).float()],1);self.gate_state=torch.cat([proto.detach(),scalars],1);assert self.gate_state.shape==(len(proto),517)
 def gains(self):return .25*self.gate(self.gate_state).tanh().squeeze(1)
 def combine(self,raw,teacher):return teacher+self.gains()[None]*(raw*1.4-teacher)
