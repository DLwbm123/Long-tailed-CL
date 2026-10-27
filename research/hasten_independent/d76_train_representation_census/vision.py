"""Native ViT prefix/tail interface for planned backbone adaptation, not a new readout."""
import copy,math
import torch
from torch import nn
import torch.nn.functional as F

def frozen_prefix(visual,pixels,split=10):
 t=visual.trunk;x=t.patch_embed(pixels);x=t._pos_embed(x);x=t.patch_drop(x);x=t.norm_pre(x)
 for block in list(t.blocks)[:split]:x=block(x)
 return x

def export_tail(visual,split=10):
 v=copy.deepcopy(visual).cpu();v.trunk.blocks=nn.Sequential(*list(v.trunk.blocks)[split:]);v.requires_grad_(False).eval();return v

def tail_embedding(tail,tokens):
 t=tail.trunk;x=t.blocks(tokens);x=t.norm(x);x=t.forward_head(x);return F.normalize(tail.head(x),dim=-1)

class LoRAQKV(nn.Module):
 def __init__(self,base,rank=8):
  super().__init__();self.base=base;self.base.requires_grad_(False);d=base.in_features;assert base.out_features==3*d
  self.qA=nn.Linear(d,rank,bias=False);self.qB=nn.Linear(rank,d,bias=False);self.vA=nn.Linear(d,rank,bias=False);self.vB=nn.Linear(rank,d,bias=False);nn.init.zeros_(self.qB.weight);nn.init.zeros_(self.vB.weight)
 def forward(self,x):
  q,k,v=self.base(x).chunk(3,dim=-1);return torch.cat([q+self.qB(self.qA(x)),k,v+self.vB(self.vA(x))],dim=-1)

def install_lora(tail):
 tail.requires_grad_(False).eval()
 for b in tail.trunk.blocks:
  assert isinstance(b.attn.qkv,nn.Linear);b.attn.qkv=LoRAQKV(b.attn.qkv).to(b.attn.qkv.weight.device)
 return [p for p in tail.parameters() if p.requires_grad]

class NativeTail(nn.Module):
 def __init__(self,bundle):
  super().__init__()
  import timm
  trunk=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0,depth=2)
  self.blocks=trunk.blocks;self.norm=trunk.norm;self.norm.eps=bundle['meta']['norm_eps'];self.proj=nn.Linear(768,512,bias=False)
  assert bundle['meta']['global_pool']=='token' and bundle['meta']['fc_norm_identity']

  for b,eps in zip(self.blocks,bundle['meta']['block_norm_eps']):b.norm1.eps=eps[0];b.norm2.eps=eps[1]
  self.load_state_dict(bundle['state'],strict=True);self.requires_grad_(False).eval()
 def forward(self,tokens):return F.normalize(self.proj(self.norm(self.blocks(tokens))[:,0]),dim=-1)

def delta_state(model):return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
def load_delta(model,state):
 expected={n for n,p in model.named_parameters() if p.requires_grad};assert expected==set(state)
 with torch.no_grad():
  for n,p in model.named_parameters():
   if p.requires_grad:p.copy_(state[n].to(p.device))

def adapt_native_tail(model):
 model.requires_grad_(False).eval()
 for b in model.blocks:
  old=b.attn.qkv;assert isinstance(old,nn.Linear);b.attn.qkv=LoRAQKV(old).to(old.weight.device)
 params=[p for p in model.parameters() if p.requires_grad];assert sum(p.numel() for p in params)==49152;return params
