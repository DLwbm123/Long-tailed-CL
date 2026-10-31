"""Explicit paper reconstruction; see PROTOCOL for unresolved author details."""
import copy,torch
from torch import nn
import torch.nn.functional as F

def unit(x):return F.normalize(x.float(),dim=-1,eps=1e-8)
class Branch(nn.Module):
 def __init__(self):
  super().__init__();self.vprompt=nn.Parameter(torch.randn(4,768)*.02);self.tprompt=nn.Parameter(torch.randn(4,512)*.02);self.pv=nn.Linear(512,512,bias=False);self.pt=nn.Linear(512,512,bias=False)
  with torch.no_grad():self.pv.weight.copy_(torch.eye(512));self.pt.weight.copy_(torch.eye(512))
class Policy(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(512,384),nn.LayerNorm(384),nn.GELU(),nn.Linear(384,192),nn.LayerNorm(192),nn.GELU(),nn.Linear(192,96),nn.LayerNorm(96),nn.GELU(),nn.Linear(96,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,x):return self.net(x).squeeze(-1)
class Adapter(nn.Module):
 def __init__(self,arm):
  super().__init__();self.arm=arm;self.branches=nn.ModuleList();self.policy=Policy();self.reference=copy.deepcopy(self.policy).requires_grad_(False);self.easy=nn.Linear(512,512);self.hard=nn.Sequential(nn.Linear(512,512),nn.GELU(),nn.Linear(512,512),nn.GELU(),nn.Linear(512,512));self.attention=nn.Sequential(nn.Linear(1536,128),nn.ReLU(),nn.Linear(128,3));self.tau=nn.Parameter(torch.zeros(()));self.log_temp=nn.Parameter(torch.tensor(.07).log());self.reference_uniform=True
 def add(self):
  for b in self.branches:b.requires_grad_(False)
  b=Branch() if not self.branches else copy.deepcopy(self.branches[-1]).requires_grad_(True);self.branches.append(b.to(self.tau.device))
  if self.arm=='UNIFORM':self.policy.requires_grad_(False)
 def update_reference(self):
  with torch.no_grad():
   for r,p in zip(self.reference.parameters(),self.policy.parameters()):r.mul_(.9).add_(p,alpha=.1)
  self.reference_uniform=False
 def prototypes(self,features,labels,n):
  centers=torch.stack([unit(features[labels==c].mean(0)) for c in range(n)]);context=centers.mean(0).detach();w=torch.zeros(len(features),device=features.device);rw=torch.zeros_like(w)
  score=self.policy(features-context);ref=self.reference(features-context).detach()
  for c in range(n):
   ix=labels==c;k=int(ix.sum());assert k>0
   w[ix]=torch.full((k,),1/k,device=w.device) if self.arm=='UNIFORM' else score[ix].softmax(0)
   rw[ix]=torch.full((k,),1/k,device=w.device) if self.reference_uniform else ref[ix].softmax(0)
  proto=unit(torch.stack([(features[labels==c]*w[labels==c,None]).sum(0) for c in range(n)]))
  sim=features@proto.T;own=sim.gather(1,labels[:,None]).squeeze(1);wrong=sim.masked_fill(F.one_hot(labels,n).bool(),-float('inf')).max(1).values;pc=proto@proto.T;collapse=(pc.sum(1)-pc.diag())/max(n-1,1);reward=own-wrong-.1*collapse[labels];adv=((reward-reward.mean())/(reward.std(unbiased=False)+1e-8)).detach();kl=torch.stack([(w[labels==c]*(w[labels==c].clamp_min(1e-8).log()-rw[labels==c].clamp_min(1e-8).log())).sum() for c in range(n)]).mean();rl=-(w.clamp_min(1e-8).log()*adv).mean()+.01*kl
  return proto,w,rl
 def logits(self,visual_by_branch,text,proto,birth):
  # Candidate-specific inference uses no true label or oracle task identifier.
  visual=torch.stack([visual_by_branch[int(birth[c])] for c in range(len(proto))],1);pv=proto[None].expand(len(visual),-1,-1);pt=text[None].expand_as(pv)
  difficulty=1-((visual*pv).sum(-1)+1)/2;easyprob=torch.sigmoid(-10*(difficulty-self.tau.sigmoid()));expert=unit(easyprob[...,None]*self.easy(visual)+(1-easyprob[...,None])*self.hard(visual));a=self.attention(torch.cat([expert,pv,pt],-1)).softmax(-1);score=a[...,0]*(expert*pv).sum(-1)+a[...,1]*(visual*pv).sum(-1)+a[...,2]*(visual*pt).sum(-1);balance=((torch.stack([easyprob,1-easyprob],-1).mean((0,1))-.5)**2).sum()
  return score/.07,difficulty,balance

def visual(base,images,branch=None):
 v=base.visual;x=v.conv1(images);x=x.flatten(2).transpose(1,2);cls=v.class_embedding.to(x.dtype)[None,None].expand(len(x),1,-1);x=torch.cat([cls,x],1)+v.positional_embedding.to(x.dtype)
 if branch is not None:x=torch.cat([x,branch.vprompt[None].expand(len(x),-1,-1).to(x.dtype)],1)
 x=v.transformer(v.ln_pre(x));x=v.ln_post(x);z=x[:,0]@v.proj
 return unit(z if branch is None else branch.pv(z.float()))
def textual(base,tokens,branch):
 # Shift template after SOS; EOT is shifted by prompt length, never inferred from labels.
 x=base.token_embedding(tokens);x=torch.cat([x[:,:1],branch.tprompt[None].expand(len(x),-1,-1),x[:,1:-4]],1);x=x+base.positional_embedding;z=base.transformer(x,attn_mask=base.attn_mask);z=base.ln_final(z);eot=tokens.argmax(-1)+4;assert int(eot.max())<77;z=z[torch.arange(len(z),device=z.device),eot]@base.text_projection
 return unit(branch.pt(z.float()))
def make_base(cfg,device):
 import open_clip
 from safetensors.torch import load_file
 base,_,preprocess=open_clip.create_model_and_transforms('ViT-B-16',pretrained=None,force_quick_gelu=False);base.load_state_dict(load_file(cfg['weights']),strict=True);base.eval().requires_grad_(False).to(device);return base,preprocess,open_clip.get_tokenizer('ViT-B-16')
