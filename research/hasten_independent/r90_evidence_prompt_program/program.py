import torch
from torch import nn
import torch.nn.functional as F
import counts
PROGRAMS=[(a,b) for a in range(4) for b in range(4)]
def unit(x):return F.normalize(x.float(),dim=-1,eps=1e-8)
class Dictionary(nn.Module):
 def __init__(self):
  super().__init__();self.atoms=nn.Parameter(torch.randn(2,3,4,768)*.02)
 def forward(self,base,images,program):
  counts.encoder(len(images));v=base.visual;x=v.conv1(images).flatten(2).transpose(1,2);cls=v.class_embedding.to(x.dtype)[None,None].expand(len(x),1,-1);x=torch.cat([cls,x],1)+v.positional_embedding.to(x.dtype)
  a,b=program
  if a:x=torch.cat([x,self.atoms[0,a-1][None].expand(len(x),-1,-1)],1)
  x=v.ln_pre(x)
  for k,block in enumerate(v.transformer.resblocks):
   if k==6 and b:x=torch.cat([x,self.atoms[1,b-1][None].expand(len(x),-1,-1)],1)
   x=block(x) if v.transformer.batch_first else block(x.transpose(0,1)).transpose(0,1)
  return unit(v.ln_post(x)[:,0]@v.proj)
def make_base(cfg):
 import open_clip
 from safetensors.torch import load_file
 base,_,tf=open_clip.create_model_and_transforms('ViT-B-16',pretrained=None,force_quick_gelu=False);base.load_state_dict(load_file(cfg['weights']),strict=True);base.eval().requires_grad_(False).cuda();return base,tf
