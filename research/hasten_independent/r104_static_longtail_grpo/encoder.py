"""Read-only original Biomed base and R67 LoRA, no new visual adaptation."""
import copy,json,torch
from torch import nn
from pathlib import Path
import torch.nn.functional as F
def make_base(cfg,device):
 import open_clip,json
 from pathlib import Path
 from open_clip.factory import _MODEL_CONFIGS
 spec=json.loads((Path(cfg['biomed_assets'])/'open_clip_config.json').read_text());mc=copy.deepcopy(spec['model_cfg']);mc['text_cfg'].update(hf_model_name=str(Path(cfg['biomed_assets'])/'text'),hf_tokenizer_name=cfg['biomed_assets'],hf_model_pretrained=False);name='fixed_medical_branch';_MODEL_CONFIGS[name]=mc
 base,_,tf=open_clip.create_model_and_transforms(name,pretrained=None,pretrained_hf=False,precision='fp32',device='cpu',**{f'image_{k}':v for k,v in spec['preprocess_cfg'].items()});state=torch.load(cfg['biomed_checkpoint'],map_location='cpu',weights_only=False);state=state.get('state_dict',state);key='text.transformer.embeddings.position_ids'
 if key in state and key not in base.state_dict():assert torch.equal(state[key],base.text.transformer.embeddings.position_ids.cpu());del state[key]
 base.load_state_dict(state,strict=True);del state;base.eval().requires_grad_(False).to(device)
 assert not any('Random' in type(x).__name__ for x in tf.transforms),'STOCHASTIC_PREPROCESS_FORBIDS_CACHE'
 return base,tf,open_clip.get_tokenizer(name)

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


def visual(base,x):return F.normalize(base.encode_image(x).float(),dim=-1)
