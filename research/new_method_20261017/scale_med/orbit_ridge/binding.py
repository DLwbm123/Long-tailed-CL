"""Exactly the existing load_adapter key mapping, restricted to pretrained biases."""
import torch
def native(v):
 out={}
 for k,x in v.items():
  if '.qkv.' in k:
   assert x.shape[0]%3==0
   for name,part in zip(('q_proj','k_proj','v_proj'),x.chunk(3,dim=0)):out[k.replace('qkv',name)]=part.clone()
  else:out[k.replace('.mlp.fc','.fc')]=x.clone()
 return out
def checkpoint(v,original_keys):
 out={}
 for k in original_keys:
  if '.qkv.' in k:out[k]=torch.cat([v[k.replace('qkv',name)] for name in ('q_proj','k_proj','v_proj')],dim=0)
  else:out[k]=v[k.replace('.mlp.fc','.fc')].clone()
 assert set(native(out))==set(v)
 return out
def selfcheck():
 a={'blocks.0.attn.qkv.bias':torch.arange(9,dtype=torch.float32),'blocks.0.mlp.fc1.bias':torch.arange(12,dtype=torch.float32),'norm.bias':torch.zeros(3)};b=native(a);c=checkpoint(b,sorted(a));assert all(torch.equal(a[k],c[k]) for k in a)
