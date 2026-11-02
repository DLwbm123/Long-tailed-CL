import torch
from torch import nn

class Prompts(nn.Module):
 def __init__(self,width=768):
  super().__init__();self.visual=nn.Parameter(torch.randn(4,width)*.02);self.text=nn.Parameter(torch.randn(4,width)*.02)

def insert_text(word,mask,prompt):
 if prompt is None:return word,mask
 n=len(prompt)
 if word.shape[1]<=n or bool(mask[:,-n:].any()):raise ValueError('prompt insertion would truncate nonpadding tokens')
 x=torch.cat([word[:,:1],prompt[None].expand(len(word),-1,-1),word[:,1:-n]],1)
 m=torch.cat([mask[:,:1],torch.ones((len(mask),n),device=mask.device,dtype=mask.dtype),mask[:,1:-n]],1)
 return x,m

def image(base,images,prompt=None):
 v=base.visual;t=v.trunk;x=t._pos_embed(t.patch_embed(images))
 if prompt is not None:x=torch.cat([x,prompt[None].expand(len(x),-1,-1).to(x.dtype)],1)
 x=t.norm_pre(t.patch_drop(x))
 if t.grad_checkpointing:
  from timm.models._manipulate import checkpoint_seq
  x=checkpoint_seq(t.blocks,x)
 else:x=t.blocks(x)
 return v.head(t.forward_head(t.norm(x)))

def text(base,tokens,prompt=None):
 t=base.text;mask=(tokens!=t.config.pad_token_id).long();word=t.transformer.get_input_embeddings()(tokens);word,mask=insert_text(word,mask,prompt)
 out=t.transformer(inputs_embeds=word,attention_mask=mask)
 return t.proj(t.pooler(out,mask))
