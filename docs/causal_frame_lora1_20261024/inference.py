"""Declared FP64 native/LoRA scores and independent NumPy 8+4 reference."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from core import Model,merged_body
from reference import np_tokens


def body_state(b,begin=0,end=12,norm=True):
    out={}
    for j in range(begin,end):out.update({f'blocks.{j-begin}.'+k:v.detach().cpu().numpy() for k,v in b.blocks[j].state_dict().items()})
    if norm:out.update({'norm.'+k:v.detach().cpu().numpy() for k,v in b.norm.state_dict().items()})
    return out


def native_features(b,tokens,micro=8):
    ref=next(b.parameters());out=[]
    with torch.no_grad():
        for i in range(0,len(tokens),micro):
            z=b.norm(b.blocks[8:](tokens[i:i+micro].to(ref)))
            out.append(F.normalize(torch.cat([z[:,0],z[:,1:].mean(1)],dim=1),dim=1).cpu())
    return torch.cat(out)


def numpy_features(tokens,base,adapter=None):
    raw=np_tokens(tokens,base if adapter is None else merged_body(base,adapter))
    z=np.concatenate([raw[:,0],raw[:,1:].mean(1)],axis=1)
    return z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)


def numpy_scores(tokens,base,adapter):
    z=numpy_features(tokens,base,adapter);head=np.asarray(adapter['head'],dtype=np.float64)
    head=head/np.maximum(np.linalg.norm(head,axis=1,keepdims=True),1e-12)
    return 25*z@head.T


def numpy_objective(tokens,labels,base,adapter,arm):
    s=numpy_scores(tokens,base,adapter)/25
    if arm=='FRAME_DR':return float(np.mean(.5*(s[np.arange(len(labels)),labels]-1)**2))
    s*=25;s-=s.max(1,keepdims=True)
    return float(np.mean(np.log(np.exp(s).sum(1))-s[np.arange(len(labels)),labels]))


def restore(b,arm,adapter,labels):
    m=Model(b,arm).to(next(b.parameters()))
    m.head=nn.Parameter(adapter['head'].clone().to(next(m.parameters())),requires_grad=arm=='LEARNED_CE');m.labels=list(labels)
    own=dict(m.named_parameters())
    with torch.no_grad():
        for k,v in adapter.items():own[k].copy_(v.to(own[k]))
    return m


def scores(m,tokens,micro=8):
    ref=next(m.parameters())
    with torch.no_grad():return torch.cat([m(tokens[i:i+micro].to(ref)).cpu() for i in range(0,len(tokens),micro)])


def frozen_parameter_delta(m,b):
    max_error=0.
    for name,value in m.named_parameters():
        if name=='head' or name.endswith(('.qA','.qB','.vA','.vB')):continue
        key=name.replace('.qkv.base.','.qkv.')
        if key.startswith('blocks.'):
            parts=key.split('.');parts[1]=str(int(parts[1])+8);key='.'.join(parts)
        original=b.state_dict()[key]
        max_error=max(max_error,float((value.detach().cpu()-original.detach().cpu()).abs().max()))
    return max_error


def numpy_embedding(images,b):
    """Independent patch convolution/CLS/position on actual saved transformed pixels."""
    assert isinstance(b.patch_embed.norm,nn.Identity) and isinstance(b.norm_pre,nn.Identity)
    assert getattr(b,'reg_token',None) is None and not b.no_embed_class and b.num_prefix_tokens==1
    x=np.asarray(images,dtype=np.float64);batch=len(x)
    patches=x.reshape(batch,3,14,16,14,16).transpose(0,2,4,1,3,5).reshape(batch,196,768)
    w=b.patch_embed.proj.weight.detach().cpu().numpy().reshape(768,768)
    bias=b.patch_embed.proj.bias.detach().cpu().numpy()
    z=patches@w.T+bias;cls=b.cls_token.detach().cpu().numpy()
    return np.concatenate([np.broadcast_to(cls,(batch,1,768)),z],1)+b.pos_embed.detach().cpu().numpy()
