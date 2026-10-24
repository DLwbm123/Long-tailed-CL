import numpy as np
import torch
from torch import nn
from native_reference import np_tokens

def body_state(b,begin=0,end=12,norm=True):
    out={}
    for j in range(begin,end):out.update({f'blocks.{j-begin}.'+k:v.detach().cpu().numpy() for k,v in b.blocks[j].state_dict().items()})
    if norm:out.update({'norm.'+k:v.detach().cpu().numpy() for k,v in b.norm.state_dict().items()})
    return out


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


def upper_state(model):
    return {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}


def numpy_affine(tokens, body):
    raw = np_tokens(tokens, body)
    descriptor = np.concatenate((raw[:, 0], raw[:, 1:].mean(1)), axis=1)
    descriptor /= np.maximum(np.linalg.norm(descriptor, axis=1, keepdims=True), 1e-12)
    return np.concatenate((descriptor, np.ones((len(descriptor), 1))), axis=1) / np.sqrt(2)
