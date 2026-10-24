"""Independent NumPy patch embedding,12native blocks, and affine descriptors."""
import numpy as np
from native_reference import np_tokens

def arrays(state):return {k:v.detach().cpu().numpy() for k,v in state.items()}

def embedding(raw,state):
    s={k:np.asarray(v,dtype=np.float64) for k,v in state.items()};x=(np.asarray(raw,dtype=np.float64)-.5)/.5;b=len(x)
    patches=x.reshape(b,3,14,16,14,16).transpose(0,2,4,1,3,5).reshape(b,196,768)
    z=patches@s['patch_embed.proj.weight'].reshape(768,768).T+s['patch_embed.proj.bias']
    return np.concatenate([np.broadcast_to(s['cls_token'],(b,1,768)),z],1)+s['pos_embed']

def tokens(raw,state):return np_tokens(embedding(raw,state),state)

def affine(z):
    d=np.concatenate([z[:,0],z[:,1:].mean(1)],1);d/=np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-12)
    return np.concatenate([d,np.ones((len(d),1))],1)/np.sqrt(2)
