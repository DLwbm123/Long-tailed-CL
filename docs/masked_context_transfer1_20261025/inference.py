"""Independent NumPy patch embedding,12native blocks, and masked objectives."""
import numpy as np
from native_reference import np_tokens

def arrays(state):return {k:v.detach().cpu().numpy() for k,v in state.items()}

def embedding(raw,state,receipt=None,masked=False):
    s={k:np.asarray(v,dtype=np.float64) for k,v in state.items()};x=(np.asarray(raw,dtype=np.float64)-.5)/.5;b=len(x)
    patches=x.reshape(b,3,14,16,14,16).transpose(0,2,4,1,3,5).reshape(b,196,768)
    z=patches@s['patch_embed.proj.weight'].reshape(768,768).T+s['patch_embed.proj.bias']
    if masked:z[:,receipt['tokens']]=0
    return np.concatenate([np.broadcast_to(s['cls_token'],(b,1,768)),z],1)+s['pos_embed']

def tokens(raw,state,receipt=None,masked=False):return np_tokens(embedding(raw,state,receipt,masked),state)

def affine(z):
    d=np.concatenate([z[:,0],z[:,1:].mean(1)],1);d/=np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-12)
    return np.concatenate([d,np.ones((len(d),1))],1)/np.sqrt(2)

def objective(z,raw,labels,receipt,decoder,head,arm,phase):
    if phase=='A' and arm!='SUPERVISED_ONLY':
        x=np.asarray(raw,dtype=np.float64);target=x.reshape(len(x),3,14,16,14,16).transpose(0,2,4,1,3,5).reshape(len(x),196,768)[:,receipt['tokens']]
        pred=z[:,1:][:,receipt['tokens']]@decoder['weight'].T+decoder['bias']
        return float(abs(pred-target).mean())
    x=affine(z)[:,:1536]*np.sqrt(2);h=head/np.maximum(np.linalg.norm(head,axis=1,keepdims=True),1e-12)
    logits=25*x@h.T;logits-=logits.max(1,keepdims=True)
    return float(np.mean(np.log(np.exp(logits).sum(1))-logits[np.arange(len(labels)),labels]))
