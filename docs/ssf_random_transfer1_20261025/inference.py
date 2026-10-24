"""Independent full NumPy embedding/attention/SSF equations, never timm execution."""
import numpy as np
from scipy.special import erf

def arrays(state):return {k:v.detach().cpu().numpy() for k,v in state.items()}

def tokens(raw,state,mods=None):
    s=state;raw=(np.asarray(raw,dtype=np.float64)-.5)/.5;b=len(raw)
    def mod(x,k):return x if mods is None else x*mods[k+'_scale']+mods[k+'_shift']
    def norm(x,k):
        d=x-np.mean(x,axis=-1,keepdims=True);return d/np.sqrt(np.mean(d*d,axis=-1,keepdims=True)+1e-6)*s[k+'weight']+s[k+'bias']
    patches=raw.reshape(b,3,14,16,14,16).transpose(0,2,4,1,3,5).reshape(b,196,768);x=mod(patches@s['patch_embed.proj.weight'].reshape(768,768).T+s['patch_embed.proj.bias'],'patch');x=np.concatenate([np.broadcast_to(s['cls_token'],(b,1,768)),x],1)+s['pos_embed']
    for i in range(12):
        p=f'blocks.{i}.';k=f'b{i}_';z=mod(norm(x,p+'norm1.'),k+'n1');z=mod(z@s[p+'attn.qkv.weight'].T+s[p+'attn.qkv.bias'],k+'qkv');batch,t,_=z.shape;q,kk,v=z.reshape(batch,t,3,12,64).transpose(2,0,3,1,4);a=(q@kk.swapaxes(-2,-1))/8;a=np.exp(a-a.max(-1,keepdims=True));a/=a.sum(-1,keepdims=True);z=(a@v).transpose(0,2,1,3).reshape(batch,t,768);x=x+mod(z@s[p+'attn.proj.weight'].T+s[p+'attn.proj.bias'],k+'proj');z=mod(norm(x,p+'norm2.'),k+'n2');z=mod(z@s[p+'mlp.fc1.weight'].T+s[p+'mlp.fc1.bias'],k+'fc1');z=.5*z*(1+erf(z/np.sqrt(2)));x=x+mod(z@s[p+'mlp.fc2.weight'].T+s[p+'mlp.fc2.bias'],k+'fc2')
    return mod(norm(x,'norm.'),'final')

def cosine_loss(z,head,labels):
    f=np.concatenate([z[:,0],z[:,1:].mean(1)],1);f/=np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-12);h=head/np.maximum(np.linalg.norm(head,axis=1,keepdims=True),1e-12);logits=25*f@h.T;m=logits.max(1,keepdims=True);logsum=np.log(np.exp(logits-m).sum(1))+m[:,0];return float(np.mean(logsum-logits[np.arange(len(labels)),labels]))
