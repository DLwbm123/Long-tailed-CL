"""Base-only native deep prompts, original-response anchoring, fixed-space exact risk."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ARMS=['FROZEN_NATIVE','PROMPT_CE','PROMPT_RC']
class NativeEvidence(nn.Module):
    def __init__(self,backbone,prompts=None):
        super().__init__();self.backbone=backbone.eval().requires_grad_(False)
        if prompts is None:self.register_parameter('prompts',None)
        else:self.prompts=nn.Parameter(prompts.clone())
    def train(self,mode=True):
        super().train(mode);self.backbone.eval();return self
    def forward(self,tokens):
        x=tokens
        for j,block in enumerate(self.backbone.blocks):
            if self.prompts is not None and j<len(self.prompts):
                n=len(self.prompts[j]);pt=self.prompts[j][None].expand(len(x),-1,-1)
                x=block(torch.cat([x[:,:1],pt,x[:,1:]],1));x=torch.cat([x[:,:1],x[:,1+n:]],1)
            else:x=block(x)
        x=self.backbone.norm(x)
        return F.normalize(torch.cat([x[:,0],x[:,1:].mean(1)],1),dim=1)
def base_objective(features,teacher,head,labels,consistency_weight,scale=25.):
    ce=F.cross_entropy(scale*features@F.normalize(head,dim=1).T,labels)
    consistency=(1-(features*teacher.detach()).sum(1)).mean()
    return ce+consistency_weight*consistency,ce,consistency

def np_evidence(tokens,state,prompts=None,heads=12,eps=1e-6):
    """Independent NumPy attention, exact GELU and deep-prompt routing."""
    from scipy.special import erf
    s={k:np.asarray(v,dtype=np.float64) for k,v in state.items()};x=np.asarray(tokens,dtype=np.float64)
    def norm(z,p):
        d=z-z.mean(-1,keepdims=True)
        return d/np.sqrt((d*d).mean(-1,keepdims=True)+eps)*s[p+'weight']+s[p+'bias']
    depth=len([k for k in s if k.startswith('blocks.') and k.endswith('.attn.qkv.weight')])
    for j in range(depth):
        n=0
        if prompts is not None and j<len(prompts):
            n=len(prompts[j]);x=np.concatenate([x[:,:1],np.broadcast_to(np.asarray(prompts[j],dtype=np.float64),(len(x),n,x.shape[-1])),x[:,1:]],1)
        p=f'blocks.{j}.';z=norm(x,p+'norm1.');qkv=z@s[p+'attn.qkv.weight'].T+s[p+'attn.qkv.bias'];b,t,triple=qkv.shape;d=triple//(3*heads)
        q,k,v=qkv.reshape(b,t,3,heads,d).transpose(2,0,3,1,4);a=(q@k.swapaxes(-1,-2))/np.sqrt(d);a=np.exp(a-a.max(-1,keepdims=True));a/=a.sum(-1,keepdims=True)
        z=(a@v).transpose(0,2,1,3).reshape(b,t,heads*d);x=x+z@s[p+'attn.proj.weight'].T+s[p+'attn.proj.bias']
        z=norm(x,p+'norm2.');z=z@s[p+'mlp.fc1.weight'].T+s[p+'mlp.fc1.bias'];z=.5*z*(1+erf(z/np.sqrt(2)));x=x+z@s[p+'mlp.fc2.weight'].T+s[p+'mlp.fc2.bias']
        if n:x=np.concatenate([x[:,:1],x[:,1+n:]],1)
    x=norm(x,'norm.');z=np.concatenate([x[:,0],x[:,1:].mean(1)],1)
    return z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
def affine(z):
    z=z.double();z=F.normalize(z,dim=1)
    return torch.cat([z,torch.ones(len(z),1,dtype=z.dtype,device=z.device)],1)/np.sqrt(2)
def patient_weights(rows):
    counts={}
    for r in rows:counts[r['group']]=counts.get(r['group'],0)+1
    return np.array([1/(len(counts)*counts[r['group']]) for r in rows]),len(counts)
def moments(phi,weights):
    w=torch.as_tensor(weights,dtype=torch.float64,device=phi.device)
    return phi.T@(w[:,None]*phi),(w[:,None]*phi).sum(0)
def solve(records,arm,ridge):
    gram=sum(r[arm]['gram'] for r in records)/len(records);rhs=torch.stack([r[arm]['mean']/len(records) for r in records],1)
    mat=gram+ridge*torch.eye(len(gram),dtype=torch.float64,device=gram.device);head=torch.linalg.solve(mat,rhs)
    error=float((mat@head-rhs).abs().max());assert error<1e-9 and torch.isfinite(head).all()
    return head,error
