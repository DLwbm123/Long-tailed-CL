"""Frozen Gaussian nonlinear features and base-only per-operation scale/shift."""
import numpy as np
import torch
from torch import nn
ARMS=['NATIVE_LINEAR','SSF_LINEAR','NATIVE_RANDOM_RELU','SSF_RANDOM_RELU']

def descriptor(z):
    x=torch.cat([z[:,0],z[:,1:].mean(1)],1)
    return x/x.norm(dim=1,keepdim=True).clamp_min(1e-12)

def affine(z,arm,W):
    x=descriptor(z)
    if 'RANDOM' in arm:x=torch.relu(x@W)
    x=x/x.norm(dim=1,keepdim=True).clamp_min(1e-12)
    return torch.cat([x,torch.ones_like(x[:,:1])],1)/(2**.5)

def numpy_affine(z,arm,W):
    x=np.concatenate([z[:,0],np.sum(z[:,1:],axis=1)/196],1);x/=np.maximum(np.sqrt(np.sum(x*x,axis=1,keepdims=True)),1e-12)
    if 'RANDOM' in arm:x=np.maximum(x@W,0)
    x/=np.maximum(np.sqrt(np.sum(x*x,axis=1,keepdims=True)),1e-12)
    return np.column_stack([x,np.ones(len(x))])/np.sqrt(2)

def pack(matrix):
    ix=torch.triu_indices(len(matrix),len(matrix));return matrix[ix[0],ix[1]]

def unpack_numpy(value,d):
    x=np.zeros((d,d),dtype=np.float64);x[np.triu_indices(d)]=value;x+=np.triu(x,1).T;return x

class SSF(nn.Module):
    def __init__(self,backbone,seed):
        super().__init__();self.backbone=backbone;self.mods=nn.ParameterDict();g=torch.Generator(device='cpu').manual_seed(seed)
        sizes={'patch':768,'final':768}
        for i in range(12):
            for k,d in [('n1',768),('qkv',2304),('proj',768),('n2',768),('fc1',3072),('fc2',768)]:sizes[f'b{i}_{k}']=d
        for k,d in sizes.items():
            self.mods[k+'_scale']=nn.Parameter(1+.02*torch.randn(d,generator=g,dtype=torch.float64));self.mods[k+'_shift']=nn.Parameter(.02*torch.randn(d,generator=g,dtype=torch.float64))
        self.head=nn.Parameter(.02*torch.randn(10,1536,generator=g,dtype=torch.float64));self.backbone.requires_grad_(False)
    def mod(self,x,k):return x*self.mods[k+'_scale']+self.mods[k+'_shift']
    def forward_features(self,raw):
        b=self.backbone;x=b.patch_embed((raw-.5)/.5);x=self.mod(x,'patch');x=b._pos_embed(x)
        for i,block in enumerate(b.blocks):
            k=f'b{i}_';z=self.mod(block.norm1(x),k+'n1');qkv=self.mod(block.attn.qkv(z),k+'qkv');batch,tokens,_=qkv.shape
            q,kk,v=qkv.reshape(batch,tokens,3,12,64).permute(2,0,3,1,4).unbind(0);a=torch.softmax((q@kk.transpose(-2,-1))*(64**-.5),dim=-1);z=(a@v).transpose(1,2).reshape(batch,tokens,768);x=x+self.mod(block.attn.proj(z),k+'proj')
            z=self.mod(block.norm2(x),k+'n2');z=self.mod(block.mlp.fc1(z),k+'fc1');z=block.mlp.act(z);x=x+self.mod(block.mlp.fc2(z),k+'fc2')
        return self.mod(b.norm(x),'final')
    def objective(self,raw,labels):
        f=descriptor(self.forward_features(raw));h=self.head/self.head.norm(dim=1,keepdim=True).clamp_min(1e-12)
        return torch.nn.functional.cross_entropy(25*(f@h.T),labels)
    def small_state(self):return {k:v.detach().cpu().clone() for k,v in self.mods.items()}
    def load_small(self,s):self.mods.load_state_dict(s,strict=True)
