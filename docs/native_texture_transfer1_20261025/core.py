"""Preserve full Native coordinates and measure complementary raw-pixel texture."""
import random
from functools import lru_cache
import numpy as np
import torch
ARMS=['NATIVE_RIDGE','NATIVE_HISTOGRAM','NATIVE_SHUFFLED_TEXTURE','NATIVE_REAL_TEXTURE']
@lru_cache(1)
def permutation():
    v=list(range(224*224));random.Random(74010).shuffle(v);return v

def texture(raw,kind):
    x=raw.double().mean(1);x=x-x.mean((1,2),keepdim=True);x=x/torch.sqrt((x*x).mean((1,2),keepdim=True)).clamp_min(1e-12)
    if kind=='hist':return torch.quantile(x.flatten(1),(torch.arange(28,dtype=torch.float64,device=x.device)+.5)/28,dim=1).T
    if kind=='shuffle':x=x.flatten(1)[:,permutation()].reshape(-1,224,224)
    values=[]
    for level in range(7):
        if x.shape[-2]%2 or x.shape[-1]%2:x=torch.nn.functional.pad(x[:,None],(0,x.shape[-1]%2,0,x.shape[-2]%2),mode='reflect')[:,0]
        a,b,c,d=x[:,::2,::2],x[:,::2,1::2],x[:,1::2,::2],x[:,1::2,1::2];bands=[(a+b+c+d)/2,(a-b+c-d)/2,(a+b-c-d)/2,(a-b-c+d)/2];e=torch.stack([(z*z).mean((1,2)) for z in bands],1);values.append(e/e.sum(1,keepdim=True).clamp_min(1e-12));x=bands[0]
    return torch.cat(values,1)

def numpy_texture(raw,kind):
    x=np.asarray(raw,dtype=np.float64).sum(1)/3;x-=x.mean((1,2),keepdims=True);x/=np.maximum(np.sqrt(np.mean(x*x,axis=(1,2),keepdims=True)),1e-12)
    if kind=='hist':return np.quantile(x.reshape(len(x),-1),(np.arange(28)+.5)/28,axis=1,method='linear').T
    if kind=='shuffle':x=x.reshape(len(x),-1)[:,np.asarray(permutation())].reshape(-1,224,224)
    out=[]
    for j in range(7):
        if x.shape[1]%2 or x.shape[2]%2:x=np.pad(x,((0,0),(0,x.shape[1]%2),(0,x.shape[2]%2)),mode='reflect')
        cells=x.reshape(len(x),x.shape[1]//2,2,x.shape[2]//2,2);a=cells[:,:,0,:,0];b=cells[:,:,0,:,1];c=cells[:,:,1,:,0];d=cells[:,:,1,:,1];ll=(a+b+c+d)*.5;lh=(a-b+c-d)*.5;hl=(a+b-c-d)*.5;hh=(a-b-c+d)*.5;energy=np.column_stack([np.mean(v*v,axis=(1,2)) for v in [ll,lh,hl,hh]]);out.append(energy/np.maximum(energy.sum(1,keepdims=True),1e-12));x=ll
    return np.concatenate(out,1)

def affine(z,arm,raw):
    x=torch.cat([z[:,0],z[:,1:].mean(1)],1);x=x/x.norm(dim=1,keepdim=True).clamp_min(1e-12);native=torch.cat([x,torch.ones_like(x[:,:1])],1)/(2**.5)
    if arm==ARMS[0]:return native
    f=texture(raw,{'NATIVE_HISTOGRAM':'hist','NATIVE_SHUFFLED_TEXTURE':'shuffle','NATIVE_REAL_TEXTURE':'real'}[arm]);f=f/f.norm(dim=1,keepdim=True).clamp_min(1e-12);return torch.cat([native,f/(2**.5)],1)

def numpy_affine(z,arm,raw):
    x=np.concatenate([z[:,0],np.sum(z[:,1:],axis=1)/196],1);x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12);native=np.column_stack([x,np.ones(len(x))])/np.sqrt(2)
    if arm==ARMS[0]:return native
    f=numpy_texture(raw,{'NATIVE_HISTOGRAM':'hist','NATIVE_SHUFFLED_TEXTURE':'shuffle','NATIVE_REAL_TEXTURE':'real'}[arm]);f/=np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-12);return np.column_stack([native,f/np.sqrt(2)])

def pack(matrix):
    ix=torch.triu_indices(len(matrix),len(matrix));return matrix[ix[0],ix[1]]
def unpack_numpy(value,d):
    x=np.zeros((d,d),dtype=np.float64);x[np.triu_indices(d)]=value;x+=np.triu(x,1).T;return x

def compact_systems(systems):
    base=systems[ARMS[0]];common=dict(matrix=base['matrix_upper'],arrival=base['arrival_upper']);out={}
    for a,s in systems.items():
        if a==ARMS[0]:out[a]=dict(rhs=s['rhs'],d=s['d']);continue
        d=s['d'];m=unpack_numpy(s['matrix_upper'].numpy(),d);v=unpack_numpy(s['arrival_upper'].numpy(),d);out[a]=dict(rhs=s['rhs'],d=d,cross=torch.from_numpy(m[:1537,1537:].copy()),aux=torch.from_numpy(m[1537:,1537:].copy()),arrival_cross=torch.from_numpy(v[:1537,1537:].copy()),arrival_aux=torch.from_numpy(v[1537:,1537:].copy()))
    return dict(common=common,arms=out)

def expand_system(compact,arm):
    s=compact['arms'][arm];base=unpack_numpy(compact['common']['matrix'].numpy(),1537);arrival=unpack_numpy(compact['common']['arrival'].numpy(),1537)
    if arm==ARMS[0]:return base,arrival,s['rhs'].numpy(),1537
    cross=s['cross'].numpy();delta=s['arrival_cross'].numpy();m=np.block([[base,cross],[cross.T,s['aux'].numpy()]]);v=np.block([[arrival,delta],[delta.T,s['arrival_aux'].numpy()]]);return m,v,s['rhs'].numpy(),s['d']
