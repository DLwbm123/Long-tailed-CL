"""Fixed spatial operators over one shared original native token field."""
import random
import numpy as np
import torch
ARMS=['NATIVE_RIDGE','BILATERAL_RIDGE','RANDOM4_RIDGE','QUADRANT4_RIDGE']

def groups(arm,seed=74008):
    if arm=='NATIVE_RIDGE':return [list(range(196))]
    if arm=='BILATERAL_RIDGE':return [[i for i in range(196) if (i%14<7)==left] for left in [True,False]]
    if arm=='RANDOM4_RIDGE':
        ix=list(range(196));random.Random(seed).shuffle(ix);return [ix[i*49:(i+1)*49] for i in range(4)]
    assert arm=='QUADRANT4_RIDGE';return [[i for i in range(196) if i//14//7==y and i%14//7==x] for y in [0,1] for x in [0,1]]

def affine(z,arm,seed=74008):
    v=torch.cat([z[:,0]]+[z[:,1:][:,g].mean(1) for g in groups(arm,seed)],1);v=v/v.norm(dim=1,keepdim=True).clamp_min(1e-12)
    return torch.cat((v,torch.ones_like(v[:,:1])),1)/(2**.5)

def numpy_affine(z,arm,seed=74008):
    batch,tokens,channels=z.shape;values=[z[:,0]];grid=z[:,1:].reshape(batch,14,14,channels)
    if arm=='NATIVE_RIDGE':values.append(grid.sum((1,2))/196)
    elif arm=='BILATERAL_RIDGE':values.extend([grid[:,:,:7].sum((1,2))/98,grid[:,:,7:].sum((1,2))/98])
    elif arm=='QUADRANT4_RIDGE':
        for y in [0,7]:
            for x in [0,7]:values.append(grid[:,y:y+7,x:x+7].sum((1,2))/49)
    else:
        assert arm=='RANDOM4_RIDGE';ix=list(range(196));random.Random(seed).shuffle(ix)
        for j in range(4):
            v=np.zeros((batch,channels),dtype=np.float64)
            for i in ix[49*j:49*(j+1)]:v+=z[:,i+1]
            values.append(v/49)
    v=np.concatenate(values,1);v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
    return np.concatenate((v,np.ones((batch,1))),1)/np.sqrt(2)

def pack(matrix):
    ix=torch.triu_indices(len(matrix),len(matrix));return matrix[ix[0],ix[1]]

def unpack_numpy(value,d):
    x=np.zeros((d,d),dtype=np.float64);x[np.triu_indices(d)]=value;x+=np.triu(x,1).T;return x
