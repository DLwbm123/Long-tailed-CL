"""Independent crop/pixel permutation/objective references, without a native model."""
import numpy as np

def crop(raw,top,left):
    x=np.asarray(raw)[:,:,top:top+112,left:left+112];q=np.maximum(0,np.minimum(111,(np.arange(224)+.5)/2-.5));lo=np.floor(q).astype(int);hi=np.minimum(lo+1,111);w=(q-lo).astype(x.dtype)
    y=x[:,:,lo,:]*(1-w)[None,None,:,None]+x[:,:,hi,:]*w[None,None,:,None]
    return y[:,:,:,lo]*(1-w)[None,None,None,:]+y[:,:,:,hi]*w[None,None,None,:]

def patchify(raw):return raw.reshape(len(raw),3,14,16,14,16).transpose(0,2,4,1,3,5).reshape(len(raw),196,768)

def corrupt(raw,receipt):
    # Explicit physical 16px cells and RGB-shared pixel indexing, independent of patchify/gather.
    out=np.empty_like(raw)
    for token,permutation in enumerate(receipt['permutations']):
        y,x=divmod(token,14);tile=raw[:,:,16*y:16*y+16,16*x:16*x+16].reshape(len(raw),3,256);out[:,:,16*y:16*y+16,16*x:16*x+16]=tile[:,:,permutation].reshape(len(raw),3,16,16)
    return out

def objective(z,target,labels,arm,decoder,word_head,disease_head):
    v=np.concatenate((z[:,0],z[:,1:].mean(1)),1);v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
    if arm=='FULL_IMAGE_SUPERVISED':logits=25*v@(disease_head/np.maximum(np.linalg.norm(disease_head,axis=1,keepdims=True),1e-12)).T;rest=0.;factor=1.
    else:
        rest=float(np.square(z[:,1:]@decoder['weight'].T+decoder['bias']-patchify(target)).mean())
        if arm=='WORD_RESTORATION':return rest
        assert arm=='VISUAL_WORD_CLASS_RESTORATION';logits=v@word_head['weight'].T+word_head['bias'];factor=.01
    logits=logits-logits.max(1,keepdims=True);ce=np.log(np.exp(logits).sum(1))-logits[np.arange(len(labels)),labels]
    return rest+factor*float(ce.mean())
