"""Fixed RGB crops before the original deterministic Biomed transform."""
import math
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from common import guard

BOXES=[(0,0,1,1),(.125,.125,.875,.875),(0,0,.6,.6),(.4,0,1,.6),(0,.4,.6,1),(.4,.4,1,1)]
def crop(image,k):
    w,h=image.size;l,t,r,b=BOXES[k]
    l=max(0,min(w-1,math.floor(w*l)));t=max(0,min(h-1,math.floor(h*t)))
    r=max(l+1,min(w,math.ceil(w*r)));b=max(t+1,min(h,math.ceil(h*b)))
    return image.crop((l,t,r,b))

@torch.no_grad()
def prefix(store,pixels):
    t=store.visual.trunk;x=t.norm_pre(t.patch_drop(t._pos_embed(t.patch_embed(pixels))))
    for block in t.blocks[:10]:x=block(x)
    return x

@torch.no_grad()
def encode_views(c,rows,ids,store,tails,counts,split):
    # General prefix is shared only while this minibatch is live; no crop token files.
    out={k:[] for k in tails}
    for lo in range(0,len(ids),8):
        guard(c);ix=ids[lo:lo+8];images=[]
        for i in ix:
            with Image.open(Path(c['image_root'])/rows[i]['relative_path']) as im:images.append(im.convert('RGB'))
        views={k:[] for k in tails}
        for v in range(6):
            pix=torch.stack([store.tf(crop(im,v)) for im in images]).cuda();tok=prefix(store,pix)
            counts[split+'_generic_view_rows']+=len(ix)
            for k,model in tails.items():
                views[k].append(model(tok).cpu());counts[split+'_teacher_view_rows']+=len(ix)
            del tok,pix
        for k in tails:
            z=torch.stack(views[k],1);assert torch.isfinite(z).all()
            out[k].append(torch.stack([z[:,0],z[:,1],F.normalize(z[:,2:].mean(1),dim=-1)],1))
    return {k:torch.cat(v) for k,v in out.items()}
