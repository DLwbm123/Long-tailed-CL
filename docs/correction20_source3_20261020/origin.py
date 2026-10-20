"""Recover block prototype consistency from already frozen joint prototypes."""
import numpy as np
import torch


def block_features(data,ids,head,b,f):
 result=np.empty((len(ids),2));bp=b.argmax(1);fp=f.argmax(1)
 for off in range(0,len(ids),2048):
  rows=ids[off:off+2048];x=data.x(rows);n=len(rows)
  for j,sl in enumerate([slice(0,768),slice(768,1536)]):
   z=x[:,sl];proto=head['prototypes'][:,sl];z=z/torch.linalg.vector_norm(z,dim=1).clamp_min(1e-300)[:,None];proto=proto/torch.linalg.vector_norm(proto,dim=1).clamp_min(1e-300)[:,None];cos=(z@proto.T).numpy();r=np.arange(n);result[off:off+n,j]=2*(cos[r,fp[off:off+n]]-cos[r,bp[off:off+n]])
 return result
