import json,os
from pathlib import Path
import numpy as np
import torch
from core import texture,numpy_texture,permutation,ARMS,affine,numpy_affine

def main():
    torch.set_num_threads(2);g=torch.Generator().manual_seed(74010);raw=torch.rand(2,3,224,224,generator=g,dtype=torch.float64);z=torch.randn(2,197,768,generator=g,dtype=torch.float64);errors=[]
    for kind in ['real','shuffle','hist']:
        e=float(abs(texture(raw,kind).numpy()-numpy_texture(raw.numpy(),kind)).max());assert e<=1e-10;errors.append(e)
    for arm in ARMS:
        e=float(abs(affine(z,arm,raw).numpy()-numpy_affine(z.numpy(),arm,raw.numpy())).max());assert e<=1e-10;errors.append(e)
    assert sorted(permutation())==list(range(224*224));x=raw.mean(1).flatten(1);assert torch.equal(torch.sort(x,dim=1).values,torch.sort(x[:,permutation()],dim=1).values)
    const=torch.ones(1,3,224,224,dtype=torch.float64)
    assert all(bool(torch.isfinite(texture(const,k)).all()) for k in ['hist','real','shuffle'])
    a,b,c,d=[torch.randn(4,4,generator=g,dtype=torch.float64) for _ in range(4)];bands=[(a+b+c+d)/2,(a-b+c-d)/2,(a+b-c-d)/2,(a-b-c+d)/2];energy=float(abs(sum((v*v).sum() for v in bands)-sum((v*v).sum() for v in [a,b,c,d])));assert energy<=1e-10
    result=dict(state='PASS',checks=11,synthetic_images=2,synthetic_token_fields=2,feature_max_error=max(errors),Haar_energy_error=energy,optimizer_steps=0,NIH_images=0,GPU=False)
    Path(os.environ['MATH_OUTPUT']).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
