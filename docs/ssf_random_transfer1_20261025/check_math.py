"""Unique bounded synthetic SSF/RP normalization and two-step SGD equation check."""
import json,os
from pathlib import Path
import numpy as np
import torch
from core import affine,numpy_affine,ARMS

def main():
    torch.set_num_threads(2);g=torch.Generator().manual_seed(74009);z=torch.randn(3,197,4,generator=g,dtype=torch.float64);W=torch.randn(8,13,generator=g,dtype=torch.float64);errors=[]
    for arm in ARMS:
        error=float(abs(affine(z,arm,W).numpy()-numpy_affine(z.numpy(),arm,W.numpy())).max());assert error<=1e-12;errors.append(error)
    x=torch.randn(3,4,generator=g,dtype=torch.float64);s=torch.randn(4,generator=g,dtype=torch.float64,requires_grad=True);b=torch.randn(4,generator=g,dtype=torch.float64,requires_grad=True);loss=((x*s+b)**2).mean();loss.backward();v=x.numpy()*s.detach().numpy()+b.detach().numpy();ge=max(float(abs(s.grad.numpy()-2*(v*x.numpy()).mean(0)/4).max()),float(abs(b.grad.numpy()-2*v.mean(0)/4).max()));assert ge<=1e-12
    old=np.array([1.,-2.]);moment=None;param=torch.tensor(old.copy(),dtype=torch.float64)
    for gradient in [np.array([.3,-.4]),np.array([-.2,.7])]:
        d=gradient+.0005*old;moment=d.copy() if moment is None else .9*moment+d;old=old-.01*moment
        # Pure equation arithmetic, no optimizer or realmodel.
        td=torch.tensor(gradient)+.0005*param;buf=td.clone() if 'buf' not in locals() else .9*buf+td;param=param-.01*buf
    se=float(abs(param.numpy()-old).max());assert se<=1e-12
    result=dict(state='PASS',checks=6,synthetic_token_rows=3,objective_CPU_forward=1,objective_CPU_backward=1,optimizer_updates=0,NIH_images=0,GPU=False,feature_error=max(errors),SSF_gradient_error=ge,SGD_equation_error=se)
    Path(os.environ['MATH_OUTPUT']).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
