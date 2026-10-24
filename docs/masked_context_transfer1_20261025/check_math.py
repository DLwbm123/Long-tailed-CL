"""One bounded new CPU objective/gradient and masking-layout check, no model or data."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
from core import patchify,mask_receipt

def main(path):
    torch.set_num_threads(2);tick=time.time();torch.manual_seed(74006)
    x=torch.randn(4,5,dtype=torch.float64,requires_grad=True);w=torch.randn(3,5,dtype=torch.float64,requires_grad=True);b=torch.randn(3,dtype=torch.float64,requires_grad=True);target=torch.randn(4,3,dtype=torch.float64)
    value=(x@w.T+b-target).abs().mean();value.backward();d=np.sign(x.detach().numpy()@w.detach().numpy().T+b.detach().numpy()-target.numpy())/12
    error=max(float(abs(x.grad.numpy()-d@w.detach().numpy()).max()),float(abs(w.grad.numpy()-d.T@x.detach().numpy()).max()),float(abs(b.grad.numpy()-d.sum(0)).max()))
    raw=torch.arange(3*224*224,dtype=torch.float64).reshape(1,3,224,224);patch=patchify(raw).numpy();expected=raw.numpy().reshape(1,3,14,16,14,16).transpose(0,2,4,1,3,5).reshape(1,196,768);layout=float(abs(patch-expected).max())
    receipts=[mask_receipt(74006+i) for i in range(10)];out=dict(state='PASS' if error<=1e-12 and layout==0 else 'FAIL',L1_derivative_error=error,patch_layout_error=layout,mask_receipts=len(receipts),tokens_each=[len(r['tokens']) for r in receipts],CPU_forward=1,CPU_backward=1,real_models=0,NIH_images=0,GPU_calls=0,optimizer_steps=0,wall_seconds=time.time()-tick);Path(path).write_text(json.dumps(out,indent=2)+'\n');assert out['state']=='PASS';print(json.dumps(out))
if __name__=='__main__':main(sys.argv[1])
