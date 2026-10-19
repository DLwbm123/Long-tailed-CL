import torch
from core import training_rms,overlap

torch.set_default_dtype(torch.float64);torch.manual_seed(74002);torch.set_num_threads(4)
x=torch.randn(23,6);x[:,-1]=1;k=4;w=torch.rand(23);w*=k/w.sum();head=torch.randn(6,k)
g=x.T@(x*w[:,None]);scores=x@head;center=scores-scores.mean(1,keepdim=True)
rms=float((center.square()*w[:,None]).sum()/(k*k))**.5
assert abs(rms-training_rms(g,head,k))<1e-12
bias=torch.randn(6,1);assert abs(training_rms(g,head+bias,k)-rms)<1e-12
assert torch.allclose((scores+7)-(scores+7).mean(1,keepdim=True),center,atol=1e-12)
z=overlap({1,2,3},{2,3,4,5});assert z['shared']==2 and z['A_only']==1 and z['J_only']==2
print('PASS: explicit weighted training RMS; common offset invariance; set partition')
