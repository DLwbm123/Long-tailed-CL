import torch
from positive import pooled_text_ce
for device in ['cpu','cuda']:
 torch.manual_seed(58);z=torch.randn(7,7,dtype=torch.float64,device=device,requires_grad=True);y=torch.arange(7,device=device)
 a=pooled_text_ce(z,y);b=torch.nn.functional.cross_entropy(z,y);ga=torch.autograd.grad(a,z,retain_graph=True)[0];gb=torch.autograd.grad(b,z)[0]
 assert torch.allclose(a,b,atol=1e-12) and torch.allclose(ga,gb,atol=1e-12)
 y=torch.tensor([0,0,0,1,1,2,3],device=device);image=torch.randn(7,9,dtype=torch.double,device=device);text=torch.randn(4,9,dtype=torch.double,device=device);z=(text[y]@image.T).detach().requires_grad_(True)
 loss=pooled_text_ce(z,y);g=torch.autograd.grad(loss,z,retain_graph=True)[0];original=torch.nn.functional.cross_entropy(z,torch.arange(7,device=device));go=torch.autograd.grad(original,z)[0]
 assert torch.isfinite(g).all() and (g-go).norm()>1e-6 and loss<=original+1e-12
 p=torch.tensor([6,2,3,0,5,1,4],device=device);assert torch.allclose(pooled_text_ce(z[p][:,p],y[p]),loss,atol=1e-12)
 zz=torch.randn(4,4,dtype=torch.double,device=device,requires_grad=True);v=pooled_text_ce(zz,torch.zeros(4,dtype=torch.long,device=device));gg=torch.autograd.grad(v,zz)[0];assert v==0 and gg.abs().max()<1e-12
print('POOLED_T2I_UNIQUE_PARITY_DUPLICATE_GRAD_PERMUTATION_ALLPOSITIVE_PASS')
