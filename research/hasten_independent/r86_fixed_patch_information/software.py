import json,time,torch
import torch.nn.functional as F
from boundary import objective,mass,fit,logits
from pooling import channels
t=time.time();torch.set_num_threads(4);g=torch.Generator().manual_seed(74002)
z=F.normalize(torch.randn(12,5,generator=g,dtype=torch.double),dim=1)
y=torch.tensor([0]*2+[1]*4+[2]*6);p=F.normalize(torch.randn(3,5,generator=g,dtype=torch.double),dim=1)
u=(torch.randn(3,6,generator=g,dtype=torch.double)*.1).requires_grad_()
zd=channels(z,z)['duplicate'];pd=channels(p,p)['duplicate'];w=mass(y,3,True)
ud=torch.cat([u[:,:-1]/2**.5,u[:,:-1]/2**.5,u[:,-1:]],dim=1)
base=20*z@p.T;based=20*zd@pd.T
assert torch.allclose(base,based,atol=1e-12,rtol=0)
loss=objective(u,z,y,base,w);duplicate=objective(ud,zd,y,based,w)
assert abs(float(loss-duplicate))<1e-11
a=torch.autograd.grad(loss,u,retain_graph=True)[0];b=torch.autograd.grad(duplicate,u)[0]
assert torch.allclose(a,b,atol=1e-11,rtol=0)
assert torch.allclose(logits(z,p),logits(zd,pd),atol=2e-5,rtol=0)
old=fit(z,y,p,True);new=fit(zd,y,pd,True)
mapped=torch.cat([old['residual'][:,:-1]/2**.5,old['residual'][:,:-1]/2**.5,old['residual'][:,-1:]],dim=1)
assert torch.allclose(new['residual'],mapped,atol=2e-5,rtol=0)
assert torch.equal(logits(z,p,old['residual']).argmax(1),logits(zd,pd,new['residual']).argmax(1))
print(json.dumps(dict(passed=True,duplicate_isometry_loss_gradient=True,duplicate_fit_reduces_to_CLS=True,
 toy_CPU_head_fits=2,visual_optimizer_steps=0,model_scores=0,VAL_reads=0,wall_seconds=time.time()-t)))
