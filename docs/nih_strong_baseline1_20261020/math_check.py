"""Small independent LA identity check; no image/model fitting."""
import json,time,math
START=time.time()
z=[.2,-.4,.7];n=[7.,2.,1.];y=1;p=[v/sum(n) for v in n];a=[math.exp(s)*q for s,q in zip(z,p)];manual=-math.log(a[y]/sum(a))
import torch
v=torch.tensor(z,dtype=torch.float64,requires_grad=True);prior=torch.tensor(p,dtype=torch.float64);loss=torch.nn.functional.cross_entropy((v+prior.log())[None],torch.tensor([y]));loss.backward();expected=[q/sum(a)-(i==y) for i,q in enumerate(a)];assert abs(float(loss.detach())-manual)<1e-12 and max(abs(float(g)-h) for g,h in zip(v.grad,expected))<1e-12
print(json.dumps(dict(state='PASS',LA_identity_and_gradient=True,engineering_seconds=time.time()-START)))
