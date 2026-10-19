import json,time
import numpy as np
import torch
from core import loss,weights,targets
start=time.time();torch.set_num_threads(4);torch.manual_seed(74002)
n,d,k=7,5,3;x=torch.randn(n,d,dtype=torch.float64);base=torch.randn(n,k,dtype=torch.float64);y=torch.arange(n)%k;w=torch.ones(n,dtype=torch.float64)/n
r=torch.tensor([0,1,0,0,1,0,0],dtype=torch.float64);keep=1-r;t=torch.ones(n,dtype=torch.float64)*.3;p=torch.ones(d,dtype=torch.float64)*.7
u=torch.randn(d,k,dtype=torch.float64,requires_grad=True)*.02
for utility in [False,True]:
 z=loss(u,x,base,y,w,r,keep,t,p,utility,.05);grad=torch.autograd.grad(z,u)[0]
 for i,j in [(0,0),(2,1),(4,2)]:
  delta=torch.zeros_like(u);delta[i,j]=1e-6
  fd=(loss(u+delta,x,base,y,w,r,keep,t,p,utility,.05)-loss(u-delta,x,base,y,w,r,keep,t,p,utility,.05))/(2e-6)
  assert abs(float(fd-grad[i,j]))<1e-7
assert np.allclose(weights(np.array([0,0,0,1]),np.array([1,1,2,3])),[.25,.25,.5,1.])
c=np.array([[3.,0,0],[3,0,0],[0,3,0]]);j=np.array([[0.,3,0],[3,0,0],[0,3,0]])
a,b,t,temperature,cap=targets(c,j,np.array([0,0,1]),np.ones(3));assert a.tolist()==[True,False,False] and b.tolist()==[False,True,True] and cap>0
print(json.dumps(dict(status='PASS',finite_difference_loss_gradient_verified=True,patient_weights_and_training_only_targets_verified=True,engineering_wall_seconds=time.time()-start)))
