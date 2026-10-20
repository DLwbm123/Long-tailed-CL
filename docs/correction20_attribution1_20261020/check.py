"""Small synthetic transport, identity-loss, and gradient qualification only."""
import time
START=time.time()
import json,os,sys
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());sys.path.insert(0,cfg['parent_code'])
import numpy as np
import torch
from core import loss
from control import transport,shuffled_loss

torch.set_num_threads(4);torch.manual_seed(74002);y=np.repeat(np.arange(20),3);w=np.tile([.01,.015,.025],20);e,d,m=transport(y,w,74002,0);assert np.array_equal((e,d,m)[0],transport(y,w,74002,0)[0])
r=np.arange(60)%3==0;k=np.arange(60)%3==1;target=np.where(k,.4,0.)
for c in range(20):
 mask=y==c;edges=y[e]==c
 for a in [r,k,target]:assert abs(w[mask]@a[mask]-m[edges]@a[d[edges]])<1e-12
n,dim,classes=60,5,20;x=torch.randn(n,dim,dtype=torch.float64);base=torch.randn(n,classes,dtype=torch.float64);ty=torch.from_numpy(y);tw=torch.from_numpy(w);tr=torch.from_numpy(r.astype(float));tk=torch.from_numpy(k.astype(float));tt=torch.from_numpy(target);pre=torch.ones(dim,dtype=torch.float64);u=(torch.randn(dim,classes,dtype=torch.float64)*.01).requires_grad_();ii=torch.arange(n)
a=loss(u,x,base,ty,tw,tr,tk,tt,pre,True,.05);b=shuffled_loss(u,x,base,ty,tw,tr,tt,tk,pre,ii,ii,tw,.05);assert abs(float(a.detach()-b.detach()))<1e-12
for ee,dd,mm in [(ii,ii,tw),(torch.from_numpy(e),torch.from_numpy(d),torch.from_numpy(m))]:
 v=shuffled_loss(u,x,base,ty,tw,tr,tt,tk,pre,ee,dd,mm,.05);g=torch.autograd.grad(v,u)[0]
 for i,j in [(0,0),(2,7),(4,19)]:
  delta=torch.zeros_like(u);delta[i,j]=1e-6;fd=(shuffled_loss(u+delta,x,base,ty,tw,tr,tt,tk,pre,ee,dd,mm,.05)-shuffled_loss(u-delta,x,base,ty,tw,tr,tt,tk,pre,ee,dd,mm,.05))/2e-6;assert abs(float(fd-g[i,j]))<1e-7
print(json.dumps(dict(state='PASS',weighted_transport_and_both_marginals_verified=True,identity_matches_original_utility_loss=True,gradient_finite_difference_verified=True,new_scientific_data_access=False,process_wall_seconds=time.time()-START)))
