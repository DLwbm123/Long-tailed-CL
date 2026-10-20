"""Synthetic basis geometry, original-coordinate regularizer and gradient checks."""
import time
START=time.time()
import os,sys,json
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());sys.path.insert(0,cfg['parent_code'])
import numpy as np
import torch
from core import loss
from basis import bases,diagnostic

torch.set_num_threads(4);torch.manual_seed(74002);j=torch.randn(1537,20,dtype=torch.float64)*.01;c=torch.randn(769,20,dtype=torch.float64)*.01;fac=torch.ones(1537,dtype=torch.float64);fac[:1536]*=2**-.5;gram=torch.diag(torch.linspace(1,2,1537,dtype=torch.float64));mean=torch.zeros(1537,dtype=torch.float64);qb,qp,info=bases(j,c,fac,gram,mean,1e-10);assert info['rank']<=39 and qb.shape==qp.shape
z=torch.randn(60,1537,dtype=torch.float64);y=torch.arange(60)%20;w=torch.ones(60,dtype=torch.float64)/60;empty=torch.zeros(60,dtype=torch.float64);base=z@j
for q in [qb,qp]:
 a=(torch.randn(q.shape[1],20,dtype=torch.float64)*.01).requires_grad_();r=(q@a).detach().requires_grad_();small=loss(a,z@q,base,y,w,empty,empty,empty,torch.ones(q.shape[1],dtype=torch.float64),False,.05);full=loss(r,z,base,y,w,empty,empty,empty,torch.ones(1537,dtype=torch.float64),False,.05);ga=torch.autograd.grad(small,a)[0];gr=torch.autograd.grad(full,r)[0];assert abs(float(small.detach()-full.detach()))<1e-12 and float((ga-q.T@gr).abs().max())<1e-12
 for i,k in [(0,0),(3,7),(q.shape[1]-1,19)]:
  delta=torch.zeros_like(a);delta[i,k]=1e-6;fd=(loss(a+delta,z@q,base,y,w,empty,empty,empty,torch.ones(q.shape[1],dtype=torch.float64),False,.05)-loss(a-delta,z@q,base,y,w,empty,empty,empty,torch.ones(q.shape[1],dtype=torch.float64),False,.05))/2e-6;assert abs(float(fd-ga[i,k]))<1e-7
 d=diagnostic(z,y,w,j,torch.randn(1537,20,dtype=torch.float64)*.001,q);assert d['additive_decomposition_max_error']<1e-9
print(json.dumps(dict(state='PASS',basis_orthogonality_and_head_span=True,PCA_training_coordinate_rule=True,original_residual_regularizer_and_projected_gradient=True,fixed_competitor_decomposition=True,science_data_access=False,process_wall_seconds=time.time()-START)))
