"""Synthetic qualification only: equivalence, chain rule, and both losses."""
import time
START=time.time()
import json
import numpy as np
import torch
from core import loss
from solver import solve

torch.set_num_threads(4);torch.manual_seed(74002);n,d,k=13,5,3;x=torch.randn(n,d,dtype=torch.float64);x[:,1]=x[:,0]*.99+x[:,1]*.01;base=torch.randn(n,k,dtype=torch.float64);y=torch.arange(n)%k;w=torch.ones(n,dtype=torch.float64)/n;r=(y==0).double();keep=(y!=0).double();target=torch.ones(n,dtype=torch.float64)*.4;pre=torch.linspace(.4,1.5,d,dtype=torch.float64)
cfg=dict(maxiter=500,maxfun=15000,maxls=30,maxcor=20,ftol=1e-10,gtol=1e-6);proxy=x.T@(w[:,None]*x)/k+torch.diag(.1*pre**2);checks=[]
for utility in [False,True]:
 def fun(v):
  u=torch.tensor(v.reshape(d,k),dtype=torch.float64,requires_grad=True);f=loss(u,x,base,y,w,r,keep,target,pre,utility,.05);f.backward();return float(f.detach()),u.grad.numpy().ravel()
 l=torch.linalg.cholesky(proxy);v=torch.randn(d,k,dtype=torch.float64)*.01;u=torch.linalg.solve_triangular(l.T,v,upper=True);a,ga=fun(u.numpy().ravel());gu=torch.from_numpy(ga.reshape(d,k));gv=torch.linalg.solve_triangular(l,gu,upper=False)
 for i,j in [(0,0),(2,1),(4,2)]:
  delta=torch.zeros_like(v);delta[i,j]=1e-6;plus=torch.linalg.solve_triangular(l.T,v+delta,upper=True);minus=torch.linalg.solve_triangular(l.T,v-delta,upper=True);fd=(fun(plus.numpy().ravel())[0]-fun(minus.numpy().ravel())[0])/2e-6;assert abs(fd-float(gv[i,j]))<1e-7
 candidate,audit=solve(fun,np.zeros(d*k),proxy,cfg,lambda v:None);assert audit['success'];checks.append(dict(utility=utility,**audit))
print(json.dumps(dict(state='PASS',objective_and_gradient_coordinate_equivalence=True,scientific_data_access=False,synthetic_solver_calls=2,checks=checks,process_wall_seconds=time.time()-START)))
