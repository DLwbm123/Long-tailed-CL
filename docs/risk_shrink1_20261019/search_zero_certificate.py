import json,os,time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
r=Path(os.environ['RUN_OUTPUT']);p=np.load(r/'problem_2.private.npz');bs=p['bs'];sc=np.maximum(p['baselines'],1e-8);t=2*bs/sc[:,None];k=t.shape[1];g=len(t)
start=time.time()
a=linprog(2*p['b'],A_ub=t,b_ub=np.zeros(g),A_eq=np.ones((1,k)),b_eq=[1.],bounds=[(0,None)]*k,method='highs')
b=linprog(np.r_[np.zeros(g),-1.],A_ub=np.c_[-t.T,np.ones(k)],b_ub=np.zeros(k),A_eq=np.array([np.r_[np.ones(g),0.]]),b_eq=[1.],bounds=[(0,None)]*g+[(None,None)],method='highs')
print('PRIMAL',a.message,a.status,a.fun)
print('DUAL',b.message,b.status,b.fun)
if a.success:print('DIRECTION',a.x.tolist(),'GRAD_MAX',float((t@a.x).max()))
if b.success:
 lam=b.x[:-1];v=t.T@lam
 print('GAMMA',b.x[-1],v.tolist());print('SUPPORT',[(i,float(x)) for i,x in enumerate(lam) if x>0])
 np.savez(r/'zero_direction_certificate.private.npz',multipliers_normalized=lam,gamma=b.x[-1])
print('WALL',time.time()-start)
