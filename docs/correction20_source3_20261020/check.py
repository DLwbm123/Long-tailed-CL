import time,json,numpy as np,torch
START=time.time()
from origin import block_features
class Data:
 def x(self,ids):return torch.tensor([[1.,0.]*768+[1.],[0.,1.]*768+[1.]],dtype=torch.float64)[ids]
x=Data();p=torch.tensor([[1.,0.]*768,[0.,1.]*768],dtype=torch.float64);p=torch.cat([p,torch.ones(18,1536,dtype=torch.float64)],0);b=np.zeros((2,20));f=b.copy();b[:,0]=1;f[:,1]=1
v=block_features(x,np.array([0,1]),dict(prototypes=p),b,f);assert np.allclose(v,np.array([[-2.,-2.],[2.,2.]]))
p[:,:768]*=3;p[:,768:]*=7;vv=block_features(x,np.array([0,1]),dict(prototypes=p),b,f);assert np.allclose(v,vv)
print(json.dumps(dict(state='PASS',block_scale_invariance_checked=True,engineering_seconds=time.time()-START)))
