import torch,json,time
from boundary import mass,objective,fit,logits
t=time.time();torch.set_num_threads(4);g=torch.Generator().manual_seed(74002)
z=torch.nn.functional.normalize(torch.randn(11,5,generator=g,dtype=torch.double),dim=1)
y=torch.tensor([0]*6+[1]*3+[2]*2);p=torch.nn.functional.normalize(torch.randn(3,5,generator=g,dtype=torch.double),dim=1)
v=(torch.randn(3,6,generator=g,dtype=torch.double)*.03).requires_grad_();base=logits(z,p)
assert torch.equal(logits(z,p,torch.zeros_like(v)),base)
for balanced in [False,True]:
 w=mass(y,3,balanced);loss=objective(v,z,y,base,w);grad=torch.autograd.grad(loss,v)[0]
 for index in [(0,0),(1,4),(2,5)]:
  delta=torch.zeros_like(v);delta[index]=1e-6
  numerical=(objective(v+delta,z,y,base,w)-objective(v-delta,z,y,base,w))/2e-6
  assert abs(float(numerical-grad[index]))<1e-7
 result=fit(z,y,p,balanced);assert result['max_abs_gradient']<1e-5
 if balanced:assert all(abs(float(w[y==c].sum())-1)<1e-12 for c in range(3))
print(json.dumps(dict(passed=True,zero_residual_NCM_exact=True,gradient_finite_difference=True,
                     equal_total_mass=True,toy_CPU_head_fits=2,visual_optimizer_steps=0,
                     model_scores=0,VAL_reads=0,wall_seconds=time.time()-t)))
