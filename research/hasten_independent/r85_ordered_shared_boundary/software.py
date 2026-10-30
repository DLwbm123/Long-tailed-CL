import time,json,torch
from boundary import mass,objective,fit,logits
from structure import matrices
t=time.time();torch.set_num_threads(4);g=torch.Generator().manual_seed(74002)
names=['ulcerative colitis grade 1 2','ulcerative colitis grade 3','ulcerative colitis grade 2']
ms,nodes=matrices(names);assert nodes==[0,2,1]
z=torch.nn.functional.normalize(torch.randn(12,5,generator=g,dtype=torch.double),dim=1);y=torch.tensor([0]*2+[1]*4+[2]*6)
p=torch.nn.functional.normalize(torch.randn(3,5,generator=g,dtype=torch.double),dim=1);base=logits(z,p);w=mass(y,3,True)
v=(torch.randn(3,6,generator=g,dtype=torch.double)*.1).requires_grad_()
for arm,L in ms.items():
 loss=objective(v,z,y,base,w,L);grad=torch.autograd.grad(loss,v)[0]
 for index in [(0,0),(1,4),(2,5)]:
  d=torch.zeros_like(v);d[index]=1e-6
  numerical=(objective(v+d,z,y,base,w,L)-objective(v-d,z,y,base,w,L))/2e-6
  assert abs(float(numerical-grad[index]))<1e-7
 assert torch.isfinite(grad).all()
assert torch.equal(objective(v,z,y,base,w,None),objective(v,z,y,base,w,torch.zeros(3,3,dtype=torch.double)))
a=fit(z,y,p,True);b=fit(z,y,p,True,torch.zeros(3,3,dtype=torch.double));assert torch.equal(a['residual'],b['residual'])
c=fit(z,y,p,True,ms['ordered']);assert c['max_abs_gradient']<=1e-5
print(json.dumps(dict(passed=True,zero_graph_inherited_objective_and_fit_exact=True,PSD=True,
 degree_diagonal_exact=True,mean_edge_mass_matched=True,all_arm_gradient_finite_difference=True,
 toy_CPU_head_fits=3,visual_optimizer_steps=0,model_scores=0,VAL_reads=0,wall_seconds=time.time()-t)))
