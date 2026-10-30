"""Synthetic invariants only; no real scores/images or efficacy screening."""
import json,time,torch,copy,resource
from decision import inputs,outcomes,Policy,uniform,joint,kl,draw
from fit import optimize
start=time.time();cpu=time.process_time();torch.set_num_threads(4)
s=torch.tensor([[3.,2.,0.],[2.,2.,0.],[0.,3.,2.],[0.,1.,2.],[2.,0.,3.],[2.,0.,1.]],dtype=torch.double);y=torch.tensor([1,1,2,2,0,0]);proto=torch.eye(3,dtype=torch.double);c=inputs(s,proto,torch.ones(3,dtype=torch.double),torch.zeros(3));o=outcomes(c,y,[0,1,2],[0,2]);p=uniform(c);assert abs(float(joint(p).sum())-1)<1e-12 and torch.allclose(p[1].sum(1),torch.ones(len(p[0]),dtype=torch.double));assert c['mask'][0].sum()==1
for a in range(len(c['actions'])):
 for b in range(len(c['actions'])):
  if not c['mask'][a,b]:continue
  z=s.clone()
  for action in [a,b]:
   if action:
    t,d=c['actions'][action];rows=c['groups'][action];z[rows,t]=s[rows,d];z[rows,d]=s[rows,t]
  pred=z.argmax(1);corr=torch.bincount(y[pred==y],minlength=3).double();ce=torch.nn.functional.cross_entropy(z,y,reduction='none');loss=torch.zeros(3,dtype=torch.double).index_add_(0,y,ce)
  assert torch.equal(corr,o['base_correct']+o['delta_correct'][a]+o['delta_correct'][b])
  assert torch.allclose(loss,o['base_ce']+o['delta_ce'][a]+o['delta_ce'][b],atol=1e-12,rtol=0)
assert any(c['groups'][i][0] for i in range(1,len(c['actions'])))
torch.manual_seed(44);m=Policy();pp=m.distribution(c);assert torch.allclose(pp[0],p[0]) and torch.allclose(pp[1],p[1]);assert abs(float(kl(pp,c)))<1e-12
loss=-(joint(pp)*o['reward_z']).sum();loss.backward();param=m.net[-1].weight;analytic=float(param.grad[0,0]);eps=1e-6
with torch.no_grad():
 param[0,0]+=eps;plus=float(-(joint(m.distribution(c))*o['reward_z']).sum());param[0,0]-=2*eps;minus=float(-(joint(m.distribution(c))*o['reward_z']).sum());param[0,0]+=eps
assert abs(analytic-(plus-minus)/(2*eps))<1e-7
cfg=dict(groups=1,epochs=1,group_size=8,beta=.04,clip=.2);steps=0
for arm in ['G','H','CE','E']:
 torch.manual_seed(44);model=Policy();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=0);v=optimize(model,opt,c,o,arm,cfg,89);steps+=v['steps'];assert abs(float(v['final']['joint'].sum())-1)<1e-12
# all-terminal-tie case: one legal stop trajectory, finite zero advantages
cc=inputs(torch.tensor([[1.,0.]],dtype=torch.double),torch.eye(2,dtype=torch.double),torch.ones(2,dtype=torch.double),torch.zeros(2));oo=outcomes(cc,torch.tensor([0]),[0],[0]);oo['reward'].zero_();oo['reward_z'].zero_();torch.manual_seed(44);mm=Policy();opt=torch.optim.AdamW(mm.parameters(),lr=.0003,weight_decay=0);v=optimize(mm,opt,cc,oo,'G',cfg,89);steps+=1;assert all(x==0 for x in v['history'][0]['advantage'])
print(json.dumps(dict(passed=True,synthetic_policy_optimizer_steps=steps,visual_optimizer_steps=0,new_real_model_scores=0,VAL_TEST_reads=0,two_slot_decoder_additive_counts_CE=True,legal_tree_probability=True,stop_semantics=True,one_support_action_freedom=True,exact_gradient_finite_difference_error=abs(analytic-(plus-minus)/(2*eps)),CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start,platform='CPU',peak_RSS_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))
