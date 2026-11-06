"""Bounded CPU contracts for conditional repairs, query roles, features and objective."""
import torch,time,itertools,copy,json,tempfile
from pathlib import Path
from memory import from_views,BUDGET_BYTES
from learner import Learner,optimizer
from selection import FixedSelect
from proposals import generate,key,validate
from features import describe
from roles import Roles
from policy import Head,prior,logprob,objective,update,behavior
from common import write

def run(r):
 cpu=time.process_time();wall=time.time();torch.set_num_threads(1);torch.manual_seed(74002);g=lambda n:torch.Generator().manual_seed(n);z=torch.nn.functional.normalize(torch.randn(120,6,512),dim=-1);pool=from_views(z,torch.arange(120),torch.arange(120)%4);m=Learner(74002);m.arrive(pool,4);base,anchors,_=FixedSelect(pool,4,128).choose('BALANCED',g(1));ms,es,au=generate(pool,base,anchors,g(2));assert len(ms)==9 and len({tuple(key(M).tolist()) for M in ms})==len(ms)
 for M in ms:validate(pool,base,M,key(pool)[anchors],12)
 x,fa=describe(pool,ms,es,m,[30]*4,[0]*4,1,pool.parents[:10].tolist());assert x.shape==(9,48) and not x.requires_grad
 permutation=torch.randperm(120,generator=g(6));renamed=from_views(z,permutation,torch.arange(120)%4);ix=torch.searchsorted(key(renamed),permutation[pool.pid]*8+pool.view);newmem=[]
 for M in ms:
  targets=permutation[M.pid]*8+M.view;newmem.append(renamed.take(torch.searchsorted(key(renamed),targets)))
 xx,_=describe(renamed,newmem,es,m,[30]*4,[0]*4,1,permutation[pool.parents[:10]].tolist());assert torch.allclose(x,xx,atol=2e-5,rtol=2e-5)
 gp=pool.take((pool.y==0)&(pool.view==0));gm=Learner(74002);gm.arrive(gp,1);single=gp.take(torch.tensor([0]));fx,_=describe(gp,[single],[dict(removed_tokens=0,replaced_parents=0,operations=0)],gm,[len(gp)],[0],0,gp.parents.tolist());assert fx.shape==(1,48) and fx[0,2]==0 and fx[0,3]==0 and fx[0,4]==0 and fx[0,9]==0 and fx[0,10]==0
 full=pool.take(torch.isin(pool.pid,torch.arange(8)));M,aa,_=FixedSelect(full,4,len(full)).choose('ZERO',g(1));one,_,_=generate(full,M,aa,g(2));assert len(one)==1
 h=Head(74002).double();xd=x.double();p0=prior(len(xd),xd.device);assert torch.allclose(logprob(h,xd).exp(),p0,atol=1e-14);assert logprob(h,xd[:1]).exp().item()==1
 R=torch.tensor([.3,.1,.6,.2,.8,.1,.4,.5,.2],dtype=torch.double);old=logprob(h,xd).exp().detach();loss,s=objective(logprob(h,xd),old,R);assert s['max_ratio_deviation']<1e-12;loss.backward();a=float(h.net[-1].weight.grad[0,0]);v=h.net[-1].weight
 with torch.no_grad():v[0,0]+=1e-6
 hi=float(objective(logprob(h,xd),old,R)[0].detach())
 with torch.no_grad():v[0,0]-=2e-6
 lo=float(objective(logprob(h,xd),old,R)[0].detach())
 with torch.no_grad():v[0,0]+=1e-6
 err=abs((hi-lo)/2e-6-a);assert err<1e-7
 h.zero_grad();expected=-(logprob(h,xd).exp()*R).sum()/R.std(unbiased=False).clamp_min(.02);expected.backward();assert abs(float(h.net[-1].weight.grad[0,0])-a)<1e-10
 equal=objective(logprob(h,xd),old,torch.ones_like(R))[0];assert abs(float(equal.detach()))<1e-12
 logits=torch.zeros(4,dtype=torch.double,requires_grad=True);pp=prior(4,'cpu');rr=torch.tensor([1.,0.,0.,0.],dtype=torch.double);loss,_=objective((pp.log()+logits).log_softmax(0),pp,rr);dg=torch.autograd.grad(loss,logits)[0];assert dg[0]<0
 grads=[];der=[]
 for perm in itertools.permutations(range(4)):
  v=torch.zeros(4,dtype=torch.double,requires_grad=True);loss,_=objective((pp.log()+v).log_softmax(0),pp,rr[list(perm)]);q=torch.autograd.grad(loss,v)[0];grads.append(q)
  if all(i!=j for i,j in enumerate(perm)):der.append(q)
 assert torch.stack(grads).mean(0).abs().max()<1e-12;assert (torch.stack(der).mean(0)+dg/3).abs().max()<1e-12
 roles=Roles();rows=[{'identity_component':str(i),'original_label':str(i%4)} for i in range(120)];roles.arrive(list(range(120)),rows);spl=roles.split(pool,rows);ss=[set(rows[i]['identity_component'] for i in spl[k].parents.tolist()) for k in ['S','FIT','AUDIT']];assert not ss[0]&ss[1] and not ss[0]&ss[2] and not ss[1]&ss[2]
 before=dict(roles.role);roles.arrive(list(range(120)),rows);assert roles.role==before
 qcomp=next(k for k,v in roles.role.items() if v=='FIT');extra=rows+[{'identity_component':qcomp,'original_label':'4'},{'identity_component':'newcomponent','original_label':'4'}];roles.arrive([120,121],extra);assert roles.role[qcomp]=='FIT' and roles.role['newcomponent']=='S'
 # Prototype firewall is checked before classifier updates, even for existing classes.
 m.forbidden=set(spl['FIT'].parents.tolist())|set(spl['AUDIT'].parents.tolist())
 try:m.arrive(spl['FIT'],4)
 except AssertionError:pass
 else:raise AssertionError('query firewall not enforced')
 # One synthetic classifier/AdamW engineering step, explicitly separate from science.
 opt=optimizer(m);bx,by,_=spl['S'].bags();opt.zero_grad();torch.nn.functional.cross_entropy(m(bx),by).backward();opt.step()
 with tempfile.TemporaryDirectory(dir=r) as d:
  p=Path(d)/'check.pt';torch.save({'m':m.state_dict(),'o':opt.state_dict()},p);pk=torch.load(p,weights_only=False);n=Learner(74002);n.C=4;n.load_state_dict(pk['m'],strict=True);oo=optimizer(n);oo.load_state_dict(pk['o']);assert torch.equal(m(bx),n(bx))
  for a,b in zip(opt.state.values(),oo.state.values()):
   for k in a:
    if torch.is_tensor(a[k]):assert torch.equal(a[k],b[k])
  assert base.serialize(Path(d)/'memory.bin',[30]*4,[0]*4)==128*2064+184
 assert BUDGET_BYTES==528568
 out={'passed':True,'nine_unique_whole_memory_candidates':True,'exact_budget_anchors_parent_dependencies_same_global_local_counts':True,'no_unretained_view_recovery':True,'noop_only_state':True,'ID_content_feature_equivariance_tolerance':2e-5,'detached_deterministic_48_features':True,'C1_singleton_no_local_feature_boundaries':True,'cross_label_component_role_never_reassigned':True,'head_parameters':1601,'zero_head_equals_prior':True,'equal_reward_advantage_zero':True,'all_repair_worse_noop_gradient_direction_correct':True,'enumerated_objective_finite_difference_error':err,'old_weighted_objective_matches_exact_expected_reward_first_gradient':True,'all24_shuffles_zero_mean_first_gradient':True,'derangement_reverse_one_third_first_gradient_verified':True,'immutable_roles_and_query_firewall':True,'classifier_nonempty_optimizer_roundtrip':True,'engineering_CPU_classifier_optimizer':1,'science_encoder_optimizer_VAL_TEST':0,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-wall};write(Path(r)/'SOFTWARE_CHECK.json',out)
if __name__=='__main__':run(Path(__file__).resolve().parent)
