"""CPU engineering checks, including exact two-decision toy policy gradients."""
import itertools,json,torch
from decision import *
def check():
 torch.manual_seed(74002);torch.set_num_threads(4)
 labels=torch.tensor([0]*3+[1]*120+[2]*377);support=list(range(500));common=list(range(100,292));g=torch.Generator().manual_seed(80)
 for action in [-1,0,1,2]:
  ii=allocation(labels,support,common,action,g);assert len(set(ii))==256
  if action==0:assert set(range(3))<=set(ii)
 masks=legal_actions(labels,support,common);assert masks.tolist()==[True,True,True,True]
 u,d=utility(torch.tensor([.2,.5,.8]),torch.tensor([.3,.4,.7]),torch.tensor([True,False,True]));assert torch.allclose(d,torch.tensor(.05))
 # Finite policy expectation enumeration equals its score-function gradient.
 theta=torch.tensor([.1,-.2],dtype=torch.double,requires_grad=True);p=theta.softmax(0);total=theta.sum()*0;sf=torch.zeros(2,dtype=torch.double)
 for a,b in itertools.product(range(2),repeat=2):
  reward=torch.tensor([[.1,-.3],[.5,.2]],dtype=torch.double)[a,b];prob=p[a]*p[b];total=total+prob*reward
  score=torch.autograd.grad(p[a].log()+p[b].log(),theta,retain_graph=True)[0];sf+=prob.detach()*reward*score
 exact=torch.autograd.grad(total,theta)[0];assert torch.allclose(exact,sf,atol=1e-12)
 c=Controller();state=torch.randn(3,8);mask=torch.ones(4,dtype=torch.bool);assert torch.equal(c(state,mask),torch.zeros(4))
 states=[state.clone() for _ in range(16)];mm=[mask.clone() for _ in range(16)];aa=[i%4 for i in range(16)];old=torch.full((8,2),-torch.tensor(4.).log());r=torch.tensor([-.9,-.8,-.7,-.6,-.5,-.4,-.3,-.2]);adv=(r-r.mean())/(r.std(unbiased=False)+1e-8)
 loss,st=grpo_loss(c,states,mm,aa,old,adv);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in c.parameters())
 # A common balanced baseline cancels exactly under group normalization.
 assert torch.allclose(adv,((r-.12)-(r-.12).mean())/((r-.12).std(unbiased=False)+1e-8),atol=1e-6)
 # Permutation fixture is a derangement, including all-negative absolute returns.
 perm=torch.roll(torch.arange(8),1);assert bool((perm!=torch.arange(8)).all()) and bool((r<0).all())
 # The experimental actor remains uniform under arbitrary states and legal masks.
 actor=Controller();before={k:v.clone() for k,v in actor.state_dict().items()}
 for n in [3,13,23]:
  for step in range(3):
   state=torch.randn(n,8);legal=torch.arange(n+1)%3!=1;legal[0]=True
   with torch.no_grad():prob=actor(state,legal).softmax(0)
   assert torch.allclose(prob[legal],torch.full_like(prob[legal],1/int(legal.sum())),atol=1e-7,rtol=0) and bool((prob[~legal]==0).all())
 assert all(torch.equal(v,before[k]) for k,v in actor.state_dict().items())
 import copy
 x=torch.nn.Parameter(torch.tensor([.2,-.4]));opt=torch.optim.AdamW([x],lr=.001);(x.square().sum()).backward();opt.step();snapshot=x.detach().clone();optimizer=copy.deepcopy(opt.state_dict());outcomes=[]
 for _ in range(2):
  with torch.no_grad():x.copy_(snapshot)
  opt.load_state_dict(copy.deepcopy(optimizer));opt.zero_grad();(x.square().sum()).backward();opt.step();outcomes.append((x.detach().clone(),copy.deepcopy(opt.state_dict())))
 assert torch.equal(outcomes[0][0],outcomes[1][0]) and torch.equal(outcomes[0][1]['state'][0]['exp_avg'],outcomes[1][1]['state'][0]['exp_avg'])
 return dict(masked_uniform_unchanged_actor=True,toy_AdamW_branch_restore=True,passed=True,unique_allocation=True,toy_trajectory_exact_gradient=True,controller_finite_gradient=True,common_baseline_cancellation=True,derangement=True,new_model_scores=0,visual_optimizer_steps=0,toy_optimizer_steps=3)
if __name__=='__main__':print(json.dumps(check()))
