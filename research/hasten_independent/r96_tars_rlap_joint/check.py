"""New joint-objective integration checks; no encoder/optimizer/VAL."""
import itertools,time,torch
from joint import loss
from decision import exact_group_surrogate,tailness
from common import write

def run(path):
 start=time.time();cpu=time.process_time();x=torch.tensor([[.3,-.4]],dtype=torch.double,requires_grad=True);old=torch.tensor([[.2,-.1]],dtype=torch.double);ref=torch.tensor([[-.2,.4]],dtype=torch.double);y=torch.tensor([0]);w=torch.tensor([2.,1.],dtype=torch.double);p=old.softmax(1)[0];expected=x.sum()*0
 for seq in itertools.product(range(2),repeat=8):
  a=torch.tensor([seq]);r=(a==0).double();adv=(r-r.mean(1,keepdim=True))/(r.std(1,keepdim=True,correction=0)+1e-8);value,_=loss(x,y,old,a,adv,ref,w);prob=p[a[0]].prod();expected=expected+prob*value
 lp=x.log_softmax(1);kl=(lp.exp()*(lp-ref.log_softmax(1))).sum();exact=-2*exact_group_surrogate(lp,old.log_softmax(1),y).mean()+.04*kl;err=float((expected-exact).abs().detach());g1=torch.autograd.grad(expected,x,retain_graph=True)[0];g2=torch.autograd.grad(exact,x)[0];ge=float((g1-g2).abs().max());assert err<1e-12 and ge<1e-12,(err,ge)
 # Exact zero advantage stays zero; old behavior remains fixed after current logits move.
 for a in [torch.zeros(1,8,dtype=torch.long),torch.ones(1,8,dtype=torch.long)]:
  zero=torch.zeros(1,8);v,st=loss(x,y,old,a,zero,ref,w);assert st['zero_advantage_groups']==1 and torch.isfinite(v)
 assert torch.equal(old,torch.tensor([[.2,-.1]],dtype=torch.double))
 assert torch.equal(tailness(torch.tensor([2.,2.]),torch.tensor([0.,0.])),torch.ones(2,dtype=torch.double))
 out={'passed':True,'sampled_G8_expectation_vs_exact_value_error':err,'gradient_error':ge,'fixed_behavior':True,'zero_groups_retained':True,'tailness_midpoint_median':True,'encoder':0,'optimizer':0,'VAL_TEST':0,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start};write(path,out)
if __name__=='__main__':
 import sys
 run(sys.argv[1])
