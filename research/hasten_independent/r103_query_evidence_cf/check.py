"""Small, bounded correctness tests; never scientific performance gates."""
import copy,json
import torch
from pathlib import Path
from grpo_reference import self_test,exact_group_surrogate,_brute_force,old_tail_penalty,sampled_surrogate
from router import Router,features
from fusion import Fusion,loss_part
from objectives import objective
from common import write

def run(root):
    torch.set_num_threads(1);out=self_test();torch.manual_seed(74002)
    logits=torch.tensor([[.05,.9,.05],[.9,.05,.05]],dtype=torch.double).log().requires_grad_();old=torch.tensor([[.8,.1,.1],[.2,.6,.2]],dtype=torch.double);rr=torch.tensor([[0.,1.,-1.],[0.,1.,0.]],dtype=torch.double)
    x=exact_group_surrogate(logits,old,rr);y=_brute_force(logits,old,rr);gx,=torch.autograd.grad(x.sum(),logits,retain_graph=True);gy,=torch.autograd.grad(y.sum(),logits);assert (x-y).abs().max()<1e-11 and (gx-gy).abs().max()<1e-11
    fm=Fusion();z=torch.nn.functional.normalize(torch.randn(6,512),dim=-1);v=z.roll(1,0);P=torch.nn.functional.normalize(torch.randn(3,512),dim=-1).requires_grad_();score=fm(z,v,P);assert (score-20*z@P.T).abs().max()<1e-4
    opt=torch.optim.AdamW(fm.parameters(),lr=.0001);opt.zero_grad();score.sum().backward();assert fm.U.weight.grad.norm()>0 and fm.V.weight.grad.norm()==0;opt.step();opt.zero_grad();fm(z,v,P.detach()).sum().backward();assert fm.V.weight.grad.norm()>0
    # Feature construction uses predicted class statistics, never query labels.
    ss=torch.stack([20*z@P.detach().T]*3,1);zz=torch.stack([z,v,v],1);phi=features(ss,zz,torch.tensor([2,3,1]),torch.tensor([0,0,0]),0);rm=Router();assert sum(p.numel() for p in rm.parameters())==672
    assert (rm(phi).double().softmax(-1)-torch.tensor([.8,.1,.1])).abs().max()<1e-7 and (rm(phi).argmax(-1)==0).all()
    oo=torch.optim.AdamW(rm.parameters(),lr=.001);cc=torch.tensor([[1,0,1],[0,1,0],[1,1,0],[0,1,1],[1,0,0],[0,0,0]])
    old=rm(phi).double().softmax(-1).detach().clone();actions=torch.tensor([[0,1,2,0,1,2,0,1]]*6);saved=old.clone();deviation=[]
    for k in range(4):
        oo.zero_grad();cur=rm(phi);deviation.append(float((cur.double().softmax(-1)/old-1).abs().max().detach()));objective('GRPO_CF',cur,old,cc,torch.ones(6,dtype=torch.bool),3,3,actions).backward();oo.step()
    assert torch.equal(old,saved) and deviation[0]<1e-6 and max(deviation[1:])>1e-5
    # Uniform class then uniform row estimate equals full class-macro harm in expectation.
    lg=torch.randn(6,3,dtype=torch.double,requires_grad=True);prot=torch.tensor([1,1,1,0,0,0],dtype=torch.bool);classes=torch.tensor([0,0,1,2,2,2]);prob=torch.tensor([1/6,1/6,1/3,1/9,1/9,1/9],dtype=torch.double)
    h=cc[:,0].double()*(lg.softmax(-1)*(1-cc.double())).sum(-1);full=torch.stack([h[classes==cl].mean() for cl in [0,1]]).mean();expected=(prob*prot*h).sum()*3/2;assert (full-expected).abs()<1e-12
    sample_index=torch.multinomial(prob,24000,replacement=True,generator=torch.Generator().manual_seed(74002));mc=(prot[sample_index]*h[sample_index]).mean()*3/2;assert abs(float((mc-full).detach()))<.02
    out.update(software_engineering_optimizer_steps=5,software_encoder_VAL_TEST=0,class_balanced_harm_small_MC_passed=True,nontrivial_clip_exact_value_gradient=True,zero_U_first_V_gradient_correct=True,second_step_V_gradient_nonzero=True,old_fixed_four_updates=True,ratio_deviation_after_first=max(deviation[1:]),class_balanced_harm_expectation_exact=True,feature_dim=19,router_params=672,formal_optimizer_encoder_VAL_TEST=0)
    write(Path(root)/'SOFTWARE_CHECK.json',out);return out

if __name__=='__main__':run(Path(__file__).resolve().parent)
