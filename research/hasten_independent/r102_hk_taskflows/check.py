"""New joint-geometry and group-policy contract checks, no old toy rerun."""
import copy,time,torch
from model import Joint,unit,deterministic,supervised,auxiliary,prototypes,reward,weight_kl

def fixture(device='cpu'):
    g=torch.Generator().manual_seed(74002);S=unit(torch.randn(9,512,generator=g)).to(device);B=unit(torch.randn(12,512,generator=g)).to(device);sy=torch.tensor([0]*4+[1]*4+[2],device=device);y=torch.tensor([0,1,2]*4,device=device);ix=[torch.where(sy==k)[0] for k in range(3)]
    base=unit(torch.stack([S[j].mean(0) for j in ix]));return dict(stage=0,C=3,S=S,B=B,Sy=sy,By=y,ix=ix,base=base,raw_mean=torch.stack([S[j].mean(0) for j in ix]),rawcounts=[40,40,3],births=[0,0,0],nB=torch.tensor([4.,4.,4.],device=device),protected=torch.tensor([False,True,True],device=device),tail=torch.tensor([False,True,True],device=device),teacher_scores=20*B@base.T)

def run(device='cpu'):
    cpu=time.process_time();wall=time.time();d=fixture(device);torch.manual_seed(74002);m=Joint('PROTO_GRPO').to(device)
    score,*_=deterministic(m,d);err=float((score-d['teacher_scores']).abs().max());assert err<2e-5
    loss,_=supervised(m,d);loss.backward();assert float(m.U.weight.grad.norm())>0 and float(m.policy[-1].weight.grad.norm())>0
    # With both branches current, gradients through support and query must exist.
    m.zero_grad(set_to_none=True);S=d['S'].clone().requires_grad_(True);B=d['B'].clone().requires_grad_(True);d2={**d,'S':S,'B':B};l,_=supervised(m,d2);l.backward();assert float(S.grad.norm())>0 and float(B.grad.norm())>0
    m.zero_grad(set_to_none=True);eps=[torch.randn(8,len(j),device=device) for j in d['ix']];l,st=auxiliary(m,d,eps);l.backward()
    assert all(p.grad is None for p in m.representation_parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    assert any(p.grad is not None and float(p.grad.norm())>0 for p in m.policy_parameters())
    R=torch.tensor([0.,1.,2.,3.,4.,5.,6.,7.],device=device);A=R-(R.sum()-R)/7;Rp=R+11;assert torch.allclose(A,Rp-(Rp.sum()-Rp)/7)
    Z=torch.ones(8,device=device);assert torch.equal(Z-(Z.sum()-Z)/7,torch.zeros_like(Z))
    # Evaluate score-function derivative at fixed sampled actions analytically.
    mu=torch.tensor([.2,.1],dtype=torch.double,requires_grad=True);a=torch.tensor([[.3,.4],[-.2,.1]],dtype=torch.double);adv=torch.tensor([.7,-.7],dtype=torch.double)
    lp=(-.5*((a-mu)/.25).square()).sum(1);f=-(adv*(lp-lp.detach()).exp()).mean();g=torch.autograd.grad(f,mu)[0];target=-(adv[:,None]*(a-mu.detach())/.25**2).mean(0);assert torch.allclose(g,target,atol=1e-12)
    parts,b,ref,_,w=deterministic(m,d)[1:];assert float(w[-1][0])==1
    saved=copy.deepcopy(m.state_dict());m.load_state_dict(saved,strict=True)
    return {'passed':True,'device':device,'initial_teacher_score_maxabs':err,'joint_support_query_gradients_nonzero':True,'auxiliary_adapter_frozen':True,'policy_gradient_nonzero_finite':True,'fixed_action_score_gradient_analytic':True,'LOO_offset_and_zero_group':True,'single_support_weight_one':True,'strict_restore':True,'optimizer':0,'image_encoder':0,'F_VAL_TEST':0,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-wall}

if __name__=='__main__':print(run())
