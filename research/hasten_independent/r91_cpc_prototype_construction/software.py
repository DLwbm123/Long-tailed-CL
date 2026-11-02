"""New CPC objectives checked against finite exact enumeration; no real data."""
import itertools,time,torch
from core import *
def grad(v,x):return torch.autograd.grad(v,x,retain_graph=True)[0]
def run():
 start=time.time();cpu=time.process_time();torch.manual_seed(91001);torch.set_num_threads(4)
 y=torch.tensor([0,0,1,1,2,2]);scores=torch.randn(6,3,3,dtype=DT);scores[0]=0 # exact stable-index tie fixture
 tail=torch.tensor([False,True,True]);protected=torch.ones(3,dtype=torch.bool);ref=torch.tensor([.7,.5,.8],dtype=DT);actions=torch.tensor(list(itertools.product(range(3),repeat=3)));v,rec,ce=utility(scores,y,actions,tail,protected,ref)
 theta=torch.randn(3,3,dtype=DT,requires_grad=True);p=theta.softmax(1);pr=p[torch.arange(3)[None],actions].prod(1);b_exact,rr=exact_B(p,wins(scores,y),y,tail);b_enum=(pr*v[:,0]).sum();be=float((b_exact-b_enum).abs());bg=float((grad(b_exact,theta)-grad(b_enum,theta)).abs().max());assert be<1e-10 and bg<1e-10
 old=p.detach();q=conditional(scores,y,actions,tail,protected,ref);adv=(q-(q*old[None]).sum(-1,keepdim=True)).detach();counter=(pr.detach()*(old[None]*adv*(p/old)[None]).sum((1,2))).sum();risk_enum=-(pr*v[:,1]).sum();rg=float((grad(counter,theta)-grad(risk_enum,theta)).abs().max());assert rg<1e-10
 qc=conditional(scores,y,actions,tail,protected,ref,kind='CE');cg=(pr.detach()*(p[None]*qc).sum((1,2))).sum();ce_enum=(pr*ce).sum();ceg=float((grad(cg,theta)-grad(ce_enum,theta)).abs().max());assert ceg<1e-10
 current=torch.randn(3,3,dtype=DT,requires_grad=True);logp=current.log_softmax(1);oldlog=old.log();ratio=(logp-oldlog).exp();terms=torch.minimum(ratio[None]*adv,ratio[None].clamp(.8,1.2)*adv);integrated=(old[None]*terms).sum((1,2));enumerated=torch.zeros(len(actions),dtype=DT)
 for local in actions:
  weight=old[torch.arange(3),local].prod();enumerated+=weight*terms[:,torch.arange(3),local].sum(1)
 cliperr=float((integrated-enumerated).abs().max());assert cliperr<1e-10 and (grad(integrated.sum(),current)-grad(enumerated.sum(),current)).abs().max()<1e-10
 zeroq=torch.ones(8,3,3,dtype=DT);assert risk_loss(logp,oldlog,old,zeroq).item()==0
 h1=grad(risk_loss(logp,oldlog,old,q),current);h2=grad(risk_loss(logp,oldlog,old,q,shuffle=True),current);assert (h1-h2).abs().max()>1e-7
 # Singleton correction is a real construction action; zero output restored exactly.
 g=Builder();z=[unit(torch.randn(1,512,dtype=DT)) for _ in range(3)];st,p0=state(z,[0,1,2],0,[3,6,6],[0,0,0]);assert st.shape==(3,1541) and st[:,-3].eq(0).all()
 with torch.no_grad():g.net[-1].bias.copy_(torch.randn(512,dtype=DT)*.01)
 cand,r=g(st,p0);assert torch.equal(cand[:,0],p0) and r.norm(dim=1).min()>0 and r.norm(dim=1).max()<=.1+1e-12
 perm=torch.tensor([2,0,1]);st2,p02=state([z[i] for i in perm],[2,0,1],0,[3,6,6],[0,0,0]);cand2,_=g(st2,p02);assert torch.allclose(cand2,cand[perm],atol=1e-12,rtol=0)
 checkpoint={k:v.clone() for k,v in g.state_dict().items()};other=Builder();other.load_state_dict(checkpoint,strict=True);assert all(torch.equal(v,other.state_dict()[k]) for k,v in checkpoint.items())
 actor=Actor();ast=torch.cat([st,r.detach(),r.detach().norm(dim=1,keepdim=True)],1);lp=actor(ast);oldlp=lp.detach().clone();aa=sampled(oldlp.exp(),torch.Generator().manual_seed(91002));uu=utility(scores,y,aa,tail,protected,ref)[0][:,2];loss=group_loss(lp,oldlp,aa,uu)+.04*kl(lp);loss.backward();assert all(t.grad is None or torch.isfinite(t.grad).all() for t in actor.parameters())
 return {'passed':True,'exact_B_error':be,'exact_B_gradient_error':bg,'conditional_risk_gradient_error':rg,'conditional_CE_gradient_error':ceg,'clipped_local_integral_error':cliperr,'H_changes_gradient':True,'singleton_freedom':True,'NOOP_exact':True,'permutation_equivariance':True,'strict_restore':True,'new_optimizer_steps':0,'real_encoder_classifier_forward':0,'VAL_TEST':0,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start}
if __name__=='__main__':print(__import__('json').dumps(run()))
