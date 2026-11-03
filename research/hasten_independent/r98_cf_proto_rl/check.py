"""Only new mechanism tests; fixed MC seed/sample size, no rescue resampling."""
import copy,io,time,math,torch
from prototype_policy import Policy,build_state,build_prototypes,gaussian_logp,SIGMA
from utility import utility,class_values,counterfactual_credit
from common import write

def run(path):
 start=time.time();cpu=time.process_time();torch.manual_seed(981);support=[torch.nn.functional.normalize(torch.randn(n,512),dim=-1) for n in [1,4,6,3]];ep=build_state(support,[3,8,12,6],[0,0,1,1],1);a=torch.zeros(8,4,4,requires_grad=True);p,w=build_prototypes(a,ep,True);assert torch.allclose(p,ep['p0'][None].expand_as(p),atol=2e-6);p.sum().backward();assert torch.isfinite(a.grad).all() and a.grad[:,1:].norm()>0 and a.grad[:,0].norm()==0
 for v,z in zip(w,support):assert (v>=.75/len(z)-1e-7).all() and (v<=.25+.75/len(z)+1e-7).all() and torch.allclose(v.sum(1),torch.ones(8),atol=2e-6)
 epr=build_state([v.flip(0) for v in support],[3,8,12,6],[0,0,1,1],1);assert torch.allclose(epr['state'],ep['state'],atol=2e-5)
 perm=[2,0,3,1];epp=build_state([support[i] for i in perm],[[3,8,12,6][i] for i in perm],[[0,0,1,1][i] for i in perm],1);assert torch.allclose(epp['state'],ep['state'][perm],atol=2e-5)
 y=torch.tensor([0,1,2,3,0,1,2,3]);scores=torch.randn(8,8,4);rs=torch.randn(8,4);protected=torch.tensor([True,False,True,False]);rv=class_values(rs[None],y)[0];U,D=counterfactual_credit(scores,rs,y,rv,protected,ep['active']);err=0.
 for c in [1,2,3]:
  z=scores.clone();z[:,:,c]=rs[:,c];want=U-utility(z,y,rv,protected);err=max(err,float((want-D[:,c]).abs().max()))
 assert err<1e-12
 # The same full utility sees collateral damage for both estimators.
 yy=torch.tensor([0,1]);ref=torch.tensor([[[2.,1.],[1.,2.]]]);bad=torch.tensor([[[3.,1.],[4.,2.]]]);rval=class_values(ref,yy)[0];vals=class_values(bad,yy)[0];assert vals[0]>rval[0] and vals[1]<rval[1] and utility(bad,yy,rval,torch.ones(2,dtype=torch.bool))<utility(ref,yy,rval,torch.ones(2,dtype=torch.bool))
 mu=torch.randn(3,2,dtype=torch.double)*.1;target=torch.tensor([[.3,-.2],[.4,.1],[-.1,.2]],dtype=torch.double);N=65536;g=torch.Generator().manual_seed(982);eps=torch.randn(N,3,2,dtype=torch.double,generator=g);action=mu[None]+SIGMA*eps;rr=torch.zeros_like(mu)
 def fun(x):return -(x-target).square().sum((-1,-2))+.3*x[...,0,0]*x[...,1,0]
 R=fun(action);analytic=-2*(mu-target);analytic[0,0]+=.3*mu[1,0];analytic[1,0]+=.3*mu[0,0];cf=torch.empty_like(action)
 for c in range(3):
  other=action.clone();other[:,c]=rr[c];cf[:,c]=(R-fun(other))[:,None]*eps[:,c]/SIGMA
 glob=(R-fun(rr))[...,None,None]*eps/SIGMA;pathg=-2*(action-target);pathg[:,0,0]+=.3*action[:,1,0];pathg[:,1,0]+=.3*action[:,0,0];mc={}
 for name,v in [('CF',cf),('GLOBAL',glob),('PATH',pathg)]:
  se=v.std(0,correction=1)/math.sqrt(N);error=(v.mean(0)-analytic).abs();assert (error<=5*se+1e-5).all(),(name,error,se);mc[name]={'maxabs_mean_error':float(error.max()),'max_standard_error':float(se.max()),'fixed_MC':N}
 m=Policy();ep['query']=torch.nn.functional.normalize(torch.randn(8,512),dim=-1);ep['y']=y;ep['nB']=torch.ones(4)*2;ep['protected']=protected;refm=copy.deepcopy(m).requires_grad_(False)
 from losses import loss_for_arm
 val,_,aux,_=loss_for_arm('CF',m,refm,ep,torch.randn(8,4,4));aux.backward();assert m.net[-1].weight.grad.norm()>0 and not ep['state'].requires_grad and all(not x.requires_grad for x in ep['support']);m.zero_grad();v,_,aux,_=loss_for_arm('PATH',m,refm,ep,torch.randn(8,4,4));aux.backward();assert m.net[-1].weight.grad.norm()>0
 before={k:v.clone() for k,v in m.state_dict().items()};opt=torch.optim.AdamW(m.parameters(),lr=0,weight_decay=0);opt.step();assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items());b=io.BytesIO();torch.save(m.state_dict(),b);b.seek(0);m2=Policy();m2.load_state_dict(torch.load(b,weights_only=False),strict=True);assert torch.equal(m(ep['state']),m2(ep['state']))
 write(path,{'passed':True,'uniform_NOOP_maxabs':float((p-ep['p0'][None]).abs().max()),'CF_column_vs_brute_maxabs':err,'MC_gradient_test':mc,'row_and_class_permutation':True,'single_support_inactive':True,'collateral_harm_counted':True,'score_function_and_PATH_gradients':True,'LR_zero_strict_restore':True,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'engineering_optimizer_steps':1,'image_encoder_rows':0,'VAL_TEST':0})
