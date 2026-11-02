import campaign as c
import torch,time,json
from pathlib import Path
c.guard(c.cfg);torch.manual_seed(93001);C=23;counts=c.rcfg['train_counts'];labels=torch.cat([torch.full((n,),i,dtype=torch.long) for i,n in enumerate(counts)]);N=len(labels);ids=list(range(N));d={'C':C,'z':c.unit(torch.randn(N,512,dtype=torch.float64)),'y':labels,'roles':{'S':ids},'index':{i:i for i in ids},'birth':[j for j,cc in enumerate(c.rcfg['stages']) for _ in cc]};z=c.unit(torch.randn(1713,512,dtype=torch.float64));y=torch.arange(1713)%C;g,opt=c.init();times=[];cp=time.process_time();wt=time.time()
for slot in range(5):
 tt=[]
 for repeat in range(2):
  c.guard(c.cfg);start=time.time();g.zero_grad();loss,_,_=c.objective(g,d,5,slot,list(range(C)),z,y,1.);loss.backward();tt.append(time.time()-start);assert torch.isfinite(loss)
 times.append({'slot':slot,'pass_seconds':tt})
start=time.time();opt.step();optim_time=time.time()-start
qz=c.unit(torch.randn(1703,512,dtype=torch.float64));qy=torch.arange(1703)%C;start=time.time();torch.set_grad_enabled(False)
for _ in range(4):
 gg,_=c.init();a,pa,b,pb,_=c.pair(d,5,4,list(range(C)));ca,_=gg(a,pa);cb,_=gg(b,pb);sa=c.score(qz,ca);sb=c.score(qz,cb);_=c.consistency(sa[:,:,2]/20,sb[:,:,2]/20,qy)
 for k in range(3):
  for sc in [sa,sb]:_=c.recall(sc[:,:,k],qy,C)
eval_time=time.time()-start;pair_time=max(t['pass_seconds'][1] for t in times);projected=pair_time*6144*4*4+optim_time*6144*4+eval_time*120*4+3600;remaining=c.cfg['deadline']-time.time();passed=projected<remaining-180
c.write(c.root/'BUDGET_DIRECT.json',{'passed':passed,'method':'direct worst-size same-function synthetic profile, five caps maximum fixed second pass','cap_times':times,'worst_pair_seconds':pair_time,'synthetic_optimizer_step_seconds':optim_time,'synthetic_eval_context_seconds':eval_time,'safety_factor':4,'reserve_seconds':3600,'projected_seconds':projected,'remaining_seconds':remaining,'engineering_synthetic_pair_backward':10,'engineering_synthetic_optimizer_steps':1,'engineering_synthetic_eval_contexts':1,'formal_optimizer':0,'encoder':0,'VAL_TEST':0,'CPU_seconds':time.process_time()-cp,'wall_seconds':time.time()-wt,'synthetic_support_rows':N,'synthetic_B_rows':1713,'synthetic_F_shaped_rows':1703})
c.write(c.root/'PROFILE_STATUS.json',{'phase':'DIRECT_PROFILE_COMPLETE','passed':passed,'formal_optimizer':0,'encoder':0,'VAL_TEST':0})
