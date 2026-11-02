import os,time,json,copy,importlib.util,runpy
from pathlib import Path
import torch
import campaign as c
start=time.time();cpu=time.process_time();spec=importlib.util.spec_from_file_location('reference_impl',c.root/'engineering_preopt'/'campaign.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
assert not list(c.root.glob('builder_*.private.pt'))
maxstate=0.;maxloss=0.;maxgrad=0.;membership_contexts=0;state_pairs=0;gradient_pair_backwards=0;pool_time={'reference':0.,'cached_first':0.,'cached_repeat':0.}
for s in range(6):
 c.guard(c.cfg);d=c.load_stage(c.rcfg,s);t=time.time();pools={j:sorted(i for i in d['roles']['S'] if int(d['y'][d['index'][i]])==j) for j in range(d['C'])};pool_time['reference']+=time.time()-t;t=time.time();cached=c.support_pools(d);pool_time['cached_first']+=time.time()-t;t=time.time();assert c.support_pools(d) is cached;pool_time['cached_repeat']+=time.time()-t;assert pools==cached
 for idx in list(range(1024))+[50000+slot+5*rep for slot in range(5) for rep in range(4)]:
  cap=[1,2,4,8,None][idx%5]
  for j in range(d['C']):
   for pool in [pools[j]]:
    n=len(pool);m=n if cap is None else min(cap,n);a=(74002+100003*s+idx//5)%n;b=(a+max(1,m))%n if m<n else a
    original_a=[pool[(a+k)%n] for k in range(m)];original_b=[pool[(b+k)%n] for k in range(m)];newpool=cached[j];assert original_a==[newpool[(a+k)%n] for k in range(m)] and original_b==[newpool[(b+k)%n] for k in range(m)]
  membership_contexts+=1
 if s in [0,5]:
  for classes in [list(range(d['C'])),[j for j in range(d['C']) if j%3!=1]]:
   for idx in list(range(5))+list(range(50000,50005)):
    aa=ref.pair(d,s,idx,classes);bb=c.pair(d,s,idx,classes)
    for a,b in zip(aa[:4],bb[:4]):maxstate=max(maxstate,float((a-b).abs().max()));assert torch.equal(a,b)
    assert aa[4]==bb[4];state_pairs+=1
 if s==0:
  classes=list(range(d['C']));z,y,_=c.query(d,'W',classes)
  for lam in [0.,1.]:
   for idx in [2,4]:
    c.guard(c.cfg);g1,_=c.init();torch.manual_seed(9303)
    with torch.no_grad():g1.net[-1].weight.normal_(0,.001);g1.net[-1].bias.normal_(0,.001)
    g2=copy.deepcopy(g1);l1,_,_=ref.objective(g1,d,s,idx,classes,z,y,lam);l2,_,_=c.objective(g2,d,s,idx,classes,z,y,lam);maxloss=max(maxloss,abs(float(l1.detach()-l2.detach())));l1.backward();l2.backward();gradient_pair_backwards+=2
    for a,b in zip(g1.parameters(),g2.parameters()):assert a.grad is not None and b.grad is not None;maxgrad=max(maxgrad,float((a.grad-b.grad).abs().max()))
assert maxstate==0 and maxloss<1e-12 and maxgrad<1e-12
c.write(c.root/'OPTIMIZATION_EQUIVALENCE.json',{'passed':True,'all_six_stage_pool_ID_lists_equal':True,'membership_contexts':membership_contexts,'state_pair_comparisons':state_pairs,'max_state_error':maxstate,'max_loss_error':maxloss,'max_parameter_gradient_error':maxgrad,'lambda_controls':[0,1],'gradient_actual_TRAIN_pair_backwards':gradient_pair_backwards,'formal_optimizer':0,'encoder':0,'VAL_TEST':0,'pool_build_wall_seconds':pool_time,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'scope':'Same ordered IDs, states, coverage and losses/gradients; only immutable metadata caching changed. No parameter fitting.'})
runpy.run_path(str(c.root/'profile_optimized.py'),run_name='__main__')
