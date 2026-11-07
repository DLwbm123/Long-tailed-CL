"""Worst-shape engineering admission of the complete P1, never a science gate."""
import time,torch
from .base.memory import from_views
from .base.learner import Learner,optimizer,learn
from .base.contracts import VisibleState,DataAccessLedger
from .base.state_features import describe
from .base.branch_env import PulseEnvironment
from .candidates import generate
from .common import write,guard

def run(cfg,encoder,rows,first_ids,counter):
    start=time.time();torch.cuda.synchronize()
    generic=None;encode_seconds=[]
    for _ in range(2):
        t=time.time();generic=encoder.encode(cfg,rows,first_ids[:30],counter,'native',first_ids);torch.cuda.synchronize();encode_seconds.append(time.time()-t)
    def bank(n,offset,labels):
        z=generic[torch.arange(n,device='cuda')%len(generic)]
        return from_views(z,torch.arange(offset,offset+n,device='cuda'),labels.cuda())
    n=3331;labels=torch.cat([torch.arange(20),torch.full((n-20,),20)])
    pool=bank(n,0,labels);future=[bank(64,10000+h*100,torch.full((64,),21+h)) for h in range(2)]
    full_query=bank(2048,20000,torch.arange(2048)%23)
    queries=[full_query.take(torch.where(full_query.y<C)[0]) for C in [21,22,23]]
    allids=list(range(n))+[int(i) for d in future for i in d.parents]+list(range(20000,22048))
    groups={i:str(i) for i in allids};roles={str(i):'AUDIT' if i>=20000 else 'S' for i in allids}
    ledger=DataAccessLedger({i:0 for i in allids},groups,roles,0)
    m=Learner(110).cuda();m.arrive(pool,21);o=optimizer(m);m.trained_groups={str(i) for i in range(n)}
    learn(m,o,pool,200,110,counter,None,64,'native_prefix')
    state=VisibleState(pool,None,tuple([1]*20+[n-20]),tuple([0]*21),0,256,groups,False,ledger)
    t=time.time();desc=describe(state,m,o,counter,seed=110);candidates,audit=generate(state,m,desc,110);torch.cuda.synchronize();proposal_seconds=time.time()-t
    # Repeated legal branches force the worst count only in this engineering fixture.
    fixture=[candidates[i%len(candidates)] for i in range(16)]
    env=PulseEnvironment(m,o,state,future,queries,set(range(7)),ledger,counter,post_steps=100,pre_steps=200,batch=64)
    t=time.time()
    for candidate in fixture:
        env.rollout(candidate.memory,31001)
    torch.cuda.synchronize();branch_seconds=time.time()-t
    # Both flows independently admit their complete 16-root/2-tape plan.
    update_rate=branch_seconds/(16*700)
    projection=2*(32*branch_seconds+16*proposal_seconds+23900*update_rate+6821*encode_seconds[-1]/30)+600+7200
    passed=projection<cfg['deadline']-time.time()
    result=dict(passed=passed,full_P1_projected_seconds=projection,native_wall_seconds=time.time()-start,
                encode_30_parent_seconds=encode_seconds,proposal_seconds=proposal_seconds,full16_H3_branch_seconds=branch_seconds,
                engineering_fixture_classes=[21,22,23],root_parent_pool=3331,query_parent_pool=2048,
                distinct_generated_candidates=len(candidates),executed_fixture_branches=16,
                repeated_generic_features_and_labels_are_engineering_only=True,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                science_optimizer=0,counts=dict(counter))
    guard(cfg);write(cfg['output']+'/NATIVE_ADMISSION.json',result)
    return result

