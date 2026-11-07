"""Worst-shape native engineering cost; never a classifier-effect gate."""
import importlib,time
from pathlib import Path
import numpy as np
import torch
from . import evidence as E
from . import roots
from .supply import birth_roles,write


def fixture(generic,n,offset,labels,stage=0):
    device=generic.device;ids=torch.arange(offset,offset+n,device=device)
    bank=E.mem.from_views(generic[torch.arange(n,device=device)%len(generic)],ids,labels.to(device))
    flags=torch.zeros(len(bank),device=device,dtype=torch.uint8)
    stats=np.zeros(23,dtype=E.STATS);stats['birth']=-1
    for c in sorted(set(labels.tolist())):
        ix=bank.gidx[bank.y[bank.gidx]==c];flags[ix[0]]=4;stats[c]=(len(ix),stage,len(ix)+2)
    C=sorted(set(labels.tolist()));qids=torch.arange(20000+offset,20000+offset+2*len(C),device=device)
    qlabels=torch.tensor(C+C,device=device);q=E.Bank(generic[torch.arange(len(qids),device=device)%len(generic),0],qids,qlabels,torch.zeros(len(qids),device=device,dtype=torch.long))
    qflags=torch.tensor([1]*len(C)+[2]*len(C),device=device,dtype=torch.uint8)
    keys=torch.cat([bank.pid*8+bank.view,q.pid*8+q.view]);flags=torch.cat([flags,qflags])[keys.argsort()]
    return E.Evidence(E.mem.join(bank,q),flags,stats)


def run(cfg,encoder,rows,first,counter):
    common=importlib.import_module('r110_ecv_opportunity.common');start=time.time();counter.phase='native'
    roles=birth_roles(first,rows);ids=[i for i in first if roles[i]=='S'][:30];assert len(ids)==30
    times=[];generic=None
    for _ in range(2):
        t=time.time();generic=encoder.encode(cfg,rows,ids,counter,'native',first);torch.cuda.synchronize();times.append(time.time()-t)
    labels=torch.cat([torch.arange(20),torch.full((3331-20,),20)])
    pool=fixture(generic,3331,0,labels);future=[fixture(generic,64,10000+1000*h,torch.full((64,),21+h),1+h) for h in range(2)]
    m=roots.learner.Learner(112).cuda();m.forbidden={int(i) for x in [pool]+future for role in [1,2] for i in x.role(role).parents}
    m.arrive(pool.role(0),21);o=roots.learner.optimizer(m);roots.learner.learn(m,o,pool.role(0),200,112,counter,None,64,'native_prefix')
    t=time.time();candidates,ledger=roots.generate(pool,None,m,o,0,112,counter);torch.cuda.synchronize();proposal=time.time()-t
    # Repetition forces the full twelve-branch engineering bound only.
    t=time.time()
    for j in range(12):roots.rollout(pool,future,m,o,None,candidates[j%len(candidates)],31001,counter)
    torch.cuda.synchronize();branch=time.time()-t;update=branch/(12*700)
    # Per worker: two seeds, 32 roots, two tapes, twelve branches, maximum
    # 1400 prefix updates per root, 3000 donor updates, 32 virtual updates.
    projection=2*(64*branch+32*proposal+47832*update+6821*times[-1]/30)+600+7200
    passed=projection<cfg['deadline']-time.time()
    result={'passed':passed,'complete_scientific_projected_seconds':projection,'native_wall_seconds':time.time()-start,'encode_30_S_seconds':times,
            'proposal_seconds':proposal,'twelve_H3_branch_seconds':branch,'engineering_only':True,'repeated_generic_features_are_not_scientific_data':True,
            'maximum_science_updates_both_workers':1170864,'full_seeds_per_stream':2,'full_roots_per_stream':32,'tapes':2,'H':3,'maximum_unique_candidates':12,
            'root_support_parents':3331,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'counts':dict(counter)}
    common.guard(cfg);write(Path(cfg['output'])/'NATIVE_ADMISSION.json',result)
    del generic,pool,future,m,o,candidates;torch.cuda.empty_cache();return result
