"""CPU-only end-to-end synthetic exercise; never performance evidence."""
import copy
import json
import time
import torch
from .memory import from_views
from .learner import Learner,optimizer,learn
from .contracts import VisibleState,DataAccessLedger
from .branch_env import PulseEnvironment
from .evaluate_controller import prepare,select
from .paired_labels import utility,near_optimal
from .verifier import Verifier
from .policy import ResidualPolicy
from .train_controller import train_verifier,train_sft,collect_trajectories,train_pg,train_more_sup,shuffle_rewards


def synthetic_environment(seed=109):
    torch.manual_seed(seed)
    n=8;C=4
    z=torch.nn.functional.normalize(torch.randn(C*n,6,512),dim=-1)
    bank=from_views(z,torch.arange(C*n),torch.arange(C*n)//n)
    support=bank.parents_only(torch.tensor([i for i in range(C*n) if i%n<6]))
    root=support.parents_only(torch.arange(2*n))
    future=[support.parents_only(torch.arange(c*n,(c+1)*n)) for c in [2,3]]
    queries=[bank.parents_only(torch.tensor([i for i in range(cc*n) if i%n>=6])) for cc in [2,3,4]]
    groups={i:str(i) for i in range(C*n)}
    ledger=DataAccessLedger({i:0 for i in groups},groups,{str(i):'S' if i%n<6 else 'FIT' for i in groups},0)
    model=Learner(seed);model.arrive(root,2);opt=optimizer(model);counter={}
    learn(model,opt,root,2,seed,counter,None,8,'engineering')
    model.trained_groups={str(int(i)) for i in root.parents}
    state=VisibleState(root,None,(6,6),(0,0),0,16,groups,True,ledger)
    env=PulseEnvironment(model,opt,state,future,queries,{0},ledger,counter,post_steps=2,pre_steps=2,batch=8)
    return env,model,opt,state,counter


def record_pairs(env,model,opt,state,counter,seed=109,split='fit'):
    candidates,classes,actions=prepare(state,model,opt,seed,counter)
    outcomes=env.compare(candidates,seed)
    targets=[];masks=[]
    for item in outcomes:
        h1=utility(item['endpoints'][:1]);full=item['summary']
        row=[];mask=[]
        for value,ends in [(h1,item['endpoints'][:1]),(full,item['endpoints'])]:
            tails=[e['tail'] for e in ends if e['tail'] is not None]
            row.extend([value['R'],value['macro'],sum(tails)/len(tails) if tails else 0.])
            mask.extend([True,True,bool(tails)])
        if len(item['endpoints'])<3: mask[3:]=[False]*3
        targets.append(row);masks.append(mask)
    dynamic=torch.stack([o['dynamic'] for o in outcomes])
    returns=[o['summary']['R'] for o in outcomes]
    record=dict(classes=classes,actions=actions,targets=torch.tensor(targets,device=classes.device),target_mask=torch.tensor(masks,device=classes.device),dynamic=dynamic,
                dynamic_mask=torch.ones_like(dynamic,dtype=torch.bool),nearopt=near_optimal(returns,positive_evidence=max(returns)>.005).to(classes.device),split=split,
                return_column=3 if len(env.queries)==3 else 0,
                root_groups=sorted({env.ledger.groups[int(i)] for i in state.pool.parents}),
                provenance='synthetic' if state.independent_groups_verified else 'identity-components; patient independence unverified')
    return record,candidates,outcomes


def run(output):
    from pathlib import Path
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    start=time.time();cpu=time.process_time();torch.set_num_threads(1)
    env,model,opt,state,counter=synthetic_environment()
    record,candidates,outcomes=record_pairs(env,model,opt,state,counter)
    verifier=Verifier();direct=Verifier(False);direct.load_state_dict(verifier.state_dict())
    dv=train_verifier(direct,[record],epochs=2);vv=train_verifier(verifier,[record],epochs=2)
    policy=ResidualPolicy();ss=train_sft(policy,verifier,[record],epochs=2)
    verifier.eval().requires_grad_(False)
    ref=copy.deepcopy(policy).eval().requires_grad_(False)
    builder=lambda s,m,o,seed:prepare(s,m,o,seed,counter)
    batch=collect_trajectories(env,policy,verifier,builder,candidates,109,G=2,payload=('synthetic',16,3))
    torch.save(batch,output/'SYNTHETIC_TRAJECTORIES.private.pt')
    torch.save(ref.state_dict(),output/'SYNTHETIC_SFT.private.pt')
    torch.save(verifier.state_dict(),output/'SYNTHETIC_VERIFIER.private.pt')
    assert all(len(t['steps'])==3 for t in batch['trajectories'])
    more=copy.deepcopy(policy);shuffled=copy.deepcopy(policy)
    pg=train_pg(policy,ref,verifier,batch,.02)
    sup=train_more_sup(more,batch)
    sh=train_pg(shuffled,ref,verifier,shuffle_rewards(batch,19),.02)
    for mode in ['VERIFIER_GREEDY','DIRECT_RETURN','Q_PRIOR','NEAROPT_SFT','ECV_RL']:
        v=direct if mode=='DIRECT_RETURN' else verifier
        v.eval().requires_grad_(False)
        p=copy.deepcopy(policy if mode=='ECV_RL' else ref).eval().requires_grad_(False)
        select(state,model,opt,109,counter,v,p,mode)
    result=dict(passed=True,synthetic_only=True,scientific_updates=0,VAL_TEST_rows=0,
                root_candidates=len(candidates),H3_real_decisions_per_trajectory=3,
                engineering_counts=counter,verifier_updates=vv['updates'],direct_return_updates=dv['updates'],sft_updates=ss['updates'],pg_updates=pg['updates'],more_sup_updates=sup['updates'],shuffle_updates=sh['updates'],
                verifier_parameters=sum(p.numel() for p in verifier.parameters()),policy_parameters=sum(p.numel() for p in policy.parameters()),
                CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start)
    assert result['verifier_parameters']<1_000_000
    (output/'CPU_SMOKE.json').write_text(json.dumps(result,indent=2))
    torch.save(record,output/'SYNTHETIC_PAIRS.private.pt')
    torch.save(dict(verifier=verifier.state_dict(),policy=policy.state_dict(),source='synthetic engineering only'),output/'SYNTHETIC_MODELS.private.pt')
    return result

