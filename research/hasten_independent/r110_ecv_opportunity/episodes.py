"""Legal donor snapshots -> fresh, query-disjoint pseudo-incremental roots."""
import random
import torch
from .base.memory import Bank,join
from .base.learner import Learner,optimizer,learn,frozen
from .base.contracts import VisibleState,DataAccessLedger,token_keys
from .base.anchor_candidates import anchors
from .base.branch_env import PulseEnvironment
from .base.paired_labels import Counts,paired,utility


class Roles:
    def __init__(self,seed):self.seed=seed;self.role={}
    def arrive(self,ids,rows):
        classes={}
        for i in ids:classes.setdefault(int(rows[i]['original_label']),set()).add(rows[i]['identity_component'])
        for label,groups in classes.items():
            units=sorted(groups);random.Random(self.seed+label).shuffle(units)
            q=max(1,int(.2*len(units))) if len(units)>=3 else 0
            for j,u in enumerate(units):
                role='FIT' if j<q else 'AUDIT' if j<2*q else 'S'
                assert u not in self.role;self.role[u]=role
    def split(self,bank,rows):
        return {role:bank.parents_only(torch.tensor([int(i) for i in bank.parents if self.role[rows[int(i)]['identity_component']]==role],device=bank.z.device,dtype=torch.long)) for role in ['S','FIT','AUDIT']}


def relabel(bank,mapping):
    return Bank(bank.z,bank.pid,torch.tensor(mapping,device=bank.z.device)[bank.y],bank.view)


def root(previous,current,roles,rows,cfg,stage,index,counter):
    assert set(current.y.tolist())==set(cfg['stages'][stage]),'CURRENT_STAGE_BINDING'
    assert previous is None or len(previous)<=256,'OVERSIZED_DONOR_HISTORY'
    actual_pool=join(previous,current);split=roles.split(actual_pool,rows)
    supported=sorted(set(split['S'].y.tolist()));new=sorted(set(current.y.tolist()));seed=75110+100*stage+index
    futures=[c for c in new if c in supported];random.Random(seed).shuffle(futures)
    if len(futures)<2:return None,{'reason':'LESS_THAN_TWO_CURRENT_SUPPORT_CLASSES'}
    future_classes=futures[:2];prefix=sorted(set(supported)-set(future_classes));order=prefix+future_classes
    if not prefix:return None,{'reason':'NO_PREFIX_SUPPORT'}
    mapping=[-1]*(max(new)+1)
    for i,c in enumerate(order):mapping[c]=i
    selected={role:bank.take(torch.where(torch.isin(bank.y,torch.tensor(order,device=bank.z.device)))[0]) for role,bank in split.items()}
    banks={role:relabel(bank,mapping) for role,bank in selected.items()}
    groups={int(i):rows[int(i)]['identity_component'] for i in actual_pool.parents}
    class_birth={c:t for t,cls in enumerate(cfg['stages']) for c in cls}
    access=DataAccessLedger({int(i):class_birth[int(c)] for i,c in zip(actual_pool.parents.tolist(),actual_pool.y[actual_pool.gidx].tolist())},groups,roles.role,stage)
    E=sorted(set(banks['FIT'].y.tolist())&set(banks['AUDIT'].y.tolist()))
    if not (set(range(len(prefix)))&set(E)):return None,{'reason':'NO_COMMON_E_PREFIX','supported_classes':len(order)}
    model=Learner(seed).to(current.z.device);opt=optimizer(model)
    query_ids={int(i) for role in ['FIT','AUDIT'] for i in banks[role].parents}
    model.forbidden=query_ids;model.trained_groups=set();memory=None;seen=[];birth=[]
    prefix_phases=[]
    for t in range(stage+1):
        cls=[mapping[c] for c in prefix if c in cfg['stages'][t]]
        if cls:prefix_phases.append((t,cls))
    for j,(t,cls) in enumerate(prefix_phases):
        data=banks['S'].take(torch.where(torch.isin(banks['S'].y,torch.tensor(cls,device=current.z.device)))[0]);teacher=frozen(model) if model.C else None
        seen.extend([int((data.y[data.gidx]==c).sum()) for c in cls]);birth.extend([t]*len(cls))
        pool=join(memory,data);access.check(pool,'support');model.arrive(data,len(seen));model.trained_groups.update(groups[int(i)] for i in data.parents)
        learn(model,opt,pool,200,seed+j*2,counter,teacher,64,'science_prefix')
        visible=VisibleState(pool,memory,tuple(seen),tuple(birth),stage,256,groups,False,access,teacher)
        if j<len(prefix_phases)-1:
            memory,_=anchors(visible,model,seed+2000+j);learn(model,opt,memory,100,seed+j*2+1,counter,teacher,64,'science_prefix')
    future=[banks['S'].take(torch.where(banks['S'].y==mapping[c])[0]) for c in future_classes]
    queries=[];subsets={};query_by_role={}
    for h,C in enumerate([len(prefix),len(prefix)+1,len(order)]):
        eligible=[c for c in E if c<C];subsets[h]={};parts=[]
        for role in ['FIT','AUDIT']:
            bank=banks[role].take(torch.where(torch.isin(banks[role].y,torch.tensor(eligible,device=current.z.device)))[0])
            parts.append(bank);subsets[h][role+'_large']=set(bank.parents.tolist())
            small=[]
            for c in eligible:
                ids=bank.parents[bank.y[bank.gidx]==c].tolist();random.Random(seed+17*c).shuffle(ids);small.extend(ids[:2])
            subsets[h][role+'_small']=set(small)
        queries.append(join(*parts))
    # Every old query is a view actually retained in this donor's previous memory.
    previous_keys=set() if previous is None else set(token_keys(previous))
    current_keys=set(token_keys(current))
    assert all(set(token_keys(q)) <= previous_keys|current_keys for q in queries)
    oldtail={mapping[c] for c in prefix if c<sum(map(len,cfg['stages'][:stage])) and cfg['train_counts'][c]<250}
    env=PulseEnvironment(model,opt,visible,future,queries,oldtail,access,counter,teacher,100,200,64)
    meta={'seed':seed,'real_stage':stage,'scenario':index,'real_seen_classes':max(new)+1,'supported_shadow_classes':len(order),'prefix_classes':len(prefix),
          'unavailable_support_classes':sorted(set(range(max(new)+1))-set(order)),'root_tokens':len(visible.pool),'capacity':256,'pressure':len(visible.pool)/256,
          'oldtail_classes':len(oldtail),'E_counts':[len(set(q.y.tolist())) for q in queries],
          'query_counts':[{k:len(v) for k,v in subsets[h].items()} for h in range(3)],'prefix_phases':len(prefix_phases),'clinical_patient_independence_verified':False}
    return (env,model,opt,visible,subsets,meta),None


def slice_counts(counts,query,ids):
    _,labels,parents=query.bags();labels=labels.cpu();parents=parents.cpu()
    mask=torch.tensor([int(i) in ids for i in parents]);y=labels[mask];good=counts.good[mask]
    return Counts(torch.bincount(y[good],minlength=len(counts.correct)),torch.bincount(y,minlength=len(counts.correct)),tuple(parents[mask].tolist()),good,counts.brier[mask])


def score_runs(runs,env,subsets):
    out={}
    for name in ['FIT_small','FIT_large','AUDIT_small','AUDIT_large']:
        records=[]
        for run in runs:
            endpoints=[]
            for h in range(3):
                a=slice_counts(run['endpoints'][h],env.queries[h],subsets[h][name]);b=slice_counts(runs[0]['endpoints'][h],env.queries[h],subsets[h][name])
                endpoints.append(paired(a,b,env.oldtail))
            records.append({'H1':utility(endpoints[:1]),'H3':utility(endpoints),'endpoints':endpoints})
        out[name]=records
    return out

