"""Fixed pseudo-arrivals, query-disjoint prefix fitting and pinned local edits."""
import importlib,random
import torch
from . import evidence as E

base='r110_ecv_opportunity.base.'
learner=importlib.import_module(base+'learner')
contracts=importlib.import_module(base+'contracts')
old_edit=importlib.import_module(base+'anchor_candidates')._edit
describe=importlib.import_module(base+'state_features').describe
branch=importlib.import_module(base+'branch_env')
paired_labels=importlib.import_module(base+'paired_labels')


def visible(pool,previous,stage,teacher=None):
    S=pool.role(0);C=pool.C;B=256-len(pool.role(1))-len(pool.role(2));groups={int(i):int(i) for i in S.parents}
    ledger=contracts.DataAccessLedger({i:0 for i in groups},groups,{i:'S' for i in groups},0)
    return contracts.VisibleState(S,None if previous is None else previous.role(0),tuple(int(x) for x in pool.stats['count'][:C]),tuple(int(x) for x in pool.stats['birth'][:C]),stage,B,groups,False,ledger,teacher)


def layout(previous,current,cfg,stage,index):
    actual=E.join(previous,current);new=cfg['stages'][stage];seed=75110+100*stage+index;future=list(new);random.Random(seed).shuffle(future);future=future[:2]
    prefix=sorted(set(range(max(new)+1))-set(future));order=prefix+future;mapping=[-1]*(max(new)+1)
    for j,c in enumerate(order):mapping[c]=j
    phases=[]
    for t,cs in enumerate(cfg['stages'][:stage+1]):
        cls=[c for c in prefix if c in cs]
        if cls:phases.append(E.classes(actual,cls,mapping,True))
    futures=[E.classes(actual,[c],mapping,True) for c in future]
    tail={mapping[c] for c in prefix if int(actual.stats[c]['birth'])<stage and int(actual.stats[c]['birth_total'])<250}
    meta={'seed':seed,'stage':stage,'scenario':index,'real_to_shadow':mapping,'shadow_to_real':order,'prefix_classes':len(prefix),'real_oldtail_shadow':sorted(tail),'future_real_classes':future,'prefix_phases':len(phases),'patient_denominator':'NA'}
    return phases,futures,tail,meta


def root(previous,current,cfg,stage,index,counter=None,train=False):
    phases,futures,tail,meta=layout(previous,current,cfg,stage,index);seed=meta['seed'];kept=None;m=None;o=None;teacher=None
    if train:
        m=learner.Learner(seed).to(current.bank.z.device);o=learner.optimizer(m)
        m.forbidden={int(i) for x in phases+futures for r in [1,2] for i in x.role(r).parents}
    for j,data in enumerate(phases):
        pool=E.join(kept,data)
        if train:
            teacher=learner.frozen(m) if m.C else None;m.arrive(data.role(0),pool.C)
            learner.learn(m,o,pool.role(0),200,seed+j*2,counter,teacher,64,'science_prefix')
        if j<len(phases)-1:
            kept=E.compressed(pool,seed+2000+j)
            if train:learner.learn(m,o,kept.role(0),100,seed+j*2+1,counter,teacher,64,'science_prefix')
    meta.update(S_root_tokens=len(pool.role(0)),S_capacity=256-2*pool.C,total_query_tokens=2*pool.C,
                prefix_support_complete=set(pool.role(0).y.tolist())==set(range(pool.C)),
                full_oldtail_roles=all(all(int((pool.role(r).y==c).sum())>0 for r in [0,1,2]) for c in tail))
    return pool,kept,futures,tail,meta,m,o,teacher


def generate(pool,previous,m,o,stage,seed,counter,teacher=None,construction=False):
    z,b,ledger=E.anchors(pool,seed);state=visible(pool,previous,stage,teacher);out=[];seen=set();core=set(pool.core_keys().tolist())
    def add(bank,family,name,anchor):
        count=len(set(contracts.token_keys(anchor))-set(contracts.token_keys(bank)));c=contracts.Candidate(bank,family,name,count)
        contracts.validate_candidate(c,state,anchor);assert core<=set(contracts.token_keys(bank));pool.with_support(bank)
        if c.key not in seen:seen.add(c.key);out.append(c)
    add(z,'anchor','E-ZERO',z);add(b,'anchor','E-BAL',b)
    if construction:
        for ai,(name,anchor) in enumerate([('E-ZERO',z),('E-BAL',b)]):
            eligible=~torch.isin(state.pool.pid*8+state.pool.view,pool.core_keys())
            bank,n=old_edit(state,anchor,torch.zeros(len(state.pool),device=state.pool.z.device),2,seed+100*ai+4,True,eligible)
            add(bank,'random',name,anchor)
        return out,ledger
    description=describe(state,m,o,counter,teacher=teacher,seed=seed)
    for ai,(name,anchor) in enumerate([('E-ZERO',z),('E-BAL',b)]):
        S=state.pool;y=S.y[S.gidx];selected=torch.isin(S.parents,anchor.parents);zz=description['z'];cover=torch.zeros(len(y),device=S.z.device)
        for c in range(m.C):
            mask=y==c;cover[mask]=1-(zz[mask]@zz[mask&selected].T).max(1).values
        for j,(family,limit) in enumerate([('risk',1),('risk',2),('risk',4),('cover',4),('random',2)]):
            priority=(description['parent_risk'] if family=='risk' else cover if family=='cover' else torch.zeros_like(cover))[S.parent_index]
            eligible=~torch.isin(S.pid*8+S.view,pool.core_keys())
            if family=='risk':eligible &= description['scored'][S.parent_index]
            bank,n=old_edit(state,anchor,priority,min(limit,max(1,len(anchor)//10)),seed+100*ai+j,family=='random',eligible)
            add(bank,family,name,anchor)
    assert len(out)<=12
    return out,ledger


def rollout(pool,futures,m,o,teacher,candidate,seed,counter):
    model,opt=learner.clone_pair(m,o,seed);kept=pool.with_support(candidate.memory);results=[];audit=[];records=[]
    for h in range(3):
        if h:
            data=futures[h-1];full=E.join(kept,data);teacher=learner.frozen(model);model.arrive(data.role(0),full.C)
            branch.paired_learn(model,opt,full.role(0),200,branch.event_seed(seed,f'{h}:pre'),counter,teacher,64)
            kept=E.compressed(full,branch.event_seed(seed,f'{h}:zero'))
        branch.paired_learn(model,opt,kept.role(0),100,branch.event_seed(seed,f'{h}:post'),counter,teacher,64)
        # Only the isolated evaluator receives query banks. Neither result is
        # returned to candidate generation or any later learning step.
        results.append(paired_labels.evaluate(model,kept.role(1),counter))
        audit.append(paired_labels.evaluate(model,kept.role(2),counter))
        assert bool(torch.isfinite(results[-1].brier).all()) and bool(torch.isfinite(audit[-1].brier).all()),'NONFINITE_QUERY_SCORE'
        records.append({'C':kept.C,'total_tokens':len(kept.bank),'total_bytes':len(kept.bank)*2064+184,'S_tokens':len(kept.role(0)),'S_capacity':256-2*kept.C,'FIT_tokens':len(kept.role(1)),'AUDIT_tokens':len(kept.role(2))})
    return {'FIT':results,'AUDIT':audit,'bytes':records}


def score(runs,tail):
    out={}
    for role in ['FIT','AUDIT']:
        out[role]=[]
        for run in runs:
            points=[paired_labels.paired(a,b,tail) for a,b in zip(run[role],runs[0][role])]
            full=all(all(c<len(x['denominator']) and x['denominator'][c]>0 for c in tail) for x in points)
            u=paired_labels.utility(points)
            out[role].append({'R':u['R'] if full or not tail else None,'descriptive_shadow_R':u['R'],'macro':u['macro'],'endpoints':points,'full_oldtail_evaluable':bool(tail) and full,
                              'qualified':bool(tail) and full and all(x['tail']>=-.005 for x in points)})
    return out
