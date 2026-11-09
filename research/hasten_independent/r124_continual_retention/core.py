"""Frozen reward, access, and scoring rules for generative continual RL."""
import json
import time
from pathlib import Path

ARMS=['GRPO','TARS','RAPO','TARS_RAPO','TARS_CE']
STAGES=[list(range(0,5)),list(range(5,10)),list(range(10,15)),list(range(15,19)),list(range(19,23))]


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


class Access:
    def __init__(self,train_labels,val_labels):
        self.train_labels=list(train_labels);self.val_labels=list(val_labels);self.journal=[]

    def check(self,split,ids,stage,arm,purpose,sealed=False):
        ids=[int(i) for i in ids]
        labels=self.train_labels if split=='TRAIN' else self.val_labels
        allowed=set(STAGES[stage]) if split=='TRAIN' else set(c for task in STAGES[:stage+1] for c in task)
        if split not in ['TRAIN','VAL'] or not ids or any(i<0 or i>=len(labels) for i in ids):raise PermissionError('INVALID_ACCESS')
        if split=='VAL' and (purpose!='isolated_audit' or not sealed):raise PermissionError('VAL_BEFORE_SEAL_OR_TRAINING_USE')
        if any(labels[i] not in allowed for i in ids):raise PermissionError('OLD_OR_FUTURE_TRAIN_OR_FUTURE_VAL')
        self.journal.append({'split':split,'ids':ids,'stage':stage,'arm':arm,'purpose':purpose,'all_stage_models_sealed':sealed,'epoch':time.time()})
        return ids


def rewards_and_advantages(task,old_logps,anchor_logps,mask,rapo,sigma):
    import torch
    drift=((old_logps-anchor_logps)*mask).sum(-1)/mask.sum(-1).clamp_min(1)
    drift=drift.clamp_min(0).detach()
    retention=torch.exp(-20*drift).reshape_as(task).detach()
    total=task+.5*retention if rapo else task
    if rapo:
        sigma=.999*float(sigma)+.001*float(total.std(unbiased=False))
        adv=(total-total.mean(-1,keepdim=True))/(sigma+1e-6)
    else:
        adv=(total-total.mean(-1,keepdim=True))/(total.std(-1,unbiased=False,keepdim=True)+1e-6)
    return adv.detach(),retention,total.detach(),drift,float(sigma)


def weights_for_current(counts,correct):
    import torch
    n=torch.as_tensor(counts,dtype=torch.float64);g=torch.as_tensor(correct,dtype=torch.float64)
    assert bool((n>0).all())
    raw=(n.mean()/n*(1-g/n)).pow(.75);median=raw.quantile(.5)
    w=(raw/median.clamp_min(1e-8)).clamp_min(1);z=float((n*w).sum()/n.sum());w=w/z
    assert torch.isfinite(w).all() and abs(float((n*w).sum()/n.sum())-1)<1e-12
    return w,z,float(median)


def score(correct,den,seen,counts):
    def mean(ids):
        ids=[c for c in ids if c in seen and den[c]>0]
        return sum(correct[c]/den[c] for c in ids)/len(ids) if ids else None
    return {'correct':correct,'denominator':den,'seen':list(seen),'macro':mean(seen),
            'head':mean([c for c,n in enumerate(counts) if n>=250]),
            'tail':mean([c for c,n in enumerate(counts) if n<250]),
            'tiny':mean([c for c,n in enumerate(counts) if n<25]),
            'micro':sum(correct)/sum(den)}


def summarize(evaluations,counts):
    by={(r['arm'],r['stage']):r for r in evaluations};assert len(by)==25
    post={a:{k:sum(by[a,s][k] for s in range(1,5))/4 for k in ['macro','tail']} for a in ARMS}
    forgetting={}
    for a in ARMS:
        values=[]
        for c in range(23):
            available=[by[a,s]['correct'][c]/by[a,s]['denominator'][c] for s in range(5) if by[a,s]['denominator'][c]>0]
            values.append(max(available)-available[-1])
        forgetting[a]=sum(values)/23
    old_tail_deficits=[]
    for stage in range(1,5):
        old=[c for task in STAGES[:stage] for c in task if counts[c]<250]
        if old:
            old_tail_deficits.append(sum(by['TARS_RAPO',stage]['correct'][c]/by['TARS_RAPO',stage]['denominator'][c]
                -by['TARS',stage]['correct'][c]/by['TARS',stage]['denominator'][c] for c in old)/len(old))
    initial_head=[c for c in STAGES[0] if counts[c]>=250]
    head_drop=sum(by['TARS_RAPO',4]['correct'][c]/by['TARS_RAPO',4]['denominator'][c]
                  -by['TARS_RAPO',0]['correct'][c]/by['TARS_RAPO',0]['denominator'][c] for c in initial_head)/len(initial_head)
    main=post['TARS_RAPO'];gates={
        'post_macro_over_TARS_ge_1pp':main['macro']-post['TARS']['macro']>=.01,
        'post_tail_over_TARS_ge_2pp':main['tail']-post['TARS']['tail']>=.02,
        'initial_task_head_drop_ge_minus1pp':head_drop>=-.01,
        'worst_old_tail_over_TARS_ge_minus1pp':min(old_tail_deficits)>=-.01}
    for control in ['RAPO','TARS_CE']:
        for metric in ['macro','tail']:gates[f'post_{metric}_over_{control}_positive']=main[metric]>post[control][metric]
    gates['classwise_forgetting_less_than_TARS']=forgetting['TARS_RAPO']<forgetting['TARS']
    assert len(gates)==9
    return {'post_stage_means':post,'mean_classwise_forgetting':forgetting,'initial_task_head_change':head_drop,
        'worst_old_tail_over_TARS':min(old_tail_deficits),'gates':gates,'all_gates_pass':all(gates.values()),
        'stage_readouts':evaluations,'development_only_no_independent_patient_claim':True}


def fixtures():
    import torch
    task=torch.tensor([[0.,0.],[1.,1.]])
    mask=torch.ones(4,2);lp=torch.zeros(4,2)
    adv,ret,total,drift,sigma=rewards_and_advantages(task,lp,lp,mask,True,1.)
    assert torch.equal(adv,torch.zeros_like(adv)) and torch.equal(ret,torch.ones_like(ret))
    assert sigma<1 and sigma>0
    task=torch.tensor([[0.,1.],[0.,0.]])
    old=torch.tensor([[0.,0.],[.1,.1],[0.,0.],[.2,.2]])
    adv,ret,total,drift,next_sigma=rewards_and_advantages(task,old,lp,mask,True,sigma)
    assert total[0,1]>total[0,0] and adv[1,0]>adv[1,1] and next_sigma!=sigma
    w,z,_=weights_for_current([100,10],[90,0]);assert abs(float((w*torch.tensor([100,10])).sum()/110)-1)<1e-12
    access=Access([0,5],[0,5]);access.check('TRAIN',[0],0,'fixture','current')
    rejected=0
    for args in [('TRAIN',[1],0,'x','train',False),('TRAIN',[0],1,'x','train',False),
                 ('VAL',[0],0,'x','train',True),('VAL',[1],0,'x','isolated_audit',True),('VAL',[0],0,'x','isolated_audit',False)]:
        try:access.check(*args)
        except PermissionError:rejected+=1
    assert rejected==5
    return {'passed':True,'native_model_queries_updates':0,'access_denial_cases':5,
        'retention_reward_before_group_mean':True,'constant_groups_zero_advantage':True,
        'EMA_persists_without_task_reset':True,'TARS_natural_mean_weight_exact_one':True}
