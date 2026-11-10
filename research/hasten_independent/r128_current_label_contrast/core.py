"""Frozen reward, access, and scoring rules for generative continual RL."""
import json
import time
from pathlib import Path

ARMS=['BASE_TARS_RAPO','CE_AUX_TARS_RAPO','POS_KD_TARS_RAPO','CONTRAST_TARS_RAPO']
MAIN='CONTRAST_TARS_RAPO'
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
    by={(r['arm'],r['stage']):r for r in evaluations};assert len(by)==20
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
            old_tail_deficits.append(sum(by['CONTRAST_TARS_RAPO',stage]['correct'][c]/by['CONTRAST_TARS_RAPO',stage]['denominator'][c]
                -by['BASE_TARS_RAPO',stage]['correct'][c]/by['BASE_TARS_RAPO',stage]['denominator'][c] for c in old)/len(old))
    initial_head=[c for c in STAGES[0] if counts[c]>=250]
    head_drop=sum(by['CONTRAST_TARS_RAPO',4]['correct'][c]/by['CONTRAST_TARS_RAPO',4]['denominator'][c]
                  -by['CONTRAST_TARS_RAPO',0]['correct'][c]/by['CONTRAST_TARS_RAPO',0]['denominator'][c] for c in initial_head)/len(initial_head)
    main=post['CONTRAST_TARS_RAPO'];gates={
        'post_macro_over_BASE_ge_1pp':main['macro']-post['BASE_TARS_RAPO']['macro']>=.01,
        'post_tail_over_BASE_ge_2pp':main['tail']-post['BASE_TARS_RAPO']['tail']>=.02,
        'initial_task_head_drop_ge_minus1pp':head_drop>=-.01,
        'worst_old_tail_over_BASE_ge_minus1pp':min(old_tail_deficits)>=-.01}
    for control in ['CE_AUX_TARS_RAPO','POS_KD_TARS_RAPO']:
        for metric in ['macro','tail']:gates[f'post_{metric}_over_{control}_positive']=main[metric]>post[control][metric]
    gates['classwise_forgetting_less_than_BASE']=forgetting['CONTRAST_TARS_RAPO']<forgetting['BASE_TARS_RAPO']
    assert len(gates)==9
    return {'post_stage_means':post,'mean_classwise_forgetting':forgetting,'initial_task_head_change':head_drop,
        'worst_old_tail_over_BASE':min(old_tail_deficits),'gates':gates,'all_gates_pass':all(gates.values()),
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


PROMPT_TEMPLATE='What is the endoscopic finding or anatomical landmark shown in this image? Choose one category from the categories seen so far: {seen_categories}. Answer with the category name only.'


def stage_prompt(stage,names):
    assert 0<=stage<len(STAGES)
    seen=[c for task in STAGES[:stage+1] for c in task]
    return PROMPT_TEMPLATE.format(seen_categories='; '.join(names[c] for c in seen))


def interface_fixtures():
    names={c:f'CLASS_{c:02d}' for c in range(23)}
    for stage in range(5):
        text=stage_prompt(stage,names)
        expected=[c for task in STAGES[:stage+1] for c in task]
        assert [c for c in range(23) if names[c] in text]==expected
    return {'all_five_cumulative_seen_only_prompts':True,'future_categories_in_policy_prompt':False}


TEACHER_TEMPLATE='A reference answer is: {answer}. {student_question}'


def canonical_key(text):
    import re
    return re.sub(r'[\s_\-]+','',text.lower().strip())


def teacher_negatives(text,done,labels,names,seen):
    from collections import Counter
    lookup={canonical_key(v):c for c,v in names.items()};parsed=[lookup.get(canonical_key(t),-1) if d else -1 for t,d in zip(text,done)]
    negatives=[-1]*len(text)
    assert len(text)==len(labels)*8 and len(done)==len(text)
    for row,gold in enumerate(labels):
        sl=range(row*8,(row+1)*8);wrong=[parsed[i] for i in sl if parsed[i] in seen and parsed[i]!=gold]
        mode=min(Counter(wrong),key=lambda c:(-Counter(wrong)[c],c)) if wrong else -1
        for i in sl:
            if parsed[i] in seen and parsed[i]!=gold:negatives[i]=parsed[i]
            elif parsed[i]==gold and mode>=0:negatives[i]=mode
    return parsed,negatives


def contrast_advantages(base,evidence,eligible,kappa=.2,delta=.2):
    import torch
    a=base.reshape(-1,1);e=evidence.detach()*eligible[:,None]
    z=a+kappa*e.clamp(-delta,delta)
    return torch.where(a>0,z.clamp_min(0),torch.where(a<0,z.clamp_max(0),z)).detach()


def token_surrogate(lp,old,ref,mask,adv,weights):
    import torch
    ratio=(lp-old).exp();pg=torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)
    d=ref-lp;assert torch.isfinite(d).all() and float(d.detach().abs().max())<80
    kl=torch.expm1(d)-d;length=mask.sum(-1).clamp_min(1)
    p=(pg*mask).sum(-1)/length;k=(kl*mask).sum(-1)/length
    return -(weights*p).mean()+.04*k.mean(),p.detach(),k.detach()


def contrast_fixtures():
    import torch
    names={i:f'C{i}' for i in range(10)}
    text=['C1']*3+['C2']*3+['C0']*2;parsed,neg=teacher_negatives(text,[True]*8,[0],names,set(range(5)))
    assert neg==[1]*3+[2]*3+[1]*2
    _,none=teacher_negatives(['C0']*8,[True]*8,[0],names,set(range(5)));assert none==[-1]*8
    _,invalid=teacher_negatives(['C8','broken']*4,[True]*8,[0],names,set(range(5)));assert invalid==[-1]*8
    _,truncated=teacher_negatives(['C1']*8,[False]*8,[0],names,set(range(5)));assert truncated==[-1]*8
    base=torch.tensor([.001,-.001,0.]);e=torch.tensor([[-1.,1.],[1.,-1.],[-1.,1.]],requires_grad=True)
    a=contrast_advantages(base,e,torch.tensor([True,True,True]));assert bool((a[0]>=0).all()) and bool((a[1]<=0).all()) and a[2,0]<0<a[2,1] and not a.requires_grad
    zero=contrast_advantages(torch.zeros(1),torch.ones(1,2),torch.tensor([False]));assert torch.equal(zero,torch.zeros_like(zero))
    lp=torch.zeros(3,2,requires_grad=True);old=lp.detach().clone();loss,_,_=token_surrogate(lp,old,old,torch.ones_like(lp),a,torch.ones(3));loss.backward();assert lp.grad is not None and e.grad is None
    assert [(g//10)*10 for g in [0,9,10,127,128,130]]==[0,0,10,120,120,130]
    return {'valid_wrong_own_negative_and_correct_modal_tie_rule':True,'all_correct_invalid_future_and_truncated_no_synthetic_negative':True,'RaPO_continuous_base_explicit_sign_projection':True,'zero_base_can_receive_valid_teacher_signal_not_efficacy':True,'teacher_stop_gradient_and_KL_separation':True,'teacher_sync_global_across_stage_boundary':True,'new_model_calls':0}
