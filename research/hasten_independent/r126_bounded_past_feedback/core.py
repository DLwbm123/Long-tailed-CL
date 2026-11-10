"""Frozen reward, access, and scoring rules for generative continual RL."""
import json
import time
from pathlib import Path

ARMS=['CURRENT_TARS_RAPO','OLD_PROXY','OLD_TASK_TARS_RAPO','OLD_TASK_RAPO','OLD_TASK_CE']
MAIN='OLD_TASK_TARS_RAPO'
STAGES=[list(range(0,5)),list(range(5,10)),list(range(10,15)),list(range(15,19)),list(range(19,23))]


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


class Evidence:
    def __init__(self,labels,out=None):
        self.labels=list(labels);self.out=Path(out) if out else None
        self.features={};self.saved_labels={};self.born={};self.events=[]

    def selected(self,stage):
        import torch
        assert len(self.events)==stage
        ids=[]
        for c in STAGES[stage]:
            available=[i for i,y in enumerate(self.labels) if y==c]
            g=torch.Generator().manual_seed(12654005+c)
            ids.extend([available[i] for i in torch.randperm(len(available),generator=g)[:2].tolist()])
        return ids

    def admit(self,stage,read_current):
        import torch
        ids=self.selected(stage);features=read_current(ids).detach().cpu().clone()
        assert features.shape==(len(ids),64,2048) and features.dtype==torch.bfloat16
        for i,row in zip(ids,features):
            assert i not in self.features and self.labels[i] in STAGES[stage]
            self.features[i]=row;self.saved_labels[i]=self.labels[i];self.born[i]=stage
        assert len(self.features)<=46
        event={'stage':stage,'ids':ids,'labels':[self.labels[i] for i in ids],
               'rows':len(ids),'total_retained_rows':len(self.features),'epoch':time.time()}
        self.events.append(event)
        if self.out:
            torch.save({'stage':stage,'ids':ids,'labels':event['labels'],'features':features},self.out/f'BUFFER_BIRTH_{stage}.private.pt')
            write(self.out/'BUFFER_EVENTS.private.json',self.events)
            write(self.out/'BUFFER_SUMMARY.json',{'births':[{'stage':e['stage'],'rows':e['rows']} for e in self.events],
                'retained_components':len(self.features),'max_components_per_class':2,
                'feature_body_bytes':len(self.features)*64*2048*2,'no_old_full_TRAIN_feature_reads':True})
        return len(ids)

    def inputs(self,ids,stage):
        import torch
        assert all(i in self.features and self.born[i]<stage for i in ids)
        return torch.stack([self.features[i] for i in ids])

    def old_tape(self,stage):
        import torch
        classes=[c for task in STAGES[:stage] for c in task];g=torch.Generator().manual_seed(12674005+stage)
        result=[]
        for _ in range(128):
            row=[]
            for _ in range(4):
                c=classes[int(torch.randint(len(classes),(1,),generator=g))]
                options=[i for i,y in self.saved_labels.items() if y==c and self.born[i]<stage]
                row.append(options[int(torch.randint(len(options),(1,),generator=g))])
            result.append(row)
        return torch.tensor(result)


class Access:
    def __init__(self,train_labels,val_labels,pool=None):
        self.train_labels=list(train_labels);self.val_labels=list(val_labels);self.pool=pool;self.journal=[]

    def check(self,split,ids,stage,arm,purpose,sealed=False,step=None):
        ids=[int(i) for i in ids]
        if split not in ['TRAIN','BUFFER','VAL'] or not ids:raise PermissionError('INVALID_ACCESS')
        labels=self.val_labels if split=='VAL' else self.train_labels
        if any(i<0 or i>=len(labels) for i in ids):raise PermissionError('INVALID_ACCESS')
        if split=='BUFFER':
            if self.pool is None or arm=='CURRENT_TARS_RAPO' or purpose!='old_training':raise PermissionError('BUFFER_ROLE_DENIED')
            if any(i not in self.pool.features or self.pool.born[i]>=stage for i in ids):raise PermissionError('NONSELECTED_OR_FUTURE_BUFFER')
        else:
            allowed=STAGES[stage] if split=='TRAIN' else [c for task in STAGES[:stage+1] for c in task]
            if split=='VAL' and (purpose!='isolated_audit' or not sealed):raise PermissionError('VAL_BEFORE_SEAL_OR_TRAINING_USE')
            if any(labels[i] not in allowed for i in ids):raise PermissionError('OLD_OR_FUTURE_TRAIN_OR_FUTURE_VAL')
        self.journal.append({'split':split,'ids':ids,'stage':stage,'arm':arm,'purpose':purpose,
                             'step':step,'all_stage_models_sealed':sealed,'epoch':time.time()})
        return ids


def used_task_rewards(actual,old_flags,arm):
    import torch
    old_flags=torch.as_tensor(old_flags,device=actual.device,dtype=torch.bool)
    assert old_flags.shape==(len(actual),)
    if arm=='CURRENT_TARS_RAPO':assert not bool(old_flags.any())
    return torch.where(old_flags[:,None],torch.zeros_like(actual),actual) if arm=='OLD_PROXY' else actual.clone()


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
            old_tail_deficits.append(sum(by['OLD_TASK_TARS_RAPO',stage]['correct'][c]/by['OLD_TASK_TARS_RAPO',stage]['denominator'][c]
                -by['OLD_PROXY',stage]['correct'][c]/by['OLD_PROXY',stage]['denominator'][c] for c in old)/len(old))
    initial_head=[c for c in STAGES[0] if counts[c]>=250]
    head_drop=sum(by['OLD_TASK_TARS_RAPO',4]['correct'][c]/by['OLD_TASK_TARS_RAPO',4]['denominator'][c]
                  -by['OLD_TASK_TARS_RAPO',0]['correct'][c]/by['OLD_TASK_TARS_RAPO',0]['denominator'][c] for c in initial_head)/len(initial_head)
    main=post['OLD_TASK_TARS_RAPO'];gates={
        'post_macro_over_OLD_PROXY_ge_1pp':main['macro']-post['OLD_PROXY']['macro']>=.01,
        'post_tail_over_OLD_PROXY_ge_2pp':main['tail']-post['OLD_PROXY']['tail']>=.02,
        'initial_task_head_drop_ge_minus1pp':head_drop>=-.01,
        'worst_old_tail_over_OLD_PROXY_ge_minus1pp':min(old_tail_deficits)>=-.01}
    for control in ['OLD_TASK_RAPO','OLD_TASK_CE']:
        for metric in ['macro','tail']:gates[f'post_{metric}_over_{control}_positive']=main[metric]>post[control][metric]
    gates['classwise_forgetting_less_than_OLD_PROXY']=forgetting['OLD_TASK_TARS_RAPO']<forgetting['OLD_PROXY']
    assert len(gates)==9
    return {'post_stage_means':post,'mean_classwise_forgetting':forgetting,'initial_task_head_change':head_drop,
        'worst_old_tail_over_OLD_PROXY':min(old_tail_deficits),'gates':gates,'all_gates_pass':all(gates.values()),
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


def bounded_fixtures():
    import torch
    y=[c for c in range(23) for _ in range(3)];pool=Evidence(y)
    def read(ids):return torch.tensor(ids).reshape(-1,1,1).expand(-1,64,2048).to(torch.bfloat16).clone()
    assert pool.admit(0,read)==10
    chosen=next(iter(pool.features));unselected=next(i for i,c in enumerate(y) if c==y[chosen] and i not in pool.features)
    a=Access(y,y,pool);a.check('BUFFER',[chosen],1,'OLD_PROXY','old_training');assert pool.inputs([chosen],1)[0,0,0]==chosen
    rejected=0
    for args in [('TRAIN',[chosen],1,'OLD_PROXY','current_training'),('BUFFER',[unselected],1,'OLD_PROXY','old_training'),
                 ('BUFFER',[chosen],0,'OLD_PROXY','old_training'),('BUFFER',[chosen],1,'CURRENT_TARS_RAPO','old_training'),
                 ('TRAIN',[30],1,'OLD_PROXY','current_training')]:
        try:a.check(*args)
        except PermissionError:rejected+=1
    assert rejected==5
    task=torch.tensor([[0.,1.],[0.,1.]])
    changed=torch.tensor([[0.,1.],[1.,0.]])
    flags=torch.tensor([False,True]);lp=torch.zeros(4,2);mask=torch.ones_like(lp)
    used=used_task_rewards(task,flags,'OLD_PROXY');other=used_task_rewards(changed,flags,'OLD_PROXY')
    assert torch.equal(used,other) and torch.equal(used[1],torch.zeros(2))
    adv,*_,sigma=rewards_and_advantages(used,lp,lp,mask,True,1.)
    alt,*_,sigma2=rewards_and_advantages(other,lp,lp,mask,True,1.)
    assert torch.equal(adv,alt) and sigma==sigma2
    def pg_gradient(adv):
        current=torch.zeros(2,2,requires_grad=True);(-(current*adv).mean()).backward();return current.grad
    assert torch.equal(pg_gradient(adv),pg_gradient(alt))
    main=used_task_rewards(changed,flags,'OLD_TASK_TARS_RAPO')
    main_adv,*_=rewards_and_advantages(main,lp,lp,mask,True,1.)
    assert not torch.equal(pg_gradient(main_adv),pg_gradient(adv))
    for stage in range(1,5):pool.admit(stage,read)
    assert len(pool.features)==46 and sum(e['rows'] for e in pool.events)==46
    return {'bounded_selected_buffer_capacity_and_sources_verified':True,'new_buffer_access_denial_cases':5,
            'old_accuracy_mask_before_advantage_and_EMA_verified':True,'old_label_perturbation_leaves_PROXY_PG_gradient_unchanged':True,
            'TASK_PG_responds_to_old_correctness':True,'fixture_model_queries_updates':0}
