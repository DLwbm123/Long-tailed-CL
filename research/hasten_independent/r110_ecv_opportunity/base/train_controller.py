"""Real-label verifier/SFT and paired-reference PPO. No synthetic rewards."""
import copy
import hashlib
import torch
import torch.nn.functional as F
from .policy import verifier_prior, sft_loss
from .paired_labels import near_optimal


def pipeline_signature():
    from .anchor_candidates import generate,_edit
    from .state_features import describe,action_features
    functions=[generate,_edit,describe,action_features,verifier_prior]
    import inspect
    return hashlib.sha256(b''.join(f.__code__.co_code+inspect.getsource(f).encode() for f in functions)).hexdigest()


def fingerprint(modules, payload=()):
    h=hashlib.sha256(repr(payload).encode())
    for module in modules:
        for name,value in module.state_dict().items():
            h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train_verifier(verifier, records, epochs=20, lr=.001):
    assert records and all(r['split']=='fit' for r in records)
    verifier.fit_scales(records)
    opt=torch.optim.AdamW(verifier.parameters(),lr=lr)
    history=[]
    for _ in range(epochs):
        for r in records:
            values,dynamic=verifier(r['classes'],r['actions'])
            mask=r['target_mask']; target=r['targets']
            assert mask.any()
            loss=F.huber_loss((values/verifier.return_scale)[mask],(target/verifier.return_scale)[mask])
            if verifier.use_dynamics:
                dm=r['dynamic_mask']
                if dm.any(): loss=loss+F.huber_loss((dynamic/verifier.dynamic_scale)[dm],(r['dynamic']/verifier.dynamic_scale)[dm])
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(verifier.parameters(),1.);opt.step()
            history.append(float(loss.detach()))
    return dict(updates=len(history),losses=history)


def train_sft(policy,verifier,records,epochs=20,lr=.001):
    assert all(r['split']=='fit' for r in records)
    opt=torch.optim.AdamW(policy.parameters(),lr=lr); losses=[]
    verifier.eval().requires_grad_(False)
    with torch.no_grad():
        for name in ['class_mean','class_std','action_mean','action_std']:
            getattr(policy,name).copy_(getattr(verifier,name))
    for _ in range(epochs):
        for r in records:
            with torch.no_grad(): scores,_=verifier(r['classes'],r['actions']);prior=verifier_prior(scores[:,r.get('return_column',0)])
            logp=policy.log_probs(r['classes'],r['actions'],prior)
            loss=sft_loss(logp,r['nearopt'])
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1.);opt.step();losses.append(float(loss.detach()))
    return dict(updates=len(losses),losses=losses)


def collect_trajectories(env,policy,verifier,selector_builder,root_candidates,seed,G=8,payload=(),return_column=3):
    """selector_builder constructs query-free features at each new real decision."""
    assert G>=1
    frozen_policy=copy.deepcopy(policy).eval().requires_grad_(False)
    verifier.eval().requires_grad_(False)
    pipeline=pipeline_signature();sealed_payload=(payload,pipeline,return_column)
    seal=fingerprint([frozen_policy,verifier],sealed_payload)
    zero_records=[]
    def zero_select(state,model,opt,decision_seed):
        candidates,classes,actions=selector_builder(state,model,opt,decision_seed)
        with torch.no_grad():
            score,_=verifier(classes,actions);q=verifier_prior(score[:,return_column]);lp=frozen_policy.log_probs(classes,actions,q)
        rec=dict(classes=classes,actions=actions,prior=q,old_logp=lp,action=0,candidate_keys=[a.key for a in candidates],seal=seal)
        zero_records.append(rec)
        return candidates[0].memory,rec
    # The reference is actually executed; it is never a predicted counterfactual.
    root_state=env.state; root_model,root_opt,_=env.snapshot.restore()
    candidates,classes,actions=selector_builder(root_state,root_model,root_opt,seed)
    assert [c.key for c in candidates]==[c.key for c in root_candidates]
    with torch.no_grad():
        values,_=verifier(classes,actions);prior=verifier_prior(values[:,return_column]);root_lp=frozen_policy.log_probs(classes,actions,prior)
    zero=env.rollout(candidates[0].memory,seed,zero_select)
    zero_step=dict(classes=classes,actions=actions,prior=prior,old_logp=root_lp,action=0,candidate_keys=[a.key for a in candidates],seal=seal)
    trajectories=[]
    for g in range(G):
        generator=torch.Generator(device=classes.device).manual_seed(seed+10000+g)
        records=[]
        def choose(state,model,opt,decision_seed):
            cc,xx,aa=selector_builder(state,model,opt,decision_seed)
            with torch.no_grad():
                scores,_=verifier(xx,aa);q=verifier_prior(scores[:,return_column]);lp=frozen_policy.log_probs(xx,aa,q)
                a=int(torch.multinomial(lp.exp(),1,generator=generator))
            record=dict(classes=xx,actions=aa,prior=q,old_logp=lp,action=a,candidate_keys=[c.key for c in cc],seal=seal)
            records.append(record)
            return cc[a].memory,record
        with torch.no_grad():a=int(torch.multinomial(root_lp.exp(),1,generator=generator))
        first=dict(classes=classes,actions=actions,prior=prior,old_logp=root_lp,action=a,candidate_keys=[c.key for c in candidates],seal=seal)
        run=env.rollout(candidates[a].memory,seed,choose)
        from .paired_labels import paired,utility
        reward=utility([paired(x,y,env.oldtail) for x,y in zip(run['endpoints'],zero['endpoints'])])['R']
        trajectories.append(dict(steps=[first]+records,R=reward,source='real_execution',root=seed))
    assert fingerprint([frozen_policy,verifier],sealed_payload)==seal
    return dict(trajectories=trajectories,zero=dict(steps=[zero_step]+zero_records,R=0.,source='real_execution',root=seed),behavior=frozen_policy,seal=seal,payload=sealed_payload,pipeline=pipeline,return_column=return_column)


def train_pg(policy,reference,verifier,batch,sigma_fit,epochs=2,lr=1e-4,kl_weight=.01,kl_stop=.02):
    assert sigma_fit>=.02 and epochs>=1
    assert pipeline_signature()==batch['pipeline'], 'CANDIDATE_PIPELINE_CHANGED'
    assert fingerprint([batch['behavior'],verifier],batch['payload'])==batch['seal'], 'ROLLOUT_OBJECT_CHANGED'
    trajectories=batch['trajectories'];assert trajectories
    opt=torch.optim.AdamW(policy.parameters(),lr=lr)
    history=[]
    for _ in range(epochs):
        losses=[];kls=[]
        for trajectory in trajectories:
            assert trajectory['source']=='real_execution'
            advantage=trajectory['R']/sigma_fit
            for s in trajectory['steps']:
                with torch.no_grad():
                    values,_=verifier(s['classes'],s['actions'])
                    recomputed_prior=verifier_prior(values[:,batch['return_column']])
                    assert torch.allclose(recomputed_prior,s['prior'],atol=1e-7), 'FROZEN_PRIOR_CHANGED'
                    check=batch['behavior'].log_probs(s['classes'],s['actions'],s['prior'])
                    assert torch.allclose(check,s['old_logp'],atol=1e-7), 'BEHAVIOR_RECOMPUTE_MISMATCH'
                    ref=reference.log_probs(s['classes'],s['actions'],s['prior'])
                new=policy.log_probs(s['classes'],s['actions'],s['prior'])
                ratio=(new[s['action']]-s['old_logp'][s['action']]).exp()
                surrogate=torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage)
                kl=(new.exp()*(new-ref)).sum()
                losses.append(-surrogate+kl_weight*kl);kls.append(kl.detach())
        # Sum over each trajectory's genuine decisions, average over trajectories.
        loss=torch.stack(losses).sum()/len(trajectories)
        kl=float(torch.stack(kls).mean())
        if kl>kl_stop:
            history.append(dict(stopped_before_update=True,kl=kl));break
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1.);opt.step()
        history.append(dict(loss=float(loss.detach()),kl=kl,stopped_before_update=False))
    return dict(updates=sum(not x['stopped_before_update'] for x in history),history=history)


def train_more_sup(policy,batch,epochs=2,lr=1e-4,epsilon=.005):
    """Near-optimal COMPLETE trajectories from the same root, never mixed states.

    A trajectory label does not invent counterfactual labels for its later states.
    Uses exactly the same newly executed feedback batch as PG, including ZERO.
    """
    trajectories=[batch['zero']]+batch['trajectories']
    assert len({t['root'] for t in trajectories})==1
    unique={}
    for t in trajectories:
        key=tuple((tuple(s['candidate_keys']),s['action']) for s in t['steps'])
        if key in unique:
            assert abs(unique[key]['R']-t['R'])<1e-8, 'DUPLICATE_TRAJECTORY_RETURN_MISMATCH'
        else:unique[key]=t
    trajectories=list(unique.values())
    returns=[t['R'] for t in trajectories]
    keep=near_optimal(returns,epsilon,positive_evidence=max(returns)>epsilon)
    opt=torch.optim.AdamW(policy.parameters(),lr=lr)
    history=[]
    for _ in range(epochs):
        lps=[]
        for t,ok in zip(trajectories,keep):
            if ok:
                lps.append(torch.stack([policy.log_probs(s['classes'],s['actions'],s['prior'])[s['action']] for s in t['steps']]).sum())
        loss=-torch.logsumexp(torch.stack(lps),0)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1.);opt.step();history.append(float(loss.detach()))
    return dict(updates=len(history),losses=history,label_unit='same-root complete trajectory',feedback_trajectories=len(trajectories))


def shuffle_rewards(batch,seed):
    result=copy.deepcopy(batch)
    rng=torch.Generator().manual_seed(seed)
    order=torch.randperm(len(result['trajectories']),generator=rng).tolist()
    rewards=[t['R'] for t in result['trajectories']]
    for t,j in zip(result['trajectories'],order):t['R']=rewards[j]
    assert result['zero']['R']==0
    result['shuffle']={'permutation':order,'fixed_points':sum(i==j for i,j in enumerate(order)),'allows_identity':True}
    return result

