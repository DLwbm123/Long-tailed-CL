"""Frozen deployment choices, independent of reward query labels."""
import torch
from .state_features import describe,action_features
from .anchor_candidates import generate
from .policy import verifier_prior
from .contracts import validate_candidate


def prepare(state,model,opt,seed,counter):
    desc=describe(state,model,opt,counter,teacher=state.teacher,seed=seed)
    candidates=generate(state,model,desc,seed)
    actions=torch.stack([action_features(state.pool,c.memory,candidates[0].memory,state,desc) for c in candidates])
    return candidates,desc['classes'],actions


def select(state,model,opt,seed,counter,verifier=None,policy=None,mode='ZERO',return_column=3):
    candidates,classes,actions=prepare(state,model,opt,seed,counter)
    if mode=='ZERO': index=0
    elif mode=='BALANCED':
        index=next((i for i,c in enumerate(candidates) if c.family=='anchor' and c.anchor=='BAL'),0)
    else:
        assert verifier is not None and not verifier.training and not any(p.requires_grad for p in verifier.parameters()), 'VERIFIER_NOT_FROZEN'
        with torch.no_grad():
            scores,_=verifier(classes,actions)
            if mode in {'DIRECT_RETURN','VERIFIER_GREEDY'}:
                index=int(scores[:,return_column].argmax()) if float(scores[:,return_column].max())>0 else 0
            else:
                q=verifier_prior(scores[:,return_column])
                if mode=='Q_PRIOR': p=q
                else:
                    assert policy is not None and not policy.training and not any(p.requires_grad for p in policy.parameters()), 'POLICY_NOT_FROZEN'
                    p=policy.log_probs(classes,actions,q).exp()
                index=int(torch.multinomial(p,1,generator=torch.Generator(device=p.device).manual_seed(seed)))
    validate_candidate(candidates[index],state)
    return candidates[index].memory,dict(action=index,family=candidates[index].family,anchor=candidates[index].anchor,candidates=len(candidates),candidate_keys=[c.key for c in candidates])

