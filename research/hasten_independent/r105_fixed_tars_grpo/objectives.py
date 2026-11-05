"""Matched per-query objectives; TARS rescales the surrogate after normalization."""
import torch
import torch.nn.functional as F
from grpo_reference import sampled_surrogate,exact_group_surrogate,raw_expected_gain,reference_kl

def tailness(counts,correct):
    counts=counts.double();correct=correct.double();assert (counts>0).all() and ((correct>=0)&(correct<=counts)).all()
    raw=(counts.mean()/counts*(1-correct/counts)).pow(.75)
    return (raw/torch.quantile(raw,.5).clamp_min(1e-8)).clamp_min(1).detach()

def objective(arm,logits,old,correct,weights,actions=None):
    weighted=arm.startswith('TARS_');kind=arm.removeprefix('TARS_');reward=correct.double()-correct[:,0:1].double()
    if kind=='SUP':per=F.cross_entropy(logits.double(),reward.argmax(-1),reduction='none')
    elif kind=='RAW_EXACT':per=-raw_expected_gain(logits,reward)
    elif kind=='GROUP_EXACT':per=-exact_group_surrogate(logits,old,reward)
    elif kind=='GRPO':per=-sampled_surrogate(logits,old,reward,actions)
    else:raise ValueError(arm)
    assert weights.shape==per.shape and not weights.requires_grad
    return ((weights if weighted else torch.ones_like(weights))*per).mean()+.01*reference_kl(logits).mean()
