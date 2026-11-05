"""Matched objectives: raw exact and normalized-group exact are distinct."""
import torch
from grpo_reference import sampled_surrogate,exact_group_surrogate,raw_expected_gain,reference_kl,old_tail_penalty

def regularizer(logits,correct,protected,C,OT):
    return old_tail_penalty(logits,correct,protected,eligible_class_count=C,protected_class_count=OT)+.01*reference_kl(logits).mean()

def objective(arm,logits,old,correct,protected,C,OT,actions=None):
    r=correct.double()-correct[:,0:1].double()
    if arm=='SUP_CF':
        target=r.argmax(-1);base=torch.nn.functional.cross_entropy(logits.double(),target)
    elif arm=='RAW_EXACT_CF':base=-raw_expected_gain(logits,r).mean()
    elif arm=='GROUP_EXACT_CF':base=-exact_group_surrogate(logits,old,r).mean()
    else:base=-sampled_surrogate(logits,old,r,actions).mean()
    return base+regularizer(logits,correct,protected,C,OT)
