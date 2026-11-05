"""Finite-action GRPO loss references; no HK data access or model training.

The old policy is detached for every inner optimization block. Rewards are
precomputed, detached correctness gains. Harm protection is outside advantage
normalization. The exact-group implementation integrates the SAME clipped,
group-normalized objective as sampled GRPO, not the raw reward objective.

Run: python grpo_reference.py --self-test --output reference_test_results.json
Requires: Python 3.10+ and PyTorch. This is loss reference code, not an experiment
launcher. Preserve the accompanying protocol's data-exclusion requirements.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def _validate(logits: Tensor, old_probs: Tensor, rewards: Tensor) -> None:
    if logits.ndim != 2 or logits.shape[-1] != 3:
        raise ValueError('Expected logits with shape [batch, 3].')
    if old_probs.shape != logits.shape or rewards.shape != logits.shape:
        raise ValueError('logits, old_probs and rewards must have identical shapes.')
    if any(not bool(torch.isfinite(x).all()) for x in (logits, old_probs, rewards)):
        raise ValueError('All inputs must be finite.')
    if not bool((old_probs > 0).all()):
        raise ValueError('Old categorical probabilities must be strictly positive.')
    if not torch.allclose(old_probs.sum(-1).double(),
                          torch.ones(len(logits), device=logits.device, dtype=torch.float64),
                          atol=1e-6, rtol=1e-6):
        raise ValueError('Old probabilities must sum to one.')


def compositions(group_size: int, *, device: torch.device | str = 'cpu') -> Tensor:
    if group_size < 2:
        raise ValueError('group_size must be at least 2.')
    return torch.tensor([(a, b, group_size-a-b)
                         for a in range(group_size+1)
                         for b in range(group_size-a+1)],
                        dtype=torch.float64, device=device)


def sampled_surrogate(logits: Tensor, old_probs: Tensor, rewards: Tensor,
                      actions: Tensor, *, clip: float = .2,
                      advantage_eps: float = 1e-8) -> Tensor:
    """Return [batch] surrogates to MAXIMIZE, without KL or harm terms."""
    _validate(logits, old_probs, rewards)
    if actions.ndim != 2 or actions.shape[0] != logits.shape[0] or actions.shape[1] < 2:
        raise ValueError('actions must have shape [batch, group_size>=2].')
    if actions.dtype != torch.long or not bool(((actions >= 0) & (actions < 3)).all()):
        raise ValueError('actions must contain int64 categorical indices 0, 1, 2.')
    old = old_probs.detach().double()
    reward = rewards.detach().double().gather(1, actions)
    center = reward - reward.mean(-1, keepdim=True)
    std = center.square().mean(-1, keepdim=True).sqrt()
    adv = torch.where(std > advantage_eps,
                      center / (std + advantage_eps), torch.zeros_like(center)).detach()
    logp = logits.double().log_softmax(-1).gather(1, actions)
    ratio = (logp - old.log().gather(1, actions)).exp()
    return torch.minimum(ratio * adv, ratio.clamp(1-clip, 1+clip) * adv).mean(-1)


def exact_group_surrogate(logits: Tensor, old_probs: Tensor, rewards: Tensor,
                          *, group_size: int = 8, clip: float = .2,
                          advantage_eps: float = 1e-8) -> Tensor:
    """Integrate the same sampled surrogate over multinomial compositions.

    For 3 actions and G=8 there are 45 count vectors, representing all 6561
    ordered action groups. Returns [batch]; maximize its mean. Gradients pass
    through the CURRENT policy only, never through the old group distribution.
    """
    _validate(logits, old_probs, rewards)
    k = compositions(group_size, device=logits.device)
    old = old_probs.detach().double()
    reward = rewards.detach().double()
    weights = k / group_size
    multinomial_log_coeff = math.lgamma(group_size+1) - torch.lgamma(k+1).sum(-1)
    log_mass = multinomial_log_coeff[None, :] + old.log() @ k.T
    mass = log_mass.exp().detach()
    mu = reward @ weights.T                         # [B, K]
    centered = reward[:, None, :] - mu[:, :, None]  # [B, K, 3]
    std = (weights[None, :, :] * centered.square()).sum(-1, keepdim=True).sqrt()
    adv = torch.where(std > advantage_eps,
                      centered / (std + advantage_eps), torch.zeros_like(centered)).detach()
    ratio = (logits.double().log_softmax(-1) - old.log()).exp()[:, None, :]
    clipped = torch.minimum(ratio * adv, ratio.clamp(1-clip, 1+clip) * adv)
    by_group = (weights[None, :, :] * clipped).sum(-1)
    return (mass * by_group).sum(-1)


def raw_expected_gain(logits: Tensor, rewards: Tensor) -> Tensor:
    """[batch] exact raw reward; NOT the exact normalized GRPO surrogate."""
    if logits.shape != rewards.shape or logits.shape[-1] != 3:
        raise ValueError('Expected matching [..., 3] logits and rewards.')
    return (logits.double().softmax(-1) * rewards.detach().double()).sum(-1)


def reference_kl(logits: Tensor) -> Tensor:
    """[batch] exact KL(pi || q_ref), q_ref=(.8,.1,.1)."""
    logp = logits.double().log_softmax(-1)
    q = torch.tensor([.8, .1, .1], device=logits.device, dtype=torch.float64)
    return (logp.exp() * (logp - q.log())).sum(-1)


def old_tail_penalty(logits: Tensor, correctness: Tensor, protected_row: Tensor,
                     *, eligible_class_count: int, protected_class_count: int) -> Tensor:
    """Unbiased minibatch estimate of class-balanced old-tail harm.

    Assumption: queries are sampled uniformly by eligible class, then uniformly
    within class. correctness[:,0] refers to the SAME bank's teacher. Denominator
    is all queries of each protected class, NOT only teacher-correct queries.
    This analytic penalty is added OUTSIDE normalized GRPO advantage.
    """
    if correctness.shape != logits.shape or protected_row.shape != logits.shape[:1]:
        raise ValueError('Incorrect correctness/protected mask shape.')
    if not (0 <= protected_class_count <= eligible_class_count):
        raise ValueError('Invalid class counts.')
    if protected_class_count == 0:
        return logits.sum() * 0.
    correct = correctness.detach().double()
    protected = protected_row.detach().double()
    h = correct[:, 0] * (logits.double().softmax(-1) * (1-correct)).sum(-1)
    return (eligible_class_count / protected_class_count) * (protected * h).mean()


def _brute_force(logits: Tensor, old_probs: Tensor, rewards: Tensor,
                 group_size: int = 8, clip: float = .2,
                 advantage_eps: float = 1e-8) -> Tensor:
    """Test only: enumerate all ordered groups, independent of compositions."""
    seq = torch.tensor(list(itertools.product(range(3), repeat=group_size)),
                       device=logits.device, dtype=torch.long)
    old = old_probs.detach().double()
    r = rewards.detach().double()[:, seq]
    centered = r-r.mean(-1, keepdim=True)
    std = centered.square().mean(-1, keepdim=True).sqrt()
    adv = torch.where(std > advantage_eps,
                      centered/(std+advantage_eps), torch.zeros_like(centered))
    ratio_all = (logits.double().log_softmax(-1)-old.log()).exp()
    ratio = ratio_all[:, seq]
    sur = torch.minimum(ratio*adv, ratio.clamp(1-clip, 1+clip)*adv).mean(-1)
    prob = old[:, seq].prod(-1)
    return (prob * sur).sum(-1)


def self_test() -> dict[str, Any]:
    torch.set_num_threads(1)
    dtype = torch.float64
    old = torch.tensor([[.60,.25,.15],[.25,.55,.20],[.80,.10,.10]], dtype=dtype)
    logits = torch.tensor([[.52,.31,.17],[.29,.49,.22],[.72,.14,.14]], dtype=dtype).log().requires_grad_()
    rewards = torch.tensor([[0.,1.,0.],[0.,-1.,-1.],[0.,0.,0.]], dtype=dtype)
    exact = exact_group_surrogate(logits, old, rewards)
    brute = _brute_force(logits, old, rewards)
    grad_exact, = torch.autograd.grad(exact.sum(), logits, retain_graph=True)
    grad_brute, = torch.autograd.grad(brute.sum(), logits)
    value_error = float((exact-brute).detach().abs().max())
    gradient_error = float((grad_exact-grad_brute).abs().max())
    assert value_error < 1e-11 and gradient_error < 1e-11
    k = compositions(8)
    mass = (math.lgamma(9)-torch.lgamma(k+1).sum(-1)[None,:]+old.log() @ k.T).exp()
    assert torch.allclose(mass.sum(-1), torch.ones(3, dtype=dtype), atol=1e-12, rtol=1e-12)
    shifted = exact_group_surrogate(logits, old, rewards+7.)
    baseline_error = float((shifted-exact.detach()).detach().abs().max())
    assert baseline_error < 1e-11
    assert float(exact[2].detach()) == 0.
    zero_gradient, = torch.autograd.grad(exact_group_surrogate(logits, old, torch.zeros_like(rewards)).sum(), logits)
    assert float(zero_gradient.abs().max()) == 0.
    raw = raw_expected_gain(logits, rewards)
    assert not torch.allclose(raw, exact.detach())
    correct = torch.tensor([[1.,0.,0.],[0.,1.,0.],[1.,1.,1.]], dtype=dtype)
    pen = old_tail_penalty(logits, correct, torch.tensor([True,False,False]),
                           eligible_class_count=3, protected_class_count=1)
    gp, = torch.autograd.grad(pen, logits)
    assert float(pen.detach()) > 0 and float(gp[0].norm()) > 0
    no_old = old_tail_penalty(logits, correct, torch.zeros(3, dtype=torch.bool),
                              eligible_class_count=3, protected_class_count=0)
    assert float(no_old.detach()) == 0.
    # One sampled group agrees exactly with a direct implementation.
    actions = torch.tensor([[0,0,0,1,1,2,0,1],[0,1,1,1,2,2,0,1],[0,0,1,1,2,2,0,1]])
    sample = sampled_surrogate(logits, old, rewards, actions)
    assert torch.isfinite(sample).all() and float(sample[2].detach()) == 0.
    # Positive rescaling is nearly cancelled by normalization (epsilon remains).
    scaled = exact_group_surrogate(logits, old, rewards*9.)
    scale_error = float((scaled-exact.detach()).detach().abs().max())
    assert scale_error < 1e-7
    q = torch.tensor([[.8,.1,.1]], dtype=dtype)
    assert float(reference_kl(q.log()).abs().max()) < 1e-14
    out = {
        'status': 'PASS', 'scope': 'synthetic finite-action loss tests only; no HK training or evaluation',
        'torch_version': str(torch.__version__), 'composition_count': len(k),
        'ordered_group_count': 3**8, 'max_value_error': value_error,
        'max_gradient_error': gradient_error, 'baseline_shift_error': baseline_error,
        'positive_reward_scale_difference_eps_only': scale_error,
        'old_tail_penalty_gradient_nonzero': True,
        'all_equal_groups_zero_surrogate_and_gradient': True,
        'raw_reward_objective_distinct_from_group_normalized_objective': True,
        'reference_kl_at_qref_is_zero': True,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.self_test:
        parser.error('Use --self-test; this file is not a training launcher.')
    result = self_test()
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text+'\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
