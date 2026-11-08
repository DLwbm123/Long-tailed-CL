"""Post-completion synthetic audit; never modifies the frozen R113 updater."""
import copy
import csv
import json
import os
import time
from pathlib import Path
import torch
from . import trajectory as T


def norm(grads):
    return float(sum(g.double().square().sum() for g in grads).sqrt())


def fixture(dtype, C, N, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        policy = T.Policy().to(dtype=dtype)
        rng = torch.Generator().manual_seed(seed + 1)
        packed = [(torch.randn(C - 2 + h, 14, generator=rng, dtype=torch.float32).to(dtype),
                   torch.randn(N, C - 2 + h, 10, generator=rng, dtype=torch.float32).to(dtype),
                   torch.full((N,), 1 / N, dtype=dtype)) for h in range(3)]
    reference = copy.deepcopy(policy).requires_grad_(False)
    behavior = copy.deepcopy(policy).requires_grad_(False)
    runs = []
    for g in range(4):
        steps = []
        for h, (classes, actions, prior) in enumerate(packed):
            with torch.no_grad():
                lp = behavior.log_probs(classes, actions, prior)
            steps.append({'classes': classes, 'actions': actions, 'prior': prior,
                          'old_logp': lp, 'choice': (g + h) % N})
        runs.append({'steps': steps, 'R': 0., 'immediate': [0., 0., 0.]})
    return policy, reference, {'runs': runs, 'behavior': behavior}, packed


def run(output):
    start = time.time()
    torch.set_num_threads(1)
    rows = []
    for dtype in [torch.float32, torch.float64]:
        for C in [5, 10, 23]:
            for N in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                for seed in [113000, 113001]:
                    policy, reference, batch, packed = fixture(dtype, C, N, seed)
                    opt = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=.001)
                    params = list(policy.parameters())
                    pgs, kls = [], []
                    for trace in batch['runs']:
                        for s in trace['steps']:
                            lp = policy.log_probs(s['classes'], s['actions'], s['prior'])
                            with torch.no_grad():
                                ref = reference.log_probs(s['classes'], s['actions'], s['prior'])
                            ratio = (lp[s['choice']] - s['old_logp'][s['choice']]).exp()
                            pgs.append(-torch.minimum(ratio * 0., ratio.clamp(.8, 1.2) * 0.))
                            kls.append(.01 * (lp.exp() * (lp - ref)).sum())
                    pg = torch.stack(pgs).mean()
                    kl = torch.stack(kls).mean()
                    pg_grad = norm(torch.autograd.grad(pg, params, retain_graph=True))
                    kl_grad = norm(torch.autograd.grad(kl, params))
                    assert pg_grad == 0 and not bool(T.advantages(batch, False).any())
                    before = {k: v.clone() for k, v in policy.state_dict().items()}
                    with torch.no_grad():
                        initial = policy.log_probs(*packed[0]).exp()
                    assert int(initial.argmax()) == 0
                    # Two calls retain AdamW state and match the frozen two-epoch update.
                    for epoch in [1, 2]:
                        history = T.update(policy, reference, opt, batch, epochs=1)
                        with torch.no_grad():
                            probs = policy.log_probs(*packed[0]).exp()
                        rows.append({'dtype': str(dtype), 'classes_terminal': C, 'candidates': N,
                                     'fixture_seed': seed, 'epoch': epoch, 'all_rewards': 0.,
                                     'policy_gradient_norm_at_initialization': pg_grad,
                                     'weighted_KL_gradient_norm_at_initialization': kl_grad,
                                     'optimizer_gradient_norm': history[0]['grad_norm'],
                                     'KL_value': history[0]['KL'],
                                     'max_parameter_change': max(float((v - before[k]).abs().max())
                                                                 for k, v in policy.state_dict().items()),
                                     'output_weight_change': float((policy.head[-1].weight - before['head.2.weight']).abs().max()),
                                     'max_probability_change': float((probs - initial).abs().max()),
                                     'probability_span': float(probs.max() - probs.min()),
                                     'best_minus_ZERO_probability': float(probs.max() - probs[0]),
                                     'argmax': int(probs.argmax())})
                    # Decay alone can move encoder weights but must preserve uniform behavior.
                    control, _, _, _ = fixture(dtype, C, N, seed)
                    oo = torch.optim.AdamW(control.parameters(), lr=1e-4, weight_decay=.001)
                    for p in control.parameters():
                        p.grad = torch.zeros_like(p)
                    oo.step()
                    with torch.no_grad():
                        cp = control.log_probs(*packed[0]).exp()
                    assert int(cp.argmax()) == 0 and bool((cp == cp[0]).all())
    # A trained off-reference actor may have a legitimate KL-only update.
    policy, reference, batch, packed = fixture(torch.float64, 5, 12, 113002)
    with torch.no_grad():
        policy.head[-1].weight.fill_(.01)
    terms = []
    for classes, actions, prior in packed:
        lp = policy.log_probs(classes, actions, prior)
        with torch.no_grad():
            ref = reference.log_probs(classes, actions, prior)
        terms.append((lp.exp() * (lp - ref)).sum())
    trained_kl_gradient = norm(torch.autograd.grad(torch.stack(terms).mean(), list(policy.parameters())))
    assert trained_kl_gradient > 1e-8
    base = Path(output)
    with base.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'synthetic_only': True, 'post_hoc_audit': True, 'real_images': 0,
               'real_classifier_updates': 0, 'AUDIT_queries': 0,
               'frozen_training_code_modified': False, 'fixtures': len(rows) // 2, 'optimizer_steps': len(rows),
               'frozen_updater_optimizer_steps': len(rows), 'decay_control_optimizer_steps': len(rows) // 2,
               'total_synthetic_optimizer_steps': 3 * len(rows) // 2,
               'after_update_observations_per_dtype': {str(d): sum(r['dtype'] == str(d) for r in rows)
                                                      for d in [torch.float32, torch.float64]},
               'zero_reward_nonZERO_argmax_cases': {str(d): sum(r['argmax'] != 0 for r in rows if r['dtype'] == str(d))
                                                    for d in [torch.float32, torch.float64]},
               'maximum_probability_span': {str(d): max(r['probability_span'] for r in rows if r['dtype'] == str(d))
                                             for d in [torch.float32, torch.float64]},
               'maximum_initial_policy_gradient': max(r['policy_gradient_norm_at_initialization'] for r in rows),
               'trained_off_reference_KL_gradient': trained_kl_gradient,
               'decay_only_uniform_behavior_checks_passed': True,
               'actual_historical_cause_reconstruction': 'NA; deployment vectors and historical descriptors were not saved',
               'wall_seconds': time.time() - start}
    base.with_suffix('.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    run(os.environ['Q_OUTPUT'])
