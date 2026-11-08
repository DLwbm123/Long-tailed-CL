"""S-only actions, closed-loop rollouts, FIT-only group-relative updates."""
import copy
import importlib
import torch

E = importlib.import_module('r112_evidence_memory.evidence')
old = importlib.import_module('r112_evidence_memory.roots')
features = importlib.import_module('r109_ecv_set_lt.state_features')
Policy = importlib.import_module('r109_ecv_set_lt.policy').ResidualPolicy


def prepare(pool, previous, model, opt, stage, seed, counter, teacher=None):
    """Same R112 candidate family; reuse one S-only description for the actor."""
    zero, balanced, ledger = E.anchors(pool, seed)
    state = old.visible(pool, previous, stage, teacher)
    desc = features.describe(state, model, opt, counter, teacher=teacher, seed=seed)
    out, seen = [], set()
    core = set(pool.core_keys().tolist())

    def add(bank, family, name, anchor):
        changed = len(set(old.contracts.token_keys(anchor)) - set(old.contracts.token_keys(bank)))
        candidate = old.contracts.Candidate(bank, family, name, changed)
        old.contracts.validate_candidate(candidate, state, anchor)
        assert core <= set(old.contracts.token_keys(bank))
        pool.with_support(bank)
        if candidate.key not in seen:
            out.append(candidate)
            seen.add(candidate.key)

    add(zero, 'anchor', 'E-ZERO', zero)
    add(balanced, 'anchor', 'E-BAL', balanced)
    S, z = state.pool, desc['z']
    y = S.y[S.gidx]
    for ai, (name, anchor) in enumerate([('E-ZERO', zero), ('E-BAL', balanced)]):
        selected = torch.isin(S.parents, anchor.parents)
        cover = torch.zeros(len(y), device=S.z.device)
        for c in range(model.C):
            mask = y == c
            cover[mask] = 1 - (z[mask] @ z[mask & selected].T).max(1).values
        for j, (family, limit) in enumerate([('risk', 1), ('risk', 2), ('risk', 4), ('cover', 4), ('random', 2)]):
            priority = (desc['parent_risk'] if family == 'risk' else cover if family == 'cover'
                        else torch.zeros_like(cover))[S.parent_index]
            eligible = ~torch.isin(S.pid * 8 + S.view, pool.core_keys())
            if family == 'risk':
                eligible &= desc['scored'][S.parent_index]
            bank, _ = old.old_edit(state, anchor, priority, min(limit, max(1, len(anchor) // 10)),
                                   seed + 100 * ai + j, family == 'random', eligible)
            add(bank, family, name, anchor)
    actions = torch.stack([features.action_features(S, c.memory, zero, state, desc) for c in out])
    classes = desc['classes']
    assert len(out) <= 12 and torch.isfinite(actions).all() and torch.isfinite(classes).all()
    prior = torch.full((len(out),), 1 / len(out), device=classes.device)
    return out, classes, actions, prior, ledger


def state_record(policy, packed, choice=None, generator=None):
    cc, classes, actions, prior, _ = packed
    with torch.no_grad():
        lp = policy.log_probs(classes, actions, prior)
        i = int(torch.multinomial(lp.exp(), 1, generator=generator)) if choice is None else choice
    return i, {'classes': classes.detach(), 'actions': actions.detach(), 'prior': prior.detach(),
               'old_logp': lp.detach(), 'choice': i}


def rollout(pool, future, model, opt, teacher, policy, packed, seed, draw_seed,
            stage, counter, zero=False, pre=200, post=100):
    """Each later action uses this trajectory's retained memory and weights."""
    m, o = old.learner.clone_pair(model, opt, seed)
    previous = None
    steps, endpoints, allocations = [], [], []
    generator = torch.Generator(device=pool.bank.z.device).manual_seed(draw_seed)
    for h in range(3):
        if h:
            previous = kept
            data = future[h - 1]
            pool = E.join(kept, data)
            teacher = old.learner.frozen(m)
            m.arrive(data.role(0), pool.C)
            old.branch.paired_learn(m, o, pool.role(0), pre,
                                   old.branch.event_seed(seed, f'{h}:pre'), counter, teacher, 64)
            packed = prepare(pool, previous, m, o, stage + h,
                             old.branch.event_seed(seed, f'{h}:candidate') % (2**31), counter, teacher)
        i, rec = state_record(policy, packed, 0 if zero else None, generator)
        kept = pool.with_support(packed[0][i].memory)
        steps.append(rec)
        old.branch.paired_learn(m, o, kept.role(0), post,
                               old.branch.event_seed(seed, f'{h}:post'), counter, teacher, 64)
        # AUDIT is deliberately absent from every policy-training rollout.
        endpoints.append(old.paired_labels.evaluate(m, kept.role(1), counter))
        allocations.append({'C': kept.C, 'tokens': len(kept.bank), 'bytes': len(kept.bank) * 2064 + 184,
                            'S_by_class': torch.bincount(kept.role(0).y, minlength=kept.C).tolist()})
    return {'steps': steps, 'endpoints': endpoints, 'allocations': allocations}


def collect(pool, future, model, opt, teacher, policy, packed, tail, seed, stage,
            counter, group=4, pre=200, post=100):
    behavior = copy.deepcopy(policy).eval().requires_grad_(False)
    zero = rollout(pool, future, model, opt, teacher, behavior, packed, seed, seed + 9000,
                   stage, counter, True, pre, post)
    runs = []
    for g in range(group):
        run = rollout(pool, future, model, opt, teacher, behavior, packed, seed,
                      seed + 10000 + g, stage, counter, False, pre, post)
        points = [old.paired_labels.paired(a, b, tail) for a, b in zip(run['endpoints'], zero['endpoints'])]
        run['R'] = old.paired_labels.utility(points)['R']
        run['immediate'] = [p['macro'] - 2 * max(0, -p['tail'] - .005)
                            if p['tail'] is not None else p['macro'] for p in points]
        run['points'] = points
        runs.append(run)
    zero.update(R=0., immediate=[0., 0., 0.])
    return {'runs': runs, 'zero': zero, 'behavior': behavior}


def advantages(batch, immediate):
    values = torch.tensor([r['immediate'] if immediate else [r['R']] * 3 for r in batch['runs']])
    center = values.mean(0, keepdim=True)
    return (values - center) / (values.std(0, unbiased=False, keepdim=True) + 1e-6)


def update(policy, reference, optimizer, batch, immediate=False, epochs=2):
    adv = advantages(batch, immediate)
    history = []
    for _ in range(epochs):
        losses, kls = [], []
        for g, run in enumerate(batch['runs']):
            for h, s in enumerate(run['steps']):
                lp = policy.log_probs(s['classes'], s['actions'], s['prior'])
                with torch.no_grad():
                    check = batch['behavior'].log_probs(s['classes'], s['actions'], s['prior'])
                    assert torch.allclose(check, s['old_logp'], atol=1e-6)
                    ref = reference.log_probs(s['classes'], s['actions'], s['prior'])
                ratio = (lp[s['choice']] - s['old_logp'][s['choice']]).exp()
                a = adv[g, h].to(lp.device)
                kl = (lp.exp() * (lp - ref)).sum()
                losses.append(-torch.minimum(ratio * a, ratio.clamp(.8, 1.2) * a) + .01 * kl)
                kls.append(kl.detach())
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
        assert torch.isfinite(loss) and torch.isfinite(norm)
        optimizer.step()
        history.append({'loss': float(loss.detach()), 'KL': float(torch.stack(kls).mean()),
                        'grad_norm': float(norm), 'reward_std': float(torch.tensor([r['R'] for r in batch['runs']]).std(unbiased=False))})
    return history


def supervise(policy, optimizer, batch, epochs=2):
    """Exactly the LONG group's feedback; imitate its best complete trajectory."""
    runs = [batch['zero']] + batch['runs']
    best = max(range(len(runs)), key=lambda i: (runs[i]['R'], -i))
    if runs[best]['R'] <= 0:
        best = 0
    history = []
    for _ in range(epochs):
        loss = -torch.stack([policy.log_probs(s['classes'], s['actions'], s['prior'])[s['choice']]
                             for s in runs[best]['steps']]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
        assert torch.isfinite(loss) and torch.isfinite(norm)
        optimizer.step()
        history.append({'loss': float(loss.detach()), 'chosen_trajectory': best,
                        'label_R': runs[best]['R'], 'grad_norm': float(norm)})
    return history


def receipt(batch):
    return {'group_returns': [r['R'] for r in batch['runs']],
            'steps': [[{'action': s['choice'], 'candidate_count': len(s['prior']),
                        'chosen_probability': float(s['old_logp'][s['choice']].exp())}
                       for s in r['steps']] for r in batch['runs']],
            'allocations': [r['allocations'] for r in batch['runs']],
            'FIT_points': [r['points'] for r in batch['runs']],
            'AUDIT_training_reads': 0}
