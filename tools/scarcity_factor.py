"""Fixed scarcity controls with a shared half-base competition floor."""
import math

import torch

from decision_probability import choose


KEYS = ['zero', 'full_plus', 'full_minus', 'row_plus', 'row_minus', 'within_plus', 'within_minus']
FAMILIES = {name: [0, KEYS.index(name+'_plus'), KEYS.index(name+'_minus')] for name in ('full', 'row', 'within')}


def allocations(base, scarcity):
    assert base.shape == scarcity.shape and torch.isfinite(scarcity).all()
    mass = base.sum(1, keepdim=True)
    assert (mass > 0).all() and torch.isfinite(base).all() and (base >= 0).all()
    result = {'zero': base.clone()}; support = base > 0
    for sign, coefficient in [('plus', 1.), ('minus', -1.)]:
        tilt = math.log(2.)*(coefficient*scarcity).tanh()
        logits = (base.clamp_min(1e-30).log()+tilt).masked_fill(~support, -torch.inf)
        adaptive = base.sum()*logits.flatten().softmax(0).reshape_as(base)
        row_mass = adaptive.sum(1, keepdim=True)
        pieces = dict(full=adaptive, row=base*row_mass/mass, within=adaptive*mass/row_mass)
        for mode, piece in pieces.items():
            value = .5*base+.5*piece
            assert torch.isfinite(value).all() and torch.allclose(value.sum(), base.sum())
            assert (value >= .5*base-1e-12).all() and torch.equal(value == 0, base == 0)
            result[mode+'_'+sign] = value
        assert torch.allclose(result['within_'+sign].sum(1, keepdim=True), mass)
        assert torch.allclose(result['row_'+sign]/result['row_'+sign].sum(1, keepdim=True), base/mass)
        reconstructed = (2*result['row_'+sign]-base).sum(1,keepdim=True)/mass*(2*result['within_'+sign]-base)
        assert torch.allclose(reconstructed, adaptive)
    return result


def select(probabilities, correctness, labels, identities, old, tail):
    """Deletion stability covers current meta identities only, not old-class uncertainty."""
    k = probabilities.shape[2]; current = list(range(old,k))
    assert correctness.shape == (len(labels),len(KEYS)) and len(identities) == len(labels)
    meta = probabilities.new_zeros((len(KEYS),k))
    groups = {}
    for index, identity in enumerate(identities): groups.setdefault(identity, []).append(index)
    assert all(len(set(labels[i] for i in indices)) == 1 for indices in groups.values())
    totals = {}; counts = {}; support = {}
    for c in current:
        indices = [i for i,label in enumerate(labels) if label == c]
        assert indices
        totals[c] = correctness[indices].double().sum(0); counts[c] = len(indices)
        meta[:,c] = totals[c]/counts[c]
        support[c] = len({identities[i] for i in indices})
    chosen = {}; audits = {}; stability = {}
    meta_list = meta.cpu().tolist()
    deletions = []
    for indices in groups.values():
        c = labels[indices[0]]; remaining = counts[c]-len(indices)
        if not remaining:
            deletions.append((c,None)); continue
        acc = ((totals[c]-correctness[indices].double().sum(0))/remaining).tolist()
        deletions.append((c,[(acc[i]-acc[0])-(meta_list[i][c]-meta_list[0][c]) for i in range(len(KEYS))]))
    group_sizes = dict(all=k,old=old,current=k-old,tail=len(tail))
    for family, candidates in FAMILIES.items():
        point, audit = choose(probabilities,meta,old,tail,candidates)
        matches = 0; histogram = {KEYS[i]: 0 for i in candidates}; unsupported = 0
        for c, delta in deletions:
            if delta is None:
                unsupported += 1; continue
            eligible = {}
            for i in candidates:
                gains = {g:terms['mean_gain']+(delta[i]/group_sizes[g] if g in ('all','current') or g == 'tail' and c in tail else 0.)
                         for g,terms in audit[i]['groups'].items()}
                if gains['all'] > audit[i]['numerical_resolution'] and all(v >= -1e-12 for g,v in gains.items() if g != 'all'):
                    eligible[i] = gains['all']
            winner = min(eligible,key=lambda i:(-eligible[i],i)) if eligible else 0
            histogram[KEYS[winner]] += 1; matches += int(winner == point)
        stable = point if not unsupported and matches == len(groups) else 0
        chosen[family+'_point'] = KEYS[point]; chosen[family+'_stable'] = KEYS[stable]
        audits[family] = {KEYS[i]: v for i,v in audit.items()}
        stability[family] = dict(identity_deletions=len(groups),same_winner=matches,
            unsupported_deletions=unsupported,winner_counts=histogram,
            primary_point=KEYS[point],stable_choice=KEYS[stable],population_confidence_guarantee=False)
    # Anonymous class/identity aggregates suffice to independently reproduce deletions.
    anonymous = [dict(label=labels[indices[0]],images=len(indices),
        correct=correctness[indices].sum(0).tolist()) for indices in groups.values()]
    return chosen, audits, stability, meta, anonymous, support


def self_check():
    torch.set_num_threads(2)
    base = torch.tensor([[0.,.2,.8],[.4,0.,.6],[.7,.3,0.]],dtype=torch.float64)
    counts = torch.tensor([3.,30.,300.],dtype=torch.float64)
    scarcity = (counts.log()[None,:]-counts.log()[:,None]).tanh()
    values = allocations(base,scarcity)
    from edge_competition import allocation
    for sign,coefficient in [('plus',1.),('minus',-1.)]:
        assert torch.allclose(values['full_'+sign],allocation(base,scarcity[:,:,None],torch.tensor([coefficient],dtype=torch.float64)))
    assert not torch.allclose(values['full_plus'].sum(1),base.sum(1))
    assert torch.allclose(allocations(base,torch.zeros_like(base))['within_plus'],base)
    p = torch.full((8,7,2),.5,dtype=torch.float64); p[:,5,0] = .6
    correct = torch.ones((3,7),dtype=torch.bool)
    chosen,_,audit,_,_,_ = select(p,correct,[1,1,1],['a','b','c'],1,[0])
    assert chosen['within_stable'] == 'within_plus' and audit['within']['same_winner'] == 3
    chosen,_,audit,_,_,_ = select(p,correct,[1,1,1],['a','a','a'],1,[0])
    assert chosen['within_stable'] == 'zero' and audit['within']['unsupported_deletions'] == 1
    # Deleting the only beneficial current identity changes the winner to zero.
    p[:,5,0] = .5; correct[:,5] = True; correct[:,0] = torch.tensor([False,True,True])
    chosen,_,audit,_,_,_ = select(p,correct,[1,1,1],['a','b','c'],1,[0])
    assert chosen['within_point'] == 'within_plus' and chosen['within_stable'] == 'zero'
    # The scalar deletion shortcut must agree with explicitly recomputing the selector.
    gen = torch.Generator().manual_seed(43)
    p = torch.rand((8,7,3),generator=gen,dtype=torch.float64)
    correct = torch.rand((9,7),generator=gen)>.4; labels = [1]*4+[2]*5; ids = list(range(9))
    _,_,audit,_,_,_ = select(p,correct,labels,ids,1,[0,2])
    for family,candidates in FAMILIES.items():
        histogram = {KEYS[i]:0 for i in candidates}
        for excluded in ids:
            meta = torch.zeros((7,3),dtype=torch.float64)
            for c in (1,2):meta[:,c] = correct[[i for i in ids if i != excluded and labels[i] == c]].double().mean(0)
            winner,_ = choose(p,meta,1,[0,2],candidates); histogram[KEYS[winner]] += 1
        assert histogram == audit[family]['winner_counts']
