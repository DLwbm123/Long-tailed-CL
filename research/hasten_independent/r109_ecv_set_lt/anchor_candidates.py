"""Existing ZERO/BAL anchors; equal-byte, same-class and view-kind edits."""
import torch
from .legacy_manager import Choice, ValueNet
from .contracts import Candidate, token_keys, validate_candidate


def anchors(state, model, seed):
    state.validate(model)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0); value = ValueNet().to(state.pool.z.device)
    choice = Choice(state.pool, model, state.observed, state.birth, state.stage, state.capacity, value)
    zero, _, _ = choice.construct('RANDOM', generator=torch.Generator(device=state.pool.z.device).manual_seed(seed))
    bal, _, _ = choice.construct('BALANCED')
    return zero, bal


def _edit(state, anchor, priority, steps, seed, random=False, eligible=None):
    pool = state.pool; device = pool.z.device
    keys = pool.pid*8+pool.view
    original = torch.searchsorted(keys, torch.tensor(token_keys(anchor), device=device))
    selected = set(original.tolist()); original_set = set(selected)
    added, removed = set(), set()
    labels=pool.y.tolist(); views=pool.view.tolist(); parents=pool.pid.tolist()
    priorities=priority.tolist()
    allowed=[True]*len(pool) if eligible is None else eligible.tolist()
    protected=set()
    for c in range(len(state.observed)):
        ix=pool.gidx[pool.y[pool.gidx]==c]
        center=torch.nn.functional.normalize(pool.z[ix].mean(0),dim=0)
        protected.add(int(ix[(pool.z[ix]@center).argmax()]))
    buckets={}
    for i in range(len(pool)):
        if i not in original_set:
            buckets.setdefault((labels[i],views[i]>0),[]).append(i)
    for key,items in buckets.items():
        if random:
            order=torch.randperm(len(items),generator=torch.Generator().manual_seed(seed)).tolist()
            buckets[key]=[items[j] for j in order]
        else:
            buckets[key]=sorted(items,key=lambda j:(-priorities[j],j))
    rng = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        pairs = []
        retained_globals={parents[j] for j in selected if views[j]==0}
        retained_locals={parents[j] for j in selected if views[j]>0}
        for old in sorted(selected - added):
            if old in protected:continue
            if not allowed[old]:continue
            # Parent globals with retained locals cannot be singly evicted.
            if views[old] == 0 and parents[old] in retained_locals:
                continue
            for new in buckets.get((labels[old],views[old]>0),[]):
                if new in selected or new in original_set or new in removed:
                    continue
                if not allowed[new]:continue
                rest = selected - {old}
                if views[new] > 0 and parents[new] not in retained_globals:
                    continue
                if state.groups is not None:
                    previous = {state.groups[parents[j]] for j in selected if labels[j] == labels[old]}
                    following = {state.groups[parents[j]] for j in rest | {new} if labels[j] == labels[old]}
                    if len(following) < len(previous):
                        continue
                gain = priorities[new]-priorities[old]
                if random or gain > 1e-12:
                    pairs.append((gain, old, new))
                    # First legal entrant maximizes gain for this removal.
                    break
        if not pairs:
            break
        if random:
            _, old, new = pairs[int(torch.randint(len(pairs), (1,), generator=rng))]
        else:
            _, old, new = max(pairs, key=lambda t: (t[0], -t[1], -t[2]))
        selected.remove(old); selected.add(new); added.add(new); removed.add(old)
    return pool.take(torch.tensor(sorted(selected), device=device)), len(removed)


def generate(state, model, description, seed=109):
    zero, bal = anchors(state, model, seed)
    output = []; seen = set()
    def add(bank, family, anchor, count):
        candidate = Candidate(bank, family, anchor, count)
        validate_candidate(candidate, state, zero if anchor == 'ZERO' else bal)
        if candidate.key not in seen:
            seen.add(candidate.key); output.append(candidate)
    add(zero, 'anchor', 'ZERO', 0); add(bal, 'anchor', 'BAL', 0)
    for ai, (name, anchor) in enumerate([('ZERO', zero), ('BAL', bal)]):
        chosen = torch.isin(state.pool.parents, anchor.parents)
        y = state.pool.y[state.pool.gidx]; z = description['z']
        cover = torch.zeros(len(y), device=z.device)
        for c in range(model.C):
            ix = y == c
            cover[ix] = 1-(z[ix]@z[ix & chosen].T).max(1).values
        priorities = {'risk': description['parent_risk'][state.pool.parent_index],
                      'raw': description['parent_delta'][state.pool.parent_index],
                      'cover': cover[state.pool.parent_index]}
        for j, (family, target) in enumerate([('risk',1),('risk',2),('risk',4),('raw',4),('cover',4),('random',2),('random',4)]):
            limit = min(target, 4, max(1, int(.1*len(anchor))))
            priority = priorities.get(family, torch.zeros(len(state.pool), device=z.device))
            eligible=description['scored'][state.pool.parent_index] if family in {'risk','raw'} else None
            memory, count = _edit(state, anchor, priority, limit, seed+100*ai+j, family=='random',eligible)
            add(memory, family, name, count)
    assert 1 <= len(output) <= 16 and output[0].anchor == 'ZERO'
    return output

