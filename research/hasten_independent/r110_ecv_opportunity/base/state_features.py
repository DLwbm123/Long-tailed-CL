"""Class-equivariant, query-free state with explicit missing evidence."""
import hashlib
import torch
import torch.nn.functional as F
from .learner import clone_pair, learn
from .risk_transport import geometry, transport

CLASS_DIM = 14
ACTION_DIM = 10


def split_half(z, group_ids=None, verified=False, repeats=4, seed=109):
    missing = dict(value=0., spread=0., available=False, n=0, chi=None)
    if group_ids is None or not verified:
        return missing
    units = sorted(set(group_ids))
    missing['n'] = len(units)
    if len(units) < 5:
        return missing
    x = torch.stack([z[torch.tensor([g == u for g in group_ids], device=z.device)].mean(0) for u in units])
    n = len(x); a = n // 2; b = n - a
    values = []
    for k in range(repeats):
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed+k)).to(z.device)
        left, right = x[order[:a]].mean(0), x[order[a:]].mean(0)
        if min(float(left.norm()), float(right.norm())) < 1e-10:
            return missing
        values.append((2*a*b/n**2) * (1-F.cosine_similarity(left[None], right[None]).item()))
    return dict(value=sum(values)/repeats, spread=float(torch.tensor(values).std(unbiased=False)), available=True, n=n, chi=2*a*b/n**2)


def _hash_rank(pid, seed):
    return hashlib.blake2b(f'{seed}:{pid}'.encode(), digest_size=8).digest()


@torch.no_grad()
def mapped(model, bank):
    x, y, ids = bank.bags()
    z = F.normalize(x[:, :512] + model.U(F.gelu(model.V(x))), dim=-1)
    return x, y, ids, z


def describe(state, model, opt, counter, teacher=None, seed=109, score_cap=128, sigma_floor=.02):
    state.validate(model)
    assert score_cap >= model.C, 'ROTATING_SCORE_COVERAGE_NOT_IMPLEMENTED_FOR_C_GT_CAP'
    pool = state.pool
    x, y, ids, z = mapped(model, pool)
    # Round-robin class coverage, fixed identity-key sampling, no duplication.
    unit_members=[];unit_labels=[];lists=[]
    for c in range(model.C):
        grouped={}
        for j in torch.where(y==c)[0].tolist():
            key=state.groups[int(ids[j])] if state.groups is not None else int(ids[j])
            grouped.setdefault(key,[]).append(j)
        items=[]
        for key in sorted(grouped,key=lambda u:_hash_rank(u,seed)):
            items.append(len(unit_members));unit_members.append(grouped[key]);unit_labels.append(c)
        lists.append(items)
    indices = []
    for j in range(max(map(len, lists))):
        for items in lists:
            if j < len(items) and len(indices) < score_cap:
                indices.append(items[j])
    unit_index=torch.tensor(indices,device=z.device)
    unit_y=torch.tensor(unit_labels,device=z.device)
    virtual, vo = clone_pair(model, opt, seed)
    learn(virtual, vo, pool, 1, seed, counter, teacher, 64, 'engineering_virtual')
    with torch.no_grad():
        all_logits = model(x)
        ce = F.cross_entropy(all_logits, y, reduction='none')
        parent_change=F.cross_entropy(virtual(x),y,reduction='none')-ce
        delta=torch.stack([parent_change[unit_members[u]].mean() for u in indices]).detach()
        unit_z=F.normalize(torch.stack([z[unit_members[u]].mean(0) for u in indices]),dim=-1)
        probs = all_logits.softmax(-1)
        other = all_logits.clone(); other.scatter_(1, y[:, None], -1e6)
        margin = all_logits.gather(1, y[:, None]).squeeze(1)-other.max(1).values
        if model.C == 1:
            margin.zero_()
        brier = (probs-F.one_hot(y, model.C)).square().sum(1)
    parent_risk = torch.zeros(len(ids), device=z.device)
    parent_delta = torch.zeros_like(parent_risk)
    scored = torch.zeros(len(ids), dtype=torch.bool, device=z.device)
    tokens = []
    diagnostics = []
    for c in range(model.C):
        ci = torch.where(y == c)[0]; sel = torch.where(unit_y[unit_index] == c)[0]
        zz = unit_z[sel]
        C, S = geometry(zz)
        tr = transport(delta[sel], C, S, sigma_floor=sigma_floor)
        for j,selected_position in enumerate(sel.tolist()):
            members=unit_members[indices[selected_position]]
            parent_risk[members]=tr['p'][j];parent_delta[members]=delta[selected_position];scored[members]=True
        groups = [state.groups[int(ids[j])] for j in ci] if state.groups is not None else None
        nu = split_half(z[ci], groups, state.independent_groups_verified)
        mem = state.memory
        mc = 0 if mem is None else int((mem.y == c).sum())
        vals = [__import__('math').log1p(int((y == c).sum()))/9,
                __import__('math').log1p(nu['n'])/9, mc/state.capacity,
                (state.stage-state.birth[c])/4, float(state.birth[c] == state.stage),
                float(ce[ci].mean()), float(margin[ci].mean())/40,
                float(brier[ci].mean()), nu['value'], nu['spread'], float(nu['available']),
                float(delta[sel].mean()), float(tr['w'].sum()), len(sel)/len(lists[c])]
        tokens.append(vals); diagnostics.append({'class_position': c, 'noise': nu, 'scored_units': len(sel), 'available_units': len(lists[c])})
    return dict(classes=torch.tensor(tokens, dtype=x.dtype, device=x.device).detach(), z=z.detach(),
                parent_risk=parent_risk, parent_delta=parent_delta, scored=scored, diagnostics=diagnostics)


def action_features(pool, memory, zero, state, description):
    pids = pool.parents
    selected = torch.isin(pids, memory.parents)
    base = torch.isin(pids, zero.parents)
    akeys = set((memory.pid*8+memory.view).tolist()); zkeys = set((zero.pid*8+zero.view).tolist())
    z = description['z']; C = len(state.observed)
    coverage = []
    redundancy = []
    for c in range(C):
        allc = pool.y[pool.gidx] == c; chosen = allc & selected
        similarity = z[allc] @ z[chosen].T
        coverage.append(similarity.max(1).values.mean())
        zz = z[chosen]
        redundancy.append((zz@zz.T)[~torch.eye(len(zz), device=z.device, dtype=torch.bool)].mean() if len(zz)>1 else z.new_tensor(0.))
    rows=[]
    for c in range(C):
        pc=pool.y[pool.gidx]==c;sc=pc&selected;bc=pc&base
        ck=set((pool.pid[pool.y==c]*8+pool.view[pool.y==c]).tolist())
        vals=[int((memory.y==c).sum())/state.capacity,int(sc.sum())/max(1,int((memory.y==c).sum())),
              len((akeys-zkeys)&ck)/max(1,len(memory)),len((zkeys-akeys)&ck)/max(1,len(zero)),
              float(description['parent_risk'][sc].sum()),float(description['parent_delta'][sc].mean()),
              float(coverage[c]),float(redundancy[c]),int(sc.sum())/max(1,int(pc.sum())),
              int((sc&bc).sum())/max(1,int(bc.sum()))]
        rows.append(vals)
    return torch.tensor(rows,device=z.device,dtype=z.dtype).detach()

