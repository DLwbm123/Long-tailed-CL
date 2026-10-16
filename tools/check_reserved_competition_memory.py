"""Frozen synthetic 2x2 test: slot assignment x response/gradient memory."""
import math
import os
import time

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from check_lowrank_noise_memory import lowrank_fixture
from check_progressive_decision_memory import DT, moments_objective, project_ball

MODES = ('fixed_moment', 'fixed_decision', 'assigned_moment', 'assigned_decision')
BETA = 8.


def initial_model(d, seed):
    g = torch.Generator().manual_seed(seed + 2100)
    return (torch.randn(d, 8, generator=g, dtype=DT) / math.sqrt(8), torch.zeros(d, 8, dtype=DT))


def clone_model(model, grad=False):
    return tuple(p.detach().clone().requires_grad_(grad) for p in model)


def features(x, model):
    u, v = model
    raw = x + torch.tanh(3 * (x @ u)) @ v.T
    assert bool(torch.isfinite(raw).all()) and float(raw.detach().norm(dim=1).min()) > 1e-12
    return F.normalize(raw, dim=1)


def frame(d, k, seed):
    assert 2 <= k <= d
    g = torch.Generator().manual_seed(seed + 2000)
    basis = torch.linalg.qr(torch.randn(d, k, generator=g, dtype=DT), mode='reduced').Q
    return math.sqrt(k / (k - 1)) * (basis - basis.mean(1, keepdim=True))


def scores(x, model, q):
    return BETA * features(x, model) @ q


def response(x, label, model, q, slots):
    s = scores(x, model, q)
    active = [p for c, p in enumerate(slots) if c != label]
    reserve = [p for p in range(q.shape[1]) if p not in slots]
    correct = s[:, slots[label]]
    # An absent competitor set contributes a constant masked coordinate, not an infinite margin.
    return torch.stack([correct - s[:, active].max(1).values if active else correct * 0,
                        correct - s[:, reserve].max(1).values if reserve else correct * 0], 1)


def projections(d):
    assert d in (2, 4)
    h = torch.tensor([[1., 1.], [1., -1.]], dtype=DT)
    if d == 4:
        h = torch.kron(h, h)
    return torch.cat((torch.eye(d, dtype=DT), h / math.sqrt(d)))


def wasserstein_1d(a, b, weights):
    """Exact scalar empirical W1; gradients through moving atoms and probability breakpoints."""
    av = a.sort().values
    bv, order = b.sort()
    bw = weights[order]
    ac = torch.arange(1, len(a) + 1, dtype=DT) / len(a)
    bc = torch.cat((bw.cumsum(0)[:-1], bw.new_ones(1)))
    edges = torch.cat((ac.new_zeros(1), ac, bc)).sort().values
    mid = ((edges[:-1] + edges[1:]) / 2).detach()
    ai = torch.searchsorted(ac, mid).clamp_max(len(a) - 1)
    bi = torch.searchsorted(bc.detach(), mid).clamp_max(len(b) - 1)
    return ((edges[1:] - edges[:-1]) * (av[ai] - bv[bi]).abs()).sum()


def sliced(a, b, weights):
    p = projections(a.shape[1])
    aa, bb = a @ p.T, b @ p.T
    return torch.stack([wasserstein_1d(aa[:, j], bb[:, j], weights) for j in range(len(p))]).mean()


def ce(x, label, model, q, slots, weights=None):
    logp = scores(x, model, q)[:, slots].log_softmax(1)[:, label]
    return -logp.mean() if weights is None else -(logp * weights).sum()


def risk_gradient(x, label, model, q, slots, weights=None, create_graph=False):
    value = ce(x, label, model, q, slots, weights)
    grad = torch.autograd.grad(value, model, create_graph=create_graph)
    return value, torch.cat([p.reshape(-1) for p in grad])


def assign(current, old, model, q, previous, adaptive):
    available = [j for j in range(q.shape[1]) if j not in previous]
    labels = sorted(current)
    if not adaptive:
        return previous + available[:len(labels)]
    with torch.no_grad():
        collision = torch.zeros(len(available), dtype=DT)
        for c, (z, p) in old.items():
            s = scores(z, model, q)
            collision += (p[:, None] * F.softplus(s[:, available] - s[:, previous[c], None])).sum(0) / len(old)
        cost = torch.stack([-scores(current[c], model, q)[:, available].mean(0) + collision for c in labels])
    rows, columns = linear_sum_assignment(cost.numpy())
    assert list(rows) == list(range(len(labels)))
    return previous + [available[int(j)] for j in columns]


def training_batch(current, old):
    assert not set(current).intersection(old)
    seen = len(current) + len(old)
    assert set(current) | set(old) == set(range(seen))
    xs, ys, weights = [], [], []
    for c in range(seen):
        x, p = (current[c], None) if c in current else old[c]
        xs.append(x)
        ys.append(torch.full((len(x),), c, dtype=torch.long))
        weights.append(torch.full((len(x),), 1 / len(x), dtype=DT) / seen if p is None else p / seen)
    return torch.cat(xs), torch.cat(ys), torch.cat(weights)


def train(model, current, old, q, slots, steps=48):
    start = time.monotonic()
    x, y, p = training_batch(current, old)
    params = clone_model(model, True)
    old_n = len(old)
    mask = y < old_n
    with torch.no_grad():
        teacher = scores(x[mask], model, q)[:, slots[:old_n]] if old_n else None
        correct = teacher.argmax(1) == y[mask] if old_n else None
        if old_n:
            teacher -= teacher.mean(1, keepdim=True)
    for _ in range(steps):
        ss = scores(x, params, q)[:, slots]
        classification = -(ss.log_softmax(1)[torch.arange(len(x)), y] * p).sum()
        retain = classification * 0
        if old_n and bool(correct.any()):
            now = ss[mask, :old_n]
            now = now - now.mean(1, keepdim=True)
            # Old-correct only; per-class replay mass remains fixed instead of selecting survivors.
            retain = (((now - teacher).square().mean(1) * correct) * p[mask] * len(slots) / old_n).sum()
        loss = classification + retain
        gg = torch.autograd.grad(loss, params)
        with torch.no_grad():
            for a, b in zip(params, gg):
                a.sub_(b / 16)
                assert bool(torch.isfinite(a).all())
    return clone_model(params), dict(seconds=time.monotonic()-start, updates=steps,
                                     last_preupdate_classification=float(classification),
                                     last_preupdate_retention=float(retain),
                                     last_preupdate_gradient_norm=float(torch.cat([g.flatten() for g in gg]).norm()))


def initialization(current, seed):
    nodes = []
    for c in sorted(current):
        x = current[c]
        g = torch.Generator().manual_seed(seed + 1000 + c)
        # Synthetic initialization, shared across all arms; never deploy selected real feature rows.
        n = x.mean(0) + .1 * torch.randn(16, x.shape[1], generator=g, dtype=DT) / math.sqrt(x.shape[1])
        nodes.append(project_ball(n))
    return torch.stack(nodes), torch.zeros(len(nodes), 16, dtype=DT)


def compress(current, seed, mode, model_start=None, model_end=None, q=None, slots=None):
    start = time.monotonic()
    labels = sorted(current)
    n, w = initialization(current, seed)
    n.requires_grad_(); w.requires_grad_()
    opt = torch.optim.Adam([n, w], lr=.02)
    targets = []
    if mode == 'moment':
        means = torch.stack([current[c].mean(0) for c in labels])
        second = torch.stack([current[c].T @ current[c] / len(current[c]) for c in labels])
    else:
        models = [clone_model(m, True) for m in (model_start, model_end)]
        for c in labels:
            with torch.no_grad():
                resp = torch.cat([response(current[c], c, m, q, slots) for m in models], 1)
            gradients = [risk_gradient(current[c], c, m, q, slots)[1].detach() for m in models]
            targets.append((resp.detach(), gradients))
    first = None
    for step in range(48):
        if mode == 'moment':
            objective = moments_objective(n, w, means, second)
        else:
            objective = n.sum() * 0
            for i, c in enumerate(labels):
                z, p = F.normalize(n[i], dim=1), w[i].softmax(0)
                response_now = torch.cat([response(z, c, m, q, slots) for m in models], 1)
                term = sliced(targets[i][0], response_now, p) / BETA
                for m, reference in zip(models, targets[i][1]):
                    grad = risk_gradient(z, c, m, q, slots, p, True)[1]
                    term = term + .5 * (grad-reference).square().sum() / reference.square().sum().clamp_min(1e-12)
                objective = objective + term / len(labels)
        assert bool(torch.isfinite(objective))
        if first is None:
            first = float(objective.detach())
        opt.zero_grad()
        n.grad, w.grad = torch.autograd.grad(objective, (n, w))
        opt.step()
        with torch.no_grad():
            n.copy_(project_ball(n)); w.clamp_(-8, 8)
            assert float(n.norm(dim=2).min()) > 1e-12
    memory = {c: (F.normalize(n[i].detach(), dim=1), w[i].detach().softmax(0)) for i, c in enumerate(labels)}
    return memory, dict(seconds=time.monotonic()-start, updates=48, method=mode,
                        initial_objective=first, last_preupdate_objective=float(objective.detach()),
                        min_effective_nodes=min(float(1/p.square().sum()) for _, p in memory.values()))


def unit_checks():
    g = torch.Generator().manual_seed(343)
    q = frame(12, 4, 343)
    etf = torch.eye(4, dtype=DT) * 4/3 - torch.ones(4, 4, dtype=DT)/3
    etf_error = float((q.T@q-etf).abs().max()); assert etf_error < 1e-12
    x = F.normalize(torch.randn(9, 12, generator=g, dtype=DT), dim=1)
    model = clone_model(initial_model(12, 343), True)
    slots = [0, 2]
    start = scores(x, model, q)[:, slots]
    expanded = scores(x, model, q)[:, slots+[1]]
    assert torch.equal(start, expanded[:, :2])
    # Independent exact scalar W1 checks and weighted finite difference away from ties.
    a = torch.tensor([-.4, .7, 1.3], dtype=DT)
    b = torch.tensor([-.1, .9], dtype=DT, requires_grad=True)
    w = torch.tensor([.2, -.3], dtype=DT, requires_grad=True)
    v = wasserstein_1d(a, b, w.softmax(0))
    gb, gw = torch.autograd.grad(v, (b,w)); eps=1e-6
    fd = abs(float((wasserstein_1d(a,b.detach(),(w.detach()+torch.tensor([eps,0.])).softmax(0))-wasserstein_1d(a,b.detach(),(w.detach()-torch.tensor([eps,0.])).softmax(0)))/(2*eps)-gw[0]))
    assert fd < 1e-7
    assert abs(float(wasserstein_1d(a,a,torch.ones(3)/3))) < 1e-12
    # Mixed derivative used by gradient matching, not just forward-memory matching.
    z = x[:4].clone().requires_grad_(); pp=torch.ones(4,dtype=DT)/4
    ref = risk_gradient(x, 0, model, q, slots)[1].detach()
    def objective(nodes):
        gg = risk_gradient(nodes, 0, model, q, slots, pp, True)[1]
        return (gg-ref).square().sum()
    val=objective(z); grad=torch.autograd.grad(val,z)[0]
    direction=F.normalize(torch.randn(z.numel(),generator=g,dtype=DT),dim=0).reshape_as(z)
    mixed_fd=abs(float((objective(z.detach()+eps*direction)-objective(z.detach()-eps*direction))/(2*eps)-(grad*direction).sum()))
    assert mixed_fd < 1e-6
    current={0:x[:4],1:x[4:]}
    assigned=assign(current,{},model,q,[],True)
    assert len(set(assigned))==2
    pa,_=train(model,current,{},q,assigned,1)
    pb,_=train(model,current,{},q,assigned,1)
    assert all(torch.equal(u,v) for u,v in zip(pa,pb))
    fake={0:(x[:4],pp)}
    extended=assign({1:x[4:]},fake,model,q,[assigned[0]],True)
    assert extended[0]==assigned[0] and len(set(extended))==2
    rejected=False
    try: training_batch({0:x},fake)
    except AssertionError: rejected=True
    assert rejected
    with torch.no_grad():
        r=response(x,0,model,q,[0,1]); all_margin=scores(x,model,q)[:,0]-scores(x,model,q)[:,1:].max(1).values
        assert float((r.min(1).values-all_margin).abs().max())<1e-12
    return dict(status='PASS',ETF_gram_error=etf_error,old_scores_exact_on_activation=True,
                scalar_W1_weight_FD_error=fd,gradient_matching_mixed_FD_error=mixed_fd,
                no_old_memory_same_training_exact=True,assignment_injective_and_old_prefix_exact=True,
                overlapping_support_rejected=True,reserve_margin_identity=True,synthetic_updates=2)


def run_scale(cfg, lock, journal):
    begun=time.monotonic(); x,y,counts=lowrank_fixture(cfg); d,k=cfg['dimension'],cfg['classes']
    q=frame(d,k,cfg['seed']); q_copy=q.clone(); source_copy=x.clone()
    models={m:initial_model(d,cfg['seed']) for m in MODES}
    memories={m:{} for m in MODES}; slots={m:[] for m in MODES}
    snapshots=[]; training=[]; construction=[]; allocation=[]; seen=0
    for task,add in enumerate(cfg['task_sizes'],1):
        old_seen=seen; seen+=add; current={c:x[y==c] for c in range(old_seen,seen)}
        before_memory={m:{c:(z.clone(),p.clone()) for c,(z,p) in memories[m].items()} for m in MODES}
        starts={}
        for mode in MODES:
            journal(dict(phase='synthetic_nonlinear_train',task=task,mode=mode,completed_stages=len(snapshots)))
            tm=time.monotonic(); previous=slots[mode].copy()
            slots[mode]=assign(current,memories[mode],models[mode],q,previous,mode.startswith('assigned'))
            assert slots[mode][:old_seen]==previous
            allocation.append(dict(task=task,mode=mode,seconds=time.monotonic()-tm,old_slots_unchanged=True,
                                   nondefault_new_assignments=sum(a!=b for a,b in zip(slots[mode][old_seen:],range(old_seen,seen)))))
            starts[mode]=clone_model(models[mode])
            models[mode],cost=train(models[mode],current,memories[mode],q,slots[mode])
            training.append(dict(task=task,mode=mode,**cost))
        if task==1:
            for group in ('fixed','assigned'):
                assert all(torch.equal(a,b) for a,b in zip(models[group+'_moment'],models[group+'_decision']))
        journal(dict(phase='current_memory_compression',task=task))
        baseline,cost=compress(current,cfg['seed'],'moment')
        construction.append(dict(task=task,mode='shared_moment_construction',**cost))
        for mode in MODES:
            if mode.endswith('moment'):
                new=baseline
            else:
                journal(dict(phase='decision_memory_compression',task=task,mode=mode))
                new,cost=compress(current,cfg['seed'],'decision',starts[mode],models[mode],q,slots[mode])
                construction.append(dict(task=task,mode=mode,**cost))
            for c,(z,p) in before_memory[mode].items():
                assert torch.equal(z,memories[mode][c][0]) and torch.equal(p,memories[mode][c][1])
            memories[mode].update(new)
        assert torch.equal(x,source_copy) and torch.equal(q,q_copy)
        snapshots.append((task,old_seen,seen,{m:clone_model(models[m]) for m in MODES},{m:slots[m].copy() for m in MODES}))
    # Only now create independent synthetic evaluation; all four trajectories and stages are sealed.
    ex,ey,eval_counts=lowrank_fixture(dict(cfg,seed=cfg['seed']+10000))
    rows=[]; class_rows=[]; pairs=[]; diagnostic=[]; contributions=[]
    comparisons=[('fixed_moment','fixed_decision'),('assigned_moment','assigned_decision'),
                 ('fixed_moment','assigned_moment'),('fixed_decision','assigned_decision'),('fixed_moment','assigned_decision')]
    for task,old_seen,seen,mm,ss in snapshots:
        journal(dict(phase='sealed_synthetic_evaluation',task=task)); mask=ey<seen; xx,yy=ex[mask],ey[mask]; predictions={}
        with torch.no_grad():
            for mode in MODES:
                s=scores(xx,mm[mode],q)[:,ss[mode]]; pred=s.argmax(1); predictions[mode]=pred
                recalls=[]
                for c in range(seen):
                    cm=yy==c; recall=float((pred[cm]==c).to(DT).mean()); recalls.append(recall)
                    class_rows.append(dict(task=task,mode=mode,class_index=c,count=int(cm.sum()),recall=recall))
                truth=s[torch.arange(len(yy)),yy]; wrong=s.clone(); wrong[torch.arange(len(yy)),yy]=-torch.inf
                margins=truth-wrong.max(1).values
                rows.append(dict(task=task,mode=mode,seen_classes=seen,accuracy=float((pred==yy).to(DT).mean()),
                    balanced_accuracy=sum(recalls)/seen,old_balanced_accuracy=sum(recalls[:old_seen])/old_seen if old_seen else None,
                    new_balanced_accuracy=sum(recalls[old_seen:])/(seen-old_seen),margin_min=float(margins.min()),
                    margin_q10=float(torch.quantile(margins,.1)),margin_median=float(margins.median())))
            for base,can in comparisons:
                a,b=predictions[base],predictions[can]
                for group,gm in (('all',torch.ones_like(yy,dtype=torch.bool)),('old',yy<old_seen),('new',yy>=old_seen)):
                    ac,bc=a==yy,b==yy
                    pairs.append(dict(task=task,baseline=base,candidate=can,group=group,count=int(gm.sum()),
                        rescued=int((gm&~ac&bc).sum()),harmed=int((gm&ac&~bc).sum()),
                        wrong_to_different_wrong=int((gm&~ac&~bc&(a!=b)).sum()),
                        candidate_errors_to_old=int((gm&~bc&(b<old_seen)).sum()),candidate_errors_to_new=int((gm&~bc&(b>=old_seen)).sum())))
        by={r['mode']:r for r in rows if r['task']==task}
        keys=['balanced_accuracy','new_balanced_accuracy']+(['old_balanced_accuracy'] if old_seen else [])
        passes={base:all(by['assigned_decision'][j]>=by[base][j] for j in keys) for base in ('fixed_moment','assigned_moment','fixed_decision')}
        contributions.append(dict(task=task,no_harm_vs_controls=passes,
            full_minus_fixed_moment_BA=by['assigned_decision']['balanced_accuracy']-by['fixed_moment']['balanced_accuracy'],
            memory_at_fixed_BA=by['fixed_decision']['balanced_accuracy']-by['fixed_moment']['balanced_accuracy'],
            memory_at_assigned_BA=by['assigned_decision']['balanced_accuracy']-by['assigned_moment']['balanced_accuracy'],
            assignment_at_moment_BA=by['assigned_moment']['balanced_accuracy']-by['fixed_moment']['balanced_accuracy'],
            assignment_at_decision_BA=by['assigned_decision']['balanced_accuracy']-by['fixed_decision']['balanced_accuracy'],
            interaction_BA=by['assigned_decision']['balanced_accuracy']-by['assigned_moment']['balanced_accuracy']-by['fixed_decision']['balanced_accuracy']+by['fixed_moment']['balanced_accuracy']))
        # Same-state real-vs-memory diagnostic uses old synthetic support only AFTER sealing.
        for mode in MODES:
            model=clone_model(mm[mode],True); losses=[]; errors=[]; gradients=[]
            for c in range(old_seen):
                real=x[y==c]; z,p=memories[mode][c]
                a,ag=risk_gradient(real,c,model,q,ss[mode]); b,bg=risk_gradient(z,c,model,q,ss[mode],p)
                losses.append(abs(float(a-b))); gradients.append(float((ag-bg).norm()))
                with torch.no_grad(): errors.append(float(sliced(response(real,c,model,q,ss[mode]),response(z,c,model,q,ss[mode]),p)))
            diagnostic.append(dict(task=task,mode=mode,old_classes=old_seen,
                mean_old_CE_error=sum(losses)/len(losses) if losses else None,max_old_CE_error=max(losses) if losses else None,
                max_old_gradient_error=max(gradients) if gradients else None,max_old_margin_sliced_W1=max(errors) if errors else None))
    full_signal=all(all(c['no_harm_vs_controls'].values()) for c in contributions)
    for control,field in (('fixed_moment','full_minus_fixed_moment_BA'),('assigned_moment','memory_at_assigned_BA'),('fixed_decision','assignment_at_decision_BA')):
        full_signal &= any(c[field]>0 for c in contributions)
    return dict(status='COMPLETED',case=cfg,synthetic_counts=counts,synthetic_evaluation_counts=eval_counts,
        rows=rows,class_rows=class_rows,paired_predictions=pairs,training=training,construction=construction,
        assignment_costs=allocation,diagnostics=diagnostic,contributions=contributions,
        joint_mechanism_signal_pass=bool(full_signal),all_training_sealed_before_evaluation=True,
        old_memory_unchanged=True,stable_input_and_frame_exact=True,old_slot_ids_unchanged=True,
        no_old_support_during_candidate_training=True,candidate_memory_bytes=k*16*(d+1)*8,
        adapter_parameters=2*d*8,frame_bytes=d*k*8,total_adapter_updates=4*48*len(cfg['task_sizes']),
        actual_memory_updates=3*48*len(cfg['task_sizes']),shared_moment_construction_counted_once=True,
        CPU_pipeline_seconds=time.monotonic()-begun)


def run(cfg, lock, journal):
    torch.set_num_threads(lock['threads']); torch.set_default_dtype(DT)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    result=unit_checks() if cfg['kind']=='unit' else run_scale(cfg,lock,journal)
    result.update(id=cfg['id'],GPU_initialized=torch.cuda.is_initialized(),medical_images=0,
                  medical_checkpoints=0,val_accessed=False,test_accessed=False,formal_updates=0)
    assert not result['GPU_initialized']
    return result
