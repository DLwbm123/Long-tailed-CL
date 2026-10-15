"""All CPU case settings come from the prospective source-only design lock."""
import json
import sys
import time
import torch
import core1_competition as competition
import prototype_coherent as coherent
import module_additions as modules
import native_pointwise_guard_design as native
import full_fit_guard_design as guard
from check_native_pointwise_guard_design import bank, fixture, make


def sample(n, d, label, generator):
    center = torch.zeros(d, dtype=torch.float64); center[label % d] = .8
    noise = torch.nn.functional.normalize(torch.randn(n, d, generator=generator, dtype=center.dtype), dim=1)*.6
    return torch.nn.functional.normalize(noise+center, dim=1)


def label_counts(n, raw):
    total = sum(raw); counts = [max(1, n*v//total) for v in raw]
    remainder = n-sum(counts)
    # Fixed largest-remainder allocation; these are NOT actual per-class medical fit counts.
    ids = sorted(range(len(raw)), key=lambda i: (-(n*raw[i]/total-counts[i]), i))
    for t in range(remainder): counts[ids[t % len(ids)]] += 1
    assert sum(counts) == n
    return counts


def scale_case(config):
    start = time.monotonic(); torch.set_num_threads(2)
    d = config['dimension']; k = config['classes']; old_count = config['old_classes']; n = config['fit_n']
    gen = torch.Generator().manual_seed(227+100*k+config['task'])
    counts = label_counts(n, config['current_train_counts'])
    x = torch.cat([sample(c, d, old_count+i, gen) for i, c in enumerate(counts)])
    y = torch.cat([torch.full((c,), old_count+i, dtype=torch.long) for i, c in enumerate(counts)])
    # Rank is checked without regularization; this cannot turn a singular Gram into full rank.
    rank_begin = time.monotonic()
    small = x @ x.T if n < d else x.T @ x
    eig = torch.linalg.eigvalsh(small)
    threshold = torch.finfo(x.dtype).eps*max(n, d)*float(eig.max())
    rank = int((eig > threshold).sum()); del small, eig
    rank_seconds = time.monotonic()-rank_begin
    old_samples = [sample(c, d, i, gen) for i, c in enumerate(config['old_train_counts'])]
    old = bank(old_samples) if old_count else coherent.empty(d)
    current = bank([x[y == c] for c in range(old_count, k)])
    # bank() starts labels at zero; reindex only current component labels.
    for c in current['components']: c['label'] += old_count
    all_bank = dict(mu=torch.cat([old['mu'], current['mu']]), Q=torch.cat([old['Q'], current['Q']]),
                    n=old['n']+current['n'], components=old['components']+current['components'])
    local = torch.nn.functional.normalize(torch.randn(k, 4, d, generator=gen, dtype=x.dtype), dim=-1)
    arm = 'local_transport' if config['LT'] else 'base'
    r0 = modules.metric(all_bank, arm, config['dataset'], list(range(k)), local if config['LT'] else None)[0]
    initial = torch.linalg.solve(all_bank['Q'].sum(0)/k+.001*r0, all_bank['mu'].T/k)
    a = competition.competition(all_bank, initial)
    # A fixed uniform-without-replacement batch; native class weighting uses full-fit counts.
    ids = torch.randperm(n, generator=gen)[:min(64, n)]; bx, by = x[ids], y[ids]
    weights = torch.tensor([1/counts[int(v)-old_count] for v in by], dtype=x.dtype)
    alpha = n*weights/(k*len(ids)); previous = initial.detach()
    q0, mu0, mass0 = competition.moments(old, bx, by, alpha, k)
    inv0 = coherent.proximal_base(old, r0, k)[0]
    o0, b0, s0 = native.system(q0, mu0, mass0, a, r0, previous, inverse=inv0, x=bx, alpha=alpha)
    ref, _ = s0(b0)
    if old_count:
        scale = torch.ones(d, dtype=x.dtype); scale[0] = 2.8; scale[1] = .65
        moved_old = bank([z*scale for z in old_samples])
    else: moved_old = old
    moved_bank = dict(mu=torch.cat([moved_old['mu'], current['mu']]), Q=torch.cat([moved_old['Q'], current['Q']]),
        n=moved_old['n']+current['n'], components=moved_old['components']+current['components'])
    r = modules.metric(moved_bank, arm, config['dataset'], list(range(k)), local if config['LT'] else None)[0]
    q, mu, mass = competition.moments(moved_old, bx, by, alpha, k)
    inv = coherent.proximal_base(moved_old, r, k)[0]
    op, rhs, solve = native.system(q, mu, mass, a, r, previous, inverse=inv, x=bx, alpha=alpha)
    wu, ua = solve(rhs); trace = []
    w, audit, dual = guard.constrained(op, rhs, solve, x, y, ref, wu, old_count, trace,
                                      max_atoms=64, time_limit=240.)
    if audit['status'] == 'FEASIBLE':
        assert audit['feasible_quadratic_gain'] >= -1e-8
        # Purely synthetic classification counts are secondary, not a medical gain claim.
        off = (x @ ref).argmax(1); on = (x @ w).argmax(1)
        audit['synthetic_current_rescued'] = int(((off != y) & (on == y)).sum())
        audit['synthetic_current_damaged'] = int(((off == y) & (on != y)).sum())
        if old_count:
            audit['old_moment_loss_change'] = float((coherent.old_square_losses(moved_old, w)-coherent.old_square_losses(moved_old, ref)).mean())
    return dict(status='COMPLETED_CPU_CASE', case=config, synthetic_label_counts=counts,
        actual_medical_class_fit_counts=False, observed_support_rank=rank,
        rank_check_seconds=rank_seconds, unconstrained=ua,
        result=audit, active_growth=trace, CPU_pipeline_seconds=time.monotonic()-start,
        GPU_processes=0, medical_images_read=0, val_reads=0, test_accessed=False, optimizer_updates=0)


def refresh_gradient():
    begun = time.monotonic(); torch.set_num_threads(2)
    result = []
    for lt in (False, True):
        f = fixture(6, 3, lt)
        # Reconstruct untransported old moments from the known fixed affine.
        inv_scale = torch.ones(6, dtype=torch.float64); inv_scale[0] = 1/2.8; inv_scale[1] = 1/.65
        old0 = dict(f['old'], mu=f['old']['mu']*inv_scale,
                    Q=f['old']['Q']*inv_scale[None, :, None]*inv_scale[None, None, :])
        def run(xx, grad=False):
            op, rhs, solve, wu, _, _ = make(f, xx)
            q0, mu0, mass0 = competition.moments(old0, xx, f['y'], f['alpha'], 3)
            o0, b0, s0 = native.system(q0, mu0, mass0, f['a'], f['r'], f['previous'])
            with torch.no_grad(): raw_ref, _ = s0(b0)
            # Fixed stress offset ensures active faces; its derivative is zero, declared in lock.
            fixed = torch.zeros_like(raw_ref); fixed[0, 1] = .3; fixed[0, 0] = -.3
            ref = raw_ref+fixed
            trace = []
            with torch.no_grad():
                w, audit, dual = guard.constrained(op, rhs, solve, xx.detach(), f['y'], ref.detach(), wu.detach(), 1, trace)
            assert audit['status'] == 'FEASIBLE' and audit['positive_duals'] > 0
            value = native.objective(op, rhs, w.detach())
            frozen = value
            slack = native.contrasts(xx, f['y'], w.detach())-native.contrasts(xx, f['y'], ref.detach())
            force = torch.zeros_like(w)
            for i, j, lam in dual:
                frozen = frozen-lam*slack[i, j]
                force += lam*native.atom(xx.detach(), f['y'], i, j, 3)
            term, _ = guard.refresh_term(o0, b0, s0, raw_ref.detach(), force)
            correct = frozen+term
            return value, frozen, correct, audit
        xx = f['x'].clone().requires_grad_(); v, frozen, corrected, aa = run(xx)
        naive = torch.autograd.grad(v, xx, retain_graph=True)[0]
        missing_ref = torch.autograd.grad(frozen, xx, retain_graph=True)[0]
        good = torch.autograd.grad(corrected, xx)[0]
        direction = torch.zeros_like(xx); direction[0, 1] = 1.
        eps = 1e-5
        vp = run(f['x']+eps*direction)[0]; vm = run(f['x']-eps*direction)[0]
        finite = float((vp-vm)/(2*eps))
        err = abs(finite-float((good*direction).sum()))
        assert err < 2e-6
        result.append(dict(arm='LT' if lt else 'Core', finite_difference=finite,
            full_refresh_error=err, missing_reference_derivative_error=abs(finite-float((missing_ref*direction).sum())),
            naive_unconstrained_error=abs(finite-float((naive*direction).sum())), active=aa['positive_duals']))
    # Native highdim reference VJP checked independently; no giant autograd PCG graph.
    f = fixture(1536, 23, True)
    force = native.atom(f['x'], f['y'], 0, 0, 23)
    def reference_system(xx):
        q, mu, mass = competition.moments(f['old'], xx, f['y'], f['alpha'], 23)
        inv = coherent.proximal_base(f['old'], f['r'], 23)[0]
        return native.system(q, mu, mass, f['a'], f['r'], f['previous'], inverse=inv, x=xx, alpha=f['alpha'])
    def scalar(xx):
        oo, rr, ss = reference_system(xx)
        with torch.no_grad(): ww, _ = ss(rr)
        return (force*ww).sum()
    xx = f['x'].clone().requires_grad_(); oo, rr, ss = reference_system(xx)
    with torch.no_grad(): ww, _ = ss(rr)
    term, adj = guard.refresh_term(oo, rr, ss, ww, force)
    gradient = torch.autograd.grad(term, xx)[0]
    eps = 1e-5; dd = torch.zeros_like(xx); dd[0, 1] = 1.
    finite = float((scalar(f['x']+eps*dd)-scalar(f['x']-eps*dd))/(2*eps))
    err = abs(finite-float((gradient*dd).sum())); assert err < 2e-6
    return dict(status='PASS', small_full_active_refresh=result, highdim_reference_VJP_error=err,
        highdim_reference_adjoint=adj, highdim_full_support_constrained_gradient_checked=False,
        reference_refresh_scope='a/R/previous/old moments fixed; reference solve and supports move',
        CPU_pipeline_seconds=time.monotonic()-begun, GPU_processes=0,medical_images_read=0,val_reads=0,test_accessed=False,optimizer_updates=0)


if __name__ == '__main__':
    cfg = json.loads(sys.argv[1])
    print(json.dumps(refresh_gradient() if cfg['kind'] == 'gradient' else scale_case(cfg), allow_nan=False))
