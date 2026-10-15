"""Prospective synthetic CPU checks; never opens a dataset or medical checkpoint."""
import io
import itertools
import time
import torch
import core1_competition as competition
import prototype_coherent as coherent
import module_additions as modules
import native_pointwise_guard_design as guard


def bank(samples):
    return dict(mu=torch.stack([z.mean(0) for z in samples]),
        Q=torch.stack([z.T @ z/len(z) for z in samples]), n=[len(z) for z in samples],
        components=[dict(label=i, center=z.mean(0), mass=z.new_tensor(1.),
                        radius=(z-z.mean(0)).square().sum(1).mean()) for i, z in enumerate(samples)])


def fixture(d, k, lt):
    generator = torch.Generator().manual_seed(223+d+k)
    z = torch.randn(k, 4, d, generator=generator, dtype=torch.float64)
    z = torch.nn.functional.normalize(z, dim=-1)
    samples = list(z); original = bank(samples); old_count = k-2
    old = bank(samples[:old_count]); x = torch.cat(samples[old_count:]); y = torch.arange(old_count, k).repeat_interleave(4)
    alpha = x.new_full((len(x),), 1/(4*k))
    local = torch.nn.functional.normalize(torch.randn(k, 4, d, generator=generator, dtype=x.dtype), dim=-1)
    arm = 'local_transport' if lt else 'base'
    r, _ = modules.metric(original, arm, 'HK', list(range(k)), local if lt else None)
    previous = torch.zeros((d, k), dtype=x.dtype)
    initial, _ = coherent.head(original)
    a = competition.competition(original, initial)
    q, mu, mass = competition.moments(old, x, y, alpha, k)
    op, rhs, solve = guard.system(q, mu, mass, a, r, previous)
    ref, _ = solve(rhs)
    scale = torch.ones(d, dtype=x.dtype); scale[0] = 2.8; scale[1] = .65
    transformed = [v*scale for v in samples[:old_count]]
    moved_old = bank(transformed); moved = bank(transformed+samples[old_count:])
    moved_r, _ = modules.metric(moved, arm, 'HK', list(range(k)), local if lt else None)
    return dict(old=moved_old, x=x, y=y, alpha=alpha, a=a, r=moved_r, previous=previous,
                reference=ref, k=k, old_count=old_count)


def make(f, x=None, woodbury=False):
    x = f['x'] if x is None else x
    q, mu, mass = competition.moments(f['old'], x, f['y'], f['alpha'], f['k'])
    inverse = coherent.proximal_base(f['old'], f['r'], f['k'])[0] if woodbury else None
    op, rhs, solve = guard.system(q, mu, mass, f['a'], f['r'], f['previous'],
                                inverse=inverse, x=x, alpha=f['alpha'])
    w, audit = solve(rhs)
    return op, rhs, solve, w, audit, (q, mu, mass)


def oracle(op, rhs, x, y, ref):
    shape = rhs.shape; n = rhs.numel(); eye = torch.eye(n, dtype=rhs.dtype)
    h = torch.stack([op(v.reshape(shape)).flatten() for v in eye], dim=1)
    assert torch.allclose(h, h.T, atol=1e-11)
    torch.linalg.cholesky(h)
    b = rhs.flatten(); base = torch.linalg.solve(h, b)
    rows = [guard.atom(x, y, i, j, shape[1]).flatten() for i in range(len(x)) for j in range(shape[1]) if j != y[i]]
    A = torch.stack(rows); t = A @ ref.flatten(); best = None
    for count in range(len(rows)+1):
        for ids in itertools.combinations(range(len(rows)), count):
            if ids:
                sub = A[list(ids)]; directions = torch.linalg.solve(h, sub.T)
                gram = sub @ directions
                lam = torch.linalg.lstsq(gram, t[list(ids)]-sub @ base, rcond=1e-12).solution
                w = base+directions @ lam
                if float(lam.min()) < -1e-8: continue
            else: w = base
            if float((A @ w-t).min()) < -1e-8: continue
            value = .5*w @ h @ w-b @ w
            if best is None or float(value) < best[0]: best = (float(value), w)
    assert best is not None
    return best[1].reshape(shape), h


def run():
    begun = time.monotonic(); torch.set_num_threads(2)
    out = dict(status='PASS', GPU_processes=0, medical_images_read=0, medical_checkpoints_read=0,
               val_reads=0, test_accessed=False, optimizer_updates=0, production_integrated=False,
               GPU_ready=False, cases=[])
    for lt in (False, True):
        f = fixture(6, 3, lt); op, rhs, solve, wu, _, moments = make(f)
        q, mu, mass = moments
        wn, _ = competition.solve(q, mu, mass, f['a'], regularizer=f['r'], previous=f['previous'], proximal=.01)
        assert torch.allclose(wn, wu, atol=2e-8, rtol=2e-8)
        wb = make(f, woodbury=True)[3]
        assert torch.allclose(wb, wu, atol=2e-8, rtol=2e-8)
        # Exhaustive oracle is deliberately restricted to four small synthetic constraints.
        x = f['x'][[0, 4]]; y = f['y'][[0, 4]]
        ref = wu+.3*guard.atom(x, y, 0, 0, 3)
        w, audit, dual = guard.guarded(op, rhs, solve, x, y, ref, wu, 1)
        exact, h = oracle(op, rhs, x, y, ref)
        assert torch.allclose(w, exact, atol=2e-7, rtol=2e-7)
        assert audit['active'] > 0 and audit['discriminative_delta'] > 1e-4
        off = guard.guarded(op, rhs, solve, x, y, ref, wu, 1, enabled=False)[0]
        assert torch.equal(off, wu)
        # Explicit native sample loss checks the full Hessian and linear term, including constants.
        def explicit(v):
            target = torch.nn.functional.one_hot(f['y'], 3).double()
            cur = .5*(f['alpha']*(f['x'] @ v-target).square().sum(1)).sum()
            old = coherent.old_square_losses(f['old'], v).sum()/(2*3)
            return cur+old+.0005*(v*(f['r'] @ v)).sum()+.005*(v-f['previous']).square().sum()+.5*competition.pair_loss(f['old'], f['x'], f['y'], f['alpha'], v, f['a'])
        zero = torch.zeros_like(wu, requires_grad=True)
        hh = torch.autograd.functional.hessian(explicit, zero).reshape(h.shape)
        bb = -torch.autograd.grad(explicit(zero), zero)[0]
        assert torch.allclose(hh, h, atol=1e-10) and torch.allclose(bb, rhs, atol=1e-10)
        # Moving supports/targets with block-frozen Wref require the dual term.
        ref_full = wu+.3*guard.atom(f['x'], f['y'], 0, 0, 3)
        def optimized(xx, gradient=False):
            oo, rr, ss, uu, _, _ = make(f, xx)
            with torch.no_grad(): vv, aa, dd = guard.guarded(oo, rr, ss, xx.detach(), f['y'], ref_full, uu.detach(), 1)
            val = guard.objective(oo, rr, vv.detach())
            lag = val
            slack = guard.contrasts(xx, f['y'], vv.detach())-guard.contrasts(xx, f['y'], ref_full)
            for i, j, l in dd: lag = lag-l*slack[i, j]
            return val, lag, aa
        xx = f['x'].clone().requires_grad_(); value, lag, aa = optimized(xx)
        naive = torch.autograd.grad(value, xx, retain_graph=True)[0]
        correct = torch.autograd.grad(lag, xx)[0]
        direction = torch.zeros_like(xx); direction[0, 1] = 1.
        eps = 1e-5
        finite = float((optimized(f['x']+eps*direction)[0]-optimized(f['x']-eps*direction)[0])/(2*eps))
        error = abs(finite-float((correct*direction).sum()))
        assert aa['active'] > 0 and error < 2e-6
        # Adjoint, class permutation, orthogonal coordinates, common-logit gauge.
        a0 = guard.atom(x, y, 0, 0, 3)
        assert abs(float((a0*w).sum()-guard.contrasts(x, y, w)[0, 0])) < 1e-12
        p = torch.tensor([0, 2, 1]); yp = torch.argsort(p)[y]
        def pop(z): return op(z[:, torch.argsort(p)])[:, p]
        def psolve(z):
            v, a = solve(z[:, torch.argsort(p)]); return v[:, p], a
        pw = guard.guarded(pop, rhs[:, p], psolve, x, yp, ref[:, p], wu[:, p], 1)[0]
        assert torch.allclose(pw, w[:, p], atol=2e-7)
        o = torch.linalg.qr(torch.randn(6, 6, dtype=x.dtype, generator=torch.Generator().manual_seed(223)))[0]
        def oop(z): return o.T @ op(o @ z)
        def osolve(z):
            v, a = solve(o @ z); return o.T @ v, a
        ow = guard.guarded(oop, o.T @ rhs, osolve, x @ o, y, o.T @ ref, o.T @ wu, 1)[0]
        assert torch.allclose(ow, o.T @ w, atol=2e-7)
        common = torch.arange(6, dtype=x.dtype)[:, None]*.01
        gw = guard.guarded(op, rhs, solve, x, y, ref+common, wu, 1)[0]
        assert torch.allclose(gw, w, atol=2e-7)
        buffer = io.BytesIO(); torch.save(dict(reference=ref, constrained=w, dual=dual), buffer); buffer.seek(0)
        loaded = torch.load(buffer, weights_only=True)
        assert torch.equal(loaded['constrained'], w) and loaded['dual'] == dual
        for bad in (torch.zeros_like(y), torch.full_like(y, 3)):
            try: guard.guarded(op, rhs, solve, x, bad, ref, wu, 1)
            except ValueError: pass
            else: raise AssertionError('Illegal historical/future guard labels accepted')
        actual, actual_audit, _ = guard.guarded(op, rhs, solve, f['x'], f['y'], f['reference'], wu, 1)
        out['cases'].append(dict(arm='LT' if lt else 'Core', small_oracle_error=float((w-exact).norm()),
            native_match_error=float((wn-wu).norm()), woodbury_error=float((wb-wu).norm()),
            lagrangian_envelope_error=error, naive_envelope_error=abs(finite-float((naive*direction).sum())),
            stress_reference='synthetic active-face construction, not a medical OFF head', stress=audit,
            affine_same_state_OFF=actual_audit))
    # Active full-support opposing constraints force all class contrasts to identity.
    f = fixture(4, 3, False); op, rhs, solve, wu, _, _ = make(f)
    x = torch.cat([torch.eye(4, dtype=wu.dtype), -torch.eye(4, dtype=wu.dtype)]).repeat_interleave(3, 0)
    y = torch.arange(3).repeat(8); ref = wu+torch.arange(12, dtype=wu.dtype).reshape(4, 3)*.02
    w, audit, _ = guard.guarded(op, rhs, solve, x, y, ref, wu, 0)
    assert audit['discriminative_delta'] < 2e-7
    out['full_support_identity'] = audit
    # Production-sized native tensors; 21 old and 2 current classes, no full head Hessian.
    for lt in (False, True):
        f = fixture(1536, 23, lt); op, rhs, solve, wu, solve_audit, _ = make(f, woodbury=True)
        x = f['x'][[0, 4]]; y = f['y'][[0, 4]]
        ref = wu+.02*guard.atom(x, y, 0, 0, 23)
        w, audit, _ = guard.guarded(op, rhs, solve, x, y, ref, wu, 21)
        assert audit['active'] > 0 and audit['discriminative_delta'] > 1e-6
        out['cases'].append(dict(arm='LT_highdim' if lt else 'Core_highdim', dimension=1536, classes=23,
            historical_classes=21, observed_support_rank=int(torch.linalg.matrix_rank(x)),
            old_moment_bytes=f['old']['Q'].numel()*8, unconstrained=solve_audit, stress=audit,
            limitation='rank-two observed guard support; not full-rank medical fit or production timing'))
    out['checks'] = dict(native_full_Hessian=True, dense_small_KKT_oracle=True, woodbury=True,
        active_Lagrangian_envelope=True, disabled_exact=True, class_permutation=True,
        orthogonal_coordinates=True, common_logit_gauge=True, checkpoint_roundtrip=True,
        illegal_label_scope_rejected=True, full_support_identity_retained=True,
        highdim_native=True, medical_loader=False, full_rank_highdim_guard=False,
        Wref_refresh_implicit_gradient=False, formal_protocol_frozen=False)
    out['CPU_pipeline_seconds'] = time.monotonic()-begun
    return out
