"""Small independent oracles for the new prototype/analytic mechanisms."""
import torch
from torch.nn import functional as F

from prototype_coherent import (empty, seed_components, memberships, sample_weights,
    append, translate, evidence, metric, head, proximal_base, proximal_head,
    old_square_losses, advantages, controller)


def check():
    torch.set_num_threads(4); torch.manual_seed(92)
    dtype = torch.float64
    centers = torch.eye(6, dtype=dtype)[:4]
    x = torch.cat([F.normalize(c+.4*torch.randn(64, 6, dtype=dtype), dim=1) for c in centers])
    y = torch.arange(4).repeat_interleave(64)
    old_x, old_y, z, labels = x[:128], y[:128], x[128:], y[128:]
    old_seeds = seed_components(old_x, old_y)
    old_group, old_difficulty = memberships(old_x, old_y, old_seeds)
    old_w = sample_weights(old_y, old_group, old_difficulty, torch.zeros(len(old_seeds), dtype=dtype))
    old = append(empty(6), old_x, old_y, old_group, old_w)
    seeds = seed_components(z, labels); group, difficulty = memberships(z, labels, seeds)
    action = torch.linspace(-.9, .9, len(seeds), dtype=dtype)
    weights = sample_weights(labels, group, difficulty, action)
    for c in (2, 3):
        assert torch.allclose(weights[labels == c].sum(), torch.tensor(1., dtype=dtype))
        assert (weights[labels == c] >= .5/int((labels == c).sum())).all()
    bank = append(old, z, labels, group, weights)
    for c in range(4):
        rows = y == c; w = old_w[old_y == c] if c < 2 else weights[labels == c]
        assert torch.allclose(bank['mu'][c], (w[:, None]*x[rows]).sum(0))
        assert torch.allclose(bank['Q'][c], x[rows].T @ (w[:, None]*x[rows]))
    d = torch.randn(6, dtype=dtype)*.1; moved = translate(bank, d)
    all_w = torch.cat([old_w, weights])
    for c in range(4):
        rows = y == c; direct = x[rows]+d
        assert torch.allclose(moved['Q'][c], direct.T @ (all_w[rows, None]*direct))
    W, R = head(bank); P = evidence(bank)
    T = torch.cat([torch.eye(6, dtype=dtype), P], dim=1)
    phi = x @ T; targets = F.one_hot(y, 4).to(dtype)
    full_w = all_w/4
    expanded_W = torch.linalg.solve(phi.T @ (full_w[:, None]*phi)+.001*torch.eye(phi.shape[1], dtype=dtype),
                                    phi.T @ (full_w[:, None]*targets))
    assert torch.allclose(W, T @ expanded_W, atol=1e-10, rtol=1e-9)
    assert torch.linalg.eigvalsh(R).min() > 0
    assert not torch.allclose(W, head(bank, strength=0.)[0], atol=1e-9, rtol=1e-9)
    expected_old = torch.stack([((old_x[old_y == c] @ W-F.one_hot(old_y[old_y == c], 4)).square().sum(1)
                                *old_w[old_y == c]).sum() for c in (0, 1)])
    assert torch.allclose(old_square_losses(old, W), expected_old)
    inverse, cross = proximal_base(old, R, 4)
    batch = z[:9]; target = F.one_hot(labels[:9], 4).to(dtype); bw = weights[:9]*2
    previous = torch.randn(6, 4, dtype=dtype)
    woodbury = proximal_head(batch, target, bw, inverse, cross, previous)
    A = old['Q'].sum(0)/4+.001*R+.01*torch.eye(6, dtype=dtype)+batch.T @ (bw[:, None]*batch)
    B = cross+.01*previous+batch.T @ (bw[:, None]*target)
    assert torch.allclose(woodbury, torch.linalg.solve(A, B), atol=1e-10, rtol=1e-9)
    # The head and encoder must optimize the same local objective, not just use ridge somewhere.
    gradients = []
    for detached in (False, True):
        bx = batch.clone().requires_grad_(True)
        V = proximal_head(bx, target, bw, inverse, cross, previous)
        if detached:
            V = V.detach()
        objective = (.5*(bw[:, None]*(bx @ V-target).square()).sum()
                     + .5*old_square_losses(old, V).sum()/4
                     + .0005*(V*(R @ V)).sum()+.005*(V-previous).square().sum())
        gradients.append(torch.autograd.grad(objective, bx)[0])
    assert torch.allclose(*gradients, atol=1e-10, rtol=1e-8)
    A, active = advantages(torch.ones(4, dtype=dtype)); assert not active and not A.any()
    A, active = advantages(torch.arange(4, dtype=dtype)); assert active and abs(float(A.mean())) < 1e-10
    hold_x = torch.cat([F.normalize(c+.4*torch.randn(16, 6, dtype=dtype), dim=1) for c in centers[2:]])
    hold_y = torch.arange(2, 4).repeat_interleave(16)
    for mode in ('gradient', 'group'):
        parameter = torch.nn.Parameter(torch.zeros(6, dtype=dtype))
        selected, audit = controller(old, z, labels, group, difficulty, hold_x, hold_y,
            seeds, parameter, parameter.detach().clone(), mode, torch.Generator().manual_seed(928), steps=2)
        assert audit['optimizer_steps'] == 2 and parameter.detach().norm() > 0
        assert torch.isfinite(selected).all() and (selected.abs() <= 1.).all()
    from run_prototype_coherent import fit_split, common_shift
    rows = [dict(label=c, identity_component=f'{c}-{i//2}') for c in (4, 0) for i in range(20)]
    fit, meta = fit_split(rows, 74002)
    assert set(fit).isdisjoint(meta) and sorted(fit+meta) == list(range(len(rows)))
    assert {rows[i]['identity_component'] for i in fit}.isdisjoint(
        {rows[i]['identity_component'] for i in meta})
    assert fit_split(rows, 74002) == (fit, meta) and len(meta) == 8
    assert torch.allclose(common_shift(x, x+d, y), d)
    print('PASS: weighted moments, translation, explicit evidence-head equivalence, informative old risk,')
    print('      Woodbury/proximal oracle, envelope gradient, bounded weights, and both controllers')


if __name__ == '__main__':
    check()
