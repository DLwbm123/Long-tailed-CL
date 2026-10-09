"""Small mathematical oracles for the edge action and prototype-metric solver."""
import torch
from torch.nn import functional as F

import core1_competition as competition
import edge_competition as edge
import prototype_coherent as native


def main():
    torch.manual_seed(471); torch.set_num_threads(2)
    x = F.normalize(torch.randn(24, 5, dtype=torch.float64), dim=1)
    y = torch.arange(4).repeat_interleave(6)
    group = torch.arange(8).repeat_interleave(3)
    weight = x.new_full((24,), 1/6)
    bank = native.append(native.empty(5), x, y, group, weight)
    head, metric = native.head(bank)
    base = competition.competition(bank, head)
    features = edge.descriptors(bank, head, 2)
    coefficients = torch.tensor([.4, -.5, .1, .3, -.7, .6, .2, -.1], dtype=x.dtype)
    pairs = edge.allocation(base, features, coefficients)
    assert torch.equal(edge.allocation(base, features, torch.zeros(8)), base)
    assert torch.all(pairs >= .5*base) and torch.all(pairs.diag() == 0)
    assert torch.allclose(pairs.sum(), base.sum(), atol=1e-12)
    coarse = edge.allocation(base, features[:, :, :3], coefficients[:3])
    nested = edge.allocation(base, features, torch.cat([coefficients[:3], x.new_zeros(5)]))
    assert torch.allclose(coarse, nested, atol=1e-12)
    # Within a coarse group all edge ratios match; fine actions can distinguish those edges.
    ratio = pairs/base.clamp_min(1e-30)
    coarse_ratio = coarse/base.clamp_min(1e-30)
    mask = features[:, :, 0].bool()
    assert coarse_ratio[mask].std() < 1e-12 and ratio[mask].std() > 1e-5
    permutation = torch.tensor([1, 0, 3, 2])
    inverse = permutation.argsort()
    permuted = dict(mu=bank['mu'][permutation], Q=bank['Q'][permutation],
        n=[bank['n'][i] for i in permutation],
        components=[dict(c, label=int(inverse[c['label']])) for c in bank['components']])
    permuted_features = edge.descriptors(permuted, head[:, permutation], 2)
    assert torch.allclose(permuted_features, features[permutation][:, permutation], atol=1e-10)
    assert torch.allclose(edge.allocation(base[permutation][:, permutation], permuted_features, coefficients),
                          pairs[permutation][:, permutation], atol=1e-10)
    solved, audit = competition.bank_head(bank, pairs, regularizer=metric)
    target = F.one_hot(y, 4).double()

    def objective(w):
        return (.5*(x @ w-target).square().sum(1).mean()+.0005*(w*(metric @ w)).sum()+
                .5*competition.pair_loss(native.empty(5), x, y, x.new_full((24,), 1/24), w, pairs))

    origin = torch.zeros_like(solved, requires_grad=True)
    hessian = torch.autograd.functional.hessian(objective, origin).reshape(20, 20)
    rhs = -torch.autograd.grad(objective(origin), origin)[0].flatten()
    oracle = torch.linalg.solve(hessian, rhs).reshape_as(solved)
    assert torch.allclose(solved, oracle, atol=1e-8, rtol=1e-8), audit
    neutral, _ = competition.bank_head(bank, pairs, beta=0, regularizer=metric)
    assert torch.allclose(neutral, head, atol=1e-10)
    old = native.append(native.empty(5), x[:12], y[:12], group[:12], weight[:12])
    q, mu, mass = competition.moments(old, x[12:], y[12:], x.new_full((12,), 1/24), 4)
    inv, cross = native.proximal_base(old, metric, 4)
    nw = native.proximal_head(x[12:], target[12:], x.new_full((12,), 1/24), inv, cross, head)
    fast, _ = competition.solve(q, mu, mass, pairs, previous=head, proximal=.01,
        inverse=inv, x=x[12:], alpha=x.new_full((12,), 1/24), native=nw, regularizer=metric)
    direct, _ = competition.solve(q, mu, mass, pairs, previous=head, proximal=.01, regularizer=metric)
    assert torch.allclose(fast, direct, atol=1e-8, rtol=1e-8)

    def optimized(z):
        b = native.append(native.empty(5), z, y, group, weight)
        # Hold epoch-level prototype metric and edge action fixed, as in the native training block.
        w, _ = competition.bank_head(b, pairs, regularizer=metric)
        w = w.detach()
        return (.5*(z @ w-target).square().sum(1).mean()+.0005*(w*(metric @ w)).sum()+
                .5*competition.pair_loss(native.empty(5), z, y, z.new_full((24,), 1/24), w, pairs))

    variable = x.clone().requires_grad_()
    grad = torch.autograd.grad(optimized(variable), variable)[0][4, 2]
    xp, xm = x.clone(), x.clone(); xp[4, 2] += 1e-5; xm[4, 2] -= 1e-5
    finite = (optimized(xp)-optimized(xm))/(2e-5)
    assert torch.allclose(grad, finite, atol=2e-6, rtol=2e-5)
    results = {}
    for arm in ['static_pc', 'coarse_rl', 'edge_rl', 'edge_search']:
        controller = edge.Controller(arm, 97); controller.begin_task()
        selected, _, _, audit = controller.choose(bank, old, base, metric, x[12:]+.01, y[12:], [0, 3], lambda: None)
        assert audit['head_evaluations'] == (1 if arm == 'static_pc' else 18)
        assert controller.updates == (8 if arm.endswith('_rl') else 0)
        assert audit['executed'] == (audit['proposed_reward'] > 1e-8)
        if not audit['executed']:
            assert torch.equal(selected, base)
        results[arm] = (selected, audit)
    repeat = edge.Controller('edge_rl', 97); repeat.begin_task()
    selected, _, _, audit = repeat.choose(bank, old, base, metric, x[12:]+.01, y[12:], [0, 3], lambda: None)
    assert torch.equal(selected, results['edge_rl'][0]) and audit == results['edge_rl'][1]
    bad = coefficients.clone(); bad[0] = float('nan')
    try:
        edge.allocation(base, features, bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite action accepted')
    print('PASS: budget/floor, exact no-op, coarse nesting, fine variation, permutation equivariance, '
          'explicit Hessian, native reduction, Woodbury, envelope gradient, policy accounting and reproducibility')


if __name__ == '__main__':
    main()
