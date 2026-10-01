"""Four locked stability conditions; reuse the checked A1 reward estimator."""
import torch
from nb_rl_a1_core import objective as previous_objective

CONDITIONS = {
    'A': dict(fd=10., lr=3e-4, exact=False),
    'B': dict(fd=30., lr=3e-4, exact=False),
    'C': dict(fd=10., lr=1e-4, exact=False),
    'D': dict(fd=10., lr=3e-4, exact=True),
}


def objective(z, target, W, y, weights, method, task, sigma, generator):
    condition = CONDITIONS[method]
    source = 'S' if task == 1 else 'E' if condition['exact'] else 'H'
    _, terms, coefficients, next_sigma, diag, actions = previous_objective(
        z, target, W, y, weights, source, task, sigma, generator)
    coefficients['FD'] = condition['fd']
    total = sum(coefficients[k] * value for k, value in terms.items())
    return total, terms, coefficients, next_sigma, diag, actions


def selfcheck():
    generator = torch.Generator().manual_seed(84001)
    z = torch.randn(5, 7, generator=generator, dtype=torch.float64, requires_grad=True)
    target = torch.randn(5, 7, generator=generator, dtype=torch.float64)
    W = torch.randn(7, 4, generator=generator, dtype=torch.float64)
    y = torch.tensor([0, 1, 0, 1, 1]); weights = torch.tensor([.5, 2., 1., 1.])
    first = [objective(z, target, W, y, weights, m, 1, .5, generator) for m in CONDITIONS]
    for value in first:
        torch.testing.assert_close(value[0], first[0][0], rtol=0, atol=0)
        assert value[3] == .5 and value[5] is None
        assert float(value[1]['FD']) == float(value[1]['Exact']) == 0
    a = objective(z, target, W, y, weights, 'A', 2, .5, generator)
    h = previous_objective(z, target, W, y, weights, 'H', 2, .5, generator)
    torch.testing.assert_close(a[0], h[0], rtol=0, atol=0)
    b = objective(z, target, W, y, weights, 'B', 2, .5, generator)
    d = objective(z, target, W, y, weights, 'D', 2, .5, generator)
    torch.testing.assert_close(b[0] - a[0], 20 * a[1]['FD'])
    torch.testing.assert_close(d[0] - a[0], .1 * d[1]['Exact'])
    assert torch.autograd.grad(d[1]['Exact'], z)[0].norm() > 0
    assert d[5] is None and d[2]['Exact'] == .1
    return dict(status='PASS',identical_CE_Task1=True,A_equals_H=True,
                B_changes_only_FD=True,D_changes_only_exact_auxiliary=True,no_action_sampling=True)


if __name__ == '__main__':
    import json
    print(json.dumps(selfcheck()))
