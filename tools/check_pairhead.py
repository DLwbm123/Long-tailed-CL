"""Independent empirical-margin and explicit-objective CPU checks."""
import time
import torch
from torch.nn import functional as F
import core1_competition as competition
import pcrl_control as control
import prototype_coherent as method
from run_pairhead import pair_risks, redistribute


def run():
    start = time.process_time(); torch.set_num_threads(2); torch.manual_seed(441)
    x = torch.randn(30, 5, dtype=torch.float64); y = torch.arange(5).repeat_interleave(6)
    bank = method.append(method.empty(5), x, y, y, x.new_full((30,), 1/6))
    native, _ = method.head(bank, 0.)
    base = control.allocation(competition.competition(bank, native), 3, 1)
    fixed, _ = competition.bank_head(bank, base)
    risk = pair_risks(bank, fixed)
    empirical = torch.zeros_like(risk)
    for c in range(5):
        scores = x[y == c] @ fixed
        margins = scores[:, c:c+1]-scores
        empirical[c] = F.softplus(-margins.mean(0)/(margins.var(0, unbiased=False)+1e-4).sqrt()).clamp_max(20.)
    empirical.fill_diagonal_(0)
    assert torch.allclose(risk, empirical, atol=1e-10, rtol=1e-10)
    assert torch.allclose(risk.sum(1)/4, control.old_risk(bank, fixed), atol=1e-12)
    pairs = redistribute(base, risk, 3)
    assert not torch.allclose(pairs, base)
    assert (pairs >= .5*base).all() and (pairs.diag() == 0).all()
    for ids in (slice(0, 3), slice(3, 5)):
        assert torch.allclose(pairs[:, ids].sum(1), base[:, ids].sum(1), atol=1e-12, rtol=1e-12)
    assert torch.equal(redistribute(base, torch.zeros_like(risk), 3), base)
    permutation = torch.tensor([2, 0, 1, 4, 3])
    assert torch.allclose(redistribute(base[permutation][:, permutation], risk[permutation][:, permutation], 3), pairs[permutation][:, permutation])
    two = torch.tensor([[0., .9], [1.1, 0.]], dtype=torch.float64)
    assert torch.equal(redistribute(two, torch.ones_like(two), 1), two)
    invalid = risk.clone(); invalid[0, 1] = float('nan')
    try:
        redistribute(base, invalid, 3)
        raise AssertionError('Nonfinite risk accepted')
    except ValueError:
        pass
    chosen, audit = competition.bank_head(bank, pairs)
    def objective(w):
        scores = x @ w; target = F.one_hot(y, 5).double()
        margins = scores.gather(1, y[:, None])-scores-1
        return .5*(scores-target).square().sum(1).mean()+.0005*w.square().sum()+.25*(pairs[y]*margins.square()).sum(1).mean()
    zero = torch.zeros_like(chosen, requires_grad=True)
    h = torch.autograd.functional.hessian(objective, zero).reshape(chosen.numel(), chosen.numel())
    rhs = -torch.autograd.grad(objective(zero), zero)[0].flatten()
    assert torch.allclose(chosen, torch.linalg.solve(h, rhs).reshape_as(chosen), atol=1e-8, rtol=1e-8)
    assert audit['relative_residual'] <= 1e-8
    return dict(status='PASS', checks=['empirical standardized margins', 'existing risk equivalence',
        'per-class old/new budget conservation', 'half-base retention and zero diagonal',
        'zero-risk fallback and singleton groups', 'within-age label permutation equivariance',
        'nonfinite input rejection', 'explicit-sample Hessian oracle'],
        scientific_optimizer_updates=0, gpu_seconds=0, cpu_core_seconds=time.process_time()-start)


if __name__ == '__main__':
    print(run())
