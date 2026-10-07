"""CPU oracles for the boundary refit and paired selection guard."""
import time
import numpy as np
import torch
from torch.nn import functional as F
import core1_competition as competition
import pcrl_control as control
import prototype_coherent as method
from run_finalhead import fit_bank, head, rewards, select


def run():
    start = time.process_time(); torch.set_num_threads(2); torch.manual_seed(431)
    x = torch.randn(24, 5, dtype=torch.float64); y = torch.arange(4).repeat_interleave(6)
    old = method.append(method.empty(5), x[:12], y[:12], y[:12], x.new_full((12,), 1/6))
    bank, translated = fit_bank(old, x[12:], x[12:]+.1, y[12:])
    assert bank['n'] == [6]*4 and torch.allclose(translated['mu'], old['mu']+.1)
    # An explicit-sample objective independently reconstructs the selected full head.
    z = x+.1; reference, _ = method.head(bank, 0.)
    for action in range(7):
        w, audit = head(bank, 2, action)
        pairs = control.allocation(competition.competition(bank, reference), 2, action)
        target = F.one_hot(y, 4).double()
        def objective(v):
            scores = z @ v
            margins = scores.gather(1, y[:, None])-scores-1
            return (.5*(scores-target).square().sum(1).mean()+.0005*v.square().sum()
                    +.25*(pairs[y]*margins.square()).sum(1).mean())
        w0 = torch.zeros_like(w, requires_grad=True)
        h = torch.autograd.functional.hessian(objective, w0).reshape(w.numel(), w.numel())
        rhs = -torch.autograd.grad(objective(w0), w0)[0].flatten()
        assert torch.allclose(w, torch.linalg.solve(h, rhs).reshape_as(w), atol=1e-8, rtol=1e-8)
        assert audit['relative_residual'] <= 1e-8
    rng = np.random.default_rng(56)
    risks = rng.uniform(.2, 2., (7, 4))
    oracle = [control.reward(torch.tensor(v), torch.tensor(risks[1]), 2, [0, 3]) for v in risks]
    assert np.allclose(rewards(risks, 2, [0, 3]), oracle, atol=1e-12)
    labels = np.repeat([2, 3], 6); identities = np.arange(12)
    losses = np.ones((7, 12)); historical = np.ones((7, 2))
    assert select(historical, losses, labels, identities, [0, 3], 43)[0] == 1
    losses[4] = .8; historical[4] = .8
    chosen, audit = select(historical, losses, labels, identities, [0, 3], 43)
    assert chosen == 4 and audit['lower_proxy_bounds'][4] > 0
    historical[4] = 4.
    assert select(historical, losses, labels, identities, [0, 3], 43)[0] == 1
    # Too few independent meta groups cannot establish an improvement.
    historical[4] = .8
    assert select(historical, losses, labels, labels, [0, 3], 43)[0] == 1
    permutation = rng.permutation(12)
    a = select(historical, losses, labels, identities, [0, 3], 43)
    b = select(historical, losses[:, permutation], labels[permutation], identities[permutation], [0, 3], 43)
    assert a == b
    return dict(status='PASS', checks=['seven explicit-sample Hessian oracles', 'translated moments',
        'independent scalar reward oracle', 'ties and harmful actions fall back',
        'credible improvement selected', 'small-group fallback', 'paired identity row-order invariance'],
        scientific_optimizer_updates=0, gpu_seconds=0, cpu_core_seconds=time.process_time()-start)


if __name__ == '__main__':
    print(run())
