"""Integer denominators first; absent classes never become zero recall."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class Counts:
    correct: torch.Tensor
    denominator: torch.Tensor
    ids: tuple
    good: torch.Tensor
    brier: torch.Tensor

    def ba(self, classes=None):
        mask = self.denominator > 0
        if classes is not None:
            mask &= torch.tensor([c in classes for c in range(len(mask))])
        if not mask.any():
            return None
        return float((self.correct[mask].double()/self.denominator[mask]).mean())


@torch.no_grad()
def evaluate(model, query, counter=None):
    x, y, ids = query.bags()
    assert bool((y < model.C).all()), 'UNSEEN_QUERY_LABEL'
    p = model(x).double().softmax(-1)
    good = p.argmax(1) == y
    correct = torch.bincount(y[good], minlength=model.C).cpu()
    denominator = torch.bincount(y, minlength=model.C).cpu()
    brier = (p-F.one_hot(y, model.C)).square().sum(1).cpu()
    if counter is not None:
        counter['query_rows'] = counter.get('query_rows',0)+len(y)
    return Counts(correct, denominator, tuple(ids.cpu().tolist()), good.cpu(), brier)


def paired(candidate, zero, oldtail, tolerance=.005):
    assert candidate.ids == zero.ids
    assert torch.equal(candidate.denominator, zero.denominator)
    macro = candidate.ba()-zero.ba()
    ct, zt = candidate.ba(oldtail), zero.ba(oldtail)
    tail = None if ct is None else ct-zt
    return dict(macro=macro, tail=tail,
                rescued=int((candidate.good & ~zero.good).sum()),
                harmed=int((~candidate.good & zero.good).sum()),
                correct=candidate.correct.tolist(), denominator=candidate.denominator.tolist())


def utility(endpoints, tolerance=.005):
    assert endpoints
    macro = sum(p['macro'] for p in endpoints)/len(endpoints)
    tail = [p['tail'] for p in endpoints if p['tail'] is not None]
    penalty = max([max(0., -x-tolerance) for x in tail], default=0.)
    return dict(R=macro-2*penalty, macro=macro,
                tail_available_horizons=len(tail), horizons=len(endpoints),
                tail_coverage=len(tail)/len(endpoints), tail_harm_max=max([-x for x in tail],default=None))


def near_optimal(returns, epsilon=.005, feasible=None, positive_evidence=False):
    assert len(returns) and abs(float(returns[0])) < 1e-10
    ok = [True]*len(returns) if feasible is None else feasible
    assert ok[0]
    if not positive_evidence:
        return torch.tensor([i == 0 for i in range(len(returns))])
    best = max(float(r) for r, keep in zip(returns,ok) if keep)
    return torch.tensor([keep and best-float(r) <= epsilon for r,keep in zip(returns,ok)])

