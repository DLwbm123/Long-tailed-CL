import math
from collections import Counter
import torch
from torch.nn import functional as F
from opsa import OPSAAdapter, quadrature


def weights(y, groups, classes):
    sizes = Counter(groups)
    labels = {}
    for c, g in zip(y.tolist(), groups):
        assert labels.setdefault(g, c) == c
    nc = Counter(labels.values())
    assert set(nc) == set(range(classes))
    return torch.tensor([1 / (classes * nc[int(c)] * sizes[g]) for c, g in zip(y, groups)], dtype=torch.double)


def transform(g, mapper):
    return (((g-mapper['gn'][0])/mapper['gn'][1])[:, mapper['ids']]-mapper['on'][0])/mapper['on'][1]


def to_device(mapper):
    return {k: tuple(x.cuda() for x in v) if isinstance(v, tuple) else v.cuda() for k, v in mapper.items()}


class LinearControl(OPSAAdapter):
    """Same parameters, normalization and up radius, cosine factors fixed to one."""
    def forward(self, x, *, collect_mapc=False):
        assert not collect_mapc
        a = F.linear(x / x.norm(dim=-1, keepdim=True) * math.sqrt(self.width), self.down)
        return x + F.linear(a + quadrature(a), self.effective_up())


def lr_scale(step):
    return (step+1)/200 if step < 200 else (1 + math.cos(math.pi*(step-200)/600))/2


def metrics(scores, y, groups, classes):
    assert torch.isfinite(scores).all() and scores.shape == (len(y), classes)
    pred = scores.argmax(1)
    count = torch.bincount(y, minlength=classes)
    assert (count > 0).all()
    ok = [int((pred[y == c] == c).sum()) for c in range(classes)]
    recall = [a/int(b) for a, b in zip(ok, count)]
    return dict(BA=sum(recall)/classes, correct=ok, counts=count.tolist(), recall=recall,
                group_BA=float(weights(y, groups, classes)@(pred == y).double()),
                confusion=torch.bincount(y*classes+pred, minlength=classes*classes).reshape(classes, classes).tolist())


def pairs(base, method, y, classes):
    a, b = base.argmax(1) == y, method.argmax(1) == y
    return [dict(class_id=c, rescued=int((~a & b & (y == c)).sum()),
                 harmed=int((a & ~b & (y == c)).sum())) for c in range(classes)]


def risk_gate(summary, cells, paired, hk=False):
    f, m = summary['F'], summary['M']
    other = range(len(f['correct']))
    flags = dict(BA_GAIN=m['BA'] >= f['BA']+.01-1e-12,
        CLASS1_GAIN=m['correct'][1] >= f['correct'][1]+1,
        CLASS1_H0=paired['F'][1]['harmed'] == 0,
        CELL_BA=all(c['M']['BA'] >= c['F']['BA']+(.01 if hk else 0)-1e-12 for c in cells.values()),
        CELL_CLASS1=all(c['M']['correct'][1] >= c['F']['correct'][1]+(1 if hk else 0) for c in cells.values()),
        ALLCLASS_COST=all(m['recall'][k] >= f['recall'][k]-.01-1e-12 for k in other),
        CELL_ALLCLASS_COST=all(c['M']['recall'][k] >= c['F']['recall'][k]-.01-1e-12 for c in cells.values() for k in other),
        MATCHED_CONTROLS=all(m['BA'] > summary[n]['BA'] for n in ['H', 'L', 'N']),
        CELL_CONTROLS=all(c['M']['BA'] >= c[n]['BA']-1e-12 for c in cells.values() for n in ['H', 'L', 'N']))
    return dict(**flags, pass_all=all(flags.values()))


def synthetic_check():
    torch.manual_seed(74439)
    frozen=torch.nn.Linear(8,8).double().requires_grad_(False)
    adapter=OPSAAdapter(8,4).double()
    original=frozen.weight.detach().clone()
    handle=frozen.register_forward_pre_hook(lambda module,args:(adapter(args[0],collect_mapc=True),))
    x=torch.randn(2,3,8,dtype=torch.double)
    frozen(x).square().mean().backward()
    assert all(p.grad is not None and p.grad.norm()>0 for p in adapter.parameters())
    assert frozen.weight.grad is None and torch.equal(original,frozen.weight)
    handle.remove();adapter.clear_mapc()
    y=torch.tensor([0,0,1,1]);gs=['a','a','b','c']
    q=weights(y,gs,2);torch.testing.assert_close(q,torch.full((4,),.25,dtype=torch.double))
    score=F.one_hot(y,2).double();tab={n:metrics(score,y,gs,2) for n in ['F','H','L','N','M']}
    paired={n:pairs(score,score,y,2) for n in ['F','H','L','N']}
    assert not risk_gate(tab,{'0':tab,'1':tab},paired)['pass_all']
    assert lr_scale(0)==.005 and lr_scale(199)==1. and lr_scale(200)==1.
    return dict(internal_hook_gradient=True,frozen_backbone=True,group_weight_example=True,no_gain_gate_rejected=True,schedule_checked=True)
