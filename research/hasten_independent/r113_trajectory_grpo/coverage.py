"""Use the validator's per-class centroid arithmetic when constructing anchors."""
import importlib
import torch

E = importlib.import_module('r112_evidence_memory.evidence')
OriginalChoice = E.Choice


class CanonicalChoice(OriginalChoice):
    def __init__(self, pool, C, capacity, core):
        self.__dict__.update(E.qchoice(pool, C, capacity).__dict__)
        centers, coverage = [], []
        for c in range(C):
            ix = pool.gidx[pool.y[pool.gidx] == c]
            center = torch.nn.functional.normalize(pool.z[ix].mean(0), dim=0)
            centers.append(center)
            coverage.append(ix[(pool.z[ix] @ center).argmax()])
        self.cent = torch.stack(centers)
        self.cover = torch.stack(coverage)
        keys = pool.pid * 8 + pool.view
        ix = torch.searchsorted(keys, core)
        assert bool((ix < len(keys)).all()) and torch.equal(keys[ix], core)
        self.cover = torch.cat([self.cover, ix[~torch.isin(ix, self.cover)]])
        self.optional = pool.gidx[~torch.isin(pool.gidx, self.cover)]
        self.capacity_hist = torch.bincount(self.localcap[pool.parent_index[self.optional]], minlength=6)
        self.covercap = int(self.localcap[pool.parent_index[self.cover]].sum())
        D = len(self.cover)
        ng = torch.arange(D, min(self.B, len(pool.gidx)) + 1, device=pool.z.device)
        self.ng = ng[E.legacy.top_capacity(self.capacity_hist.expand(len(ng), -1), ng - D)
                     + self.covercap >= self.B - ng]
        assert len(self.ng), 'PINNED_SUPPORT_INFEASIBLE'


def install():
    # Local to the R113 interpreter; the frozen R112 source is not modified.
    E.Choice = CanonicalChoice
