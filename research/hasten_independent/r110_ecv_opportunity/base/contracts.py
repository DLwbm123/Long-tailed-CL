"""Explicit access boundaries. IDs are access keys and never model inputs."""
from dataclasses import dataclass
from typing import Mapping
import hashlib
import torch
from .memory import Bank, TOKEN_BYTES, STATS_BYTES


def token_keys(bank):
    return tuple((bank.pid * 8 + bank.view).cpu().tolist())


def set_key(bank):
    # Explicitly requested candidate-content identity; no feature/data hashing.
    return hashlib.sha256(','.join(map(str, token_keys(bank))).encode()).hexdigest()


@dataclass(frozen=True)
class VisibleState:
    pool: Bank
    memory: Bank | None
    observed: tuple
    birth: tuple
    stage: int
    capacity: int = 256
    groups: Mapping | None = None
    independent_groups_verified: bool = False
    access: object = None
    teacher: object = None

    def validate(self, model):
        assert self.access is not None, 'MISSING_DATA_ACCESS_CONTRACT'
        self.access.check(self.pool,'support')
        assert len(self.observed) == len(self.birth) == model.C
        assert set(self.pool.y.tolist()) == set(range(model.C))
        assert self.capacity >= model.C, 'B_LT_C_UNSUPPORTED_BY_HOST'
        if self.memory is not None:
            self.memory.assert_subset(self.pool)


class DataAccessLedger:
    """Real arrival frontier is distinct from pseudo-episode time."""
    def __init__(self, arrivals, groups, roles, frontier):
        self.arrivals = dict(arrivals)
        self.groups = dict(groups)
        self.roles = dict(roles)
        self.frontier = frontier
        self.events = []

    def check(self, bank, purpose, allowed_keys=None):
        assert purpose in {'support', 'query', 'candidate'}
        ids = set(bank.pid.cpu().tolist())
        assert ids <= self.arrivals.keys(), 'UNKNOWN_IDS'
        assert all(self.arrivals[i] <= self.frontier for i in ids), 'FUTURE_IDS'
        expected = {'S'} if purpose != 'query' else {'FIT', 'CAL', 'AUDIT'}
        assert all(self.roles[self.groups[i]] in expected for i in ids), 'QUERY_SUPPORT_FIREWALL'
        if allowed_keys is not None:
            assert set(token_keys(bank)) <= set(allowed_keys), 'EVICTED_FEATURE_ACCESS'
        self.events.append({'purpose': purpose, 'parents': len(ids), 'tokens': len(bank)})

    @staticmethod
    def disjoint_roots(partitions):
        seen = set()
        for name, groups in partitions.items():
            assert name in {'fit', 'calibration', 'audit'}
            groups = set(groups)
            assert not seen & groups, 'ROOT_GROUP_REUSE'
            seen |= groups


@dataclass(frozen=True)
class Candidate:
    memory: Bank
    family: str
    anchor: str
    removals: int

    @property
    def key(self):
        return set_key(self.memory)


def validate_candidate(candidate, state, anchor=None):
    bank = candidate.memory
    bank.assert_subset(state.pool)
    assert len(bank) == min(state.capacity, len(state.pool))
    assert len(bank) * TOKEN_BYTES + STATS_BYTES <= state.capacity * TOKEN_BYTES + STATS_BYTES
    assert set(bank.y[bank.gidx].tolist()) == set(range(len(state.observed)))
    # Preserve the host's common centroid-nearest global coverage anchors.
    pool=state.pool
    for c in range(len(state.observed)):
        ix=pool.gidx[pool.y[pool.gidx]==c]
        center=torch.nn.functional.normalize(pool.z[ix].mean(0),dim=0)
        required=int((pool.pid*8+pool.view)[ix[(pool.z[ix]@center).argmax()]])
        assert required in token_keys(bank), 'HOST_COVERAGE_ANCHOR_REMOVED'
    if anchor is not None:
        C = len(state.observed)
        for local in [False, True]:
            a = torch.bincount(anchor.y[(anchor.view > 0) == local], minlength=C)
            b = torch.bincount(bank.y[(bank.view > 0) == local], minlength=C)
            assert torch.equal(a, b), 'ANCHOR_CLASS_VIEW_QUOTA_CHANGED'

