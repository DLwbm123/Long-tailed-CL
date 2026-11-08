"""Synthetic integrity and nonzero policy-gradient checks; no image access."""
import copy
import importlib
import json
import os
import time
from pathlib import Path
import torch
from . import trajectory as T
from .worker import actors


def run(output):
    start = time.time()
    torch.set_num_threads(1)
    torch.manual_seed(113)
    native = importlib.import_module('r112_evidence_memory.native')
    generic = torch.nn.functional.normalize(torch.randn(30, 6, 512), dim=-1)
    pool = native.fixture(generic, 100, 0, torch.arange(100) % 3)
    future = [native.fixture(generic, 8, 10000 + 1000 * h, torch.full((8,), 3 + h), 1 + h) for h in range(2)]
    model = T.old.learner.Learner(113)
    model.forbidden = {int(i) for x in [pool] + future for role in [1, 2] for i in x.role(role).parents}
    model.arrive(pool.role(0), 3)
    opt = T.old.learner.optimizer(model)
    count = {}
    packed = T.prepare(pool, None, model, opt, 0, 113, count)
    old_candidates, _ = T.old.generate(pool, None, model, opt, 0, 113, count)
    assert [c.key for c in packed[0]] == [c.key for c in old_candidates]
    ref, policies, opts = actors(113, 'cpu')
    batch = T.collect(pool, future, model, opt, None, policies['LONG_GRPO'], packed,
                      {0, 1, 2}, 113, 0, count, pre=2, post=2)
    assert all([a['C'] for a in r['allocations']] == [3, 4, 5] for r in batch['runs'])
    assert all(len(r['steps']) == 3 for r in batch['runs'])
    # Known synthetic rewards exercise the optimizer independently of random accuracy.
    for r, reward in zip(batch['runs'], [.1, -.1, .05, -.05]):
        r['R'] = reward
        r['immediate'] = [reward, -reward, 2 * reward]
    assert torch.allclose(T.advantages(batch, False).mean(0), torch.zeros(3), atol=1e-6)
    assert not torch.allclose(T.advantages(batch, False), T.advantages(batch, True))
    before = {k: v.clone() for k, v in policies['LONG_GRPO'].state_dict().items()}
    history = T.update(policies['LONG_GRPO'], ref, opts['LONG_GRPO'], batch)
    assert any(not torch.equal(before[k], v) for k, v in policies['LONG_GRPO'].state_dict().items())
    assert any(r['grad_norm'] > 0 for r in history)
    T.supervise(policies['SUP_SHARED'], opts['SUP_SHARED'], batch)
    assert all(not p.requires_grad for p in ref.parameters())
    # Equal group returns have exactly zero group-relative advantages.
    same = copy.deepcopy(batch)
    for r in same['runs']:
        r['R'] = -.1
    assert not bool(T.advantages(same, False).any())
    result = {'passed': True, 'synthetic_only': True, 'real_images': 0, 'real_science_updates': 0,
              'checks': ['R112 candidate equivalence', 'three genuine own-state decisions',
                         'group-centered advantages', 'long versus immediate rewards',
                         'nonzero actor gradients and parameter updates', 'frozen reference',
                         'supervised same-feedback update', 'zero-variance negative groups'],
              'wall_seconds': time.time() - start, 'engineering_counts': count}
    Path(output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    run(os.environ['Q_OUTPUT'])
