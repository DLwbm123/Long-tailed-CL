"""New low-rank synthetic noise distribution; reuse the frozen classifier unchanged."""
import os
from contextlib import contextmanager

import torch
import check_shared_zero_memory as shared
from check_progressive_decision_memory import DT, fixture as original_fixture


def lowrank_fixture(cfg):
    rank = 2 * cfg['classes']
    assert cfg['dimension'] >= rank
    x, y, counts = original_fixture(dict(cfg, dimension=rank))
    padded = torch.zeros(len(x), cfg['dimension'], dtype=DT)
    padded[:, :rank] = x
    return padded, y, counts


@contextmanager
def fixture_binding():
    # Dedicated worker only: restore even if a scientific assertion fails.
    previous = shared.fixture
    assert previous is original_fixture
    shared.fixture = lowrank_fixture
    try:
        yield
    finally:
        shared.fixture = previous


def unit_checks():
    cfg = dict(dimension=12, classes=3, n=31, seed=335, count_weights=[4, 2, 1])
    saved = dict(cfg)
    x, y, counts = lowrank_fixture(cfg)
    z, labels, reference_counts = original_fixture(dict(cfg, dimension=6))
    assert torch.equal(x[:, :6], z) and torch.equal(y, labels) and counts == reference_counts
    assert int(torch.count_nonzero(x[:, 6:])) == 0 and cfg == saved
    norm_error = float((x.norm(dim=1) - 1).abs().max())
    assert norm_error < 1e-12
    other, _, _ = lowrank_fixture(dict(cfg, seed=cfg['seed'] + 10000))
    assert not torch.equal(x, other)
    try:
        with fixture_binding():
            assert shared.fixture is lowrank_fixture
            raise ValueError('intentional binding restoration check')
    except ValueError:
        pass
    assert shared.fixture is original_fixture
    current = {j: x[y == j] for j in range(3)}
    w = torch.zeros(3, 12, dtype=DT)
    a, _ = shared.train_head(w, current, {}, 'oracle_local', 1, oracle=current)
    b, _ = shared.train_head(w, current, {}, 'quadratic_local', 1)
    assert torch.equal(a, b)
    return dict(status='PASS', unit_norm_error=norm_error, zero_tail=True,
                ambient_embedding_exact=True, configuration_unchanged=True,
                evaluation_seed_independent=True, fixture_binding_restored=True,
                first_step_without_old_memory_exact=True, synthetic_head_updates=2)


def run(cfg, lock, journal):
    torch.set_num_threads(lock['threads'])
    torch.set_default_dtype(DT)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    if cfg['kind'] == 'unit':
        result = unit_checks()
    else:
        assert cfg['active_dimension'] == 2 * cfg['classes'] <= cfg['dimension']
        with fixture_binding():
            result = shared.run_scale(cfg, lock, journal)
        result.update(fixture_active_dimension=cfg['active_dimension'],
                      noise_coordinate_second_moment=1 / cfg['active_dimension'],
                      classifier_algorithm_unchanged=True)
    assert shared.fixture is original_fixture
    result.update(id=cfg['id'], GPU_initialized=torch.cuda.is_initialized(),
                  medical_images=0, medical_checkpoints=0, val_accessed=False,
                  test_accessed=False, formal_updates=0)
    assert not result['GPU_initialized']
    return result
