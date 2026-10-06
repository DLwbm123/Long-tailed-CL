"""Lagged error-feedback math and four-arm native admission; config on stdin."""
import json
from pathlib import Path
import sys
import torch
from run_prototype_single import error_factors, update_errors

if __name__ == '__main__':
    ema = torch.full((4,), .5)
    used = error_factors(ema, [0, 1])
    assert torch.equal(used, torch.ones(4))
    update_errors(ema, torch.tensor([0, 0, 1]), torch.tensor([1, 1, 1]), [0, 1])
    assert torch.allclose(ema, torch.tensor([.55, .45, .5, .5]))
    assert torch.equal(used, torch.ones(4))  # The previous batch's weights cannot change retroactively.
    following = error_factors(ema, [0, 1])
    assert following[0] > 1 > following[1] and not following.requires_grad
    assert torch.allclose(following[:2].mean(), torch.tensor(1.))
    unchanged = ema[1:].clone()
    update_errors(ema, torch.tensor([0]), torch.tensor([0]), [0, 1])
    assert torch.equal(ema[1:], unchanged)  # Absent and old classes acquire no false FN estimate.
    extreme = error_factors(torch.tensor([0., 1., 0., 1.]), [0, 1, 2, 3])
    assert extreme.min() >= .5 and extreme.max() <= 2. and extreme.mean() == 1.
    config = json.load(sys.stdin)
    if config.get('math_only'):
        print('Lag, normalization, bounds and absent-class feedback checks PASS')
        sys.exit(0)
    root = Path(config['admission_root']); receipts = []
    def checkpoint(name, stage):
        return torch.load(root / name / f'stage_{stage}.pt', map_location='cpu', weights_only=False)
    def same(a, b):
        assert a['adapter'].keys() == b['adapter'].keys()
        assert all(torch.allclose(v, b['adapter'][k], atol=1e-7, rtol=1e-6) for k, v in a['adapter'].items())
        assert torch.allclose(a['head'], b['head'], atol=1e-7, rtol=1e-6)
    for name, feedback, multi in [('A', False, False), ('B', False, True), ('C', True, False), ('D', True, True)]:
        receipt = json.loads((root / name / 'STATUS.json').read_text())
        assert receipt['status'] == 'COMPLETE' and receipt['steps'] == 4 and receipt['stages'] == 2
        diag = json.loads((root / name / 'diagnostics.json').read_text())
        assert all(d['error_feedback'] == feedback and d['trained_encoder'] for d in diag)
        assert all(d['fd_weight'] == 10. and d['old_logit_weight'] == 0. and d['subspace_weight'] == 0. for d in diag)
        assert max(d['ridge_residual'] for d in diag) < 1e-7
        if feedback:
            assert any(d['feedback_applied_max'] > 1.000001 for d in diag)
            for d in diag:
                assert .5 <= d['feedback_applied_min'] <= d['feedback_applied_max'] <= 2.
                assert abs(sum(d['feedback_next_factors'].values()) / 2 - 1.) < 1e-6
                matrix = torch.tensor(d['training_preupdate_confusion'])
                current = [0, 1] if d['task'] == 1 else [2, 3]
                assert matrix.sum() == 64 and matrix[current].sum() == matrix.sum()
        else:
            assert all(d['feedback_error_ema'] is None for d in diag)
        if multi:
            assert diag[-1]['old_components'] > 2 and diag[-1]['graph_edges'] > 0
        receipts.append(dict(condition=name, **receipt))
    baseline = torch.load(config['baseline_checkpoint'], map_location='cpu', weights_only=False)
    same(checkpoint('A', 1), baseline)
    same(checkpoint('A', 1), checkpoint('B', 1))
    same(checkpoint('C', 1), checkpoint('D', 1))
    # First-task errors can be equal, correctly leaving neutral weights unchanged.
    control, feedback_state = checkpoint('A', 2), checkpoint('C', 2)
    assert any(not torch.equal(v, feedback_state['adapter'][k]) for k,v in control['adapter'].items())
    result = dict(status='PASS', steps=16, receipts=receipts, delayed_feedback_math=True,
        old_unobserved_fn_not_estimated=True, control_first_task_matches_baseline=True,
        first_task_multi_pair_parity=True, feedback_changes_training=True, multi_graph_active=True,
        effectiveness_claim=False)
    Path(config['output']).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
