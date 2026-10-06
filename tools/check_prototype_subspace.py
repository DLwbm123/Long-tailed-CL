"""R3 projection checks and optional native admission; private config on stdin."""
import json
from pathlib import Path
import sys
import torch
from run_prototype_single import old_head_basis, old_subspace_loss

if __name__ == '__main__':
    head = torch.tensor([[2., 0.], [0., 3.], [0., 0.]], requires_grad=True)
    q = old_head_basis(head)
    assert not q.requires_grad and q.shape == (3, 2)
    assert torch.allclose(q.T @ q, torch.eye(2))
    scaled = old_head_basis(100. * head)
    assert torch.allclose(q @ q.T, scaled @ scaled.T)
    assert old_head_basis(torch.tensor([[1., 2.], [0., 0.], [0., 0.]])).shape == (3, 1)
    assert old_head_basis(torch.zeros(3, 2)).shape == (3, 0)
    z = torch.tensor([[1., 2., 3.]], requires_grad=True)
    reference = torch.zeros_like(z, requires_grad=True)
    loss = old_subspace_loss(z, reference, q)
    assert loss.item() == 5.
    loss.backward()
    assert torch.equal(z.grad, torch.tensor([[2., 4., 0.]]))
    assert reference.grad is None and head.grad is None
    z.grad = None
    old_subspace_loss(z, reference, None).backward()
    assert torch.equal(z.grad, torch.zeros_like(z))
    assert old_subspace_loss(z, reference, old_head_basis(torch.zeros(3, 2))).item() == 0.
    config = json.load(sys.stdin)
    if config.get('math_only'):
        print('Projection, scaling, rank and stopped-gradient checks PASS')
        sys.exit(0)
    root = Path(config['admission_root'])
    receipt = json.loads((root / 'global_subspace30/STATUS.json').read_text())
    assert receipt['status'] == 'COMPLETE' and receipt['steps'] == 4 and receipt['stages'] == 2
    diag = json.loads((root / 'global_subspace30/diagnostics.json').read_text())
    assert all(d['fd_weight'] == 10. and d['old_logit_weight'] == 0. and d['subspace_weight'] == 30. for d in diag)
    assert all(d['old_subspace_rank'] == 0 and d['mean_old_subspace_loss'] == 0. for d in diag if d['task'] == 1)
    assert all(0 < d['old_subspace_rank'] <= 2 for d in diag if d['task'] == 2)
    assert diag[-1]['mean_old_subspace_loss'] > 0.
    assert max(d['ridge_residual'] for d in diag) < 1e-7
    baseline = torch.load(config['baseline_checkpoint'], map_location='cpu', weights_only=False)
    first = torch.load(root / 'global_subspace30/stage_1.pt', map_location='cpu', weights_only=False)
    assert all(torch.allclose(v, baseline['adapter'][k], atol=1e-7, rtol=1e-6) for k, v in first['adapter'].items())
    assert torch.allclose(first['head'], baseline['head'], atol=1e-7, rtol=1e-6)
    result = dict(status='PASS', steps=4, receipt=receipt, subspace_gradient_and_scale_invariance=True,
                  rank_deficiency_safe=True, first_task_matches_baseline=True, subspace_active_second_task=True,
                  effectiveness_claim=False)
    Path(config['output']).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
