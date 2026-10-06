"""R2 retention gradient and native checkpoint checks; optional config on stdin."""
import json
from pathlib import Path
import sys
import torch
from run_prototype_single import old_readout_loss

if __name__ == '__main__':
    z = torch.tensor([[1., 2., 3.]], requires_grad=True)
    reference = torch.zeros_like(z, requires_grad=True)
    head = torch.tensor([[2., 0.], [0., 3.], [0., 0.]], requires_grad=True)
    loss = old_readout_loss(z, reference, head)
    assert loss.item() == 20.
    loss.backward()
    assert torch.equal(z.grad, torch.tensor([[4., 18., 0.]]))
    assert reference.grad is None and head.grad is None
    z.grad = None
    old_readout_loss(z, reference, None).backward()
    assert torch.equal(z.grad, torch.zeros_like(z))
    config = json.load(sys.stdin)
    root = Path(config['admission_root'])
    receipts = []
    for name, fd, weight in [('global_fd40', 40., 0.), ('global_readout1', 10., 1.)]:
        receipt = json.loads((root / name / 'STATUS.json').read_text())
        assert receipt['status'] == 'COMPLETE' and receipt['steps'] == 4 and receipt['stages'] == 2
        diag = json.loads((root / name / 'diagnostics.json').read_text())
        assert all(d['fd_weight'] == fd and d['old_logit_weight'] == weight for d in diag)
        assert all(d['old_head_classes'] == (0 if d['task'] == 1 else 2) for d in diag)
        assert all(d['mean_old_readout_loss'] == 0 for d in diag if d['task'] == 1)
        if weight:
            assert diag[-1]['mean_old_readout_loss'] > 0
        assert max(d['ridge_residual'] for d in diag) < 1e-7
        receipts.append(dict(condition=name, **receipt))
    baseline = torch.load(config['baseline_checkpoint'], map_location='cpu', weights_only=False)
    first = torch.load(root / 'global_readout1/stage_1.pt', map_location='cpu', weights_only=False)
    assert all(torch.allclose(v, baseline['adapter'][k], atol=1e-7, rtol=1e-6) for k, v in first['adapter'].items())
    assert torch.allclose(first['head'], baseline['head'], atol=1e-7, rtol=1e-6)
    result = dict(status='PASS', steps=8, receipts=receipts, gradient_and_stopgrad_check=True,
                  first_task_matches_baseline=True, readout_active_second_task=True, effectiveness_claim=False)
    Path(config['output']).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
