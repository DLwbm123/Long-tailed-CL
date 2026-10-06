"""Read-only native checkpoint checks for the R1 causal controls; config on stdin."""
import json
from pathlib import Path
import sys
import torch

if __name__ == '__main__':
    config = json.load(sys.stdin); root = Path(config['admission_root'])
    def state(method, task):
        return torch.load(root / method / f'stage_{task}.pt', map_location='cpu', weights_only=False)
    records = []
    for method, steps in [('graph_stage', 4), ('no_shift', 4), ('frozen_after_first', 2)]:
        receipt = json.loads((root / method / 'STATUS.json').read_text())
        assert receipt['status'] == 'COMPLETE' and receipt['steps'] == steps and receipt['stages'] == 2
        diag = json.loads((root / method / 'diagnostics.json').read_text())
        assert max(d['ridge_residual'] for d in diag) < 1e-7
        records.append(dict(method=method, **receipt))
    a, b = state('frozen_after_first', 1), state('frozen_after_first', 2)
    assert a['adapter'].keys() == b['adapter'].keys()
    assert all(torch.equal(v, b['adapter'][k]) for k, v in a['adapter'].items())
    a, b = state('no_shift', 1), state('no_shift', 2)
    assert (a['bank']['mu'] == b['bank']['mu'][:2]).all()
    baseline = torch.load(config['baseline_checkpoint'], map_location='cpu', weights_only=False)
    first = state('graph_stage', 1)
    assert all(torch.allclose(v, baseline['adapter'][k], atol=1e-7, rtol=1e-6) for k, v in first['adapter'].items())
    assert torch.allclose(first['head'], baseline['head'], atol=1e-7, rtol=1e-6)
    last = json.loads((root / 'graph_stage/diagnostics.json').read_text())[-1]
    assert last['old_components'] > 0 and last['graph_edges'] > 0
    result = dict(status='PASS',steps=10,receipts=records,frozen_adapter_unchanged=True,
        no_shift_preserves_old_means=True,graph_stage_first_task_matches_baseline=True,
        graph_stage_second_task_uses_graph=True,effectiveness_claim=False)
    Path(config['output']).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
