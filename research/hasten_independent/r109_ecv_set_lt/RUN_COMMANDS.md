# Verified bounded engineering entry

The public package does not embed machine paths. A neutral launcher reads source and configuration paths from environment variables. For an existing Python environment with torch, numpy and Pillow, create a neutral directory and a launcher containing:

```python
import os, sys, runpy
from pathlib import Path
source = Path(os.environ['Q_SOURCE'])
sys.path.insert(0, str(source.parent))
runpy.run_module(source.name + '.cli', run_name='__main__')
```

Invoke it with a neutral interpreter/entry path, for example `python /tmp/.q/main.py`, with Q_SOURCE and Q_CONFIG supplied through the environment. Never place the project/method name in visible process arguments. The CLI records and checks its actual process arguments. Before a remote operation, follow the user's current server/storage instructions and preserve its independent hard timeout.

Required config fields: `science:false`, `command`, `output`, `deadline_epoch`, `max_wall_seconds` (≤86400), `max_updates` (≤5000), and `threads` (≤2). The checked configuration is engineering-only; scientific admission is rejected explicitly.

Supported commands:

- `audit`, `report`: read implementation contract.
- `smoke`: execute the complete synthetic H=3 workflow, including DIRECT_RETURN, dynamic verifier, SFT, PG, MORE_SUP and reward shuffle.
- `collect-pairs`: create synthetic paired feedback and save it privately.
- `train-verifier`: use `records`, optional `epochs`/`dynamics`; fits normalizers on fit records only.
- `train-sft`: use `records` and `verifier` checkpoint.
- `train-pg`: use `trajectories`, `policy`, `reference`, `verifier`, `sigma_fit`; consumes actually executed stored trajectories.
- `evaluate`: use `records`, `verifier`, `policy`; evaluates a frozen sampled selection against saved feedback, without new classifier training.
- `real-integration`: machine-private host configuration and CPU affinity; exactly 54 current first-arrival TRAIN images, full 256-token memory, 200 warmup updates, one virtual update, two fixed 100-step branches.

`dry_run:true` validates dispatch without executing the command. Real dispatch and CLI smoke logs are recorded in TEST_REPORT.md. Stored tensor files are trusted locally generated artifacts, not arbitrary downloaded checkpoints. No unexecuted CLI is described as a runnable scientific campaign.

The original formal learner/evaluator is not invoked by `evaluate`. A complete scientific runner and its native total-budget admission remain outside this engineering entry's admitted scope.

