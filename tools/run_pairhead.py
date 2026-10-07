"""One fixed readout: redistribute FIXED1 competition within each age group."""
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F
import core1_competition as competition
import pcrl_control as control
import prototype_coherent as method
from run_finalhead import device_bank
from run_prototype_single import save


def pair_risks(bank, w):
    values = []
    for c in range(len(bank['n'])):
        direction = w[:, c:c+1]-w
        mean = bank['mu'][c] @ direction
        variance = ((bank['Q'][c] @ direction)*direction).sum(0)-mean.square()
        values.append(F.softplus(-mean/(variance.clamp_min(0)+1e-4).sqrt()).clamp_max(20.))
    risk = torch.stack(values); risk.fill_diagonal_(0)
    if not torch.isfinite(risk).all():
        raise ValueError('Nonfinite pair risks')
    return risk


def redistribute(base, risk, old):
    k = len(base)
    if (base.shape != (k, k) or risk.shape != base.shape or not 0 < old < k
            or not torch.isfinite(base).all() or not torch.isfinite(risk).all()
            or (base < 0).any() or (risk < 0).any() or (base.diag() != 0).any()):
        raise ValueError('Invalid pair allocation inputs')
    result = base.clone()
    for group in (range(old), range(old, k)):
        for c in range(k):
            ids = [j for j in group if j != c]
            if not ids:
                continue
            mass = base[c, ids].sum(); total = risk[c, ids].sum()
            if total > 1e-12:
                result[c, ids] = .5*base[c, ids]+.5*mass*risk[c, ids]/total
            if not torch.allclose(result[c, ids].sum(), mass, atol=1e-12, rtol=1e-12):
                raise ValueError('Per-class age-group budget changed')
    if not torch.isfinite(result).all() or (result < .5*base).any():
        raise ValueError('Invalid retained pair mass')
    return result.detach()


@torch.no_grad()
def run(config):
    output = Path(config['output']); source = Path(config['checkpoint_directory'])
    if output.exists() or config['method'] != 'PAIRHEAD':
        raise ValueError('Fresh PAIRHEAD output required; no retry')
    receipt = json.loads((source/'STATUS.json').read_text())
    if receipt.get('status') != 'TRAINED' or receipt.get('steps') != 476:
        raise ValueError('Incomplete FIXED1 source')
    output.mkdir(parents=True); save(output/'INPUT.private.json', config)
    started = time.monotonic(); decisions = []
    def status(**fields):
        save(output/'STATUS.json', dict(steps=476, reused_encoder_updates=476,
            actual_updates=0, retained_updates=0, rollout_updates=0, policy_updates=0,
            diagnostic_gpu_seconds=0., elapsed_seconds=time.monotonic()-started, **fields))
    try:
        torch.set_num_threads(4)
        status(status='RUNNING', phase='initialize')
        (output/'stage_1.pt').symlink_to(source/'stage_1.pt')
        for task in range(2, 5):
            if time.monotonic()-started >= config['max_wall_seconds']:
                raise TimeoutError('PAIRHEAD residence cap')
            status(status='RUNNING', phase='pair_refit', task=task)
            current = torch.load(source/f'stage_{task}.pt', map_location='cpu', weights_only=False)
            if (current['seen'] != config['order'][:2*task]
                    or current['configuration']['method'] != 'FIXED1'
                    or current['configuration']['seed'] != config['seed']
                    or current['controller_action'] != 1):
                raise ValueError('Incorrect fixed encoder source')
            bank = device_bank(current['bank']); old = 2*(task-1)
            native, _ = method.head(bank, 0.)
            base = control.allocation(competition.competition(bank, native), old, 1)
            fixed, fixed_solve = competition.bank_head(bank, base, .5)
            error = float((fixed.float().cpu()-current['head']).abs().max())
            if not torch.allclose(fixed.float().cpu(), current['head'], atol=2e-6, rtol=2e-6):
                raise ValueError('FIXED1 reconstruction mismatch')
            risk = pair_risks(bank, fixed); pairs = redistribute(base, risk, old)
            chosen, solve = competition.bank_head(bank, pairs, .5)
            audit = dict(task=task, seen=current['seen'], training_counts=bank['n'],
                risk=risk.cpu().tolist(), base_pairs=base.cpu().tolist(), pairs=pairs.cpu().tolist(),
                fixed1_head_max_abs_error=error, fixed_solve=fixed_solve, final_solve=solve,
                head_change_norm=float((chosen-fixed).norm()), pair_change_norm=float((pairs-base).norm()),
                old_target_mass_error=float((pairs[:, :old].sum(1)-base[:, :old].sum(1)).abs().max()),
                new_target_mass_error=float((pairs[:, old:].sum(1)-base[:, old:].sum(1)).abs().max()),
                sealed_at=time.time(), new_optimizer_updates=0, image_reads=0)
            decisions.append(audit)
            save(output/'CONTROLLER.json', dict(decisions=decisions, adapter_updates=0, policy_updates=0))
            saved = dict(current, head=chosen.float().cpu(), controller_action=None,
                configuration=config, readout_selection=audit)
            torch.save(saved, output/f'stage_{task}.pt')
            del saved, current, bank, native, base, fixed, risk, pairs, chosen
        status(status='TRAINED', phase='SELECTION_SEALED', stages=4,
            peak_gpu_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        status(status='INCOMPLETE', error=str(exc)); raise


if __name__ == '__main__':
    run(json.load(sys.stdin))
