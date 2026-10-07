"""CPU-only audit of completed FINALHEAD1 receipts; no images/models/optimizers."""
import csv
import json
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
import time

import numpy as np


def run(root):
    started = time.process_time(); root = Path(root); public = root/'public'
    def read(path):
        return json.loads(path.read_text())
    def save(path, value):
        path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    result = read(public/'main/RESULTS.json'); ledger = read(public/'BUDGET_LEDGER.json')
    config = read(root/'cycle.private.json'); lock = read(public/'PROTOCOL_LOCK.json')
    status = read(root/'main_FINALHEAD/STATUS.json')
    evaluation = read(root/'eval_main_FINALHEAD/STATUS.json')
    rows = read(root/'main_FINALHEAD/CONTROLLER.json')['decisions']
    assert ledger['formal_ids'] == ['main_FINALHEAD'] and ledger['gpu_reserved_seconds'] == 0
    train, evaluate = ledger['jobs']
    assert train['id'] == 'main_FINALHEAD' and evaluate['id'] == 'eval_main_FINALHEAD'
    assert all(j['exit_code'] == 0 for j in ledger['jobs']) and not result['failures']
    assert status['status'] == 'TRAINED' and status['phase'] == 'SELECTION_SEALED'
    assert status['actual_updates'] == status['policy_updates'] == status['rollout_updates'] == 0
    assert status['reused_encoder_updates'] == 476 and evaluation['status'] == 'COMPLETE'
    assert evaluate['started'] >= train['started']+train['elapsed_seconds']
    assert ledger['total_gpu_seconds'] < lock['gpu_seconds_limit']
    assert [r['task'] for r in rows] == [2, 3, 4]
    with (Path(config['base']['manifest'])/'train.csv').open() as stream:
        counts = Counter(int(r['original_label']) for r in csv.DictReader(stream))
    tail = set(sorted(config['base']['order'], key=lambda c: (-counts[c], c))[4:])
    decisions = []
    for r in rows:
        risks = np.asarray(r['risks']); old = risks.shape[1]-2
        tail_ids = [i for i, c in enumerate(r['seen']) if c in tail]
        gain = (risks[1]-risks)/np.maximum(risks[1], .1); negative = np.minimum(gain, 0)
        parts = dict(mean_gain=gain.mean(1), old_penalty=-negative[:, :old].mean(1),
            new_penalty=-negative[:, old:].mean(1), tail_penalty=-negative[:, tail_ids].mean(1))
        reward = parts['mean_gain']-parts['old_penalty']-parts['new_penalty']-parts['tail_penalty']
        assert risks.shape == (7, r['task']*2) and np.isfinite(risks).all()
        assert np.allclose(reward, r['rewards'], atol=1e-12, rtol=1e-12)
        bounds = reward-r['simultaneous_correction']; bounds[1] = 0.
        assert np.allclose(bounds, r['lower_proxy_bounds'], atol=1e-12, rtol=1e-12)
        eligible = [a for a in range(7) if a != 1 and bounds[a] > 1e-8] if min(r['meta_identity_counts']) >= 2 else []
        selected = max(eligible, key=lambda a: (bounds[a], -a)) if eligible else 1
        assert selected == r['selected_action'] and eligible == r['eligible_actions']
        assert r['candidate_meta_excluded'] and r['encoder_optimizer_updates'] == 0
        assert sum(r['candidate_fit_counts'][-2:]) == r['fit_n']
        assert r['fixed1_head_max_abs_error'] <= 2e-6
        residual = max(s['relative_residual'] for s in list(r['candidate_solves'])+[r['final_solve']])
        assert residual <= 1e-8 and r['selected_at'] < evaluate['started']
        alternative = max((a for a in range(7) if a != 1), key=lambda a: reward[a])
        decisions.append(dict(task=r['task'], selected_action=selected, point_best=int(reward.argmax()),
            best_alternative=alternative, best_alternative_reward=float(reward[alternative]),
            best_alternative_bound=float(bounds[alternative]), all_alternative_point_rewards_negative=bool(np.all(np.delete(reward, 1) < 0)),
            bootstrap_correction=r['simultaneous_correction'], meta_identity_counts=r['meta_identity_counts'],
            max_solve_residual=residual, fixed1_head_max_abs_error=r['fixed1_head_max_abs_error'],
            reward_parts={k:v.tolist() for k,v in parts.items()}))
    metrics = {}
    for name, record in result['records'].items():
        m = record['metrics']; last = m['stages'][-1]
        assert abs(np.mean(list(last['per_class_recall'].values()))-m['final_balanced_accuracy']) < 1e-12
        metrics[name] = dict(ba=m['final_balanced_accuracy'], average_ba=m['average_incremental_balanced_accuracy'],
            tail=m['final_tail_recall'], forgetting=m['forgetting'],
            new_recall=float(np.mean([last['per_class_recall'][str(c)] for c in last['seen'][-2:]])))
    pairs = [p for p in result['paired_predictions'] if p['a'] == 'main_FIXED1' and p['b'] == 'main_FINALHEAD']
    assert len(pairs) == 4 and all(p['prediction_agreement'] == 1 for p in pairs)
    differences = {name:{k:100*(metrics['main_FINALHEAD'][k]-v[k]) for k in v} for name,v in metrics.items() if name != 'main_FINALHEAD'}
    d = differences['main_R']; fixed = differences['main_FIXED1']
    checks = dict(ba_gain=d['ba'] >= 1, tail=d['tail'] >= -.5, forgetting=d['forgetting'] <= 1, new_recall=d['new_recall'] >= -1,
        above_fixed1=fixed['ba'] > 0, above_pc=differences['main_PC']['ba'] > 0,
        fixed1_tail=fixed['tail'] >= -.5, fixed1_forgetting=fixed['forgetting'] <= 1, fixed1_new_recall=fixed['new_recall'] >= -1)
    assert all(checks.values()) == result['screen_success'] == False
    completed = max(j['started']+j['elapsed_seconds'] for j in ledger['jobs'])
    assert completed < lock['deadline']
    summary = dict(cycle='FINALHEAD1', status='STOPPED_SCREEN_FAILED', phase='CYCLE_CLOSED',
        source_commit=config['source_commit'], completed_at=completed,
        completed_at_shanghai=datetime.fromtimestamp(completed,timezone(timedelta(hours=8))).isoformat(),
        screen_success=False, stop_reason='NO_GAIN_OVER_FIXED1_AND_BA_GAIN_BELOW_ONE_PP',
        metrics=metrics, differences_pp=differences, screen_checks=checks, decisions=decisions,
        fixed1_prediction_parity_all_stages=True, paired_fixed1_predictions=pairs,
        gpu_seconds_used=ledger['total_gpu_seconds'], gpu_reserved_seconds=0, formal_attempts=1,
        actual_optimizer_updates=0, reused_encoder_updates=476, failures=[],
        unrun=['new seeds', 'RL revision', 'HyperKvasir transfer'], unrun_reason='frozen initial screen failed',
        test_accessed=False, independent_confirmation=False, paper_evidence_ready=False,
        completion_audit_cpu_core_seconds=time.process_time()-started, completion_audit_gpu_seconds=0)
    save(public/'COMPLETION_AUDIT.json', dict(status='PASS', train=status, evaluation=evaluation,
        evaluation_after_selection=True, independent_reward_recalculation=True,
        decisions=decisions, cpu_core_seconds=summary['completion_audit_cpu_core_seconds']))
    save(public/'FINAL_SUMMARY.json', summary); save(public/'FINAL_BUDGET_LEDGER.json', ledger)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('decisions','paired_fixed1_predictions','metrics','differences_pp')},indent=2))


if __name__ == '__main__':
    run(json.load(sys.stdin)['root'])
