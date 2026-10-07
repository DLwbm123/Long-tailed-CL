"""Completed-readout receipt audit; no model, image, optimizer or GPU access."""
from datetime import datetime, timedelta, timezone
import json
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
    lock = read(public/'PROTOCOL_LOCK.json'); config = read(root/'cycle.private.json')
    status = read(root/'main_PAIRHEAD/STATUS.json')
    evaluation = read(root/'eval_main_PAIRHEAD/STATUS.json')
    rows = read(root/'main_PAIRHEAD/CONTROLLER.json')['decisions']
    assert ledger['formal_ids'] == ['main_PAIRHEAD'] and ledger['gpu_reserved_seconds'] == 0
    train, evaluate = ledger['jobs']
    assert train['id'] == 'main_PAIRHEAD' and evaluate['id'] == 'eval_main_PAIRHEAD'
    assert all(j['exit_code'] == 0 for j in ledger['jobs']) and not result['failures']
    assert status['status'] == 'TRAINED' and status['phase'] == 'SELECTION_SEALED'
    assert status['actual_updates'] == status['rollout_updates'] == status['policy_updates'] == 0
    assert status['reused_encoder_updates'] == 476 and evaluation['status'] == 'COMPLETE'
    assert evaluate['started'] >= train['started']+train['elapsed_seconds']
    assert ledger['total_gpu_seconds'] < lock['gpu_seconds_limit']
    assert [r['task'] for r in rows] == [2, 3, 4]
    decisions = []
    for r in rows:
        a = np.asarray(r['base_pairs']); b = np.asarray(r['pairs']); risk = np.asarray(r['risk'])
        k = 2*r['task']; old = k-2
        assert a.shape == b.shape == risk.shape == (k, k)
        assert all(np.isfinite(v).all() and (v >= 0).all() for v in (a, b, risk))
        assert np.all(np.diag(b) == 0) and np.all(b >= .5*a)
        expected = a.copy(); errors = []
        for c in range(k):
            for group in (range(old), range(old, k)):
                ids = [j for j in group if j != c]
                total = sum(risk[c, j] for j in ids); mass = sum(a[c, j] for j in ids)
                for j in ids:
                    if total > 1e-12:
                        expected[c, j] = .5*a[c, j]+.5*mass*risk[c, j]/total
                errors.append(abs(sum(b[c, j] for j in ids)-mass))
        assert np.allclose(b, expected, atol=1e-12, rtol=1e-12) and max(errors) <= 1e-12
        residual = max(r[n]['relative_residual'] for n in ('fixed_solve', 'final_solve'))
        assert residual <= 1e-8 and r['fixed1_head_max_abs_error'] <= 2e-6
        assert r['sealed_at'] < evaluate['started'] and r['new_optimizer_updates'] == r['image_reads'] == 0
        decisions.append(dict(task=r['task'], max_group_mass_error=max(errors),
            max_solve_residual=residual, fixed1_head_max_abs_error=r['fixed1_head_max_abs_error'],
            pair_change_norm=r['pair_change_norm'], head_change_norm=r['head_change_norm']))
    metrics = {}
    for name, record in result['records'].items():
        m = record['metrics']; last = m['stages'][-1]
        assert abs(np.mean(list(last['per_class_recall'].values()))-m['final_balanced_accuracy']) < 1e-12
        metrics[name] = dict(ba=m['final_balanced_accuracy'], average_ba=m['average_incremental_balanced_accuracy'],
            tail=m['final_tail_recall'], forgetting=m['forgetting'],
            new_recall=float(np.mean([last['per_class_recall'][str(c)] for c in last['seen'][-2:]])))
    differences = {name:{k:100*(metrics['main_PAIRHEAD'][k]-v[k]) for k in v}
                   for name,v in metrics.items() if name != 'main_PAIRHEAD'}
    d = differences['main_R']; f = differences['main_FIXED1']
    checks = dict(ba_gain=d['ba'] >= 1, tail=d['tail'] >= -.5, forgetting=d['forgetting'] <= 1,
        new_recall=d['new_recall'] >= -1, above_fixed1=f['ba'] > 0, above_pc=differences['main_PC']['ba'] > 0,
        fixed1_tail=f['tail'] >= -.5, fixed1_forgetting=f['forgetting'] <= 1, fixed1_new_recall=f['new_recall'] >= -1)
    passed = all(checks.values()); assert passed == result['screen_success']
    pairs = [p for p in result['paired_predictions'] if p['a'] == 'main_FIXED1' and p['b'] == 'main_PAIRHEAD']
    assert len(pairs) == 4
    candidate = result['records']['main_PAIRHEAD']['metrics']['stages'][-1]
    fixed = result['records']['main_FIXED1']['metrics']['stages'][-1]
    class_delta = {c:100*(recall-fixed['per_class_recall'][c]) for c,recall in candidate['per_class_recall'].items()}
    completed = max(j['started']+j['elapsed_seconds'] for j in ledger['jobs'])
    assert completed < lock['deadline']
    summary = dict(cycle='PAIRHEAD1', status='SCREEN_PASSED' if passed else 'STOPPED_SCREEN_FAILED',
        phase='CYCLE_CLOSED', source_commit=config['source_commit'], completed_at=completed,
        completed_at_shanghai=datetime.fromtimestamp(completed, timezone(timedelta(hours=8))).isoformat(),
        screen_success=passed, failed_checks=[k for k,v in checks.items() if not v],
        metrics=metrics, differences_pp=differences, screen_checks=checks, decisions=decisions,
        final_class_recall_delta_vs_fixed1_pp=class_delta, paired_fixed1_predictions=pairs,
        gpu_seconds_used=ledger['total_gpu_seconds'], gpu_reserved_seconds=0, formal_attempts=1,
        actual_optimizer_updates=0, reused_encoder_updates=476, failures=[],
        unrun=['end-to-end training', 'new seeds', 'RL revision', 'HyperKvasir transfer'],
        unrun_reason='outside frozen single-readout scope', test_accessed=False,
        independent_confirmation=False, paper_evidence_ready=False,
        completion_audit_cpu_core_seconds=time.process_time()-started, completion_audit_gpu_seconds=0)
    save(public/'COMPLETION_AUDIT.json', dict(status='PASS', train=status, evaluation=evaluation,
        evaluation_after_seal=True, independent_allocation_recalculation=True,
        decisions=decisions, cpu_core_seconds=summary['completion_audit_cpu_core_seconds']))
    save(public/'FINAL_SUMMARY.json', summary); save(public/'FINAL_BUDGET_LEDGER.json', ledger)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    run(json.load(sys.stdin)['root'])
