"""CPU-only completion audit of existing PCRL2 receipts; no model or image access."""
import csv
import json
from collections import Counter
from datetime import datetime, timezone, timedelta
from itertools import combinations
from pathlib import Path
import sys
import time

import numpy as np


def run(root):
    started = time.process_time()
    root = Path(root)
    public = root/'public'
    def read(path):
        return json.loads(path.read_text())
    def save(path, value):
        path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    result = read(public/'main/RESULTS.json')
    ledger = read(public/'BUDGET_LEDGER.json')
    config = read(root/'cycle.private.json')
    assert len(ledger['formal_ids']) == 4 and ledger['gpu_reserved_seconds'] == 0
    assert all(j.get('exit_code') == 0 for j in ledger['jobs']) and not result['failures']
    train_jobs = [j for j in ledger['jobs'] if j['kind'] == 'train']
    eval_jobs = [j for j in ledger['jobs'] if j['kind'] == 'eval']
    assert len(train_jobs) == len(eval_jobs) == 4
    assert min(j['started'] for j in eval_jobs) >= max(j['started']+j['elapsed_seconds'] for j in train_jobs)
    with (Path(config['base']['manifest'])/'train.csv').open() as stream:
        counts = Counter(int(r['original_label']) for r in csv.DictReader(stream))
    order = config['base']['order']
    tail = set(sorted(order, key=lambda c: (-counts[c], c))[4:])
    metrics = {}
    for name, record in result['records'].items():
        m = record['metrics']; last = m['stages'][-1]
        assert abs(np.mean(list(last['per_class_recall'].values()))-m['final_balanced_accuracy']) < 1e-12
        metrics[name] = dict(ba=m['final_balanced_accuracy'], average_ba=m['average_incremental_balanced_accuracy'],
            tail=m['final_tail_recall'], forgetting=m['forgetting'],
            new_recall=float(np.mean([last['per_class_recall'][str(c)] for c in last['seen'][-2:]])))
    audits = {}; controllers = {}; residuals = {}
    for job in train_jobs:
        name = job['id']; status = read(root/name/'STATUS.json')
        assert status['status'] == 'TRAINED' and status['steps'] == 476 and status['retained_updates'] == 200
        assert status['rollout_updates'] == (0 if name == 'main_FIXED1' else 150)
        assert read(root/('eval_'+name)/'STATUS.json')['status'] == 'COMPLETE'
        diagnostics = read(root/name/'diagnostics.json')
        residuals[name] = max(s['relative_residual'] for d in diagnostics if 'competition' in d for s in d['competition']['solves'])
        audits[name] = dict(status=status, prefix=read(root/name/'PREFIX.json'),
            transitions=read(root/('eval_'+name)/'TRANSITIONS.json'))
        if name == 'main_FIXED1':
            continue
        controller = read(root/name/'CONTROLLER.json'); rows = controller['decisions']
        assert len(rows) == 16 and sum(r['rollout_updates'] for r in rows) == 150
        assert all(r['restored'] and r['max_residual'] <= 1e-8 and r['forecast_max_residual'] <= 1e-8 for r in rows)
        details = []; errors = []; signs = []; ranks = []; repeats = 0
        for index, row in enumerate(rows, 1):
            value = np.asarray(row['forecast_risks']); old = value.shape[1]-2
            tail_ids = [i for i, c in enumerate(order[:value.shape[1]]) if c in tail]
            gain = (value[0]-value)/np.maximum(value[0], .1)
            negative = np.minimum(gain, 0)
            predicted = gain.mean(1)+negative[:, :old].mean(1)+negative[:, old:].mean(1)
            if tail_ids: predicted += negative[:, tail_ids].mean(1)
            assert np.allclose(np.tanh(predicted/.01), np.asarray(row['state'])[:, 0], atol=1e-12, rtol=1e-12)
            observed = {0: 0.}
            for action, reward in zip(row['proposed_actions'], row['rewards']):
                if action in observed: assert abs(observed[action]-reward) < 1e-12
                observed[action] = reward
            repeats += len(row['proposed_actions'])-len(set(row['proposed_actions']))
            for action in set(row['proposed_actions'])-{0}:
                errors.append(abs(predicted[action]-observed[action]))
                signs.append(bool(np.sign(predicted[action]) == np.sign(observed[action])))
            for a, b in combinations(observed, 2):
                if abs(predicted[a]-predicted[b]) > 1e-10 and abs(observed[a]-observed[b]) > 1e-10:
                    ranks.append(bool(np.sign(predicted[a]-predicted[b]) == np.sign(observed[a]-observed[b])))
            selected = row['selected_action']; known_best = max(observed, key=observed.get)
            rewards = np.asarray(row['rewards'])
            advantages = (rewards-rewards.mean())/max(rewards.std(), 1e-8)
            details.append(dict(decision=index, selected=selected, forecast_best=int(predicted.argmax()),
                forecast_rewards=predicted.tolist(), selected_forecast_reward=float(predicted[selected]),
                selected_actual_reward=observed.get(selected), selected_evaluated=selected in observed,
                best_evaluated=known_best, best_evaluated_reward=observed[known_best],
                known_better=selected in observed and observed[known_best] > observed[selected]+1e-10,
                sampled_relative_advantages=advantages.tolist(),
                residual_overrides_noop=int(predicted.argmax()) == 0 and selected != 0))
        controllers[name] = dict(decisions=details, selected_actions=dict(Counter(r['selected_action'] for r in rows)),
            selection_sequence=[r['selected_action'] for r in rows], policy_updates=controller['policy_updates'],
            rollout_updates=controller['rollout_updates'], duplicate_proposals=repeats,
            forecast_mae=float(np.mean(errors)), forecast_sign_agreement=float(np.mean(signs)),
            forecast_sign_n=len(signs), forecast_pair_order_agreement=float(np.mean(ranks)), forecast_pair_order_n=len(ranks),
            forecast_match_scope='unique sampled nonzero actions per decision; rank includes known zero reference',
            residual_overrides_noop_decisions=[d['decision'] for d in details if d['residual_overrides_noop']],
            all_restored=True, max_branch_residual=max(r['max_residual'] for r in rows),
            max_forecast_residual=max(r['forecast_max_residual'] for r in rows),
            final_probabilities=rows[-1]['after_probabilities'])
    assert max(residuals.values()) <= 1e-8
    pairs = [p for p in result['paired_predictions'] if p['a'] in ('main_RL', 'main_FIXED1') and p['b'] in ('main_FIXED1','main_RESPONSE0','main_RL_SHARED','main_RL_RESPONSE')]
    fixed_parity = [p for p in pairs if p['a'] == 'main_RL' and p['b'] == 'main_FIXED1']
    assert len(fixed_parity) == 4 and all(p['prediction_agreement'] == 1 for p in fixed_parity)
    differences = {n: {k: (metrics['main_RL_RESPONSE'][k]-v[k])*100 for k in v} for n,v in metrics.items() if n != 'main_RL_RESPONSE'}
    d = differences['main_R']
    checks = dict(ba_gain=d['ba'] >= 1, tail=d['tail'] >= -.5, forgetting=d['forgetting'] <= 1, new_recall=d['new_recall'] >= -1)
    checks.update({'above_'+a: differences['main_'+a]['ba'] > 0 for a in config['controls']})
    assert all(checks.values()) == result['screen_success']
    completed = max(j['started']+j['elapsed_seconds'] for j in ledger['jobs'])
    summary = dict(cycle='PCRL2', status='STOPPED_SCREEN_FAILED', phase='CYCLE_CLOSED', source_commit=config['source_commit'],
        completed_at=completed, completed_at_shanghai=datetime.fromtimestamp(completed,timezone(timedelta(hours=8))).isoformat(),
        screen_success=False, robustness_success=False, paper_evidence_ready=False,
        stop_reason='BA_GAIN_BELOW_ONE_PP_AND_RL_BELOW_ALL_REGISTERED_CONTROLS', metrics=metrics,
        rl_differences_pp=differences, screen_checks=checks, controllers=controllers, paired_predictions=pairs,
        fixed1_reproduces_old_rl_predictions_all_stages=True, max_retained_solve_residual=residuals,
        formal_attempts=4, gpu_seconds_used=ledger['total_gpu_seconds'], gpu_reserved_seconds=0,
        diagnostic_gpu_seconds_included=ledger['diagnostic_gpu_seconds'],
        retained_updates=sum(j['retained_updates'] for j in train_jobs), rollout_updates=sum(j['rollout_updates'] for j in train_jobs),
        actual_adapter_updates=sum(j['actual_updates'] for j in train_jobs), policy_updates=sum(j['policy_updates'] for j in train_jobs),
        failures=[], unrun=['two new seeds','RL_RESPONSE_uniform','RL_RESPONSE_mean','HyperKvasir transfer'],
        unrun_reason='pre-registered initial screen failed', test_accessed=False, independent_confirmation=False,
        completion_audit_cpu_core_seconds=time.process_time()-started, completion_audit_gpu_seconds=0,
        completion_audit_optimizer_updates=0)
    save(public/'COMPLETION_AUDIT.json', audits)
    save(public/'FINAL_SUMMARY.json', summary)
    save(public/'FINAL_BUDGET_LEDGER.json', ledger)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('controllers','paired_predictions')},indent=2))
    print(json.dumps({n:{k:v for k,v in d.items() if k!='decisions'} for n,d in controllers.items()},indent=2))


if __name__ == '__main__':
    run(json.load(sys.stdin)['root'])
