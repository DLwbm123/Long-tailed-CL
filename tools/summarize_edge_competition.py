"""Validate the complete preregistered matrix and emit public aggregates."""
import json
from pathlib import Path
import sys


def public_diagnostic(d):
    # Keep decision evidence without exporting learned coefficients or pair matrices.
    c = d['controller']
    rewards = [r for group in c.get('samples', []) for r in group['rewards']]
    result = {k: d[k] for k in ('task', 'epoch', 'steps', 'arm')}
    result['controller'] = {k: c[k] for k in (
        'policy_updates', 'head_evaluations', 'proposed_reward', 'executed',
        'original_current_guard', 'original_old_guard', 'relative_pair_l1',
        'max_solve_residual')}
    result['controller']['sample_reward_summary'] = dict(
        count=len(rewards), above_execution_threshold=sum(r > 1e-8 for r in rewards),
        minimum=min(rewards) if rewards else None, maximum=max(rewards) if rewards else None)
    result['max_batch_solve_residual'] = d['max_batch_solve_residual']
    if 'boundary_competition' in d:
        b = d['boundary_competition']
        base, selected = b['base_pairs'], b['selected_pairs']
        mass = sum(map(sum, base))
        result['boundary_competition'] = dict(
            relative_pair_l1=sum(abs(a-z) for row, other in zip(base, selected)
                                 for a, z in zip(row, other))/mass,
            solve=b['solve'], new_policy_updates=b['new_policy_updates'],
            refit_uses_all_current_training=b['refit_uses_all_current_training'])
    return result


def summarize(config):
    root, output = Path(config['root']), Path(config['output'])
    protocol = json.loads((root/'PROTOCOL_LOCK.json').read_text())
    records, comparisons, costs = {}, [], {}
    lines = ['# EDGE-BUDGET1：细粒度竞争动作结果', '',
        '唯一主要候选edge_rl；医学开发集与CIFAR冻结测试分别报告，全部正负结果保留。',
        '策略优化固定表示下的即时代理；不声称长期回报、充分收敛、独立医学确认或统计显著性。', '',
        '|数据|方法|平均BA %|最终BA %|尾类 %|任务遗忘 pp|末任务新类 %|执行/决策|策略更新|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for dataset in protocol['datasets']:
        for phase in ['preflight', 'formal']:
            suite = json.loads((root/phase/dataset/'STATUS.json').read_text())
            assert suite['status'] == 'COMPLETE', (phase, dataset)
            assert len(suite['completed']) == (1 if phase == 'preflight' else 4)
            costs[f'{phase}_{dataset}'] = suite['elapsed_seconds']
        for arm in protocol['arms']:
            name = f'{dataset}_{arm}'
            path = root/'runs'/name
            status = json.loads((path/'STATUS.json').read_text())
            m = json.loads((path/'metrics.json').read_text())
            ds = json.loads((path/'diagnostics.json').read_text())
            private = json.loads((path/'INPUT.private.json').read_text())
            stages = len(private['task_sizes'])
            assert status['status'] == 'COMPLETE' and status['steps'] == protocol['expected_steps'][dataset]
            assert status['stages'] == len(m['stages']) == stages
            assert {(x['task'], x['epoch']) for x in ds} == {(t, e) for t in range(1, stages+1) for e in [1, 2]}
            assert len(ds) == 2*stages and status['max_solve_residual'] <= 1e-8
            assert m['test_accessed'] == (dataset == 'CIFAR100LT')
            assert private['seed'] == protocol['seed'] and private['split_seed'] == protocol['split_seed']
            assert private['original_deadline'] == protocol['original_deadline']
            activation = dict(decisions=len(ds), executed=sum(d['controller']['executed'] for d in ds),
                changed=sum(d['controller']['relative_pair_l1'] > 1e-10 for d in ds),
                policy_updates=sum(d['controller']['policy_updates'] for d in ds),
                head_evaluations=sum(d['controller']['head_evaluations'] for d in ds),
                original_current_guard_failures=sum(not d['controller']['original_current_guard'] for d in ds),
                original_old_guard_failures=sum(not d['controller']['original_old_guard'] for d in ds))
            assert activation['policy_updates'] == status['policy_updates'] == (len(ds)*8 if arm.endswith('_rl') else 0)
            assert activation['head_evaluations'] == len(ds)*(1 if arm == 'static_pc' else 18)
            final = m['stages'][-1]
            new_classes = final['seen'][-private['task_sizes'][-1]:]
            m['final_new_class_recall'] = sum(final['per_class_recall'][str(c)] for c in new_classes)/len(new_classes)
            records[name] = dict(status=status, metrics=m, activation=activation,
                                 diagnostics=[public_diagnostic(d) for d in ds])
            keys = ['average_incremental_balanced_accuracy', 'final_balanced_accuracy', 'final_tail_recall',
                    'task_forgetting', 'final_new_class_recall']
            vals = '|'.join(f'{100*m[k]:.4f}' for k in keys)
            lines.append(f"|{dataset}|{arm}|{vals}|{activation['executed']}/{len(ds)}|{activation['policy_updates']}|")
        candidate = records[f'{dataset}_edge_rl']['metrics']
        for control in ['static_pc', 'coarse_rl', 'edge_search']:
            base = records[f'{dataset}_{control}']['metrics']
            comparisons.append(dict(dataset=dataset, control=control,
                deltas={key: candidate[key]-base[key] for key in keys}))
    screens = {}
    for dataset in ['ISIC', 'HK']:
        rows = {r['control']:r['deltas'] for r in comparisons if r['dataset'] == dataset}
        d = rows['static_pc']
        conditions = dict(final_ba=d['final_balanced_accuracy'] >= .01,
            average_ba=d['average_incremental_balanced_accuracy'] >= 0,
            tail=d['final_tail_recall'] >= -.005, forgetting=d['task_forgetting'] <= .01,
            new_classes=d['final_new_class_recall'] >= -.01,
            beats_coarse=rows['coarse_rl']['final_balanced_accuracy'] > 0,
            beats_search=rows['edge_search']['final_balanced_accuracy'] > 0)
        screens[dataset] = dict(conditions=conditions, passed=all(conditions.values()))
    budget = dict(prior_closed_gpu_seconds=protocol['prior_closed_gpu_seconds'],
        suite_gpu_seconds=costs, additional_attempt_gpu_seconds=config.get('additional_attempt_gpu_seconds', 0.),
        original_deadline=protocol['original_deadline'])
    budget['round_gpu_seconds'] = sum(costs.values())+budget['additional_attempt_gpu_seconds']
    budget['cumulative_gpu_seconds'] = budget['prior_closed_gpu_seconds']+budget['round_gpu_seconds']
    lines += ['', '## 冻结医学初筛', '', json.dumps(screens, ensure_ascii=False, indent=2), '',
        'CIFAR只作本冻结矩阵的结果报告，不进入后续方法/种子选择。医学两个设置都通过才有跨这两个开发设置的一致信号。',
        '原逐类代理风险门与执行门不同，任何最终分数改善都不能替代旧知识或临床安全确认。', '',
        f"本轮suite驻留及额外尝试共 {budget['round_gpu_seconds']:.6f} 秒，累计 {budget['cumulative_gpu_seconds']:.6f} 秒。", '']
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [('RESULTS.json',dict(records=records,comparisons=comparisons,medical_screen=screens)),
                        ('BUDGET_LEDGER.json',budget)]:
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    (output/'REPORT_ZH.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    summarize(json.load(sys.stdin))
