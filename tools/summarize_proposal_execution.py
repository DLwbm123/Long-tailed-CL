"""Audit the two frozen proposal-execution trajectories from aggregate JSON."""
import json
from pathlib import Path
import sys


def summarize(data, destination):
    records = data['records']
    comparisons = []
    metrics = ['average_incremental_balanced_accuracy', 'final_balanced_accuracy',
               'final_tail_recall', 'task_forgetting']
    lines = ['# 医学策略提议执行诊断：完成报告', '',
             '两条新增诊断完整完成；seed74002、shuffled、每任务两轮。原 strict 和 uniform 轨迹复用首轮。',
             '医学仅使用既有 development validation，test/reserved 未访问；没有新增 CIFAR 评估。',
             '诊断执行最终策略均值提议，accepted 仍表示原安全门是否通过，不表示是否执行。', '',
             '|数据|方法|平均 BA %|最终 BA %|尾类召回 %|任务遗忘 pp|原门通过|非均匀执行|',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for dataset, stages, steps in [('ISIC', 3, 468), ('HK', 6, 176)]:
        for arm in ['cf_prototype', 'cf_group', 'proposal']:
            record = records[f'{dataset}_{arm}']
            status, m, ds = record['status'], record['metrics'], record['diagnostics']
            assert status['status'] == 'COMPLETE' and status['steps'] == steps
            assert status['stages'] == stages and len(m['stages']) == stages
            assert len(ds) == 2 * stages and not m['test_accessed']
            assert {(d['task'], d['epoch']) for d in ds} == {
                (t, e) for t in range(1, stages + 1) for e in [1, 2]}
            activation = dict(decisions=len(ds),
                policy_updates=sum(d['controller']['optimizer_steps'] for d in ds),
                original_guard_passes=sum(d['controller']['accepted'] for d in ds) if arm != 'cf_prototype' else None,
                executed_nonuniform=sum(d['activation']['weight_l1_change'] > 1e-10 for d in ds),
                current_guard_failures=sum(d['controller'].get('current_guard') is False for d in ds),
                old_guard_failures=sum(d['controller'].get('old_moment_guard') is False for d in ds),
                ess_guard_failures=sum(d['controller'].get('effective_sample_guard') is False for d in ds))
            record['activation'] = activation
            record['execution_mode'] = 'diagnostic_proposal' if arm == 'proposal' else 'strict'
            if arm == 'proposal':
                assert all(d['controller']['execution_mode'] == 'diagnostic_proposal' for d in ds)
                assert activation['policy_updates'] == len(ds) * 8
                assert all(d['controller']['selected_actions'] == d['controller']['proposed_actions'] for d in ds)
            guard = 'NA' if arm == 'cf_prototype' else f"{activation['original_guard_passes']}/{len(ds)}"
            vals = '|'.join(f'{100*m[k]:.4f}' for k in metrics)
            lines.append(f"|{dataset}|{arm}|{vals}|{guard}|{activation['executed_nonuniform']}/{len(ds)}|")
        candidate = records[f'{dataset}_proposal']['metrics']
        for arm in ['cf_group', 'cf_prototype']:
            baseline = records[f'{dataset}_{arm}']['metrics']
            comparisons.append(dict(dataset=dataset, control=arm,
                deltas={key: candidate[key]-baseline[key] for key in metrics}))
    suite_seconds = {}
    for dataset, suite in data['suites'].items():
        assert suite['status'] == 'COMPLETE' and len(suite['completed']) == 1
        assert suite['elapsed_seconds'] >= suite['completed'][0]['elapsed_seconds'] > 0
        suite_seconds[dataset] = suite['elapsed_seconds']
    cost = dict(first_round_gpu_seconds=data['first_round_gpu_seconds'],
                followup_suite_gpu_seconds=suite_seconds,
                followup_total_gpu_seconds=sum(suite_seconds.values()),
                cumulative_gpu_seconds=data['first_round_gpu_seconds']+sum(suite_seconds.values()),
                original_deadline=data['original_deadline'], all_suites_closed=True,
                extra_training_attempts=0, accounting='Suite wall residence; not GPU utilization time.')
    lines += ['', '## 对照差值', '',
              '以下差值单位为百分点；BA/尾类越高越好，任务遗忘越低越好。']
    for c in comparisons:
        lines.append(f"- {c['dataset']} proposal − {c['control']}：" +
                     '，'.join(f'{k} {100*v:+.4f}' for k, v in c['deltas'].items()) + '。')
    lines += ['', '## 解释', '',
        '本轮是端到端干预，执行不同动作后后续状态与提议会分叉，不能称逐步相同提议的离线比较。',
        '是否改善指标与是否满足原安全门分别报告；代理门失败不等于已证实临床伤害，但不能声称旧知识保护成功。',
        '单种子、两轮训练和反复使用的开发集仅支持初筛；没有独立确认、充分收敛或统计优越声明。',
        '全部正负结果保留，不按本轮结果更换阈值、种子、checkpoint或重新运行。', '',
        '## 成本与交付', '',
        f"首轮 {cost['first_round_gpu_seconds']:.6f} 秒；本轮 {cost['followup_total_gpu_seconds']:.6f} 秒；累计 {cost['cumulative_gpu_seconds']:.6f} 秒。",
        '成本使用已封口 suite 回执，包含启动、导入、训练与评估；旧成本不重置。',
        '仅交付聚合指标、诊断、协议与源码。原始图像、样本身份、私有配置、checkpoint、特征及原始日志不公开。', '']
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [('RESULTS.json', dict(records=records, comparisons=comparisons)), ('BUDGET_LEDGER.json', cost)]:
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    (output/'REPORT_ZH.md').write_text('\n'.join(lines))
    return comparisons, cost


if __name__ == '__main__':
    config = json.load(sys.stdin)
    summarize(json.loads(Path(config['snapshot']).read_text()), config['output'])
