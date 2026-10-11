"""Four-cell contribution report from anonymous metrics and scalar audits only."""
import csv
import json
import math
from pathlib import Path
import sys

from summarize_module_additions import summarize

FIELDS = ('final_balanced_accuracy', 'average_incremental_balanced_accuracy', 'final_tail_recall',
          'forgetting', 'final_old_recall', 'final_new_recall')
MODULES = ('base', 'local_transport', 'residual_drift', 'local_transport_drift')


def delta(a, b):
    return {f: 100 * (a[f] - b[f]) for f in FIELDS}


def gates(d):
    return dict(BA=d[FIELDS[0]] >= 1, tail=d[FIELDS[2]] >= -.5, forgetting=d[FIELDS[3]] <= 1,
                old=d[FIELDS[4]] >= -1, new=d[FIELDS[5]] >= -1)


def finite(value):
    if isinstance(value, dict):
        for v in value.values(): finite(v)
    elif type(value) not in (int, float, bool) or not math.isfinite(value):
        raise ValueError('Nonfinite/non-scalar diagnostic')


def report(s, out):
    p = s['program']; residuals = []; activation = {}
    if p['status'] != 'COMPLETE' or p['active'] or p['failures'] or len(p['costs']) != 12:
        raise ValueError('Complete twelve-process barrier required')
    if any(v['returncode'] or v['gpu'] not in (0, 1) for v in p['costs']):
        raise ValueError('Failed or unauthorized subprocess')
    for ds in ('ISIC', 'HK'):
        for arm in MODULES[2:]:
            name = ds + '_' + arm; v = s['runs'][name]; st = v['status']; m = v['metrics']
            if (st['status'] != 'COMPLETE' or st['steps'] != {'ISIC': 468, 'HK': 176}[ds] or
                    st['policy_updates'] or m['test_accessed'] or
                    m['evaluation_split'] != 'development_validation' or m['residual_drift_weight'] != .1):
                raise ValueError('Frozen drift status/permission differs')
            diag = v['scalar_diagnostics']; tasks = len(m['stages'])
            if len(diag) != 2 * tasks: raise ValueError('Missing epoch diagnostics')
            trace = []; norm = []
            for d in diag:
                finite(d); a = d['residual_drift']
                if a['weight'] != .1 or not a['detached'] or not a['fit_only'] or a['historical_samples']:
                    raise ValueError('Residual estimate source differs')
                if a['residual_trace'] < 0 or a['residual_frobenius_norm'] < 0:
                    raise ValueError('Invalid PSD scalar audit')
                residuals.append(d['max_batch_solve_residual'])
                if arm == 'local_transport_drift' and (d['activation']['extra_columns'] <= 0 or d['activation']['prior_delta'] <= 0):
                    raise ValueError('Missing local prior activation')
                if 'boundary_residual_drift' in d:
                    a = d['boundary_residual_drift']
                    if not a['current_train_including_meta'] or a['weight'] != .1 or a['historical_samples']:
                        raise ValueError('Boundary drift source differs')
                    trace.append(a['residual_trace']); norm.append(.1 * a['residual_frobenius_norm'])
                    residuals.append(d['boundary_solve']['relative_residual'])
            if len(trace) != tasks or any(t <= 0 for t in trace):
                raise ValueError('Nonzero boundary residual activation not established')
            if diag[-1]['steps'] != st['steps']: raise ValueError('Diagnostic budget mismatch')
            q = s['preflights'][name]; u = q['scalars']; finite({k: v for k, v in u.items() if k != 'status'})
            if (q['status']['status'] != 'COMPLETE' or q['status']['steps'] or
                    q['status']['policy_updates'] or u['status'] != 'PASS' or u['adapter_updates'] or
                    u['policy_updates'] or u['gradient_norm'] <= 0 or
                    u['residual_drift']['weight'] != .1 or not u['residual_drift']['nonzero_activation_not_proven']):
                raise ValueError('Missing real zero-update preflight')
            residuals += [st['max_solve_residual'], q['status']['max_solve_residual']]
            activation[name] = dict(boundary_trace=trace, system_delta_frobenius_norm=norm)
    if max(residuals) > 1e-8: raise ValueError('Analytic residual exceeds frozen tolerance')
    s.update(modules=list(MODULES), description='残余漂移四格贡献对照：4条新增、1288更新；A0/AL严格复用，0策略/test，单seed重复开发集。')
    summarize(s, out)
    comparisons = []
    for ds in ('ISIC', 'HK'):
        m = {a: s['runs'][ds + '_' + a]['metrics'] for a in MODULES}
        for a, b in [('residual_drift', 'base'), ('local_transport_drift', 'local_transport'),
                     ('local_transport_drift', 'residual_drift'), ('local_transport_drift', 'base')]:
            comparisons.append(dict(dataset=ds, contrast=a+' - '+b, delta_pp=delta(m[a], m[b])))
        x = delta(m['local_transport_drift'], m['local_transport']); y = delta(m['residual_drift'], m['base'])
        comparisons.append(dict(dataset=ds, contrast='interaction', delta_pp={f: x[f]-y[f] for f in FIELDS}))
    costs = sum(v['residence_seconds'] for v in p['costs'])
    ledger = s['closed_round_cost_ledger'].copy(); ledger['module_drift10_20261011'] = costs
    audit = dict(status='PASS', source_commit=s['source_commit'], formal_complete=4, formal_adapter_updates=1288,
                 policy_updates=0, test_accessed=False, zero_update_preflight_checks=4,
                 successful_subprocess_cost_records=12, formal_failures=[], max_solve_residual=max(residuals),
                 control_matching=s['control_matching'], denominators_equal_all_stages=True,
                 drift_activation=activation, initial_zero_U_preflight_not_activation_proof=True,
                 reused_trajectories=4, reused_updates_and_costs_charged=0,
                 round_gpu_process_residence_seconds=costs, closed_round_cost_ledger=ledger,
                 cumulative_gpu_process_residence_seconds=83019.61353103025+sum(ledger.values()),
                 parallel_gpu_phase_wall_seconds=p['elapsed_seconds'],
                 whole_round_wall_seconds=p['whole_round_wall_seconds'], resource_wait_seconds=p['resource_wait_seconds'],
                 cpu_check_seconds=s['cpu_check_seconds'], data_preparation_cpu_seconds=0,
                 independent_confirmation=False, extra_inference_scores=0,
                 development_goal_met=all(all(gates(delta(s['runs'][ds+'_local_transport_drift']['metrics'],
                     s['runs'][ds+'_base']['metrics'])).values()) for ds in ('ISIC', 'HK')))
    for name, value in [('AUDIT.json', audit), ('CONTRIBUTIONS.json', comparisons), ('PREFLIGHT_CHECKS.json', s['preflights'])]:
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    with (out/'PER_CLASS_RECALL.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['dataset','task','module','anonymous_class','n','recall'])
        for ds in ('ISIC','HK'):
            for arm in MODULES:
                for t in s['runs'][ds+'_'+arm]['metrics']['stages']:
                    for c,n in t['per_class_n'].items(): w.writerow([ds,t['task'],arm,c,n,t['per_class_recall'][c]])
    lines=['# 残余漂移约束：完整四格贡献对照结果','',
           'A0/AL为已交付且严格匹配的原static_pc/local_transport；AD/ALD为新训练漂移单独/组合。所有正负与交互完整公开。单seed/单order/反复development validation，test/reserved封存；不是原版LCA复现、独立患者或临床验证。', '',
           '|数据集|模块|最终BA %|平均BA %|尾类 %|遗忘 pp ↓|旧类 %|新类 %|', '|---|---|---:|---:|---:|---:|---:|---:|']
    rows=json.loads((out/'RESULTS.json').read_text())['rows']
    for r in rows: lines.append('|'+ '|'.join([r['dataset'],r['module']]+[f'{r[f]:.6f}' for f in FIELDS])+'|')
    lines += ['', '## 全部贡献增量（pp）','', '|数据集|比较|ΔBA|Δ平均BA|Δtail|Δforget ↓|Δold|Δnew|','|---|---|---:|---:|---:|---:|---:|---:|']
    for c in comparisons: lines.append('|'+ '|'.join([c['dataset'],c['contrast']]+[f'{c["delta_pp"][f]:+.6f}' for f in FIELDS])+'|')
    lines += ['', '## 相对原base的冻结门','', '|数据集|候选|BA ≥+1|tail ≥−.5|forget ≤+1|old ≥−1|new ≥−1|','|---|---|---|---|---|---|---|']
    for ds in ('ISIC','HK'):
        for a in MODULES[2:]:
            d=delta(s['runs'][ds+'_'+a]['metrics'],s['runs'][ds+'_base']['metrics'])
            lines.append('|'+ '|'.join([ds,a]+['PASS' if x else 'FAIL' for x in gates(d).values()])+'|')
            text='无下降' if d[FIELDS[5]]>=0 else ('仍下降但在−1pp保护门内' if d[FIELDS[5]]>=-1 else '仍违反−1pp保护门')
            lines.append('');lines.append(f'{ds}/{a} 新类：{text}；保护门内不称无损。');lines.append('')
    lines += ['', '## 全阶段 Acc 与 BA','', '|数据集|模块|task|n|Acc %|BA %|','|---|---|---:|---:|---:|---:|']
    for ds in ('ISIC','HK'):
        for a in MODULES:
            for t in s['runs'][ds+'_'+a]['metrics']['stages']:
                lines.append(f'|{ds}|{a}|{t["task"]}|{t["validation_n"]}|{100*t["accuracy"]:.6f}|{100*t["balanced_accuracy"]:.6f}|')
    for ds in ('ISIC','HK'):
        lines += ['', '## '+ds+' 全部最终匿名类召回', '', '|类|n|A0 %|AL %|AD %|ALD %|ΔAD−A0|ΔALD−A0|ΔALD−AL|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        stages=[s['runs'][ds+'_'+a]['metrics']['stages'][-1] for a in MODULES]
        for c,n in stages[0]['per_class_n'].items():
            v=[100*t['per_class_recall'][c] for t in stages]
            lines.append('|'+ '|'.join([c,str(n)]+[f'{x:.6f}' for x in v]+[f'{v[2]-v[0]:+.6f}',f'{v[3]-v[0]:+.6f}',f'{v[3]-v[1]:+.6f}'])+'|')
    lines += ['', '全部stage每类n/recall见PER_CLASS_RECALL.csv与RESULTS.json；HK极少类n=1/1/2/4不能夸大单例变化。数学/非零U和局部先验激活不等于分类净效用，共享当前fit漂移只是旧类漂移代理。组合增量不能全归漂移。', '',
              f'4新轨迹/1288更新、4真实零预检、12成功GPU费用、0失败/策略/test；maxsolve={max(residuals):.12g}。本轮GPU驻留{costs:.6f}秒，累计{audit["cumulative_gpu_process_residence_seconds"]:.6f}秒；并行墙钟、资源等待、CPU检查另列AUDIT，不重复计复用轨迹。无新数据准备。', '',
              '历史RFMiD费用若尚未公开闭合不加入累计：ledger中的已闭合费用各一次；当前活跃未闭合轮另列，不相加cumulative。所有失败另全计，无自动正式重跑。只公开匿名聚合/标量/协议/源码，不发布影像/权重/逐样本/统计矩阵/private配置/rawlogs。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    return audit


if __name__ == '__main__':
    cfg=json.load(sys.stdin);print(json.dumps(report(json.loads(Path(cfg['snapshot']).read_text()),Path(cfg['output'])),indent=2))
