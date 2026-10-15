"""Close out the stopped four-trajectory round using public aggregates only."""
import csv
import json
import math
from pathlib import Path

SIX = ('final_balanced_accuracy', 'average_incremental_balanced_accuracy', 'final_tail_recall', 'forgetting', 'final_old_recall', 'final_new_recall')
EXTRA = ('final_accuracy', 'average_incremental_accuracy', 'final_macro_F1', 'average_incremental_macro_F1')
METHODS = ('base', 'local_transport', 'pmt', 'local_transport_pmt', 'guard', 'LT_guard', 'AIR')
LABELS = dict(zip(METHODS, ('Core', 'LT', 'PMT', 'LT+PMT', 'PMT+guard', 'LT+PMT+guard', 'AIR')))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n'); w.writeheader()
        w.writerows({k: 'null' if v is None else v for k, v in r.items()} for r in rows)


def unique_events(diagnostics):
    events = {}
    for d in diagnostics:
        for e in d.get('snapshot_guard', {}).get('events', []):
            key = tuple(e[k] for k in ('task', 'epoch', 'event', 'steps'))
            if key in events:
                assert events[key] == e, 'Conflicting cumulative event snapshots'
            events[key] = e
    return list(events.values())


def validate(data):
    p = data['program']
    # shortcut: this reporter only supports incomplete, unevaluated rounds; use a separate reviewed reporter for complete results.
    assert p['status'] == 'INCOMPLETE' and p['phase'] == 'STOPPED_DEPENDENCIES'
    assert not p['active'] and not p['completed'] and data['all_workers_exited']
    assert p['source_commit'] == data['source_commit']
    assert p['new_preflight_operations'] == 0 and p['preflights_reused'] == 4
    assert all(not v.get('metrics') for v in data['runs'].values())
    assert all(c['operation'] == 'train' and c['returncode'] != 0 and c['gpu'] in (0, 1)
               and math.isfinite(c['GPU_process_residence_seconds']) and c['GPU_process_residence_seconds'] > 0 for c in p['costs'])
    assert len(p['costs']) == len(p['failures']) == 2


def self_check(data):
    import copy
    validate(data)
    for mutation in ('active', 'metrics', 'bad_cost', 'source'):
        bad = copy.deepcopy(data)
        if mutation == 'active': bad['program']['active'] = {'0': 'still running'}
        elif mutation == 'metrics': bad['runs']['ISIC_guard']['metrics'] = {'accuracy': .5}
        elif mutation == 'bad_cost': bad['program']['costs'][0]['GPU_process_residence_seconds'] = -1
        else: bad['source_commit'] = 'wrong'
        try: validate(bad)
        except AssertionError: pass
        else: raise AssertionError('negative case accepted: ' + mutation)
    e = dict(task=1, epoch=1, event='epoch_init', steps=0, value=1)
    assert len(unique_events([{'snapshot_guard': {'events': [e, e]}}])) == 1
    try: unique_events([{'snapshot_guard': {'events': [e, dict(e, value=2)]}}])
    except AssertionError: pass
    else: raise AssertionError('conflicting event accepted')


def main(root):
    data = json.loads((root/'RESULTS.json').read_text()); self_check(data); p = data['program']
    failures = json.loads((root/'FAILURES_ANON.json').read_text())
    oldroot = root.parent/'paired_transport1_20261015/results'
    old = json.loads((oldroot/'RESULTS.json').read_text())
    assert data['source_commit'] == 'f4b272de092a2db492126654a8bcceef071fe8e1'
    execution = []; events = []; growth = []; diagnostics = []; invalidations = []
    for job, run in data['runs'].items():
        costs = [c for c in p['costs'] if c['job'] == job]
        st = run.get('status', {}); diag = run.get('scalar_diagnostics', [])
        ev = unique_events(diag)
        execution.append(dict(job=job, status=st.get('status', 'NOT_DISPATCHED_DEPENDENCY_STOP'), observed_steps=st.get('steps'),
            planned_steps=468 if job.startswith('ISIC') else 176, saved_stage_checkpoints=len(run['checkpoint_sizes_bytes']),
            policy_updates=st.get('policy_updates'), GPU_operation_records=len(costs),
            GPU_residence_seconds=sum(c['GPU_process_residence_seconds'] for c in costs) if costs else None,
            worker_elapsed_seconds=st.get('elapsed_seconds'), measured_GPU_peak_bytes=None,
            successful_guard_events=len(ev), failed_guard_events=1 if costs else 0,
            failure='Full-fit guard unresolved; no identity fallback' if costs else None))
        for e in ev:
            assert e['status'] == 'CERTIFIED' and e['active_atom_cap'] is None
            assert e['minimum_raw_slack'] >= -2e-8 and e['stationarity'] <= 1e-8 and e['complementarity'] <= 1e-7
            assert abs(e['primal_dual_gap']) <= 1e-7*(1+abs(e['objective_candidate']))
            events.append(dict(job=job, **{k:v for k,v in e.items() if k != 'trace'}))
            growth.extend(dict(job=job, task=e['task'], epoch=e['epoch'], event=e['event'], steps=e['steps'], **t) for t in e.get('trace', []))
        for d in diag:
            for kind in ('paired_transport', 'boundary_paired_transport'):
                if kind in d: diagnostics.append(dict(job=job, task=d['task'], epoch=d['epoch'], steps=d['steps'], kind=kind, **d[kind]))
            if 'snapshot_guard' in d:
                invalidations.append(dict(job=job, task=d['task'], epoch=d['epoch'], steps=d['steps'], invalidations_snapshot=d['snapshot_guard']['invalidations'], additive=False))
    assert sum(v['observed_steps'] or 0 for v in execution) == 640 and len(events) == 14
    table = []
    for ds in ('ISIC', 'HK'):
        for method in METHODS:
            r = next((r for r in old['rows'] if r['dataset'] == ds and r['module'] == method), None)
            m = old['runs'].get(ds+'_'+method, {}).get('metrics', {})
            row = dict(dataset=ds, method=method, evidence='HISTORICAL_PUBLISHED_REUSE' if r else 'NOT_EVALUATED', **{k:r[k] if r else None for k in SIX})
            row.update({k:100*m[k] if m.get(k) is not None else None for k in EXTRA})
            air = next(r for r in old['rows'] if r['dataset']==ds and r['module']=='AIR')
            row['AIR_minus_method_BA_pp'] = air[SIX[0]]-r[SIX[0]] if r else None
            table.append(row)
    comparisons = []
    for ds in ('ISIC', 'HK'):
        lookup = {r['method']:r for r in table if r['dataset']==ds}
        for a,b in [('pmt','base'),('local_transport_pmt','local_transport'),('guard','base'),('LT_guard','local_transport'),('guard','pmt'),('LT_guard','local_transport_pmt'),('LT_guard','guard'),('guard','AIR'),('LT_guard','AIR')]:
            comparisons.append(dict(dataset=ds, comparison=a+' minus '+b, status='HISTORICAL' if lookup[a][SIX[0]] is not None and lookup[b][SIX[0]] is not None else 'NOT_EVALUATED',
                **{k:lookup[a][k]-lookup[b][k] if lookup[a][k] is not None and lookup[b][k] is not None else None for k in SIX}))
        comparisons.append(dict(dataset=ds, comparison='(LT_guard - LT) - (guard - Core)', status='NOT_EVALUATED', **dict.fromkeys(SIX)))
    classes=[]; stages=[]
    # Reuse only published counts to describe missing coverage; no manifest or model reads.
    for ds in ('ISIC','HK'):
        templates=old['runs'][ds+'_pmt']['metrics']['stages']
        for method in ('guard','LT_guard'):
            for stage in templates:
                stages.append(dict(dataset=ds,method=method,stage=stage['task'],planned_validation_n=stage['validation_n'],observed_validation_n=None,BA_percent=None,Acc_percent=None,macroF1=None,status='NOT_EVALUATED'))
                for label,n in stage['per_class_n'].items():
                    classes.append(dict(dataset=ds,method=method,stage=stage['task'],label=int(label),published_reference_val_n=n,observed_n=None,correct_n=None,recall=None,rescued=None,damaged=None,wrong_to_other_wrong=None,true_margin=None,status='NOT_EVALUATED'))
    final_classes = []
    with (oldroot/'FINAL_CLASS_COMPARISON.csv').open() as f:
        for r in csv.DictReader(f):
            out = {k:v for k,v in r.items() if 'global_only' not in k}
            out['historical_columns_source'] = '080c2a33e3bd8c6856be1954c2a75f085af98d6a'
            for method in ('guard', 'LT_guard'):
                for field in ('correct', 'recall', 'rescued', 'damaged', 'wrong_to_other_wrong', 'true_margin_mean'):
                    out[method+'_'+field] = None
            final_classes.append(out)
    assert len(final_classes) == 31
    write_csv(root/'FINAL_CLASS_COMPARISON.csv', final_classes)
    missing = dict(status='NOT_EVALUATED', same_state_OFF=None, same_state_proposal=None, independent_Core_LT=None, independent_PMT=None,
        old_new_confusion=None, PMT_float_double_prediction_difference=None, class_balanced_net_correction=None, development_gates=None, new_class_nonharm=None,
        reason='All-four training seal barrier not reached; no validation evaluation was dispatched.')
    total = sum(c['GPU_process_residence_seconds'] for c in p['costs'])
    audit = dict(status='INCOMPLETE', source_commit=data['source_commit'], expected_updates=1288, actual_updates=640,
        completed_trajectories=0, failed_train_operations=2, not_dispatched_trajectories=2, evaluation_operations=0, new_preflight_operations=0, preflights_reused=4,
        GPU_operation_records=p['costs'], new_GPU_seconds=total, failed_GPU_seconds=total,
        prior_closed_GPU_seconds=data['historical_GPU_seconds'], closed_GPU_seconds=data['historical_GPU_seconds']+total,
        parent_wall_seconds=p['whole_round_wall_seconds'], orchestration_CPU_seconds=9.429333493113518,
        CPU_solver_nested_in_GPU_residence=True, new_GPU_peak_bytes=None, monetary_cost=None,
        successful_guard_events=14, failed_guard_calls=2, planned_guard_events=54, missing_failed_solver_audits=2,
        full_round_deadline_exceeded=False, guard_timeout_confirmed=None, val_accessed=False, test_accessed=False, policy_updates=0,
        all_workers_exited=True, terminal_collection_count=1, retries=0, scientific_source_modified=False,
        cost_records_counted_individually=True, guard_event_snapshots_deduplicated_by=['job','task','epoch','event','steps'],
        historical_results_source='080c2a33e3bd8c6856be1954c2a75f085af98d6a', missing_classification_values_are_null=True)
    budget = dict(planned=json.loads((root/'BUDGET_SPEC.json').read_text()), actual=dict(updates=640, complete_training_seals=0,
        successful_guard_solver_seconds=sum(e['solver_seconds'] for e in events), successful_guard_operation_seconds=sum(e['operation_wall_seconds'] for e in events),
        successful_guard_inverse_calls=sum(e['H_inverse_calls'] for e in events), successful_guard_logical_host_input_bytes=sum(e['logical_host_input_tensor_bytes'] for e in events),
        failed_guard_solver_seconds=None, failed_guard_inverse_calls=None, complete_actual_image_forward_counts=None, complete_host_traffic_bytes=None,
        checkpoints_bytes={j:v['checkpoint_sizes_bytes'] for j,v in data['runs'].items()}, final_guard_budget=None),
        inclusion='Success-only diagnostic sums are incomplete subtotals nested in GPU residence, never added to it. Final counters and failed-call audits were not persisted. Planned values are not realized costs.')
    for name, rows in [('EXECUTION_SUMMARY',execution),('COMPARISONS',table),('CONTRIBUTION_DELTAS',comparisons),('GUARD_EVENTS',events),('ACTIVE_DUAL_GROWTH',growth),('MAP_AUDIT_PARTIAL',diagnostics),('INVALIDATION_SNAPSHOTS',invalidations),('STAGE_SUMMARY',stages),('STAGE_CLASS_COVERAGE',classes)]: write_csv(root/(name+'.csv'),rows)
    for name,obj in [('AUDIT',audit),('BUDGET_AUDIT',budget),('CONTRIBUTIONS',comparisons),('MISSING_EVALUATION',missing)]: write_json(root/(name+'.json'),obj)
    assert len(table)==14 and len(stages)==18 and len(classes)==252
    check=dict(status='PASS_REPORT_INCOMPLETE_NOT_SCIENCE_PASS',main_rows=14,missing_new_stage_rows=18,missing_new_stage_class_rows=len(classes),all_new_metrics_null=True,unique_successful_events=14,failed_events_with_missing_audits=2,negative_report_fixtures_rejected=5)
    write_json(root/'REPORT_CHECK.json',check)
    def show(v): return 'null' if v is None else f'{v:.6f}' if isinstance(v,float) else str(v)
    lines=['# Snapshot guard 正式轮：失败终态与完整成本','', '本轮 INCOMPLETE / STOPPED_DEPENDENCIES。两条 ISIC 在 task3 epoch2 的 epoch_init 守卫未取得证书；各完成320/468更新。HK未派发，全四条训练屏障未通过，因此没有任何development val评价，test/reserved持续封存。没有重试、回退identity、放宽门或修改健康源。','',
        '|轨迹|状态|实测更新/计划|已存stage|失败GPU驻留秒|','|---|---|---:|---:|---:|']
    for r in execution: lines.append(f"|{r['job']}|{r['status']}|{show(r['observed_steps'])}/{r['planned_steps']}|{r['saved_stage_checkpoints']}|{show(r['GPU_residence_seconds'])}|")
    lines += ['', 'HK的null表示未生成STATUS/成本记录；协调器确认未派发，不假造测量。ISIC两stage checkpoint仅保留在NAS，未作部分评价。', '',
        '## 14行主表（百分比；历史复用与本轮缺失分开）','',
        '|数据|方法|证据|BA|avgBA|tail|forget↓|old|new|Acc|avgAcc|F1|AIR差距|', '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in table: lines.append('| '+' | '.join([r['dataset'],LABELS[r['method']], '历史' if r['evidence'].startswith('HISTORICAL') else '未评价']+[show(r[k]) for k in SIX+EXTRA[:3]+('AIR_minus_method_BA_pp',)])+' |')
    lines += ['', '全部六指标贡献见CONTRIBUTION_DELTAS.csv：本轮对Core/LT、无约束PMT、组合与单独及交互均为null。历史PMT的ISIC旧类收益伴随新类损害、HK收益弱仍成立，不能用本轮求解证书代替医学效果。AIR存在数据/预算/预训练资源差异；这些都是反复使用的development val，非独立患者确认、临床或显著性证明。', '',
        '## 真实训练中已取得的守卫证书','', '|轨迹|task/epoch/事件|约束|正dual|保留目标收益%|solver秒|最小slack|','|---|---|---:|---:|---:|---:|---:|']
    for e in events:
        lines.append('| '+' | '.join([e['job'],f"{e['task']}/{e['epoch']}/{e['event']}",str(e['constraint_count']),str(e['positive_duals']),show(None if e['retained_gain'] is None else 100*e['retained_gain']),show(e['solver_seconds']),f"{e['minimum_raw_slack']:.10g}"])+' |')
    lines += ['', '14个成功事件按(job,task,epoch,event,steps)去除累计快照重复，完整标量/KKT/trace均保留。4个task2活跃约束事件保留25.89%–28.60%的同状态二次目标收益，不是分类收益。10个NATIVE_FEASIBLE事件正dual为0，收益接近浮点舍入且retained=null；即使identity布尔为false也不认定为有效纠错。失败task3 epoch2为9434 fit、66038条约束，失败audit未保存，不补数值或identity判断。', '',
        '## 失败原因与诊断缺口','', '两项均为ValueError: Full-fit guard unresolved; no identity fallback。源在判定非CERTIFIED后先抛错，尚未写入events，因此未保存该次termination、KKT、迭代/trace和solver时间。只能确认未获得冻结数值证书，无法区分840秒软限、优化器停止或残差未过门；不能断言超时、OOM、约束不可行或只能identity。24小时总deadline未触发。具体匿名原因见FAILURES_ANON.json，私有日志/traceback不发布。', '',
        '## 费用与缺失','', f'两次失败GPU驻留全部计入：{total:.12f}秒（{total/3600:.9f} GPU小时）。父wall {p["whole_round_wall_seconds"]:.12f}秒另列；CPU调度检查9.429333493113518秒另列。成功及失败CPU投影、等待、传输已包含GPU驻留，不相加。旧闭合{data["historical_GPU_seconds"]:.12f}秒，本轮后{audit["closed_GPU_seconds"]:.12f}秒；四条旧预检不重复收费。', '',
        '实际训练峰值显存、失败solver耗时/传输量、最终完整预算计数未落盘，均为null；不借用预检峰值、不把规划曝光当实测。成功事件资源只是局部小计，BUDGET_AUDIT完整保留包含关系、缺失及checkpoint字节。费用为资源时间，无单价不造金额。', '',
        '每stage/31类的本轮n、recall、救回/破坏、错误换错误、间隔、旧新混淆以及same-state OFF/proposal与独立控制比较均未评价，见FINAL_CLASS_COMPARISON、STAGE_CLASS_COVERAGE和MISSING_EVALUATION。已公开历史逐类结果仍在../paired_transport1_20261015/results/；未读取旧NAS模型重评。', '',
        '仅固定encoder/frame的epoch初始化与boundary最终头保护current-fit；一次optimizer更新后旧证书失效，不能承诺val新类无损。原生loss/proximal/参数不变，guard不新增提图；checkpoint无完整支持不能独立全点重验。科学源f4b272de092a2db492126654a8bcceef071fe8e1保持冻结。当前完整交付失败结果，停止本轮，不自动新增科学轮。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(check)); return audit


if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
