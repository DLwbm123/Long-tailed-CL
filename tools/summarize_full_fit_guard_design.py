"""Render every frozen CPU case, including solver caps, identities and failures."""
import csv
import json
from pathlib import Path
import subprocess


def dump_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def show(value, places=6):
    return '未解/未提供' if value is None else f'{value:.{places}g}' if isinstance(value, float) else str(value)


def summarize(directory):
    root = Path(directory)
    a = json.loads((root/'CPU_RESULTS.json').read_text()); p = a['program']
    lock = json.loads((root/'DESIGN_LOCK.json').read_text())
    assert p['status'] == 'COMPLETE_CPU_CHECKS' and p['active'] is None
    assert len(p['cases']) == len(lock['cases']) == 19
    results = {r.get('case', {}).get('id', r.get('id', 'refresh_gradient')): r for r in a['results']}
    costs = {c['id']: c for c in p['costs']}; rows = []; traces = []
    assert len(costs) == len(p['costs']), 'Unexpected duplicate case ID; retain and inspect every residence'
    for cfg in lock['cases']:
        if cfg['kind'] == 'gradient': continue
        cid = cfg['id']; record = results[cid]; v = record.get('result') or {}; cost = costs.get(cid, {})
        rows.append(dict(id=cid, dataset=cfg['dataset'], task=cfg['task'], arm='LT' if cfg['LT'] else 'Core',
            fit_n=cfg['fit_n'], classes=cfg['classes'], support_rank=record.get('observed_support_rank'),
            status=v.get('status', record['status']), constraints=v.get('constraints'), atoms=v.get('atoms'),
            positive_duals=v.get('positive_duals'), minimum_slack=v.get('minimum_slack'),
            discriminative_delta=v.get('discriminative_delta'), relative_discriminative_delta=v.get('relative_discriminative_delta'),
            discriminative_identity=v.get('discriminative_identity'), useful_synthetic_update=v.get('useful_synthetic_update'),
            available_quadratic_gain=v.get('available_quadratic_gain'), feasible_quadratic_gain=v.get('feasible_quadratic_gain'),
            retained_gain=v.get('feasible_retained_gain'), remaining_gain_upper_bound=v.get('remaining_gain_upper_bound'),
            synthetic_current_rescued=v.get('synthetic_current_rescued'),synthetic_current_damaged=v.get('synthetic_current_damaged'),
            old_moment_loss_change=v.get('old_moment_loss_change'), stationarity=v.get('stationarity'),complementarity=v.get('complementarity'),
            CPU_process_residence_seconds=cost.get('CPU_process_residence_seconds'),CPU_user_seconds=cost.get('CPU_user_seconds'),
            CPU_system_seconds=cost.get('CPU_system_seconds'),peak_RSS_MiB=cost.get('peak_RSS_MiB'),
            CPU_pipeline_seconds=record.get('CPU_pipeline_seconds'),rank_check_seconds=record.get('rank_check_seconds'),
            solver_wall_seconds=v.get('solver_wall_seconds'),inverse_total_iterations=v.get('inverse_total_iterations')))
        for t in record.get('active_growth', []): traces.append(dict(id=cid, **t))
    dump_csv(root/'CPU_SCALE_SUMMARY.csv', rows); dump_csv(root/'ACTIVE_CONSTRAINT_GROWTH.csv', traces)
    gradient = results.get('refresh_gradient', {})
    failures = json.loads((root/'CPU_FAILURES_ANON.json').read_text()) if (root/'CPU_FAILURES_ANON.json').exists() else []
    feasible = [r for r in rows if r['status'] == 'FEASIBLE']
    unresolved = [r for r in rows if r['status'] != 'FEASIBLE']
    audit = dict(status='CPU_DIAGNOSTIC_COMPLETE_NOT_GPU_READY',design_source_commit=p['source_commit'],
        reporting_source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        requested_scale_cases=18,reported_scale_cases=len(rows),feasible_cases=len(feasible),unresolved_cases=len(unresolved),
        identity_cases=[r['id'] for r in feasible if r['discriminative_identity']],
        useful_synthetic_cases=[r['id'] for r in feasible if r['useful_synthetic_update']],
        gradient_status=gradient.get('status'),
        CPU_process_residence_seconds=sum(c['CPU_process_residence_seconds'] for c in p['costs']),
        CPU_user_seconds=sum(c['CPU_user_seconds'] for c in p['costs']),
        CPU_system_seconds=sum(c['CPU_system_seconds'] for c in p['costs']),
        peak_RSS_MiB=max(c['peak_RSS_MiB'] for c in p['costs']),
        supervisor_wall_seconds=p['supervisor_wall_seconds'],independent_child_residences=len(p['costs']),
        pipeline_operation_residence_nested=True,CPU_user_system_not_added_to_wall=True,
        failures=failures,program_failures=p['failures'],
        full_highdim_support_constrained_refresh_gradient_checked=False,
        formal_experiment=False,GPU_processes=0,new_GPU_seconds=0.,closed_historical_GPU_seconds=391490.24710153416,
        medical_images_read=0,medical_checkpoints_read=0,val_reads=0,test_accessed=False,optimizer_updates=0,
        production_integrated=False,GPU_ready=False)
    (root/'AUDIT.json').write_text(json.dumps(audit, indent=2)+'\n')
    lines=['# 全 fit 规模守卫与参考头刷新：CPU完整结果', '',
        '**完成独立 CPU 诊断，仍未获生产/GPU准入。** 本报告按冻结的全部18个规模与1组梯度检查完整列出，不删除锁死、cap、超时或失败。没有医学图像/特征/checkpoint、val/test/optimizer或GPU；真实规模指公开的fit点数、1536维和阶段类别数，特征与类级fit分配仍为预声明合成构造。', '',
        f'冻结设计源 `{p["source_commit"]}`；公开冻结DESIGN后才执行。旧已交付轮未poll/collect/重评。', '',
        '## 全阶段规模、作用与资源', '',
        '| 数据/阶段/背景 | fit N / K / rank | 结果 | atoms | 相对判别变化 | 保留目标收益 | CPU驻留秒 | 峰值MiB |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["dataset"]}/T{r["task"]}/{r["arm"]} | {r["fit_n"]}/{r["classes"]}/{show(r["support_rank"])} | {r["status"]} | {show(r["atoms"])} | {show(r["relative_discriminative_delta"])} | {show(r["retained_gain"])} | {show(r["CPU_process_residence_seconds"])} | {show(r["peak_RSS_MiB"])} |')
    lines += ['',f'18例中 {len(feasible)} 例全点可行，{len(unresolved)} 例未解。判别identity（仅可行例判定）：'+(', '.join(audit['identity_cases']) or '无')+'。满足预声明“非identity且有原二次目标收益”合成门：'+(', '.join(audit['useful_synthetic_cases']) or '无')+'。', '',
        'ATOM_CAP_NOT_SOLVED、TIME_CAP_NOT_SOLVED、硬超时或工程失败均不是identity，不以null补0。cap时的candidate未满足全部fit约束，禁止当守卫输出；KKT通过的松弛目标只能给完整问题剩余可达收益的数值上界，不是严格舍入区间证书。原目标收益与范数也不是医学救回/val新类无损。首任务无旧bank的identity属于预期零运输控制。所有逐atom违约、正dual数和松弛目标见ACTIVE_CONSTRAINT_GROWTH.csv，不只选最终成功。', '',
        '完整逐例原目标、保留比例、旧矩损失变化、新合成点救回/破坏、slack/KKT/迭代/全部资源见CPU_SCALE_SUMMARY.csv及CPU_RESULTS.json。', '',
        '## 参考刷新梯度', '']
    if gradient.get('status') == 'PASS':
        lines += ['| 背景 | naive误差 | 漏Wref刷新误差 | 正确隐式刷新误差 | 活跃约束 |', '|---|---:|---:|---:|---:|']
        for g in gradient['small_full_active_refresh']:
            lines.append('| '+g['arm']+' | '+show(g['naive_unconstrained_error'])+' | '+show(g['missing_reference_derivative_error'])+' | '+show(g['full_refresh_error'])+' | '+str(g['active'])+' |')
        lines += ['', '1536×23 原生 reference VJP 独立有限差分误差 '+show(gradient['highdim_reference_VJP_error'])+'。']
    else: lines += ['刷新梯度检查未通过/未完成，原因与成本保留在AUDIT；不得称梯度生产可用。']
    lines += ['',
        '正确增补为 S=A*λ，Z=H0^-1 S，再加 <Z,dB0−dH0 Wref>。只冻结Wref而沿用无约束/半约束梯度会遗漏路径。本检查固定a/R/previous/旧矩；它们或支持提取时序改变时仍需新推导。小规模完整活跃约束与高维独立reference VJP不能替代全部高维全fit受约束刷新梯度的端到端验证；该项仍未检查，不能写全PASS。', '',
        '## 费用、失败与限制', '',
        f'{len(p["costs"])}个独立CPU子进程驻留合计 **{audit["CPU_process_residence_seconds"]:.6f}秒**，user {audit["CPU_user_seconds"]:.6f}秒、system {audit["CPU_system_seconds"]:.6f}秒；峰值RSS {audit["peak_RSS_MiB"]:.3f}MiB。父任务墙钟 {p["supervisor_wall_seconds"]:.6f}秒另列。pipeline包含于operation和驻留；CPU user/system与wall不同口径，不能全部相加。每次失败/超时均计费一次，不按operation名去重。', '',
        '新增GPU0秒，历史闭合GPU391490.24710153416秒不变。医学encoder、真实训练特征分布、GPU速度及显存均未测；不能将CPU cap等同GPU训练一定不可行，也不能由合成可行推出临床或独立患者效果。实际费用和缺失值不填0。', '',
        '工程错误去标识摘要：'+json.dumps(failures,ensure_ascii=False)+'。原始私有日志/traceback不公开。', '',
        '## 分析与下一步', '',
        '本轮设计、源代码、全部正负与资源先公开交付，再依据完整结果分析。未观察旧域外推/旧旧错误仍不可识别；训练内逐点margin约束也不是val无损保证。保留Core/LT原控制与原数据/seed/每臂参数预算，不进行val网格、挑类/点或放松守卫来制造可行性。CPU资源未解时优先判断是否值得继续该求解路线，不机械增大cap或启动GPU。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    return audit


if __name__ == '__main__':
    import sys
    print(json.dumps(summarize(sys.argv[1]), indent=2))
