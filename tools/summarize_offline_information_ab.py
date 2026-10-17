"""Publish aggregate results and all charged costs for the frozen offline A/B."""
import csv
import json
import os
from pathlib import Path


def main():
    root=Path(os.environ.get('Q406_DOC','docs/offline_information_ab1_20261018'))
    collected=json.loads((root/'local_runtime/COLLECTED.private.json').read_text())
    program=collected['PROGRAM.json'];terminal=collected['terminal'];assert not terminal['ps']
    lock=json.loads((root/'PROTOCOL_LOCK.json').read_text());expected=[j['id'] for j in lock['jobs']]
    records={j:collected[j+'.json'] for j in expected if j+'.json' in collected}
    complete=not program['failures'] and not program.get('not_run') and len(records)==len(expected)
    if complete:
        assert program['status']=='COMPLETE_AWAITING_DELIVERY'
        assert all(v['source_commit']==program['source_commit'] and v['status']=='COMPLETED' and v['role_guard_passed'] for v in records.values())
        assert sum(v['production_linear_solves'] for v in records.values())==2
        assert records['unit']['unit']['engineering_small_linear_solves']==3
        for n in ('ISIC','HK'):
            assert records[n+'_build']['statistic_bytes']==lock['expected_old_statistic_payload_bytes'][n]
            assert records[n+'_evaluate']['reference_aggregates_matched']
    def save(name,value):(root/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    def table(name,rows):
        if not rows:return
        with (root/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    evaluations=[v for v in records.values() if v['role']=='evaluate']
    classes=[r for v in evaluations for r in v['class_rows']]
    summaries=[r for v in evaluations for r in v['summaries']]
    pairs=[r for v in evaluations for r in v['paired_rows']]
    group_pairs=[r for v in evaluations for r in v['group_paired_rows']]
    differences=[];class_differences=[];gates=[];storage=[];support=[];heads=[]
    for v in evaluations:
        name=v['dataset']
        for q,g in v['gates'].items():
            gates.append(dict(dataset=name,query=q,pass_gate=g['selected']=='candidate',feasible=g['feasible'],reason=g['reason']))
            for group in ('all','old','new'):
                a=next(r for r in v['summaries'] if r['query']==q and r['group']==group and r['model']=='A')
                b=next(r for r in v['summaries'] if r['query']==q and r['group']==group and r['model']=='B')
                differences.append(dict(dataset=name,query=q,group=group,A_BA=a['BA'],B_BA=b['BA'],delta_BA_pp=100*(b['BA']-a['BA']),A_CE=a['CE'],B_CE=b['CE'],delta_CE=b['CE']-a['CE']))
        for b in v['class_rows']:
            if b['model']!='B':continue
            a=next(r for r in v['class_rows'] if r['query']==b['query'] and r['class_index']==b['class_index'] and r['model']=='A')
            class_differences.append(dict(dataset=name,query=b['query'],class_index=b['class_index'],group=b['group'],count=b['count'],A_correct=a['correct'],B_correct=b['correct'],delta_correct=b['correct']-a['correct'],delta_recall_pp=100*(b['recall']-a['recall']),delta_CE=b['CE']-a['CE']))
    for name in ('ISIC','HK'):
        build=records.get(name+'_build');fit=records.get(name+'_fit')
        if build:
            storage.append(dict(dataset=name,A_H_payload_bytes=lock['expected_H_payload_bytes'][name],B_old_statistic_payload_bytes=build['statistic_bytes'],B_statistic_file_bytes=build['statistic_file_bytes'],B_over_A_memory_ratio=build['statistic_bytes']/lock['expected_H_payload_bytes'][name],archived_old_cache_file_bytes=build['old_cache_file_bytes'],head_bytes_per_arm=lock['expected_head_bytes'][name]))
        for c,count in enumerate(lock['old_counts'][name]+lock['new_counts'][name]):
            old=len(lock['old_counts'][name]);support.append(dict(dataset=name,class_index=c,group='old' if c<old else 'new',A_count=lock['H_counts'][name][c] if c<old else count,B_count=count,class_mass=1/(old+2)))
        if fit:heads += [dict(dataset=name,**r) for r in fit['head_stats']]
    if complete:
        assert len(classes)==84 and len(summaries)==24 and len(pairs)==42 and len(group_pairs)==12 and len(class_differences)==42
        assert sum(v.get('normalized_feature_samples',0) for v in records.values())==11149
        assert sum(v.get('model_logit_sample_evaluations',0) for v in records.values())==4310
    previous=json.loads(Path('docs/nullspace_response1_20261017/COSTS.json').read_text())
    costs={k:sum(v[k] for v in program['costs']) for k in ('CPU_only_residence_seconds','GPU_process_residence_seconds','CPU_user_seconds','CPU_system_seconds')}
    costs.update(peak_worker_RSS_MiB=max((v['peak_RSS_MiB'] for v in program['costs']),default=0),supervisor_wall_seconds=program['supervisor_wall_seconds'],per_job=program['costs'],
                 previous_CPU_residence_seconds=previous['cumulative_CPU_residence_seconds'],previous_GPU_residence_seconds=previous['cumulative_GPU_residence_seconds'],
                 cumulative_CPU_residence_seconds=previous['cumulative_CPU_residence_seconds']+costs['CPU_only_residence_seconds'],cumulative_GPU_residence_seconds=previous['cumulative_GPU_residence_seconds']+costs['GPU_process_residence_seconds'],
                 client_preparation_seconds=None,reused_training_seconds_added=0,source='All seven role processes charged, including engineering checks and statistic reconstruction. Supervisor wall not added again; prior training and cache extraction not charged twice.')
    save('COSTS.json',costs)
    save('RESULTS.json',dict(status='COMPLETE_DEVELOPMENT_RESULTS' if complete else 'INCOMPLETE_STOPPED',source_commit=program['source_commit'],program=program,all_processes_retired=True,retirement_checked_at=terminal['checked_at'],records=records,
                            missing=[j for j in expected if j not in records],logs=collected['public_logs'],new_offline_information_intervention=True,preexisting_protocol_memory=False,equal_memory_budget=False,performance_upper_bound=False))
    for filename,rows in [('CLASS_RESULTS.csv',classes),('SUMMARY.csv',summaries),('COMPARE_AB.csv',differences),('CLASS_COMPARE_AB.csv',class_differences),('PAIRED_CLASS.csv',pairs),('PAIRED_GROUP.csv',group_pairs),('GATES.csv',gates),('STORAGE.csv',storage),('SUPPORT_COUNTS.csv',support),('HEAD_STATS.csv',heads)]:table(filename,rows)
    lines=['# 新增离线完整统计量 A/B 对照结果','',
           '本轮为新增离线信息对照：B 从既有旧 S 缓存新构建完整统计量；A 复用已封存 H+新 S 岭头。不是原协议下已有记忆，不是等存储比较，也不作性能上界宣称。',
           '',f"状态：{'全部7个角色完成，进程已退出' if complete else '未完整完成，保留失败及缺失，不自动重跑'}。源码冻结提交：`{program['source_commit']}`。",'',
           '## 两臂与判定','',
           '相同冻结特征、零适配器归一化、seed74002、首转移、类平衡目标、λ=0.001、logit scale=8；无bias/头列归一化。B每数据集只求解一次。两套B头封存后才打开Qfit/Qgate；门槛仅用于报告，不选择或部署模型。E/test未访问。','',
           '| 数据集 | 查询 | A 总体 BA% | B 总体 BA% | 总体Δpp | 旧类Δpp | 新类Δpp | 门槛 |','|---|---|---:|---:|---:|---:|---:|---|']
    for v in evaluations:
        for q,g in v['gates'].items():
            rows={r['group']:r for r in differences if r['dataset']==v['dataset'] and r['query']==q};a=rows['all']
            lines.append(f"| {v['dataset']} | {q} | {100*a['A_BA']:.6f} | {100*a['B_BA']:.6f} | {a['delta_BA_pp']:+.6f} | {rows['old']['delta_BA_pp']:+.6f} | {rows['new']['delta_BA_pp']:+.6f} | {'PASS' if g['selected']=='candidate' else 'FAIL'} |")
    lines += ['','通过条件为该数据集两个查询均总体BA严格改善、旧/新各下降不超过1pp；逐类全部正负差值见 CLASS_COMPARE_AB.csv，完整CE见 COMPARE_AB.csv。','', '## 类别变化与配对结果','']
    for r in class_differences:
        if r['delta_correct']:lines.append(f"- {r['dataset']} {r['query']} {r['group']} class{r['class_index']}：{r['A_correct']}/{r['count']} → {r['B_correct']}/{r['count']}，召回变化 {r['delta_recall_pp']:+.6f}pp。")
    lines += ['','| 数据集 | 查询 | 组 | 救回 | 伤害 | 错误间转换 |','|---|---|---|---:|---:|---:|']
    for r in group_pairs:lines.append(f"| {r['dataset']} | {r['query']} | {r['group']} | {r['rescued']} | {r['harmed']} | {r['wrong_to_other_wrong']} |")
    lines += ['','## 信息、存储与数值检查','']
    for r in storage:
        n=r['dataset'];fit=records.get(n+'_fit',{})
        lines.append(f"- {n}：A记忆载荷 {r['A_H_payload_bytes']:,} B；B完整旧统计载荷 {r['B_old_statistic_payload_bytes']:,} B（{r['B_old_statistic_payload_bytes']/1048576:.6f} MiB），是A的 {r['B_over_A_memory_ratio']:.4f} 倍；封装后文件 {r['B_statistic_file_bytes']:,} B。已有旧S缓存 {r['archived_old_cache_file_bytes']:,} B 单独列出，不冒充方法记忆。每臂头 {r['head_bytes_per_arm']:,} B。")
        if fit:lines.append(f"  A法方程残差 {fit['A_normal_relative_residual']:.3e}；B求解残差 {fit['solve_relative_residual']:.3e}；旧矩/均值相对变化 {fit['old_moment_relative_change']:.6g}/{fit['old_mean_relative_change']:.6g}。")
    lines += ['','数值载荷不含来源元数据/文件封装/求解工作空间；实际文件大小及进程峰值另列。工程3维例子验证分阶段累积与直接构建、显式类平衡法方程及类内复制不变性，共3次小求解；正式B求解2次、A重拟合0次。成功完成时，所有角色访问守卫通过，A逐类计数/正确数/错误方向与已发布结果一致且CE误差<1e-10。','',
              '## 成本、失败与缺失','',f"本轮CPU进程驻留 {costs['CPU_only_residence_seconds']:.6f} 秒，CPU user/system {costs['CPU_user_seconds']:.6f}/{costs['CPU_system_seconds']:.6f} 秒，GPU驻留 {costs['GPU_process_residence_seconds']:.6f} 秒；峰值worker RSS {costs['peak_worker_RSS_MiB']:.3f} MiB。supervisor wall {costs['supervisor_wall_seconds']:.6f} 秒，不重复累计。",'',
              f"历史累计CPU驻留 {costs['cumulative_CPU_residence_seconds']:.6f} 秒，GPU驻留 {costs['cumulative_GPU_residence_seconds']:.6f} 秒。旧缓存提取、H/模型历史训练不重复计费；本轮构建/检查/求解/评估全部计费，分角色见 COSTS.json。客户端准备时间未测，记null。",'',
              f"失败：{json.dumps(program['failures'],ensure_ascii=False)}；未运行：{json.dumps(program.get('not_run',[]),ensure_ascii=False)}；缺失产物：{json.dumps([j for j in expected if j not in records])}。日志摘要见RESULTS.json，弃用警告与运行错误分列；不隐藏负结果。",'',
              '## 解释边界与停止','',
              '本轮只识别当前固定岭目标下完整历史信息替换H的影响。正结果支持该信息替换的价值，不能称新方法或等预算收益；负结果不证明压缩无关、特征不可学或存在不可逾越上界。所有Qfit/Qgate均为历史开发数据，患者隔离未知；极小尾类完整报告。没有调参、扩种子、访问E或自动下一轮。本轮到交付为止。私有特征、矩、模型头和身份不公开。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(complete=complete,gates=gates,costs=costs),indent=2))


if __name__=='__main__':main()
