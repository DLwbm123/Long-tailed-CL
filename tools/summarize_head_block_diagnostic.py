"""Aggregate-only report for all four fixed readouts and all six pairs."""
import csv
import json
import os
from pathlib import Path


def main():
    root=Path(os.environ.get('Q407_DOC','docs/head_block_diagnostic1_20261018'))
    x=json.loads((root/'local_runtime/COLLECTED.private.json').read_text());program=x['PROGRAM.json'];assert not x['terminal']['ps']
    lock=json.loads((root/'PROTOCOL_LOCK.json').read_text());records=[x[n+'_blocks.json'] for n in ('ISIC','HK') if n+'_blocks.json' in x]
    complete=not program['failures'] and not program.get('not_run') and len(records)==2
    if complete:
        assert all(v['status']=='COMPLETED' and v['source_commit']==program['source_commit'] and v['published_A_B_aggregates_matched'] and v['role_guard_passed'] and v['unit']['status']=='PASS' for v in records)
        assert all(v[k]==0 for v in records for k in ('head_fits','production_linear_solves','engineering_linear_solves','gradient_evaluations','adapter_updates','encoder_updates'))
        assert sum(v['shared_feature_sample_forwards'] for v in records)==2155 and sum(v['model_logit_sample_evaluations'] for v in records)==8620
        assert sum(v['mixed_head_payload_bytes'] for v in records)==516096
        assert x['source_storage_bytes']+x['output_storage_bytes']<lock['storage_limit_bytes']
    def save(name,obj):(root/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    def table(name,rows):
        if not rows:return
        with (root/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    groups={k:[row for v in records for row in v[k]] for k in ('class_rows','summaries','confusion_rows','paired_rows','paired_group_rows','transition_rows','score_rows','effect_rows','competitor_rows')}
    if complete:
        expected=dict(class_rows=168,summaries=48,confusion_rows=2088,paired_rows=252,paired_group_rows=72,score_rows=168)
        assert all(len(groups[k])==n for k,n in expected.items())
    gates=[dict(dataset=r['dataset'],query=r['query'],model=r['model'],pass_gate=r['pass_gate'],feasible=r['feasible'],reason=r['reason'],delta_all_pp=100*r['delta']['all'],delta_old_pp=100*r['delta']['old'],delta_new_pp=100*r['delta']['new']) for v in records for r in v['gates']]
    differences=[];class_differences=[];interactions=[];class_interactions=[]
    for v in records:
        for q in ('Qfit','Qgate'):
            for a,b in lock['comparisons']:
                for group in ('all','old','new'):
                    ra=next(r for r in v['summaries'] if r['query']==q and r['model']==a and r['group']==group)
                    rb=next(r for r in v['summaries'] if r['query']==q and r['model']==b and r['group']==group)
                    differences.append(dict(dataset=v['dataset'],query=q,comparison=a+'__to__'+b,group=group,from_BA=ra['BA'],to_BA=rb['BA'],delta_BA_pp=100*(rb['BA']-ra['BA']),from_CE=ra['CE'],to_CE=rb['CE'],delta_CE=rb['CE']-ra['CE']))
                for rb in v['class_rows']:
                    if rb['query']!=q or rb['model']!=b:continue
                    ra=next(r for r in v['class_rows'] if r['query']==q and r['model']==a and r['class_index']==rb['class_index'])
                    class_differences.append(dict(dataset=v['dataset'],query=q,comparison=a+'__to__'+b,class_index=rb['class_index'],group=rb['group'],count=rb['count'],from_correct=ra['correct'],to_correct=rb['correct'],delta_correct=rb['correct']-ra['correct'],delta_recall_pp=100*(rb['recall']-ra['recall']),delta_CE=rb['CE']-ra['CE']))
            for group in ('all','old','new'):
                rr={r['model']:r for r in v['summaries'] if r['query']==q and r['group']==group}
                interactions.append(dict(dataset=v['dataset'],query=q,group=group,BA_interaction_pp=100*(rr['B']['BA']-rr['old_B_new_A']['BA']-rr['old_A_new_B']['BA']+rr['A']['BA']),CE_interaction=rr['B']['CE']-rr['old_B_new_A']['CE']-rr['old_A_new_B']['CE']+rr['A']['CE']))
            for c in range(len([r for r in v['class_rows'] if r['query']==q and r['model']=='A'])):
                rr={r['model']:r for r in v['class_rows'] if r['query']==q and r['class_index']==c}
                class_interactions.append(dict(dataset=v['dataset'],query=q,class_index=c,group=rr['A']['group'],count=rr['A']['count'],correct_interaction=rr['B']['correct']-rr['old_B_new_A']['correct']-rr['old_A_new_B']['correct']+rr['A']['correct'],recall_interaction_pp=100*(rr['B']['recall']-rr['old_B_new_A']['recall']-rr['old_A_new_B']['recall']+rr['A']['recall'])))
    names=dict(class_rows='CLASS_RESULTS.csv',summaries='SUMMARY.csv',confusion_rows='CONFUSION.csv',paired_rows='PAIRED_CLASS.csv',paired_group_rows='PAIRED_GROUP.csv',transition_rows='TRANSITIONS.csv',score_rows='SCORES.csv',effect_rows='SCORE_EFFECTS.csv',competitor_rows='COMPETITORS.csv')
    for key,name in names.items():table(name,groups[key])
    for name,rows in [('GATES.csv',gates),('COMPARE_READOUTS.csv',differences),('CLASS_COMPARE_READOUTS.csv',class_differences),('INTERACTIONS.csv',interactions),('CLASS_INTERACTIONS.csv',class_interactions)]:table(name,rows)
    previous=json.loads(Path('docs/offline_information_ab1_20261018/COSTS.json').read_text())
    costs={k:sum(v[k] for v in program['costs']) for k in ('CPU_only_residence_seconds','GPU_process_residence_seconds','CPU_user_seconds','CPU_system_seconds')}
    costs.update(peak_worker_RSS_MiB=max((v['peak_RSS_MiB'] for v in program['costs']),default=0),supervisor_wall_seconds=program['supervisor_wall_seconds'],per_job=program['costs'],previous_CPU_residence_seconds=previous['cumulative_CPU_residence_seconds'],previous_GPU_residence_seconds=previous['cumulative_GPU_residence_seconds'],cumulative_CPU_residence_seconds=previous['cumulative_CPU_residence_seconds']+costs['CPU_only_residence_seconds'],cumulative_GPU_residence_seconds=previous['cumulative_GPU_residence_seconds']+costs['GPU_process_residence_seconds'],client_preparation_seconds=None,reused_fitting_or_statistic_construction_seconds_added=0)
    save('COSTS.json',costs)
    save('RESULTS.json',dict(status='COMPLETE_DEVELOPMENT_DIAGNOSTIC' if complete else 'INCOMPLETE_STOPPED',source_commit=program['source_commit'],program=program,records=records,all_processes_retired=True,retirement_checked_at=x['terminal']['checked_at'],logs=x['public_logs'],source_storage_bytes=x['source_storage_bytes'],output_storage_bytes=x['output_storage_bytes'],diagnostic_only=True,model_selected=False,equal_memory_budget=False,performance_upper_bound=False))
    short=dict(A='A',B='B',old_B_new_A='O',old_A_new_B='N')
    lines=['# 四格分类头分块诊断结果','',
           f"状态：{'两个数据集全部完成，进程已退出' if complete else '未完整完成，保留失败和缺失，不自动重跑'}。冻结源码 `{program['source_commit']}`。",'',
           'A=[A旧,A新]，O=[B旧,A新]，N=[A旧,B新]，B=[B旧,B新]。只拼接已封存列，无拟合、求解、调参或模型选择。混合头只是诊断干预，并不保证对应共同岭目标的解；包含B的新增历史信息，不作原H预算、等预算、性能上界或已验证方法宣称。','',
           '## 四格完整结果','',
           '| 数据集 | 查询 | 读出 | 总体BA% | 旧BA% | 新BA% | 总体/旧/新 Δpp 相对A | 描述门槛 |','|---|---|---|---:|---:|---:|---|---|']
    for v in records:
        for q in ('Qfit','Qgate'):
            for model in lock['models']:
                rr={r['group']:r for r in v['summaries'] if r['query']==q and r['model']==model}
                if model=='A':delta='0 / 0 / 0';gate='参考'
                else:
                    g=next(r for r in gates if r['dataset']==v['dataset'] and r['query']==q and r['model']==model)
                    delta=f"{g['delta_all_pp']:+.6f} / {g['delta_old_pp']:+.6f} / {g['delta_new_pp']:+.6f}";gate='PASS' if g['pass_gate'] else 'FAIL'
                lines.append(f"| {v['dataset']} | {q} | {short[model]} | {100*rr['all']['BA']:.6f} | {100*rr['old']['BA']:.6f} | {100*rr['new']['BA']:.6f} | {delta} | {gate} |")
    lines += ['','门槛仅描述：同一读出在同一数据集两个查询均总体BA提高、旧/新各下降不超过1pp才算双查询通过；不据此选择/部署。全部CE与六组差值见SUMMARY/COMPARE_READOUTS，全部逐类差值含正负及零变化见CLASS_COMPARE_READOUTS。','',
              '## 全部六组配对救回与伤害','',
              '| 数据集 | 查询 | 比较 | 组 | 救回 | 伤害 | 错误间转换 |','|---|---|---|---|---:|---:|---:|']
    for r in groups['paired_group_rows']:
        ka,kb=r['comparison'].split('__to__');lines.append(f"| {r['dataset']} | {r['query']} | {short[ka]}→{short[kb]} | {r['group']} | {r['rescued']} | {r['harmed']} | {r['wrong_to_other_wrong']} |")
    lines += ['','CONFUSION.csv保留每个真类×预测类的全部单元格，包括零；TRANSITIONS.csv包含所有非零预测改变。配对救回/伤害为真实样本配对，不用净错误量替代。','',
              '## ISIC预先指定的class3/class4检查','',
              '| 查询 | 类别 | 读出 | 正确数 | 正确类分数均值 | 最大旧竞争分数均值 | 旧margin均值 | 全部竞争margin均值 |','|---|---|---|---:|---:|---:|---:|---:|']
    for r in groups['score_rows']:
        if r['dataset']!='ISIC' or r['class_index'] not in (3,4):continue
        c=next(v for v in groups['class_rows'] if v['dataset']=='ISIC' and v['query']==r['query'] and v['model']==r['model'] and v['class_index']==r['class_index'])
        lines.append(f"| {r['query']} | {r['class_index']} | {short[r['model']]} | {c['correct']}/{c['count']} | {r['true_score_mean']:.6f} | {r['old_rival_score_mean']:.6f} | {r['old_margin_mean']:.6f} | {r['margin_mean']:.6f} |")
    lines += ['','上述最大竞争分数均排除正确类本身；对于旧class3，旧margin衡量旧组内部竞争，对于新class4，旧margin衡量其与旧组的竞争。完整分布min/median/mean/max及新竞争分数见SCORES，竞争类别分布见COMPETITORS。','',
              '| 查询 | 读出 | class4全部误分类去向（类别:数量） |','|---|---|---|']
    for q in ('Qfit','Qgate'):
        for model in lock['models']:
            rr=[r for r in groups['confusion_rows'] if r['dataset']=='ISIC' and r['query']==q and r['model']==model and r['true_class']==4 and r['predicted_class']!=4 and r['count']]
            lines.append(f"| {q} | {short[model]} | {', '.join(str(r['predicted_class'])+':'+str(r['count']) for r in rr) or '无'} |")
    lines += ['','## class4分数变化与竞争者切换','',
              '| 查询 | 比较 | 子集 | n | Δ正确分数均值 | Δ固定竞争分数均值 | 竞争切换惩罚均值 | Δ实际margin均值 |','|---|---|---|---:|---:|---:|---:|---:|']
    for r in groups['effect_rows']:
        if r['dataset']!='ISIC' or r['class_index']!=4 or not r['comparison'].startswith('A__to__') or r['outcome'] not in ('all','harmed'):continue
        ka,kb=r['comparison'].split('__to__');lines.append(f"| {r['query']} | {short[ka]}→{short[kb]} | {r['outcome']} | {r['count']} | {r['true_score_delta_mean']:+.6f} | {r['fixed_rival_score_delta_mean']:+.6f} | {r['rival_switch_penalty_mean']:.6f} | {r['actual_margin_delta_mean']:+.6f} |")
    lines += ['','逐样本精确分解：Δ实际margin=Δ正确类分数−Δ固定源竞争者分数−竞争切换惩罚，最后一项非负。不同读出的harmed子集不同，不能将不同子集的均值当作同一批样本的贡献比较。所有类别、六组比较及非空结果子集均在SCORE_EFFECTS；空子集由配对零计数标明。','',
              '## 四格非加性与数值验证','',
              '| 数据集 | 查询 | 组 | BA非加性 B−O−N+A（pp） |','|---|---|---|---:|']
    for r in interactions:lines.append(f"| {r['dataset']} | {r['query']} | {r['group']} | {r['BA_interaction_pp']:+.6f} |")
    for v in records:
        lines.append(f"\n{v['dataset']}：分数加性最大误差 {max(r['score_additivity_max_error'] for r in v['identities']):.3e}，margin分解最大误差 {max(r['margin_decomposition_max_error'] for r in v['identities']):.3e}；A/B公开聚合复核通过，角色守卫与小张量工程检查通过。")
    lines += ['','分数层的B−O−N+A为零；BA/CE的非加性来自后续非线性评估，不能解释为可加的总体因果贡献。只支持当前固定样本与读出干预的判断；不能由此断言伤害对所有方法都不可避免，也未分离均值与二阶矩对B的作用。','',
              '## 全部成本、失败与边界','',
              f"本轮CPU驻留 {costs['CPU_only_residence_seconds']:.6f} 秒，user/system {costs['CPU_user_seconds']:.6f}/{costs['CPU_system_seconds']:.6f} 秒，GPU驻留 {costs['GPU_process_residence_seconds']:.6f} 秒；峰值worker RSS {costs['peak_worker_RSS_MiB']:.3f} MiB，supervisor wall {costs['supervisor_wall_seconds']:.6f} 秒（不重复加总）。",'',
              f"历史累计CPU/GPU驻留 {costs['cumulative_CPU_residence_seconds']:.6f}/{costs['cumulative_GPU_residence_seconds']:.6f} 秒。旧头拟合、B统计构建不重复计费；本轮所有检查、归一化、4头评估、配对与分数汇总均计费，客户端准备未计时记null。",'',
              f"冻结源码 {x['source_storage_bytes']:,} B、本轮运行产物 {x['output_storage_bytes']:,} B；混合头仅内存组装，总载荷516,096 B。公开备份另计。没有持久化逐样本预测/分数，没有读取S/H/完整矩/E或原图。",'',
              f"失败：{json.dumps(program['failures'],ensure_ascii=False)}；未运行：{json.dumps(program.get('not_run',[]),ensure_ascii=False)}；缺失：{json.dumps([n for n in ('ISIC','HK') if n+'_blocks.json' not in x])}。全部退出码、分角色成本、日志摘要见RESULTS/COSTS。",'',
              'Qfit/Qgate均为反复使用的开发数据，患者隔离未知；极小尾类和全部伤害不剔除。0重新拟合/岭求解/梯度/新标签/GPU，无调参、扩种子、E或自动下一轮。报告所有四格，未选择任何混合头作为正式方法。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(complete=complete,gates=gates,costs=costs,row_counts={k:len(v) for k,v in groups.items()}),indent=2))


if __name__=='__main__':main()
