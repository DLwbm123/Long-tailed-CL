"""Public aggregate report for the frozen information-by-readout matrix."""
import csv
import json
from pathlib import Path


def main():
    root=Path('docs/full_support_linear_ce1_20261018');x=json.loads((root/'local_runtime/COLLECTED.private.json').read_text())
    program=x['PROGRAM.json'];assert not x['terminal']['ps']
    lock=json.loads((root/'PROTOCOL_LOCK.json').read_text())
    records=[x[j['id']+'.json'] for j in lock['jobs'] if j['id']+'.json' in x]
    fits=[r for r in records if r['role']=='fit'];evaluations=[r for r in records if r['role']=='evaluate']
    complete=not program['failures'] and not program.get('not_run') and len(records)==5
    if complete:
        assert all(r['status']=='COMPLETED' and r['source_commit']==program['source_commit'] and r['role_guard_passed'] for r in records)
        assert all(r['published_three_cells_matched'] for r in evaluations)
        assert sum(r['formal_head_fits'] for r in records)==2 and all(r['production_linear_solves']==0 for r in records)
        assert sum(r['normalized_feature_samples'] for r in fits)==8761
        assert sum(r['normalized_feature_samples'] for r in evaluations)==2155
        assert sum(r['model_logit_sample_evaluations'] for r in evaluations)==8620
        assert sum(r['head_bytes'] for r in fits)==258048
        assert x['source_storage_bytes']+x['output_storage_bytes']<lock['storage_limit_bytes']
    def save(name,obj):(root/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    def table(name,rows):
        if not rows:return
        with (root/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    groups={k:[row for r in evaluations for row in r[k]] for k in ('class_rows','summaries','paired_rows','paired_group_rows','confusion_rows')}
    if complete:assert all(len(groups[k])==v for k,v in dict(class_rows=168,summaries=48,paired_rows=168,paired_group_rows=48,confusion_rows=2088).items())
    differences=[];class_differences=[];interactions=[];class_interactions=[];gates=[]
    def interaction(rr,key):return rr['C_F'][key]-rr['C_H'][key]-rr['R_F'][key]+rr['R_H'][key]
    for r in evaluations:
        for q in ('Qfit','Qgate'):
            for a,b in lock['comparisons']:
                comparison=a+'__to__'+b
                for group in ('all','old','new'):
                    rr={v['model']:v for v in r['summaries'] if v['query']==q and v['group']==group}
                    differences.append(dict(dataset=r['dataset'],query=q,comparison=comparison,group=group,from_BA=rr[a]['BA'],to_BA=rr[b]['BA'],delta_BA_pp=100*(rr[b]['BA']-rr[a]['BA']),from_CE=rr[a]['CE'],to_CE=rr[b]['CE'],delta_CE=rr[b]['CE']-rr[a]['CE']))
                for rb in r['class_rows']:
                    if rb['query']!=q or rb['model']!=b:continue
                    ra=next(v for v in r['class_rows'] if v['query']==q and v['model']==a and v['class_index']==rb['class_index'])
                    class_differences.append(dict(dataset=r['dataset'],query=q,comparison=comparison,class_index=rb['class_index'],group=rb['group'],count=rb['count'],from_correct=ra['correct'],to_correct=rb['correct'],delta_correct=rb['correct']-ra['correct'],delta_recall_pp=100*(rb['recall']-ra['recall']),delta_CE=rb['CE']-ra['CE']))
            for group in ('all','old','new'):
                rr={v['model']:v for v in r['summaries'] if v['query']==q and v['group']==group}
                interactions.append(dict(dataset=r['dataset'],query=q,group=group,BA_interaction_pp=100*interaction(rr,'BA'),CE_interaction=interaction(rr,'CE')))
            for c in range(len([v for v in r['class_rows'] if v['query']==q and v['model']=='R_H'])):
                rr={v['model']:v for v in r['class_rows'] if v['query']==q and v['class_index']==c}
                class_interactions.append(dict(dataset=r['dataset'],query=q,class_index=c,group=rr['R_H']['group'],count=rr['R_H']['count'],correct_interaction=interaction(rr,'correct'),recall_interaction_pp=100*interaction(rr,'recall'),CE_interaction=interaction(rr,'CE')))
        for g in r['gates']:
            gates.append(dict(dataset=r['dataset'],query=g['query'],comparison=g['comparison'],descriptive_pass=g['pass_gate'],delta_all_pp=100*g['delta']['all'],delta_old_pp=100*g['delta']['old'],delta_new_pp=100*g['delta']['new']))
    # Verify each four-cell interaction also equals the difference of both edge contrasts.
    for v in interactions:
        edges={r['comparison']:r['delta_BA_pp'] for r in differences if all(r[k]==v[k] for k in ('dataset','query','group'))}
        assert abs(v['BA_interaction_pp']-(edges['C_H__to__C_F']-edges['R_H__to__R_F']))<1e-10
        assert abs(v['BA_interaction_pp']-(edges['R_F__to__C_F']-edges['R_H__to__C_H']))<1e-10
    for name,key in [('CLASS_RESULTS.csv','class_rows'),('SUMMARY.csv','summaries'),('PAIRED_CLASS.csv','paired_rows'),('PAIRED_GROUP.csv','paired_group_rows'),('CONFUSION.csv','confusion_rows')]:table(name,groups[key])
    for name,rows in [('COMPARE_READOUTS.csv',differences),('CLASS_COMPARE_READOUTS.csv',class_differences),('INTERACTIONS.csv',interactions),('CLASS_INTERACTIONS.csv',class_interactions),('GATES.csv',gates)]:table(name,rows)
    table('FIT_RESULTS.csv',[dict(dataset=r['dataset'],**{k:v for k,v in r['fit'].items() if k!='trace'}) for r in fits])
    table('FIT_TRACE.csv',[dict(dataset=r['dataset'],**v) for r in fits for v in r['fit']['trace']])
    table('SUPPORT_COUNTS.csv',[dict(dataset=r['dataset'],class_index=c,count=n) for r in fits for c,n in enumerate(r['support_counts'])])
    previous=json.loads(Path('docs/head_block_diagnostic1_20261018/COSTS.json').read_text())
    costs={k:sum(v[k] for v in program['costs']) for k in ('CPU_only_residence_seconds','GPU_process_residence_seconds','CPU_user_seconds','CPU_system_seconds')}
    costs.update(peak_worker_RSS_MiB=max((v['peak_RSS_MiB'] for v in program['costs']),default=0),supervisor_wall_seconds=program.get('supervisor_wall_seconds'),per_job=program['costs'],previous_CPU_residence_seconds=previous['cumulative_CPU_residence_seconds'],previous_GPU_residence_seconds=previous['cumulative_GPU_residence_seconds'],cumulative_CPU_residence_seconds=previous['cumulative_CPU_residence_seconds']+costs['CPU_only_residence_seconds'],cumulative_GPU_residence_seconds=previous['cumulative_GPU_residence_seconds']+costs['GPU_process_residence_seconds'],client_preparation_seconds=None,reused_training_or_statistic_construction_seconds_added=0)
    save('COSTS.json',costs)
    missing=[j['id']+'.json' for j in lock['jobs'] if j['id']+'.json' not in x]
    save('RESULTS.json',dict(status='COMPLETE_DEVELOPMENT_DIAGNOSTIC' if complete else 'INCOMPLETE_STOPPED',source_commit=program['source_commit'],program=program,records=records,missing=missing,all_processes_retired=True,retirement_checked_at=x['terminal']['checked_at'],logs=x['public_logs'],source_storage_bytes=x['source_storage_bytes'],output_storage_bytes=x['output_storage_bytes'],model_selected=False,equal_memory_budget=False,performance_upper_bound=False))
    lines=['# 完整旧支持 × 线性 CE 四格结果','',f"状态：{'全部五个角色完成，父子进程已退出' if complete else '未完整完成，保留失败和缺失，不重试'}。冻结源码 `{program['source_commit']}`。",'',
           'R_H=H-ridge，R_F=完整旧支持ridge，C_H=原H线性CE，C_F=本轮完整旧支持线性CE。其余三格复用，仅新增两次正式CE拟合，均从R_H初始化。','',
           '## 拟合收敛与完整成本','', '| 数据集 | 迭代 | closure梯度 | 终点梯度max_abs | 收敛 | 支持CE初→末 | 目标初→末 |','|---|---:|---:|---:|---|---|---|']
    for r in fits:
        f=r['fit'];lines.append(f"| {r['dataset']} | {f['optimizer_iterations']} | {f['closure_gradient_evaluations']} | {f['final_gradient_max_abs']:.6e} | {f['converged']} | {f['initial_support_CE']:.6f}→{f['final_support_CE']:.6f} | {f['initial_objective']:.6f}→{f['final_objective']:.6f} |")
    lines += ['',f"CPU驻留 {costs['CPU_only_residence_seconds']:.6f} 秒，user/system {costs['CPU_user_seconds']:.6f}/{costs['CPU_system_seconds']:.6f} 秒，GPU驻留 {costs['GPU_process_residence_seconds']:.6f} 秒；峰值worker RSS {costs['peak_worker_RSS_MiB']:.3f} MiB。supervisor wall {costs['supervisor_wall_seconds']} 秒，不重复加总。",'',
              f"累计CPU/GPU驻留 {costs['cumulative_CPU_residence_seconds']:.6f}/{costs['cumulative_GPU_residence_seconds']:.6f} 秒。历史拟合和缓存提取不重复计费，客户端准备未单测记null。各角色、失败和超时费用完整见COSTS.json。",'',
              f"正式fit {sum(r['formal_head_fits'] for r in records)}次，closure梯度 {sum(r['fit']['closure_gradient_evaluations'] for r in fits)}次，终点梯度 {len(fits)}次；工程梯度 {sum(r.get('engineering_gradient_evaluations',0) for r in records)}次，0正式岭求解/基线重拟合/编码器或适配器更新。",'',
              '## 四格总体及分组结果','', '| 数据集 | 查询 | 格 | 总体BA% | 旧BA% | 新BA% | 总体宏CE |','|---|---|---|---:|---:|---:|---:|']
    for r in evaluations:
        for q in ('Qfit','Qgate'):
            for m in lock['models']:
                rr={v['group']:v for v in r['summaries'] if v['query']==q and v['model']==m}
                lines.append(f"| {r['dataset']} | {q} | {m} | {100*rr['all']['BA']:.6f} | {100*rr['old']['BA']:.6f} | {100*rr['new']['BA']:.6f} | {rr['all']['CE']:.6f} |")
    lines += ['','## 四条边与交互量','', '| 数据集 | 查询 | 比较 | 总体/旧/新 BA差(pp) | 描述门槛 |','|---|---|---|---|---|']
    for g in gates:lines.append(f"| {g['dataset']} | {g['query']} | {g['comparison']} | {g['delta_all_pp']:+.6f} / {g['delta_old_pp']:+.6f} / {g['delta_new_pp']:+.6f} | {'PASS' if g['descriptive_pass'] else 'FAIL'} |")
    lines += ['','门槛只描述：同一边两个查询均总体BA严格改善、旧/新各下降不超过1pp；涉及CE必须另看收敛状态，不据门槛选头或部署。全部CE差值见COMPARE_READOUTS。','',
              '| 数据集 | 查询 | 组 | I = C_F−C_H−R_F+R_H (BA pp) | CE交互 |','|---|---|---|---:|---:|']
    for r in interactions:lines.append(f"| {r['dataset']} | {r['query']} | {r['group']} | {r['BA_interaction_pp']:+.6f} | {r['CE_interaction']:+.6f} |")
    lines += ['','## 逐类全部正负与零变化','', '| 数据集 | 查询 | 比较 | 类 | n | 正确数前→后 | 召回差pp |','|---|---|---|---:|---:|---|---:|']
    for r in class_differences:lines.append(f"| {r['dataset']} | {r['query']} | {r['comparison']} | {r['class_index']} | {r['count']} | {r['from_correct']}→{r['to_correct']} | {r['delta_recall_pp']:+.6f} |")
    lines += ['','逐类CE、每类交互见CLASS_COMPARE_READOUTS/CLASS_INTERACTIONS；全部类与组的真实配对救回/伤害/错转错见PAIRED_CLASS/PAIRED_GROUP，所有混淆单元格见CONFUSION。不用微准确率或净计数替代宏BA。','',
              '## 检查、访问与停止边界','',f"失败：{program['failures']}；未运行：{program.get('not_run',[])}；缺失：{missing}。日志摘要见RESULTS。",'',
              '完整旧特征获准用于本轮离线拟合，未读取旧图像/E/test或新增标签。两个新头封存后才评估；三格旧结果逐类复核通过时记录于RESULTS。全部查询是历史开发数据，患者分组未知，极小尾类保留。', '',
              '两种损失尺度及有效正则不同，仅比较冻结配方；不声称最优CE/ridge、原H访问预算、等存储收益、性能上界、统计显著性或冻结表示不可能。非收敛终点不排除优化限制。没有调参、续跑、扩种子或自动后续实验。私有头、特征、身份和原始日志不公开。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(complete=complete,records=len(records),class_rows=len(groups['class_rows']),class_differences=len(class_differences),interactions=len(interactions),costs=costs)))


if __name__=='__main__':main()
