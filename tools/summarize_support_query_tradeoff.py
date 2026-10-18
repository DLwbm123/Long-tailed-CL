"""Compare support-domain predictions with unchanged published query results."""
import csv
import json
from pathlib import Path


def main():
    root=Path('docs/support_query_tradeoff1_20261018');x=json.loads((root/'local_runtime/COLLECTED.private.json').read_text())
    program=x['PROGRAM.json'];assert not x['terminal']['ps'];lock=json.loads((root/'PROTOCOL_LOCK.json').read_text())
    record=x.get('ISIC_audit.json');complete=not program['failures'] and not program.get('not_run') and record is not None
    def save(name,obj):(root/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    def table(name,rows):
        if not rows:return
        with (root/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    previous_root=Path('docs/full_support_linear_ce1_20261018');previous=json.loads((previous_root/'COSTS.json').read_text())
    costs={k:sum(v[k] for v in program['costs']) for k in ('CPU_only_residence_seconds','GPU_process_residence_seconds','CPU_user_seconds','CPU_system_seconds')}
    costs.update(peak_worker_RSS_MiB=max((v['peak_RSS_MiB'] for v in program['costs']),default=0),supervisor_wall_seconds=program.get('supervisor_wall_seconds'),per_job=program['costs'],previous_CPU_residence_seconds=previous['cumulative_CPU_residence_seconds'],previous_GPU_residence_seconds=previous['cumulative_GPU_residence_seconds'],cumulative_CPU_residence_seconds=previous['cumulative_CPU_residence_seconds']+costs['CPU_only_residence_seconds'],cumulative_GPU_residence_seconds=previous['cumulative_GPU_residence_seconds']+costs['GPU_process_residence_seconds'],client_preparation_seconds=None,reused_training_or_query_evaluation_seconds_added=0)
    save('COSTS.json',costs)
    differences=[];class_differences=[];hypotheses=[];domain_changes=[];query_match=False;ce_error=None
    if record:
        assert record['status']=='COMPLETED' and record['source_commit']==program['source_commit'] and record['role_guard_passed'] and record['engineering_checks_passed']
        assert all(record[k]==0 for k in ('head_fits','linear_solves','gradient_evaluations','encoder_updates','adapter_updates','new_head_bytes','query_feature_evaluations'))
        assert record['support_normalized_samples']==5526 and record['model_logit_sample_evaluations']==16578 and not record['raw_query_access'] and not record['E_access']
        prior=json.loads((previous_root/'RESULTS.json').read_text());ref=next(v for v in prior['records'] if v['dataset']=='ISIC' and v['role']=='evaluate')
        assert ref['source_commit']==lock['inputs']['full']['source']
        for key in ('class_rows','summaries','confusion_rows','paired_rows','paired_group_rows'):
            a=[v for v in record[key] if v['query']!='S_full']
            b=[v for v in ref[key] if (v['model'] in lock['models'] if 'model' in v else v['comparison'] in {a+'__to__'+b for a,b in lock['comparisons']})]
            assert a==b,'Published query aggregate changed'
        query_match=True
        fit=next(v for v in prior['records'] if v['dataset']=='ISIC' and v['role']=='fit')
        support_ce=next(v['CE'] for v in record['summaries'] if v['query']=='S_full' and v['model']=='C_F' and v['group']=='all')
        ce_error=abs(support_ce-fit['fit']['final_support_CE']);assert ce_error<1e-10
        for q in ('S_full','Qfit','Qgate'):
            for a,b in lock['comparisons']:
                comp=a+'__to__'+b
                for group in ('all','old','new'):
                    rr={v['model']:v for v in record['summaries'] if v['query']==q and v['group']==group}
                    differences.append(dict(dataset='ISIC',query=q,comparison=comp,group=group,from_BA=rr[a]['BA'],to_BA=rr[b]['BA'],delta_BA_pp=100*(rr[b]['BA']-rr[a]['BA']),delta_CE=rr[b]['CE']-rr[a]['CE']))
                for c in range(6):
                    rr={v['model']:v for v in record['class_rows'] if v['query']==q and v['class_index']==c}
                    pair=next(v for v in record['paired_rows'] if v['query']==q and v['class_index']==c and v['comparison']==comp)
                    delta=rr[b]['correct']-rr[a]['correct'];assert delta==pair['rescued']-pair['harmed']
                    change=100*(rr[b]['recall']-rr[a]['recall'])
                    class_differences.append(dict(dataset='ISIC',query=q,comparison=comp,class_index=c,group=rr[a]['group'],count=rr[a]['count'],from_correct=rr[a]['correct'],to_correct=rr[b]['correct'],delta_correct=delta,delta_recall_pp=change,overall_BA_contribution_pp=change/6,delta_CE=rr[b]['CE']-rr[a]['CE'],rescued=pair['rescued'],harmed=pair['harmed'],wrong_to_other_wrong=pair['wrong_to_other_wrong']))
            primary={v['class_index']:v for v in class_differences if v['query']==q and v['comparison']=='C_H__to__C_F'}
            hypotheses.append(dict(domain=q,class3_delta_correct=primary[3]['delta_correct'],class4_delta_correct=primary[4]['delta_correct'],class3_delta_recall_pp=primary[3]['delta_recall_pp'],class4_delta_recall_pp=primary[4]['delta_recall_pp'],class3_gain_and_class4_loss=primary[3]['delta_correct']>0 and primary[4]['delta_correct']<0))
        for q in ('Qfit','Qgate'):
            for a,b in lock['comparisons']:
                for c in range(6):
                    rr={v['query']:v for v in class_differences if v['comparison']==a+'__to__'+b and v['class_index']==c}
                    domain_changes.append(dict(query=q,comparison=a+'__to__'+b,class_index=c,support_delta_recall_pp=rr['S_full']['delta_recall_pp'],query_delta_recall_pp=rr[q]['delta_recall_pp'],query_minus_support_delta_pp=rr[q]['delta_recall_pp']-rr['S_full']['delta_recall_pp']))
        for name,key in [('CLASS_RESULTS.csv','class_rows'),('SUMMARY.csv','summaries'),('PAIRED_CLASS.csv','paired_rows'),('PAIRED_GROUP.csv','paired_group_rows'),('CONFUSION.csv','confusion_rows')]:table(name,record[key])
        margin_rows=[{**{k:v for k,v in row.items() if k not in ('margin','rank')},**{k+'_'+stat:v for k in ('margin','rank') for stat,v in row[k].items()}} for row in record['support_margin_rows']]
        table('SUPPORT_MARGINS.csv',margin_rows)
        for name,rows in [('COMPARE_READOUTS.csv',differences),('CLASS_COMPARE_READOUTS.csv',class_differences),('HYPOTHESIS.csv',hypotheses),('DOMAIN_CHANGES.csv',domain_changes)]:table(name,rows)
        for v in differences:
            if v['group']=='all':
                assert abs(v['delta_BA_pp']-sum(r['overall_BA_contribution_pp'] for r in class_differences if r['query']==v['query'] and r['comparison']==v['comparison']))<1e-10
    assert x['source_storage_bytes']+x['output_storage_bytes']<lock['storage_limit_bytes']
    save('RESULTS.json',dict(status='COMPLETE_DEVELOPMENT_AUDIT' if complete else 'INCOMPLETE_STOPPED',source_commit=program['source_commit'],program=program,record=record,published_query_rows_exactly_reused=query_match,full_CE_support_CE_reproduction_error=ce_error,hypotheses=hypotheses,all_processes_retired=True,retirement_checked_at=x['terminal']['checked_at'],logs=x['public_logs'],source_storage_bytes=x['source_storage_bytes'],output_storage_bytes=x['output_storage_bytes'],missing=[] if record else ['ISIC_audit.json'],model_selected=False))
    lines=['# ISIC支持—查询取舍审计结果','',f"状态：{'完成，父子进程已退出' if complete else '未完整完成，不自动重试'}。冻结源码 `{program['source_commit']}`。",'',
           'C_H=H线性CE，C_F=完整旧支持线性CE，R_F=完整旧支持ridge。只在5526条完整支持上评分三个封存头；Qfit/Qgate直接复用上轮已交付聚合，未打开查询特征。', '',
           '## 预先指定的class3/class4主假设','', '| 域 | class3净正确数变化 | class4净正确数变化 | class3召回差pp | class4召回差pp | class3增且class4减 |','|---|---:|---:|---:|---:|---|']
    for v in hypotheses:lines.append(f"| {v['domain']} | {v['class3_delta_correct']:+d} | {v['class4_delta_correct']:+d} | {v['class3_delta_recall_pp']:+.6f} | {v['class4_delta_recall_pp']:+.6f} | {v['class3_gain_and_class4_loss']} |")
    lines += ['','主比较为C_H→C_F。符号判定不是统计显著性，也不是新模型通过门槛；零变化不算严格支持。','',
              '## 三个头的全部类别','', '| 域 | 头 | 类别 | n | 正确数 | 召回% | CE | 错到旧/新组 |','|---|---|---:|---:|---:|---:|---:|---|']
    if record:
        for v in record['class_rows']:lines.append(f"| {v['query']} | {v['model']} | {v['class_index']} | {v['count']} | {v['correct']} | {100*v['recall']:.6f} | {v['CE']:.6f} | {v['errors_to_old']}/{v['errors_to_new']} |")
    lines += ['','## 两条比较的全部逐类正负与零变化','', '| 域 | 比较 | 类 | 正确数前→后 | Δ召回pp | 对原总体BA贡献pp | 救回/伤害/错转错 |','|---|---|---:|---|---:|---:|---|']
    for v in class_differences:lines.append(f"| {v['query']} | {v['comparison']} | {v['class_index']} | {v['from_correct']}→{v['to_correct']} | {v['delta_recall_pp']:+.6f} | {v['overall_BA_contribution_pp']:+.6f} | {v['rescued']}/{v['harmed']}/{v['wrong_to_other_wrong']} |")
    lines += ['','贡献始终使用原6类分母，不删除极小查询类。支持与查询不是同一批样本，不进行跨域逐样本配对；DOMAIN_CHANGES只是召回差的描述性比较。完整混淆含零单元格见CONFUSION，支持margin/rank只作辅助，不用其均值替代分类变化。','',
              '## 分组与成本','', '| 域 | 比较 | 组 | ΔBA pp | ΔCE |','|---|---|---|---:|---:|']
    for v in differences:lines.append(f"| {v['query']} | {v['comparison']} | {v['group']} | {v['delta_BA_pp']:+.6f} | {v['delta_CE']:+.6f} |")
    lines += ['',f"CPU驻留 {costs['CPU_only_residence_seconds']:.6f} 秒，user/system {costs['CPU_user_seconds']:.6f}/{costs['CPU_system_seconds']:.6f} 秒，GPU驻留 {costs['GPU_process_residence_seconds']:.6f} 秒；峰值worker RSS {costs['peak_worker_RSS_MiB']:.3f} MiB；supervisor wall {costs['supervisor_wall_seconds']} 秒，不重复加总。",'',
              f"累计CPU/GPU驻留 {costs['cumulative_CPU_residence_seconds']:.6f}/{costs['cumulative_GPU_residence_seconds']:.6f} 秒。0拟合/求解/梯度/编码器或适配器更新，0新增头，0查询特征评分；支持5526条、三头评分16578次。历史成本不重复计费，客户端准备未单测记null。",'',
              f"收集时服务器源码 {x['source_storage_bytes']} B、本轮输出 {x['output_storage_bytes']} B，后续公开交付备份另计。输入缓存与头未复制。",'',
              f"查询公开聚合逐行复用核对：{query_match}；C_F支持宏CE与上轮拟合终点CE误差：{ce_error}。角色守卫、计数、有限性、零适配器、配对守恒及小张量检查见RESULTS。失败：{program['failures']}；未运行：{program.get('not_run',[])}。",'',
              '## 解释与停止边界','',
              '完整S是共同评分域，不是三个头共同的训练集：C_H只训练H＋新S，C_F/R_F训练完整旧S＋新S；三个头均训练过当前新S。支持良好不等于独立泛化证据，支持错误不等于表示或所有线性头不可能。', '',
              '没有调参、重拟合、查询重评分、E/test访问或后续实验。全部类别保留，患者隔离未知，查询仍是历史开发数据。公开仅聚合结果与源码，私有特征、头、身份及原始日志不公开。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(complete=complete,hypotheses=hypotheses,query_match=query_match,ce_error=ce_error,costs=costs)))


if __name__=='__main__':main()
