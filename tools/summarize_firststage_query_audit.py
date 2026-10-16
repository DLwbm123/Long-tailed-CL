"""Report frozen query audit records without importing or executing training code."""
import json
from pathlib import Path


def main(root):
    root=Path(root)
    def save(name,obj): (root/(name+'.json')).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    data=json.loads((root/'CPU_RESULTS.json').read_text());program=data['program'];results=data['results']
    lock=json.loads((root/'DESIGN_LOCK.json').read_text());by={r['id']:r for r in results}
    complete=by.get('query_arithmetic',{}).get('status')=='PASS' and all(by.get(c['id'],{}).get('status')=='COMPLETED' for c in lock['cases'][1:])
    tables={name:[dict(case=r['id'],**v) for r in results for v in r.get(key,[])] for name,key in [('GRADIENTS','gradients'),('RESPONSES','responses'),('LOSSES','losses'),('AGGREGATES','aggregates'),('CONSTRUCTION','construction')]}
    summaries=[];coverage=[];reproduction=[];resources=[];derived=[]
    previous=json.loads((root.parent/'fullslot_retention_memory_design1_20261016'/'TEACHER_COVERAGE.json').read_text())
    for r in results:
        if r.get('status')!='COMPLETED': continue
        cid=r['id'];g=r['gradients'];resp=r['responses'];n=r['old_classes']
        assert len(g)==n*9 and len(resp)==n*6 and len(r['losses'])==n*9
        assert r['total_adapter_updates']==48 and r['actual_memory_updates']==96 and r['task2_updates']==0
        assert r['teacher_and_memories_unchanged'] and not r['GPU_initialized'] and not r['val_accessed'] and not r['test_accessed']
        for row in resp:
            for key in ('seen','reserved'):
                dist=row[key];assert len(dist['values'])==row['count']==len(dist['weights'])
                assert abs(sum(dist['weights'])-1)<1e-10
                assert all(a<=b for a,b in zip(dist['values'],dist['values'][1:]))
        for query in ('old','expanded','innovation'):
            paired={m:{a['class_index']:a for a in g if a['mode']==m and a['query']==query} for m in ('moment','decision')}
            changes=[paired['decision'][c]['absolute_error']-paired['moment'][c]['absolute_error'] for c in range(n)]
            summaries.append(dict(case=cid,query=query,classes=n,decision_better_classes=sum(a< -1e-12 for a in changes),decision_worse_classes=sum(a>1e-12 for a in changes),tied_classes=sum(abs(a)<=1e-12 for a in changes),
                **{m+'_mean_absolute_error':sum(paired[m][c]['absolute_error'] for c in range(n))/n for m in paired},
                **{m+'_mean_relative_error':sum(paired[m][c]['relative_error'] for c in range(n))/n if all(paired[m][c]['relative_error'] is not None for c in range(n)) else None for m in paired}))
        for query in ('old','expanded'):
            for mode in ('full_support','moment','decision'):
                rows=[v for v in resp if v['query']==query and v['mode']==mode]
                coverage.append(dict(case=cid,query=query,mode=mode,classes=n,
                    correct_mass=sum(v['teacher_correct_mass'] for v in rows)/n,
                    near_zero_mass=sum(v['seen']['near_zero_mass'] for v in rows)/n,
                    mean_seen_W1=sum(v['seen_W1'] for v in rows)/n,
                    mean_reserved_W1=sum(v['reserved_W1'] for v in rows)/n,
                    mean_sliced_W1=sum(v['sliced_W1'] for v in rows)/n))
                if query=='old' and mode!='full_support':
                    expected={v['class_index']:v['weighted_old_correct'] for v in previous if v['case']==cid and v['task']==2 and v['mode']=='old_'+mode}
                    err=max(abs(v['teacher_correct_mass']-expected[v['class_index']]) for v in rows)
                    reproduction.append(dict(case=cid,mode=mode,max_class_coverage_error=err,pass_tolerance_1e_minus10=err<1e-10))
        for query in ('old','expanded'):
            selected={m:{v['class_index']:v for v in resp if v['query']==query and v['mode']==m} for m in ('full_support','moment','decision')}
            derived.append(dict(case=cid,query=query,classes=n,
                mean_class_absolute_coverage_error={m:sum(abs(selected[m][c]['teacher_correct_mass']-selected['full_support'][c]['teacher_correct_mass']) for c in range(n))/n for m in ('moment','decision')},
                decision_seen_W1_better_classes=sum(selected['decision'][c]['seen_W1']<selected['moment'][c]['seen_W1'] for c in range(n))))
        resources.append(dict(case=cid,adapter_updates=r['total_adapter_updates'],memory_updates=r['actual_memory_updates'],task2_updates=0,teacher_train_seconds=r['training']['seconds'],memory_construction_seconds=sum(v['seconds'] for v in r['construction']),diagnostic_seconds=r['diagnostic_seconds'],pipeline_seconds=r['CPU_pipeline_seconds']))
    for name,rows in tables.items():save(name,rows)
    save('QUERY_SUMMARY',summaries);save('RESPONSE_SUMMARY',coverage);save('RECONSTRUCTION_CHECK',reproduction);save('RESOURCE_SUMMARY',resources)
    save('DERIVED_RESPONSE_CHECK',dict(posthoc_descriptive_from_frozen_records=True,rows=derived))
    total=sum(c['CPU_process_residence_seconds'] for c in program['costs'])
    audit=dict(status='COMPLETE_FIXED_STATE_AUDIT' if complete else 'INCOMPLETE',science_source_commit=program['source_commit'],failures=program['failures'],costs=program['costs'],new_CPU_residence_seconds=total,prior_CPU_residence_seconds=lock['prior_CPU_residence_seconds'],cumulative_CPU_residence_seconds=lock['prior_CPU_residence_seconds']+total,parent_wall_seconds_separate=program.get('supervisor_wall_seconds'),CPU_user_seconds=sum(c['CPU_user_seconds'] for c in program['costs']),CPU_system_seconds=sum(c['CPU_system_seconds'] for c in program['costs']),peak_RSS_MiB=max((c['peak_RSS_MiB'] for c in program['costs']),default=None),client_collector_seconds=None,new_GPU_seconds=0,historical_GPU_seconds=lock['historical_GPU_seconds'],internal_costs_included_not_added=True,adapter_updates=sum(v['adapter_updates'] for v in resources),memory_updates=sum(v['memory_updates'] for v in resources),task2_updates=0,reconstruction_coverage_check=complete and all(v['pass_tolerance_1e_minus10'] for v in reproduction),classification_efficacy=None,medical_effect=None,auto_next_round=False)
    if complete:
        assert len(tables['GRADIENTS'])==171 and len(tables['RESPONSES'])==114 and len(tables['LOSSES'])==171
        assert audit['adapter_updates']==144 and audit['memory_updates']==288 and len(reproduction)==6
    save('AUDIT',audit);save('REPORT_CHECK',dict(complete=complete,table_counts={k:len(v) for k,v in tables.items()},distribution_mass_and_order_checked=True,updates_checked=complete,reconstruction_coverage_check=audit['reconstruction_coverage_check'],missing_filled_with_zero=False))
    def fmt(v):return 'null' if v is None else f'{v:.6f}'
    lines=['# 第一阶段固定状态查询审计','',f"状态：{audit['status']}；覆盖复现：{audit['reconstruction_coverage_check']}。这里只审计固定教师的训练反应，不是新分类结果。",'',
        '原第一阶段教师重建一次，两种记忆各构造一次；原参数、种子、优化器和48步不变。没有第二阶段更新，没有新类别样本或医学val/test。', '',
        '## 逐类梯度误差','', '相对误差=误差L2/完整支持梯度L2，再对类别等权平均；不是旧版压缩目标的半平方相对误差。绝对误差同样保留，跨查询不能仅凭相对误差大小认定迁移失败。innovation=expanded−old。','',
        '|规模|查询|moment相对误差|decision相对误差|moment绝对误差|decision绝对误差|decision较好/较差/相同类数|','|---|---|---:|---:|---:|---:|---|']
    for s in summaries:
        lines.append('|'+ '|'.join([s['case'],s['query'],fmt(s['moment_mean_relative_error']),fmt(s['decision_mean_relative_error']),fmt(s['moment_mean_absolute_error']),fmt(s['decision_mean_absolute_error']),f"{s['decision_better_classes']}/{s['decision_worse_classes']}/{s['tied_classes']}"])+'|')
    lines+=['','GRADIENTS含每类U/V分块、范数、绝对/相对误差、余弦；AGGREGATES同时含宏梯度误差及抵消比。不能只挑宏梯度误差作为主结果。','', '## 响应与覆盖','', '|规模|查询|表示|教师正确质量%|近零质量%|已见间隔W1|预留间隔W1|','|---|---|---|---:|---:|---:|---:|']
    for v in coverage:lines.append('|'+ '|'.join([v['case'],v['query'],v['mode'],fmt(100*v['correct_mass']),fmt(100*v['near_zero_mass']),fmt(v['mean_seen_W1']),fmt(v['mean_reserved_W1'])])+'|')
    lines+=['','RESPONSES发布全部逐类加权经验原子与固定分位数；不是只报告有利类别或图。覆盖目标应与完整合成支持对照，不以接近100%为自动成功。','', '## 新增运行费用','', '|进程|驻留秒|user秒|system秒|peakRSS MiB|返回码|','|---|---:|---:|---:|---:|---:|']
    for c in program['costs']:lines.append('|'+ '|'.join([c['id'],fmt(c['CPU_process_residence_seconds']),fmt(c['CPU_user_seconds']),fmt(c['CPU_system_seconds']),fmt(c['peak_RSS_MiB']),str(c['returncode'])])+'|')
    lines += ['',f"新增CPU驻留{total:.9f}秒；累计{audit['cumulative_CPU_residence_seconds']:.9f}秒。父wall {audit['parent_wall_seconds_separate']}秒单列；内部重建和诊断包含在驻留中，不相加。新增GPU0，历史GPU沿用{audit['historical_GPU_seconds']}秒。客户端编排未测null。",'',f"实际adapter更新{audit['adapter_updates']}、memory更新{audit['memory_updates']}；第二阶段更新0。失败：{json.dumps(program['failures'],ensure_ascii=False)}。",'', '## 边界','', '单个固定种子每规模；梯度接近不保证优化轨迹或分类表现。完整支持仅合成，非真实医学特征。保留私有重建状态，不发布模型张量/私有路径/原始日志；未记录历史Adam位移，不作梯度冲突或全局最优声明。完成分析后停止，不自动下一轮。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps(audit))


if __name__=='__main__':
    import sys
    main(sys.argv[1])
