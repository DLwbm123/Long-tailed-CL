"""Report frozen query audit records without importing or executing training code."""
import json
from pathlib import Path


def main(root):
    root=Path(root)
    def save(name,obj): (root/(name+'.json')).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    data=json.loads((root/'CPU_RESULTS.json').read_text());program=data['program'];results=data['results']
    lock=json.loads((root/'DESIGN_LOCK.json').read_text());by={r['id']:r for r in results}
    complete=by.get('dualquery_arithmetic',{}).get('status')=='PASS' and all(by.get(c['id'],{}).get('status')=='COMPLETED' for c in lock['cases'][1:])
    tables={name:[dict(case=r['id'],**v) for r in results for v in r.get(key,[])] for name,key in [('GRADIENTS','gradients'),('RESPONSES','responses'),('LOSSES','losses'),('AGGREGATES','aggregates'),('CONSTRUCTION','construction')]}
    summaries=[];coverage=[];reproduction=[];resources=[];derived=[]
    previous=json.loads((root.parent/'fullslot_retention_memory_design1_20261016'/'TEACHER_COVERAGE.json').read_text())
    for r in results:
        if r.get('status')!='COMPLETED': continue
        cid=r['id'];g=r['gradients'];resp=r['responses'];n=r['old_classes']
        assert len(g)==n*9 and len(resp)==n*6 and len(r['losses'])==n*9
        assert r['total_adapter_updates']==0 and r['actual_memory_updates']==48 and r['task2_updates']==0
        assert r['teacher_and_memories_unchanged'] and not r['GPU_initialized'] and not r['val_accessed'] and not r['test_accessed']
        for row in resp:
            for key in ('seen','reserved'):
                dist=row[key];assert len(dist['values'])==row['count']==len(dist['weights'])
                assert abs(sum(dist['weights'])-1)<1e-10
                assert all(a<=b for a,b in zip(dist['values'],dist['values'][1:]))
        for query in ('old','expanded','innovation'):
            paired={m:{a['class_index']:a for a in g if a['mode']==m and a['query']==query} for m in ('moment','candidate')}
            changes=[paired['candidate'][c]['absolute_error']-paired['moment'][c]['absolute_error'] for c in range(n)]
            summaries.append(dict(case=cid,query=query,classes=n,candidate_better_classes=sum(a< -1e-12 for a in changes),candidate_worse_classes=sum(a>1e-12 for a in changes),tied_classes=sum(abs(a)<=1e-12 for a in changes),
                **{m+'_mean_absolute_error':sum(paired[m][c]['absolute_error'] for c in range(n))/n for m in paired},
                **{m+'_mean_relative_error':sum(paired[m][c]['relative_error'] for c in range(n))/n if all(paired[m][c]['relative_error'] is not None for c in range(n)) else None for m in paired}))
        for query in ('old','expanded'):
            for mode in ('full_support','moment','candidate'):
                rows=[v for v in resp if v['query']==query and v['mode']==mode]
                coverage.append(dict(case=cid,query=query,mode=mode,classes=n,
                    correct_mass=sum(v['teacher_correct_mass'] for v in rows)/n,
                    near_zero_mass=sum(v['seen']['near_zero_mass'] for v in rows)/n,
                    mean_seen_W1=sum(v['seen_W1'] for v in rows)/n,
                    mean_reserved_W1=sum(v['reserved_W1'] for v in rows)/n,
                    mean_sliced_W1=sum(v['sliced_W1'] for v in rows)/n))
                if query=='old' and mode=='moment':
                    expected={v['class_index']:v['weighted_old_correct'] for v in previous if v['case']==cid and v['task']==2 and v['mode']=='old_'+mode}
                    err=max(abs(v['teacher_correct_mass']-expected[v['class_index']]) for v in rows)
                    reproduction.append(dict(case=cid,mode=mode,max_class_coverage_error=err,pass_tolerance_1e_minus10=err<1e-10))
        for query in ('old','expanded'):
            selected={m:{v['class_index']:v for v in resp if v['query']==query and v['mode']==m} for m in ('full_support','moment','candidate')}
            derived.append(dict(case=cid,query=query,classes=n,
                mean_class_absolute_coverage_error={m:sum(abs(selected[m][c]['teacher_correct_mass']-selected['full_support'][c]['teacher_correct_mass']) for c in range(n))/n for m in ('moment','candidate')},
                candidate_seen_W1_better_classes=sum(selected['candidate'][c]['seen_W1']<selected['moment'][c]['seen_W1'] for c in range(n))))
        resources.append(dict(case=cid,adapter_updates=r['total_adapter_updates'],memory_updates=r['actual_memory_updates'],task2_updates=0,teacher_train_seconds=r['training']['seconds'],load_seconds=r['load_seconds'],memory_construction_seconds=sum(v['seconds'] for v in r['construction']),diagnostic_seconds=r['diagnostic_seconds'],pipeline_seconds=r['CPU_pipeline_seconds']))
    for name,rows in tables.items():save(name,rows)
    save('QUERY_SUMMARY',summaries);save('RESPONSE_SUMMARY',coverage);save('REUSE_CHECK',reproduction);save('RESOURCE_SUMMARY',resources)
    save('DERIVED_RESPONSE_CHECK',dict(posthoc_descriptive_from_frozen_records=True,rows=derived))
    total=sum(c['CPU_process_residence_seconds'] for c in program['costs'])
    audit=dict(status='COMPLETE_FIXED_STATE_AUDIT' if complete else 'INCOMPLETE',science_source_commit=program['source_commit'],failures=program['failures'],costs=program['costs'],new_CPU_residence_seconds=total,prior_CPU_residence_seconds=lock['prior_CPU_residence_seconds'],cumulative_CPU_residence_seconds=lock['prior_CPU_residence_seconds']+total,parent_wall_seconds_separate=program.get('supervisor_wall_seconds'),CPU_user_seconds=sum(c['CPU_user_seconds'] for c in program['costs']),CPU_system_seconds=sum(c['CPU_system_seconds'] for c in program['costs']),peak_RSS_MiB=max((c['peak_RSS_MiB'] for c in program['costs']),default=None),client_collector_seconds=None,new_GPU_seconds=0,historical_GPU_seconds=lock['historical_GPU_seconds'],internal_costs_included_not_added=True,adapter_updates=sum(v['adapter_updates'] for v in resources),memory_updates=sum(v['memory_updates'] for v in resources),task2_updates=0,baseline_coverage_check=complete and all(v['pass_tolerance_1e_minus10'] for v in reproduction),classification_efficacy=None,medical_effect=None,auto_next_round=False)
    if complete:
        assert len(tables['GRADIENTS'])==171 and len(tables['RESPONSES'])==114 and len(tables['LOSSES'])==171
        assert audit['adapter_updates']==0 and audit['memory_updates']==144 and len(reproduction)==3
    save('AUDIT',audit);save('REPORT_CHECK',dict(complete=complete,table_counts={k:len(v) for k,v in tables.items()},distribution_mass_and_order_checked=True,updates_checked=complete,baseline_coverage_check=audit['baseline_coverage_check'],missing_filled_with_zero=False))
    def fmt(v):return 'null' if v is None else f'{v:.6f}'
    prior=root.parent/'firststage_query_audit_design1_20261016'
    prior_g=json.loads((prior/'GRADIENTS.json').read_text());prior_r=json.loads((prior/'RESPONSES.json').read_text())
    pg={(v['case'],v['class_index'],v['mode'],v['query']):v for v in prior_g}
    pr={(v['case'],v['class_index'],v['mode'],v['query']):v for v in prior_r}
    nowg={(v['case'],v['class_index'],v['mode'],v['query']):v for v in tables['GRADIENTS']}
    nowr={(v['case'],v['class_index'],v['mode'],v['query']):v for v in tables['RESPONSES']}
    pairs=[];response_deltas=[];reuse=[];gates=[];final_objectives=[]
    for r in results:
        if r.get('status')!='COMPLETED':continue
        cid=r['id'];n=r['old_classes'];max_diff=0.
        scales={v['class_index']:v['scale'] for v in r['construction'][0]['target_gradient_norms']}
        for c in range(n):
            for query in ('old','expanded','innovation'):
                for mode in ('full_support','moment'):
                    a=nowg[cid,c,mode,query];b=pg[cid,c,mode,query]
                    max_diff=max(max_diff,abs(a['absolute_error']-b['absolute_error']),abs(a['norm']-b['norm']))
                can=nowg[cid,c,'candidate',query]
                for base in ('moment','previous_decision'):
                    ref=nowg[cid,c,'moment',query] if base=='moment' else pg[cid,c,'decision',query]
                    pairs.append(dict(case=cid,class_index=c,query=query,baseline=base,candidate_error=can['absolute_error'],baseline_error=ref['absolute_error'],error_difference=can['absolute_error']-ref['absolute_error']))
            for query in ('old','expanded'):
                can=nowr[cid,c,'candidate',query];truth=nowr[cid,c,'full_support',query]
                for base in ('moment','previous_decision'):
                    ref=nowr[cid,c,'moment',query] if base=='moment' else pr[cid,c,'decision',query]
                    response_deltas.append(dict(case=cid,class_index=c,query=query,baseline=base,
                        seen_W1_difference=can['seen_W1']-ref['seen_W1'],reserved_W1_difference=can['reserved_W1']-ref['reserved_W1'],sliced_W1_difference=can['sliced_W1']-ref['sliced_W1'],
                        coverage_absolute_error_difference=abs(can['teacher_correct_mass']-truth['teacher_correct_mass'])-abs(ref['teacher_correct_mass']-truth['teacher_correct_mass']),
                        near_zero_absolute_error_difference=abs(can['seen']['near_zero_mass']-truth['seen']['near_zero_mass'])-abs(ref['seen']['near_zero_mass']-truth['seen']['near_zero_mass'])))
            for mode in ('moment','candidate'):
                a=nowg[cid,c,mode,'old']['absolute_error']**2;b=nowg[cid,c,mode,'innovation']['absolute_error']**2
                final_objectives.append(dict(case=cid,class_index=c,mode=mode,old_error_squared=a,innovation_error_squared=b,scale=scales[c],final_objective=max(a,b)/scales[c]))
        reuse.append(dict(case=cid,gradient_norm_error_max_abs_difference=max_diff,pass_tolerance_1e_minus10=max_diff<1e-10))
        rows=[v for v in pairs if v['case']==cid and v['baseline']=='moment']
        gates.append(dict(case=cid,no_class_query_harm=all(v['error_difference']<=1e-12 for v in rows),strict_some_improvement=any(v['error_difference']< -1e-12 for v in rows),better=sum(v['error_difference']< -1e-12 for v in rows),worse=sum(v['error_difference']>1e-12 for v in rows),tied=sum(abs(v['error_difference'])<=1e-12 for v in rows)))
    branches=[]
    for r in results:
        if r.get('status')!='COMPLETED':continue
        rows=r['construction'][0]['dominance']
        assert all(v['old']+v['innovation']+v['tied']==48 for v in rows)
        branches.append(dict(case=r['id'],class_steps=r['old_classes']*48,old=sum(v['old'] for v in rows),innovation=sum(v['innovation'] for v in rows),tied=sum(v['tied'] for v in rows)))
    save('BRANCH_SUMMARY',branches)
    save('GRADIENT_PAIRS',pairs);save('RESPONSE_DEGRADATION',response_deltas);save('BASELINE_GRADIENT_REUSE',reuse);save('MECHANISM_FLAGS',gates);save('FINAL_OBJECTIVES',final_objectives)
    audit.update(baseline_gradient_check=complete and all(v['pass_tolerance_1e_minus10'] for v in reuse),descriptive_all_class_query_no_harm=complete and all(v['no_class_query_harm'] and v['strict_some_improvement'] for v in gates))
    save('AUDIT',audit)
    lines=['# 单候选双查询记忆验证','',f"状态：{audit['status']}；复用覆盖/梯度检查：{audit['baseline_coverage_check']}/{audit['baseline_gradient_check']}；描述性逐类三查询无损改善：{audit['descriptive_all_class_query_no_harm']}。",'',
        '第一阶段共同教师和moment直接复用保存状态。仅构造candidate，原初始化与16节点/48步Adam不变；训练前快照及响应W1移出目标，固定教师上拟合旧梯度与新增竞争梯度的最大平方残差，以共同完整支持尺度归一。没有教师或第二阶段更新。','',
        '## 梯度误差','', '下表是逐类相对L2误差均值，不是训练的最大平方残差目标。所有类的绝对误差/余弦/U-V分块见GRADIENTS；宏梯度及抵消比见AGGREGATES。','',
        '|规模|查询|moment相对误差|candidate相对误差|moment绝对误差|candidate绝对误差|candidate较好/较差/相同类数|','|---|---|---:|---:|---:|---:|---|']
    for s in summaries:lines.append('|'+ '|'.join([s['case'],s['query'],fmt(s['moment_mean_relative_error']),fmt(s['candidate_mean_relative_error']),fmt(s['moment_mean_absolute_error']),fmt(s['candidate_mean_absolute_error']),f"{s['candidate_better_classes']}/{s['candidate_worse_classes']}/{s['tied_classes']}"])+'|')
    lines+=['','GRADIENT_PAIRS同时保留相对moment与上轮原decision的逐类变化。原decision从已发表记录引用，未重建或重跑。MECHANISM_FLAGS只是冻结的描述性无损标记，不是分类收益或医学有效性。','', '## 响应与覆盖','', '|规模|查询|表示|教师正确质量%|近零质量%|已见间隔W1|预留间隔W1|','|---|---|---|---:|---:|---:|---:|']
    for v in coverage:lines.append('|'+ '|'.join([v['case'],v['query'],v['mode'],fmt(100*v['correct_mass']),fmt(100*v['near_zero_mass']),fmt(v['mean_seen_W1']),fmt(v['mean_reserved_W1'])])+'|')
    lines+=['','RESPONSES包含每类全部加权经验原子/固定分位数。RESPONSE_DEGRADATION中正差表示相对对照退化，负差表示改善，包含两种查询、两种对照及覆盖/近零质量绝对误差。覆盖率高不自动等于保真。','', '## 费用','', '|进程|驻留秒|user秒|system秒|peakRSS MiB|返回码|','|---|---:|---:|---:|---:|---:|']
    for c in program['costs']:lines.append('|'+ '|'.join([c['id'],fmt(c['CPU_process_residence_seconds']),fmt(c['CPU_user_seconds']),fmt(c['CPU_system_seconds']),fmt(c['peak_RSS_MiB']),str(c['returncode'])])+'|')
    lines+=['',f"新增CPU驻留{total:.9f}秒，累计{audit['cumulative_CPU_residence_seconds']:.9f}秒；父wall{audit['parent_wall_seconds_separate']}秒另列。内部加载/构造/诊断已含驻留，不相加。新增GPU0，客户端未单测null。",'',f"教师更新{audit['adapter_updates']}，candidate memory更新{audit['memory_updates']}，第二阶段更新0。失败：{json.dumps(program['failures'],ensure_ascii=False)}。",'', '单种子每规模、固定合成训练支持上的梯度诊断，没有独立分类评估，不保证未来教师变化下有效。没有节点几何、历史Adam方向或最优性结论。完整保留负结果，终点固定48步；不调参、不重试、不自动下一轮。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps(audit))


if __name__=='__main__':
    import sys
    main(sys.argv[1])
