"""Report frozen endpoint records; standard library only, no model or optimizer calls."""
import json
from collections import Counter
from pathlib import Path


def main(root):
    root=Path(root)
    def read(p):return json.loads(p.read_text())
    def save(name,obj):(root/(name+'.json')).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    data=read(root/'CPU_RESULTS.json');program=data['program'];results=data['results'];lock=read(root/'DESIGN_LOCK.json')
    scales=[r for r in results if r.get('status')=='COMPLETED']
    complete=any(r['id']=='weight_arithmetic' and r.get('status')=='PASS' for r in results) and len(scales)==3 and not program['failures']
    tables={name:[dict(case=r['id'],**v) for r in scales for v in r[key]] for name,key in [('GRADIENTS','gradients'),('RESPONSES','responses'),('LOSSES','losses'),('AGGREGATES','aggregates'),('CONSTRUCTION','construction'),('ACCEPTANCE','acceptance')]}
    prior=root.parent/'firststage_query_audit_design1_20261016'
    priorg={(v['case'],v['class_index'],v['mode'],v['query']):v for v in read(prior/'GRADIENTS.json')}
    priorr={(v['case'],v['class_index'],v['mode'],v['query']):v for v in read(prior/'RESPONSES.json')}
    oldcost={v['case']:v for v in read(prior/'CONSTRUCTION.json') if v['mode']=='moment'}
    teacher={v['case']:v for v in read(prior/'RESOURCE_SUMMARY.json')}
    summaries=[];response_summary=[];pairs=[];degradation=[];reuse=[];resources=[];gates=[];objectives=[]
    for r in scales:
        cid=r['id'];n=r['old_classes'];g={(v['class_index'],v['mode'],v['query']):v for v in r['gradients']};rr={(v['class_index'],v['mode'],v['query']):v for v in r['responses']}
        assert len(g)==n*12 and len(rr)==n*8 and len(r['losses'])==n*12
        assert r['total_adapter_updates']==r['task2_updates']==0 and r['actual_memory_updates']==48
        assert r['teacher_and_memories_unchanged'] and r['fixed_nodes_unchanged'] and not r['GPU_initialized'] and not r['val_accessed'] and not r['test_accessed']
        for row in r['responses']:
            for key in ('seen','reserved'):
                dist=row[key];assert len(dist['values'])==row['count']==len(dist['weights'])
                assert abs(sum(dist['weights'])-1)<1e-10 and min(dist['weights'])>=0
                assert all(a<=b for a,b in zip(dist['values'],dist['values'][1:]))
        maxdiff=0.;reasons=Counter();accepted=[];gradient_only=[]
        for c in range(n):
            checks={};strict=False
            for query in ('old','innovation','expanded'):
                base=g[c,'moment',query];can=g[c,'candidate',query];chosen=g[c,'selected',query]
                delta=can['absolute_error']-base['absolute_error'];strict=strict or delta< -1e-12
                checks['gradient_'+query]=delta<=1e-12
                for mode in ('candidate','selected'):
                    row=g[c,mode,query]
                    pairs.append(dict(case=cid,class_index=c,mode=mode,query=query,error_difference=row['absolute_error']-base['absolute_error'],baseline_error=base['absolute_error'],absolute_error=row['absolute_error']))
                for mode in ('full_support','moment'):
                    v=g[c,mode,query];old=priorg[cid,c,mode,query]
                    maxdiff=max(maxdiff,abs(v['norm']-old['norm']),abs(v['absolute_error']-old['absolute_error']))
            gradient_only.append(all(checks.values()) and strict)
            for query in ('old','expanded'):
                base=rr[c,'moment',query];truth=rr[c,'full_support',query]
                for mode in ('candidate','selected'):
                    can=rr[c,mode,query]
                    delta={k:can[k]-base[k] for k in ('seen_W1','reserved_W1','sliced_W1')}
                    delta['coverage_error']=abs(can['teacher_correct_mass']-truth['teacher_correct_mass'])-abs(base['teacher_correct_mass']-truth['teacher_correct_mass'])
                    delta['near_zero_error']=abs(can['seen']['near_zero_mass']-truth['seen']['near_zero_mass'])-abs(base['seen']['near_zero_mass']-truth['seen']['near_zero_mass'])
                    degradation.append(dict(case=cid,class_index=c,query=query,mode=mode,**delta))
                    if mode=='candidate':checks.update({query+'_'+k:v<=1e-12 for k,v in delta.items()})
                for mode in ('full_support','moment'):
                    maxdiff=max(maxdiff,abs(rr[c,mode,query]['teacher_correct_mass']-priorr[cid,c,mode,query]['teacher_correct_mass']))
                for key in ('seen','reserved'):
                    assert rr[c,'candidate',query][key]['values']==base[key]['values'], 'Fixed-node response locations moved'
            assert len(checks)==13
            accept=all(checks.values()) and strict;record=r['acceptance'][c]
            fail=[k for k,v in checks.items() if not v]+([] if strict else ['no_strict_gradient_improvement'])
            assert record['class_index']==c and record['accepted']==accept and record['checks']==checks and record['failed_conditions']==fail
            source='candidate' if accept else 'moment';assert record['selected_source']==source
            for rows in (r['gradients'],r['responses'],r['losses']):
                for selected in [v for v in rows if v['class_index']==c and v['mode']=='selected']:
                    origin=next(v for v in rows if v['class_index']==c and v['mode']==source and v['query']==selected['query'])
                    assert {k:v for k,v in selected.items() if k not in ('mode','selected_source')}=={k:v for k,v in origin.items() if k!='mode'}
            accepted.append(accept);reasons.update(fail)
            baseline=r['construction'][0]['baseline_errors_squared'][c]['values']
            for mode in ('moment','candidate','selected'):
                errors=[g[c,mode,q]['absolute_error']**2 for q in ('old','innovation','expanded')]
                objectives.append(dict(case=cid,class_index=c,mode=mode,squared_errors=errors,baseline_squared_errors=baseline,objective=sum(e/max(b,1e-12) for e,b in zip(errors,baseline))/3))
        reuse.append(dict(case=cid,max_baseline_gradient_and_coverage_difference=maxdiff,pass_tolerance_1e_minus10=maxdiff<1e-10))
        gates.append(dict(case=cid,classes=n,accepted=sum(accepted),fallback=n-sum(accepted),accepted_classes=[c for c,a in enumerate(accepted) if a],gradient_only_eligible=sum(gradient_only),failure_counts=dict(reasons),useful_correction_found=any(accepted)))
        for query in ('old','innovation','expanded'):
            row=dict(case=cid,query=query,classes=n)
            for mode in ('moment','candidate','selected'):
                entries=[g[c,mode,query] for c in range(n)]
                row[mode+'_mean_relative_error']=sum(v['relative_error'] for v in entries)/n if all(v['relative_error'] is not None for v in entries) else None
                row[mode+'_mean_absolute_error']=sum(v['absolute_error'] for v in entries)/n
                if mode!='moment':
                    delta=[g[c,mode,query]['absolute_error']-g[c,'moment',query]['absolute_error'] for c in range(n)]
                    for label,predicate in [('better',lambda x:x< -1e-12),('worse',lambda x:x>1e-12),('tied',lambda x:abs(x)<=1e-12)]:row[mode+'_'+label]=sum(predicate(x) for x in delta)
            summaries.append(row)
        for query in ('old','expanded'):
            for mode in ('full_support','moment','candidate','selected'):
                rows=[rr[c,mode,query] for c in range(n)]
                response_summary.append(dict(case=cid,query=query,mode=mode,correct_mass=sum(v['teacher_correct_mass'] for v in rows)/n,near_zero_mass=sum(v['seen']['near_zero_mass'] for v in rows)/n,**{k:sum(v[k] for v in rows)/n for k in ('seen_W1','reserved_W1','sliced_W1')}))
        resources.append(dict(case=cid,new_teacher_updates=0,new_weight_updates=48,new_refinement_seconds=r['construction'][0]['seconds'],load_seconds=r['load_seconds'],diagnostic_seconds=r['diagnostic_seconds'],pipeline_seconds=r['CPU_pipeline_seconds'],historical_moment_seconds=oldcost[cid]['seconds'],historical_moment_updates=48,historical_teacher_seconds=teacher[cid]['teacher_train_seconds'],historical_teacher_updates=48,from_scratch_memory_updates=96,historical_subcosts_already_in_prior_total=True))
    for name,rows in tables.items():save(name,rows)
    for name,rows in [('QUERY_SUMMARY',summaries),('RESPONSE_SUMMARY',response_summary),('GRADIENT_PAIRS',pairs),('RESPONSE_DEGRADATION',degradation),('BASELINE_REUSE',reuse),('RESOURCE_SUMMARY',resources),('ACCEPTANCE_SUMMARY',gates),('FINAL_OBJECTIVES',objectives)]:save(name,rows)
    total=sum(c['CPU_process_residence_seconds'] for c in program['costs'])
    audit=dict(status='COMPLETE_WEIGHT_REFINEMENT_AUDIT' if complete else 'INCOMPLETE',science_source_commit=program['source_commit'],failures=program['failures'],costs=program['costs'],new_CPU_residence_seconds=total,prior_CPU_residence_seconds=lock['prior_CPU_residence_seconds'],cumulative_CPU_residence_seconds=lock['prior_CPU_residence_seconds']+total,parent_wall_seconds_separate=program.get('supervisor_wall_seconds'),CPU_user_seconds=sum(c['CPU_user_seconds'] for c in program['costs']),CPU_system_seconds=sum(c['CPU_system_seconds'] for c in program['costs']),peak_RSS_MiB=max((c['peak_RSS_MiB'] for c in program['costs']),default=None),client_collector_seconds=None,new_GPU_seconds=0,historical_GPU_seconds=lock['historical_GPU_seconds'],internal_costs_included_not_added=True,adapter_updates=0,memory_updates=sum(v['new_weight_updates'] for v in resources),task2_updates=0,baseline_reuse_check=complete and all(v['pass_tolerance_1e_minus10'] for v in reuse),endpoint_acceptance_independently_checked=True,accepted_classes=sum(v['accepted'] for v in gates),fallback_classes=sum(v['fallback'] for v in gates),useful_correction_each_scale=complete and all(v['useful_correction_found'] for v in gates),classification_efficacy=None,medical_effect=None,auto_next_round=False)
    if complete:
        assert len(tables['GRADIENTS'])==228 and len(tables['RESPONSES'])==152 and len(tables['LOSSES'])==228
        assert audit['memory_updates']==144 and len(tables['ACCEPTANCE'])==19 and len(reuse)==3
        assert all(v['pass_tolerance_1e_minus10'] for v in reuse)
        assert all(v['error_difference']<=1e-12 for v in pairs if v['mode']=='selected')
        assert all(all(v[k]<=1e-12 for k in ('seen_W1','reserved_W1','sliced_W1','coverage_error','near_zero_error')) for v in degradation if v['mode']=='selected')
    save('AUDIT',audit);save('REPORT_CHECK',dict(complete=complete,table_counts={k:len(v) for k,v in tables.items()},endpoint_rule_checked=True,selected_records_match_chosen_source=True,distributions_checked=True,fixed_node_response_locations_checked=True,baseline_reuse_check=audit['baseline_reuse_check']))
    def fmt(v):return 'null' if v is None else f'{v:.6f}'
    lines=['# 固定moment节点权重修正结果','',f"状态：{audit['status']}；接纳{audit['accepted_classes']}/19，回退{audit['fallback_classes']}/19；每规模至少一个可用修正：{audit['useful_correction_each_scale']}。",'',
        '共同教师、moment节点直接复用保存状态；每规模48步Adam只修正权重。三查询平方误差分别按原moment误差归一后均值优化；只检查固定终点。原始候选与逐类接纳后的selected分开保留；回退不计为修正收益。','',
        '|规模|接纳/总类数|仅梯度条件合格类数|接纳类|','|---|---:|---:|---|']
    for v in gates:lines.append(f"|{v['case']}|{v['accepted']}/{v['classes']}|{v['gradient_only_eligible']}|{v['accepted_classes']}|")
    lines+=['','## 梯度','', '逐类相对L2误差均值；绝对值与U/V分块、余弦见GRADIENTS，宏梯度见AGGREGATES。','', '|规模|查询|moment|原始候选|接纳后|候选好/坏/平|接纳后好/坏/平|','|---|---|---:|---:|---:|---|---|']
    for v in summaries:lines.append('|'+ '|'.join([v['case'],v['query'],*[fmt(v[m+'_mean_relative_error']) for m in ('moment','candidate','selected')],*[f"{v[m+'_better']}/{v[m+'_worse']}/{v[m+'_tied']}" for m in ('candidate','selected')]])+'|')
    lines+=['','## 响应','', 'RESPONSES包含完整经验原子；RESPONSE_DEGRADATION逐类列出候选和接纳后相对moment的五项响应差，正数为退化。固定节点只保证原子位置不变，不保证加权分布保真。ACCEPTANCE保留每类13项检查及全部拒绝原因。','', '|规模|查询|表示|正确质量%|近零质量%|seen W1|reserved W1|sliced W1|','|---|---|---|---:|---:|---:|---:|---:|']
    for v in response_summary:lines.append('|'+ '|'.join([v['case'],v['query'],v['mode'],fmt(100*v['correct_mass']),fmt(100*v['near_zero_mass']),*[fmt(v[k]) for k in ('seen_W1','reserved_W1','sliced_W1')]])+'|')
    lines+=['','## 新增与历史费用','', '|规模|新增权重构造秒|历史moment秒|历史教师秒|新增/历史moment步数|从头memory总步数|','|---|---:|---:|---:|---|---:|']
    for v in resources:lines.append(f"|{v['case']}|{fmt(v['new_refinement_seconds'])}|{fmt(v['historical_moment_seconds'])}|{fmt(v['historical_teacher_seconds'])}|48/48|96|")
    lines+=['','以上历史成本来自第一阶段共同教师审计，已含历史累计，不再重复相加；该历史轮还构造过decision并执行诊断，所以其完整174.582071413秒不等于本表所列子成本。新增内部构造/加载/诊断已含本轮进程驻留，不能重复相加。','', '|进程|驻留秒|user秒|system秒|峰值MiB|退出码|','|---|---:|---:|---:|---:|---:|']
    for c in program['costs']:lines.append('|'+ '|'.join([c['id'],*[fmt(c[k]) for k in ('CPU_process_residence_seconds','CPU_user_seconds','CPU_system_seconds','peak_RSS_MiB')],str(c['returncode'])])+'|')
    lines +=['',f"新增CPU驻留{total:.9f}秒；历史{audit['prior_CPU_residence_seconds']:.9f}秒；累计{audit['cumulative_CPU_residence_seconds']:.9f}秒。父wall另列{audit['parent_wall_seconds_separate']}秒；客户端整理未单测null。新增GPU0，历史GPU{audit['historical_GPU_seconds']}秒。",'', '总计144权重更新、0教师更新、0第二阶段更新。全部候选退化、拒绝、回退仍公开；无超预算补跑。单种子合成支持的固定教师诊断，不是分类/医学效果证据；不自动下一轮。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps(audit))


if __name__=='__main__':
    import sys
    main(sys.argv[1])
