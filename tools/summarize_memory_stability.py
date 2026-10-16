"""Report paired replications and uncertainty from frozen anonymous records."""
import json
from pathlib import Path


def main(root):
    root=Path(root)
    def read(p):return json.loads(p.read_text())
    def save(name,value):(root/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    lock=read(root/'DESIGN_LOCK.json');data=read(root/'CPU_RESULTS.json');program=data['program'];by={v['id']:v for v in data['results']};ev=by.get('balanced_evaluation',{})
    complete=by.get('stability_arithmetic',{}).get('status')=='PASS' and not program['failures'] and all(by.get(v['id'],{}).get('status')=='COMPLETED' for v in lock['cases'][1:])
    classes=ev.get('class_rows',[]);pairs=ev.get('paired',[]);summaries=ev.get('summaries',[]);intervals=ev.get('intervals',[]);decisions=[];replication=[];acceptance=[];resources=[];construction_tables={k:[] for k in ('gradients','responses','losses','aggregates')};training=[];components=[]
    for cfg in lock['cases'][1:-1]:
        r=by.get(cfg['id'],{})
        if r.get('status')!='COMPLETED':continue
        assert r['adapter_updates']==144 and r['memory_updates']==96 and r['sealed'] and not r['evaluation_generated']
        assert r['teacher_and_memories_unchanged'] and not r['old_support_used_in_next_stage_training']
        assert r['teacher_training']['updates']==r['moment_construction']['updates']==48 and len(r['training'])==2 and all(v['updates']==48 for v in r['training'])
        f=r['refinement'];assert f['actual_memory_updates']==48 and f['fixed_nodes_unchanged'] and f['teacher_and_memories_unchanged']
        assert r['accepted_classes']==[v['class_index'] for v in r['acceptance'] if v['accepted']]
        for v in r['acceptance']:
            assert len(v['checks'])==13 and v['accepted']==(all(v['checks'].values()) and v['strict_gradient_improvement'])
            assert v['selected_source']==('candidate' if v['accepted'] else 'moment')
            acceptance.append(dict(scene=cfg['id'],scale=cfg['scale'],training_seed=cfg['seed'],**v))
        for k in construction_tables:construction_tables[k].extend(dict(scene=cfg['id'],**v) for v in f[k])
        for v in f['responses']:
            for k in ('seen','reserved'):
                dist=v[k];assert len(dist['values'])==len(dist['weights'])==v['count'] and abs(sum(dist['weights'])-1)<1e-10
        training.extend(dict(scene=cfg['id'],stage='next',**v) for v in r['training'])
        training.append(dict(scene=cfg['id'],stage='teacher',mode='common',**r['teacher_training']))
        components.append(dict(scene=cfg['id'],moment=r['moment_construction'],weight_refinement=f['construction'][0]))
        resources.append(dict(scene=cfg['id'],scale=cfg['scale'],seed=cfg['seed'],teacher_updates=48,next_stage_updates=96,moment_updates=48,weight_updates=48,teacher_seconds=r['teacher_training']['seconds'],moment_seconds=r['moment_construction']['seconds'],weight_seconds=f['construction'][0]['seconds'],weight_diagnostic_seconds=f['diagnostic_seconds'],next_stage_seconds=sum(v['seconds'] for v in r['training']),pipeline_seconds=r['pipeline_seconds'],accepted_classes=len(r['accepted_classes']),old_classes=cfg['task_sizes'][0]))
    for sid in dict.fromkeys(v['scene'] for v in classes):
        cr=[v for v in classes if v['scene']==sid];pp=[v for v in pairs if v['scene']==sid];ci={v['group']:v for v in intervals if v['scene']==sid}
        assert len(cr)==len(pp) and [v['class_index'] for v in cr]==list(range(len(cr)))
        for r,p in zip(cr,pp):
            assert r['class_index']==p['class_index'] and r['count']==p['count']==1000
            assert p['both_correct']+p['rescued']+p['harmed']+p['both_wrong']==1000
            assert r['moment_correct']==p['both_correct']+p['harmed'] and r['selected_correct']==p['both_correct']+p['rescued']
            assert abs(r['recall_difference']-(p['rescued']-p['harmed'])/1000)<1e-12
            assert p['moment_errors_to_old']+p['moment_errors_to_new']==1000-r['moment_correct']
            assert p['selected_errors_to_old']+p['selected_errors_to_new']==1000-r['selected_correct']
        for group in ('all','old','new','accepted_old','fallback_old'):
            subset=[v for v in cr if group=='all' or v['group']==group or v['subgroup']==group]
            s=next(v for v in summaries if v['scene']==sid and v['group']==group)
            assert s['count']==len(subset)*1000 and s['classes']==len(subset)
            if not subset:assert s['moment_BA'] is s['selected_BA'] is s['BA_difference'] is None;continue
            a=sum(v['moment_correct'] for v in subset)/(1000*len(subset));b=sum(v['selected_correct'] for v in subset)/(1000*len(subset))
            assert abs(s['moment_BA']-a)<1e-12 and abs(s['selected_BA']-b)<1e-12
            assert abs(s['BA_difference']-(b-a))<1e-12
            if group in ci:
                v=ci[group];assert abs(v['BA_difference']-(b-a))<1e-12 and -1<=v['lower']<=v['upper']<=1
                assert v['bootstrap_draws']==10000 and v['bootstrap_seed']==lock['bootstrap_seeds'][sid] and v['confidence']==.95
        positive=ci['all']['BA_difference']>1e-12;point_cost=all(ci[k]['BA_difference']>=-.01-1e-12 for k in ('old','new'))
        interval_cost=all(ci[k]['lower']>=-.01-1e-12 for k in ('old','new'));interval_gain=ci['all']['lower']>1e-12
        status='SUPPORTED' if positive and point_cost and interval_cost and interval_gain else 'COST_EXCEEDS_LIMIT' if not point_cost else 'NO_POSITIVE_GAIN' if not positive else 'INCONCLUSIVE'
        d=dict(scene=sid,scale=cr[0]['scale'],confirmatory=cr[0]['confirmatory'],status=status,positive_overall_BA=positive,point_cost_acceptable=point_cost,interval_cost_acceptable=interval_cost,interval_gain_positive=interval_gain)
        assert d==next(v for v in ev['decisions'] if v['scene']==sid);decisions.append(d)
    for base in lock['base_cases']:
        scale=base['id'];ds=[v for v in decisions if v['scale']==scale and v['confirmatory']]
        group_stats={}
        for group in ('all','old','new'):
            vals=[v['BA_difference'] for v in summaries if v['scale']==scale and v['confirmatory'] and v['group']==group]
            group_stats[group]=dict(mean=sum(vals)/len(vals) if vals else None,minimum=min(vals) if vals else None,maximum=max(vals) if vals else None)
        replication.append(dict(scale=scale,new_seeds_completed=len(ds),both_new_seeds_positive=all(v['positive_overall_BA'] for v in ds) if len(ds)==2 else None,both_point_costs_acceptable=all(v['point_cost_acceptable'] for v in ds) if len(ds)==2 else None,both_supported=all(v['status']=='SUPPORTED' for v in ds) if len(ds)==2 else None,new_seed_group_statistics=group_stats))
    history=dict(prior_CPU_residence_seconds=lock['prior_CPU_residence_seconds'],historical_GPU_seconds=lock['historical_GPU_seconds'],original_seed_construction=read(root.parent/'weight_refinement_design1_20261016'/'RESOURCE_SUMMARY.json'),previous_paired_validation=read(root.parent/'selected_memory_transfer_design1_20261016'/'AUDIT.json'),all_historical_subcosts_already_in_prior_total=True)
    total=sum(v['CPU_process_residence_seconds'] for v in program['costs'])
    audit=dict(status='COMPLETE_TWO_SEED_STABILITY_VALIDATION' if complete else 'INCOMPLETE',science_source_commit=program['source_commit'],joint_supported=all(v['both_supported'] for v in replication) if complete else None,failures=program['failures'],new_CPU_residence_seconds=total,prior_CPU_residence_seconds=lock['prior_CPU_residence_seconds'],cumulative_CPU_residence_seconds=lock['prior_CPU_residence_seconds']+total,parent_wall_seconds_separate=program.get('supervisor_wall_seconds'),CPU_user_seconds=sum(v['CPU_user_seconds'] for v in program['costs']),CPU_system_seconds=sum(v['CPU_system_seconds'] for v in program['costs']),peak_RSS_MiB=max((v['peak_RSS_MiB'] for v in program['costs']),default=None),new_GPU_seconds=0,historical_GPU_seconds=lock['historical_GPU_seconds'],adapter_updates=sum(v['teacher_updates']+v['next_stage_updates'] for v in resources),memory_updates=sum(v['moment_updates']+v['weight_updates'] for v in resources),client_collector_seconds=None,internal_costs_included_not_added=True,evaluation_and_bootstrap_seconds=ev.get('evaluation_seconds'),uncertainty='conditional pointwise paired percentile95; not training-seed population or simultaneous confidence',auto_next_round=False,costs=program['costs'])
    if complete:
        assert len(classes)==len(pairs)==63 and len(summaries)==30 and len(intervals)==18 and len(decisions)==6
        assert len(acceptance)==34 and len(construction_tables['gradients'])==408 and len(construction_tables['responses'])==272
        assert audit['adapter_updates']==576 and audit['memory_updates']==384
        assert all(v['all_eight_new_trajectories_sealed_before_generation'] and v['samples_per_seen_class']==1000 for v in ev['evaluation'])
        assert all(not v['GPU_initialized'] and not v['test_accessed'] and not v['val_accessed'] for v in data['results'])
    for name,v in [('AUDIT',audit),('CLASS_RECALL',classes),('PAIRED_PREDICTIONS',pairs),('SUMMARY',summaries),('INTERVALS',intervals),('DECISIONS',decisions),('REPLICATION',replication),('ACCEPTANCE',acceptance),('RESOURCE_SUMMARY',resources),('HISTORICAL_COSTS',history),('TRAINING',training),('CONSTRUCTION',components),('EVALUATION',ev.get('evaluation',[]))]:save(name,v)
    for name,v in construction_tables.items():save('MEMORY_'+name.upper(),v)
    save('REPORT_CHECK',dict(complete=complete,class_count=len(classes),paired_count=len(pairs),interval_count=len(intervals),acceptance_count=len(acceptance),integer_contingencies_checked=True,group_metrics_recomputed=True,decisions_independently_checked=True,bootstrap_not_rerun_by_reporter=True,missing_filled_with_zero=False))
    def fmt(v):return 'NA' if v is None else f'{v:.6f}'
    def pct(v):return fmt(None if v is None else 100*v)
    lines=['# 整体BA稳定性与旧新代价验证','',f"执行状态：{audit['status']}；四个新增场景联合支持：{audit['joint_supported']}。",'',
        'ISIC/HK各新增两个训练种子，原有种子只作development参考。每类1000个样本、一个bank供同规模三种子及两组共享；所有新增轨迹封存后才生成。整体BA为主目标，旧/新组代价各不超过1pp；逐类分类结果仅诊断。','',
        '|场景|用途|组|moment BA%|selected BA%|差pp|条件95%区间pp|场景判定|','|---|---|---|---:|---:|---:|---|---|']
    for s in summaries:
        if s['group'] not in ('all','old','new'):continue
        v=next(v for v in intervals if v['scene']==s['scene'] and v['group']==s['group']);d=next(v for v in decisions if v['scene']==s['scene'])
        lines.append('|'+ '|'.join([s['scene'],'新增' if s['confirmatory'] else '开发参照',s['group'],pct(s['moment_BA']),pct(s['selected_BA']),pct(s['BA_difference']),f"[{pct(v['lower'])}, {pct(v['upper'])}]",d['status']])+'|')
    lines+=['','区间为固定训练场景下的评估采样不确定性，不代表跨训练种子总体置信，也不声称同时95%覆盖。两个新增种子分别判断；不把旧种子计入确认性通过率，也不把共享评估重复当作额外独立样本。','', '|规模|新增两种子整体差均值pp|范围pp|方向均为正|点估计代价均可接受|两场景均支持|','|---|---:|---|---|---|---|']
    for v in replication:
        a=v['new_seed_group_statistics']['all'];lines.append(f"|{v['scale']}|{pct(a['mean'])}|[{pct(a['minimum'])}, {pct(a['maximum'])}]|{v['both_new_seeds_positive']}|{v['both_point_costs_acceptable']}|{v['both_supported']}|")
    lines+=['','## 构造接纳与新增费用','', '|场景|接纳/旧类|教师秒|moment秒|权重秒|下一阶段两组秒|','|---|---:|---:|---:|---:|---:|']
    for v in resources:lines.append('|'+ '|'.join([v['scene'],f"{v['accepted_classes']}/{v['old_classes']}",*[fmt(v[k]) for k in ('teacher_seconds','moment_seconds','weight_seconds','next_stage_seconds')]])+'|')
    lines+=['','每场景教师48、moment48、权重48、下一阶段两组各48步。新记忆的原始候选/接纳/回退和全部响应原子见MEMORY_*与ACCEPTANCE；所有63个类×种子分类结果见CLASS_RECALL，成对救回/伤害及错向旧新类见PAIRED_PREDICTIONS。','', '|进程|驻留秒|user秒|system秒|峰值MiB|退出码|','|---|---:|---:|---:|---:|---:|']
    for v in program['costs']:lines.append('|'+ '|'.join([v['id'],*[fmt(v[k]) for k in ('CPU_process_residence_seconds','CPU_user_seconds','CPU_system_seconds','peak_RSS_MiB')],str(v['returncode'])])+'|')
    lines+=['',f"新增CPU驻留{total:.9f}秒，历史{audit['prior_CPU_residence_seconds']:.9f}秒，累计{audit['cumulative_CPU_residence_seconds']:.9f}秒；父wall另列{audit['parent_wall_seconds_separate']}秒。新增GPU0，历史GPU{audit['historical_GPU_seconds']}秒；客户端整理null。",'', '所有内部构造、诊断、训练、评估和bootstrap均已含驻留，不重复相加。HISTORICAL_COSTS保留原教师/moment/权重修正和上一轮成对验证成本，已在历史总账；原有模型没有重新训练。', '', '完整记录不确定、负收益、超代价及全回退结果；不补种子、不放宽门槛、不自动下一轮。仅两个新增种子的合成下一阶段验证，不代表多阶段、总体稳定性或医学有效性。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps(audit))


if __name__=='__main__':
    import sys
    main(sys.argv[1])
