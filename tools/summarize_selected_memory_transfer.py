"""Independent integer-count audit and report; no training or model imports."""
import json
from pathlib import Path


def main(root):
    root=Path(root)
    def read(p):return json.loads(p.read_text())
    def save(name,value):(root/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    data=read(root/'CPU_RESULTS.json');program=data['program'];by={r['id']:r for r in data['results']};lock=read(root/'DESIGN_LOCK.json')
    ids=['ISIC_scale','HK_scale'];ev=by.get('independent_evaluation',{})
    complete=by.get('paired_arithmetic',{}).get('status')=='PASS' and all(by.get(k,{}).get('status')=='COMPLETED' for k in ids+['independent_evaluation']) and not program['failures']
    classes=ev.get('class_rows',[]);pairs=ev.get('paired',[]);rows=ev.get('rows',[]);gates=[];deltas=[];recall=[];pair_groups=[];max_gate_float_difference=0.
    for cid in ids:
        cc=[r for r in classes if r['case']==cid]
        if not cc:continue
        old=[r for r in cc if r['group']=='old'];new=[r for r in cc if r['group']=='new']
        gains={group:sum((r['selected_correct']-r['moment_correct'])/r['count'] for r in cr)/len(cr) for group,cr in [('old',old),('new',new)]}
        no_harm=all(r['selected_correct']>=r['moment_correct'] for r in cc)
        result=dict(case=cid,pass_strict=gains['old']>1e-12 and no_harm,old_BA_gain=gains['old'],new_BA_gain=gains['new'],every_class_no_harm=no_harm,harmed_classes=[r['class_index'] for r in cc if r['selected_correct']<r['moment_correct']],improved_old_classes=[r['class_index'] for r in old if r['selected_correct']>r['moment_correct']])
        stored=next(v for v in ev['gates'] if v['case']==cid)
        for key in result:
            if key in ('old_BA_gain','new_BA_gain'):
                difference=abs(result[key]-stored[key]);max_gate_float_difference=max(max_gate_float_difference,difference);assert difference<1e-12
            else:assert result[key]==stored[key]
        gates.append(result)
        for r in cc:
            p=next(p for p in pairs if p['case']==cid and p['class_index']==r['class_index'])
            assert r['count']==p['count']>0 and p['both_correct']+p['rescued']+p['harmed']+p['both_wrong']==p['count']
            assert r['moment_correct']==p['both_correct']+p['harmed'] and r['selected_correct']==p['both_correct']+p['rescued']
            assert p['moment_errors_to_old']+p['moment_errors_to_new']==r['count']-r['moment_correct']
            assert p['selected_errors_to_old']+p['selected_errors_to_new']==r['count']-r['selected_correct']
            recall.append(dict(r,**{m+'_recall':r[m+'_correct']/r['count'] for m in ('start','moment','selected')},recall_difference=(r['selected_correct']-r['moment_correct'])/r['count']))
        for group in ('all','old','new','accepted_old','fallback_old'):
            subset=[r for r in cc if group=='all' or r['group']==group or r['subgroup']==group]
            table={v['mode']:v for v in rows if v['case']==cid and v['group']==group}
            for m,v in table.items():
                expected=sum(r[m+'_correct']/r['count'] for r in subset)/len(subset)
                assert abs(expected-v['balanced_accuracy'])<1e-12
                assert v['count']==sum(r['count'] for r in subset)
                assert abs(v['accuracy']-sum(r[m+'_correct'] for r in subset)/v['count'])<1e-12
                if v['forgetting_from_shared_expanded_start'] is not None:
                    assert abs(v['forgetting_from_shared_expanded_start']-(table['start']['balanced_accuracy']-v['balanced_accuracy']))<1e-12
            deltas.append(dict(case=cid,group=group,classes=len(subset),count=sum(r['count'] for r in subset),start_BA=table['start']['balanced_accuracy'],moment_BA=table['moment']['balanced_accuracy'],selected_BA=table['selected']['balanced_accuracy'],BA_difference=table['selected']['balanced_accuracy']-table['moment']['balanced_accuracy'],moment_forgetting=table['moment']['forgetting_from_shared_expanded_start'],selected_forgetting=table['selected']['forgetting_from_shared_expanded_start']))
            pp=[p for p in pairs if p['case']==cid and (group=='all' or p['group']==group or p['subgroup']==group)]
            pair_groups.append(dict(case=cid,group=group,**{k:sum(p[k] for p in pp) for k in ('count','both_correct','rescued','harmed','both_wrong','wrong_to_different_wrong','moment_errors_to_old','moment_errors_to_new','selected_errors_to_old','selected_errors_to_new')}))
    old_resources={r['case']:r for r in read(root.parent/'weight_refinement_design1_20261016'/'RESOURCE_SUMMARY.json')}
    costs=[];training=[]
    for cid in ids:
        r=by.get(cid,{})
        if r.get('status')!='COMPLETED':continue
        assert r['adapter_updates']==96 and r['memory_updates']==0 and r['sealed'] and not r['evaluation_generated'] and not r['old_support_used_in_training'] and r['teacher_and_memories_unchanged']
        assert [v['mode'] for v in r['training']]==['moment','selected'] and all(v['updates']==48 and v['retention_scope']=='old' for v in r['training'])
        assert r['accepted_classes']==lock['accepted_classes'][cid]
        training.extend(dict(case=cid,**v) for v in r['training'])
        hist=old_resources[cid]
        costs.append(dict(case=cid,new_adapter_updates=96,new_memory_updates=0,load_seconds=r['load_seconds'],new_training_seconds=sum(v['seconds'] for v in r['training']),pipeline_seconds=r['pipeline_seconds'],historical_teacher_seconds=hist['historical_teacher_seconds'],historical_moment_seconds=hist['historical_moment_seconds'],historical_weight_refinement_seconds=hist['new_refinement_seconds'],historical_teacher_updates=48,historical_moment_updates=48,historical_weight_updates=48,historical_subcosts_already_in_prior_total=True))
    if complete:
        assert len(classes)==21 and len(rows)==30 and len(pairs)==21 and len(training)==4
        assert all(v['all_four_trajectories_sealed_before_generation'] for v in ev['evaluation'])
        assert [v['seed'] for v in ev['evaluation']]==[20264,20265]
        assert by['paired_arithmetic']['small_training_updates']==0 and ev['adapter_updates']==ev['memory_updates']==0
        assert all(not r['GPU_initialized'] and not r['val_accessed'] and not r['test_accessed'] for r in data['results'])
    total=sum(v['CPU_process_residence_seconds'] for v in program['costs'])
    audit=dict(status='COMPLETE_PAIRED_NEXT_STAGE_VALIDATION' if complete else 'INCOMPLETE',science_source_commit=program['source_commit'],joint_pass=all(v['pass_strict'] for v in gates) if complete else None,strict_gate_independently_verified=complete,failures=program['failures'],adapter_updates=sum(v['new_adapter_updates'] for v in costs),memory_updates=0,new_CPU_residence_seconds=total,prior_CPU_residence_seconds=lock['prior_CPU_residence_seconds'],cumulative_CPU_residence_seconds=lock['prior_CPU_residence_seconds']+total,parent_wall_seconds_separate=program.get('supervisor_wall_seconds'),CPU_user_seconds=sum(v['CPU_user_seconds'] for v in program['costs']),CPU_system_seconds=sum(v['CPU_system_seconds'] for v in program['costs']),peak_RSS_MiB=max((v['peak_RSS_MiB'] for v in program['costs']),default=None),new_GPU_seconds=0,historical_GPU_seconds=lock['historical_GPU_seconds'],client_collector_seconds=None,internal_costs_included_not_added=True,evaluation_seconds=ev.get('evaluation_seconds'),medical_effect=None,multi_stage_effect=None,auto_next_round=False,costs=program['costs'])
    if complete:assert audit['adapter_updates']==192 and audit['joint_pass']==ev['joint_pass']
    for name,content in [('AUDIT',audit),('STRICT_GATES',gates),('CLASS_RECALL',recall),('PAIRED_PREDICTIONS',pairs),('PAIRED_GROUPS',pair_groups),('SUMMARY',rows),('DELTAS',deltas),('TRAINING',training),('RESOURCE_SUMMARY',costs),('SHARED_START',ev.get('baseline',[])),('EVALUATION',ev.get('evaluation',[]))]:save(name,content)
    save('REPORT_CHECK',dict(complete=complete,strict_gates_recomputed=complete,class_count=len(classes),summary_count=len(rows),paired_count=len(pairs),training_arm_count=len(training),integer_contingency_checked=True,group_metrics_recomputed=True,missing_filled_with_zero=False,max_gate_float_difference=max_gate_float_difference))
    def fmt(v):return 'NA' if v is None else f'{v:.6f}'
    def pct(v):return fmt(None if v is None else 100*v)
    lines=['# selected记忆的下一阶段成对验证','',f"状态：{audit['status']}；两个规模严格联合通过：{audit['joint_pass']}。",'',
        '共同第一阶段教师与moment/selected全部复用；仅ISIC/HK下一阶段各两组48步，总192适配器更新。small只做零更新身份检查。四条轨迹封存后才生成新的seed+20000合成评估；全部旧/新类在联合slots中分类。','',
        '|规模|分组|起点BA%|moment终点BA%|selected终点BA%|差值pp|moment遗忘pp|selected遗忘pp|','|---|---|---:|---:|---:|---:|---:|---:|']
    for v in deltas:lines.append('|'+ '|'.join([v['case'],v['group'],*[pct(v[k]) for k in ('start_BA','moment_BA','selected_BA','BA_difference','moment_forgetting','selected_forgetting')]])+'|')
    lines+=['','遗忘以共同起点在扩展slots的BA减终点BA定义；负值表示净改善。SHARED_START另列旧slots-only起点，避免混淆静态扩展竞争。','', '|规模|严格通过|旧BA差pp|新BA差pp|受损类|改善旧类|','|---|---|---:|---:|---|---|']
    for v in gates:lines.append(f"|{v['case']}|{v['pass_strict']}|{pct(v['old_BA_gain'])}|{pct(v['new_BA_gain'])}|{v['harmed_classes']}|{v['improved_old_classes']}|")
    lines+=['','## 逐类结果','', '|规模|类|分组|样本数|moment正确|selected正确|recall差pp|','|---|---:|---|---:|---:|---:|---:|']
    for v in recall:lines.append(f"|{v['case']}|{v['class_index']}|{v['subgroup']}|{v['count']}|{v['moment_correct']}|{v['selected_correct']}|{pct(v['recall_difference'])}|")
    lines+=['','PAIRED_PREDICTIONS/PAIRED_GROUPS保留每类/各分组救回与伤害数量、共同正确/共同错误、错向旧/新类计数。类别recall不下降不意味着每个样本不受损；逐类净效果与样本迁移分别报告。','', '## 新增与历史费用','', '|规模|本次训练秒|历史教师秒|历史moment秒|历史权重修正秒|','|---|---:|---:|---:|---:|']
    for v in costs:lines.append('|'+ '|'.join([v['case'],*[fmt(v[k]) for k in ('new_training_seconds','historical_teacher_seconds','historical_moment_seconds','historical_weight_refinement_seconds')]])+'|')
    lines+=['','本次每规模两组各48适配器步；历史教师48、moment48、权重48步另列。历史子成本均已包含历史累计，内部本次计时也已包含驻留，不重复相加。','', '|进程|驻留秒|user秒|system秒|峰值MiB|返回码|','|---|---:|---:|---:|---:|---:|']
    for v in program['costs']:lines.append('|'+ '|'.join([v['id'],*[fmt(v[k]) for k in ('CPU_process_residence_seconds','CPU_user_seconds','CPU_system_seconds','peak_RSS_MiB')],str(v['returncode'])])+'|')
    lines+=['',f"新增CPU驻留{total:.9f}秒，历史{audit['prior_CPU_residence_seconds']:.9f}秒，累计{audit['cumulative_CPU_residence_seconds']:.9f}秒；父wall另列{audit['parent_wall_seconds_separate']}秒。新增GPU0，历史GPU{audit['historical_GPU_seconds']}秒；客户端未单测null。",'', '单次固定合成场景的下一阶段检验；不证明总体零伤害、多阶段或医学有效性。任何平均提升不能覆盖逐类受损；不放宽门槛、不补步数、不自动下一轮。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps(audit))


if __name__=='__main__':
    import sys
    main(sys.argv[1])
