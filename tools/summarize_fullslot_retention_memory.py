"""Anonymous reporting only; checks identities against recorded aggregates, never reruns science."""
import json
import sys
from pathlib import Path

MODES=['old_moment','full_moment','old_decision','full_decision']
PAIRS=[('old_moment','full_moment'),('old_decision','full_decision'),('old_moment','old_decision'),('full_moment','full_decision')]
METRICS=['balanced_accuracy','old_balanced_accuracy','new_balanced_accuracy']


def main(root):
    def save(name,value): (root/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    data=json.loads((root/'CPU_RESULTS.json').read_text());lock=json.loads((root/'DESIGN_LOCK.json').read_text());program=data['program'];results=data['results'];by={r['id']:r for r in results}
    complete=by.get(lock['cases'][0]['id'],{}).get('status')=='PASS' and all(by.get(c['id'],{}).get('status')=='COMPLETED' for c in lock['cases'][1:])
    tables={name:[dict(case=v['id'],**row) for v in results for row in v.get(key,[])] for name,key in [('CPU_SUMMARY','rows'),('CLASS_RECALL','class_rows'),('PAIRED_PREDICTIONS','paired_predictions'),('TRAINING_COSTS','training'),('CONSTRUCTION_COSTS','construction'),('MEMORY_DIAGNOSTICS','diagnostics'),('CONTRIBUTIONS','contributions')]}
    coverage=[];movement=[];components=[];norms=[];differences=[];classdiff=[];gates=[];resources=[];identities=0
    for v in results[1:]:
        for t in v.get('training',[]):
            coverage.extend(dict(case=v['id'],task=t['task'],mode=t['mode'],**z) for z in t['teacher_coverage'])
            movement.extend(dict(case=v['id'],task=t['task'],mode=t['mode'],**z) for z in t['movement'])
        for c in v.get('construction',[]):
            components.extend(dict(case=v['id'],task=c['task'],mode=c['mode'],**z) for z in c['objective_components'])
            norms.extend(dict(case=v['id'],task=c['task'],mode=c['mode'],**z) for z in c['target_gradient_norms'])
        if v.get('status')!='COMPLETED':continue
        for task in sorted({r['task'] for r in v['rows']}):
            rows={r['mode']:r for r in v['rows'] if r['task']==task}
            for mode,row in rows.items():
                cr=[c for c in v['class_rows'] if c['task']==task and c['mode']==mode]
                assert abs(sum(c['recall'] for c in cr)/len(cr)-row['balanced_accuracy'])<1e-12;identities+=1
            for a,b in PAIRS:
                diff={m+'_pp':None if rows[a][m] is None else 100*(rows[b][m]-rows[a][m]) for m in METRICS}
                differences.append(dict(case=v['id'],task=task,baseline=a,candidate=b,**diff))
                ac={c['class_index']:c for c in v['class_rows'] if c['task']==task and c['mode']==a}
                for c in v['class_rows']:
                    if c['task']==task and c['mode']==b:classdiff.append(dict(case=v['id'],task=task,baseline=a,candidate=b,class_index=c['class_index'],count=c['count'],recall_difference_pp=100*(c['recall']-ac[c['class_index']]['recall'])))
                p=next(z for z in v['paired_predictions'] if z['task']==task and z['baseline']==a and z['candidate']==b and z['group']=='all')
                assert abs((p['rescued']-p['harmed'])/p['count']-(rows[b]['accuracy']-rows[a]['accuracy']))<1e-12;identities+=1
        for a,b in PAIRS:
            ds=[d for d in differences if d['case']==v['id'] and d['baseline']==a and d['candidate']==b]
            gates.append(dict(case=v['id'],baseline=a,candidate=b,no_old_new_harm_all_stages=all(all(d[m+'_pp'] is None or d[m+'_pp']>=0 for m in METRICS) for d in ds),strict_BA_some_stage=any(d['balanced_accuracy_pp']>0 for d in ds)))
        assert sum(t['updates'] for t in v['training'])==v['total_adapter_updates']
        assert sum(t['updates'] for t in v['construction'])==v['actual_memory_updates']
        resources.append(dict(case=v['id'],memory_bytes_per_arm=v['candidate_memory_bytes'],adapter_parameters=v['adapter_parameters'],frame_bytes=v['frame_bytes'],training_seconds=sum(t['seconds'] for t in v['training']),construction_seconds=sum(t['seconds'] for t in v['construction']),pipeline_seconds=v['CPU_pipeline_seconds'],adapter_updates=v['total_adapter_updates'],memory_updates=v['actual_memory_updates']))
    tables.update(TEACHER_COVERAGE=coverage,LOGIT_MOVEMENT=movement,COMPRESSION_COMPONENTS=components,TARGET_GRADIENT_NORMS=norms,STAGE_DIFFERENCES=differences,CLASS_CONTRIBUTIONS=classdiff,PAIR_GATES=gates,RESOURCE_SUMMARY=resources)
    for name,rows in tables.items():save(name,rows)
    for c in coverage:
        assert 0<=c['weighted_old_correct']<=1+1e-12 and 0<=c['weighted_full_correct']<=1+1e-12
    if complete:
        assert len(tables['CPU_SUMMARY'])==48 and len(tables['CLASS_RECALL'])==540 and len(coverage)==400 and len(movement)==72
        assert len(components)==48 and len(norms)==140 and sum(r['adapter_updates'] for r in resources)==2304 and sum(r['memory_updates'] for r in resources)==1728
        for v in results:assert not v['GPU_initialized'] and not v['val_accessed'] and not v['test_accessed']
    primary=complete and all(g['no_old_new_harm_all_stages'] and g['strict_BA_some_stage'] for g in gates if g['case'] in ('ISIC_scale','HK_scale') and g['baseline']=='full_moment' and g['candidate']=='full_decision')
    total=sum(c['CPU_process_residence_seconds'] for c in program['costs'])
    audit=dict(status='SYNTHETIC_FACTORIAL_COMPLETE' if complete else 'INCOMPLETE',primary_D_vs_B_signal=bool(primary),joint_D_vs_all_signal=complete and all(by[c]['joint_mechanism_signal_pass'] for c in ('ISIC_scale','HK_scale')),science_source=program['source_commit'],costs=program['costs'],failures=program['failures'],CPU_residence_seconds=total,prior_CPU_residence_seconds=lock['prior_CPU_residence_seconds'],cumulative_CPU_residence_seconds=lock['prior_CPU_residence_seconds']+total,parent_wall_seconds_separate=program.get('supervisor_wall_seconds'),CPU_user_seconds=sum(c['CPU_user_seconds'] for c in program['costs']),CPU_system_seconds=sum(c['CPU_system_seconds'] for c in program['costs']),peak_RSS_MiB=max((c['peak_RSS_MiB'] for c in program['costs']),default=None),client_collector_seconds=None,new_GPU_seconds=0,historical_GPU_seconds_unchanged=lock['historical_GPU_seconds'],internal_costs_included_not_added=True,medical_effect=False,GPU_ready=False,original_full_query_gate=None)
    save('AUDIT',audit);save('CPU_REPORT_CHECK',dict(complete=complete,counts={k:len(v) for k,v in tables.items()},BA_and_paired_accuracy_identities=identities,update_counts_checked=complete,missing_filled_with_zero=False))
    lines=['# 全槽位保持 × 记忆：合成CPU结果','',f"状态 {audit['status']}；主门 D相对B：{primary}；D对全部控制联合门：{audit['joint_D_vs_all_signal']}。",'', 'A=旧保持+moment；B=全槽保持+moment；C=旧保持+判别；D=全槽保持+判别。均固定槽位，旧原数学源码未改。','', '|规模|stage|A BA%|B BA%|C BA%|D BA%|','|---|---:|---:|---:|---:|---:|']
    for v in results[1:]:
        for task in sorted({r['task'] for r in v.get('rows',[])}):
            rows={r['mode']:r for r in v['rows'] if r['task']==task}
            lines.append('|'+ '|'.join([v['id'],str(task)]+[f"{100*rows[m]['balanced_accuracy']:.6f}" for m in MODES])+'|')
    lines+=['','全部Acc/oldBA/newBA见CPU_SUMMARY，逐类见CLASS_RECALL，全部简单效应与交互及救回破坏见STAGE_DIFFERENCES/CONTRIBUTIONS/PAIRED_PREDICTIONS。教师覆盖为合成memory上的类别概率质量，不是医学覆盖或真实旧支持正确率；全槽正确率仅诊断。','',f'独立CPU驻留 {total:.9f}秒，累计 {audit["cumulative_CPU_residence_seconds"]:.9f}秒。父wall {audit["parent_wall_seconds_separate"]}秒另列，peakRSS {audit["peak_RSS_MiB"]}MiB。内部训练/诊断/压缩/评价含驻留，不相加。新增GPU0。', '', '运行失败：'+json.dumps(program['failures'],ensure_ascii=False)+'。缺失不补0。','', '固定输入/ETF/旧memory不变不等旧类预测保护；全槽保持可能限制新类学习。压缩目标只记录最后更新前，缺最优性证书。没有生产checkpoint/预训练encoder/医学数据验证。完成后分析并停止，不自动下一轮。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    analyze(root,tables,audit)
    print(json.dumps(audit))



def analyze(root,tables,audit):
    if audit['status']!='SYNTHETIC_FACTORIAL_COMPLETE':return
    rows=tables['CPU_SUMMARY'];training=tables['TRAINING_COSTS']
    lines=['# 四臂机制分析与证据边界','',f"主门 D相对B={audit['primary_D_vs_B_signal']}，D对全部3控制={audit['joint_D_vs_all_signal']}。四臂保持同参数/步数/数据预算；不依据结果改门或开启下一轮。",'',
           'A=old_moment，B=full_moment，C=old_decision，D=full_decision。该轮固定槽位，隔离保持范围与记忆。对照科学数学不变；诊断只复用已有前向和梯度。','',
           '|规模|stage|比较|ΔBA pp|Δ旧BA pp|Δ新BA pp|','|---|---:|---|---:|---:|---:|']
    names={'old_moment':'A','full_moment':'B','old_decision':'C','full_decision':'D'}
    for d in tables['STAGE_DIFFERENCES']:
        vals=[d[m+'_pp'] for m in METRICS]
        lines.append('|'+ '|'.join([d['case'],str(d['task']),names[d['candidate']]+'−'+names[d['baseline']]]+['null' if v is None else f'{v:+.6f}' for v in vals])+'|')
    lines+=['','教师覆盖率：每类旧memory概率质量之和，宏平均；正确判定始终在旧已激活类别内，不用全槽argmax替换。有效保持总质量等于此宏平均。它不是旧真实支持正确率，不能用高覆盖替代真实旧类评价。','',
            '|规模|stage|A覆盖%|B覆盖%|C覆盖%|D覆盖%|','|---|---:|---:|---:|---:|---:|']
    for cid in ('small','ISIC_scale','HK_scale'):
        for task in sorted({r['task'] for r in rows if r['case']==cid}):
            tr={r['mode']:r for r in training if r['case']==cid and r['task']==task}
            lines.append('|'+ '|'.join([cid,str(task)]+['null' if tr[m]['coverage_mean'] is None else f"{100*tr[m]['coverage_mean']:.6f}" for m in MODES])+'|')
    stage2=[]
    for cid in ('small','ISIC_scale','HK_scale'):
        for a,b in [('old_moment','full_moment'),('old_decision','full_decision')]:
            aa=[r['weighted_old_correct'] for r in tables['TEACHER_COVERAGE'] if r['case']==cid and r['task']==2 and r['mode']==a]
            bb=[r['weighted_old_correct'] for r in tables['TEACHER_COVERAGE'] if r['case']==cid and r['task']==2 and r['mode']==b]
            stage2.append(dict(case=cid,baseline=a,candidate=b,max_class_coverage_difference=max(abs(x-y) for x,y in zip(aa,bb)),recorded_coverage_equal=aa==bb))
    (root/'STAGE2_COVERAGE_PAIR_CHECK.json').write_text(json.dumps(stage2,indent=2)+'\n')
    lines+=['','第二阶段同记忆的两种保持臂覆盖差见STAGE2_COVERAGE_PAIR_CHECK。后续各臂模型已不同，覆盖变化是轨迹结果，不能仅凭相关性判断损害由覆盖率导致。逐类最坏值和全槽正确率在TEACHER_COVERAGE公开，未重新选择样本。','',
            '教师在旧已见类别内正确，不保证面对全部预留槽也正确。若旧正确覆盖高而全槽正确覆盖低，全槽保持可能同时保存尚未激活方向的竞争优势；本轮只测此差距，不能把保留教师关系自动当作正确未来保护。未按全槽正确率改mask或补margin。','',
            '旧保持误差只看旧slots内部。全保持按K个槽均方，旧保持按O个槽均方。令δ是学生减教师分数，R=K−O，则点级恒等式：MSE_all=(O/K)MSE_old+(R/K)MSE_other+(O*R/K²)(meanδ_old−meanδ_other)²。对任意固定样本权重也成立。故系数1不等旧旧子空间梯度强度相同；新增方向与旧旧权重稀释同时存在。这是预声明机制的性质，不能结果后补权重。诊断MSE覆盖全部旧memory，实际保持只覆盖教师旧argmax正确部分，两者不能混同。','',
            '新/预留组分数变化诊断见LOGIT_MOVEMENT。每个数来自第0/47步更新前已有score；不是最终更新后测量。负的旧对新mean_shift意味着旧组相对新组平均分数下降，但平均不能保证逐样本间隔或分类。','',
            '压缩分项见COMPRESSION_COMPONENTS；响应项和两个快照梯度项都在旧冻结公式中计算，仅取已有值。TARGET_GRADIENT_NORMS记录分母是否触底，不把分母很小直接认定为致因。压缩最后更新前目标下降不是最优证书或未来保护证明。','',
            '|规模|臂|末阶段旧CE最大误差|末阶段旧梯度最大误差|末阶段margin SW最大误差|','|---|---|---:|---:|---:|']
    for cid in ('small','ISIC_scale','HK_scale'):
        last=max(r['task'] for r in rows if r['case']==cid)
        for d in tables['MEMORY_DIAGNOSTICS']:
            if d['case']==cid and d['task']==last:lines.append(f"|{cid}|{names[d['mode']]}|{d['max_old_CE_error']:.6f}|{d['max_old_gradient_error']:.6f}|{d['max_old_margin_sliced_W1']:.6f}|")
    lines+=['','每个误差都在对应臂同一模型状态上比较其memory与真实旧合成支持；不同臂模型不同，跨臂误差不是固定模型压缩因果实验。旧支持只封存后诊断，没有反馈训练。','',
            '判断坚持预注册：D改善C但低于B仅缓解判别记忆损害；旧收益伴新损害不通过；B优于A而D低于B支持保留moment。正交互不等D优于各单独改动。未测反事实共同后续模型/最优性证书，不把有限序列归因为唯一机制。','',
            '|规模|训练秒|压缩秒|单臂memory bytes|adapter参数数|','|---|---:|---:|---:|---:|']
    for r in tables['RESOURCE_SUMMARY']:lines.append(f"|{r['case']}|{r['training_seconds']:.6f}|{r['construction_seconds']:.6f}|{r['memory_bytes_per_arm']}|{r['adapter_parameters']}|")
    lines+=['',f"4独立CPU驻留{audit['CPU_residence_seconds']:.9f}秒，父wall {audit['parent_wall_seconds_separate']:.9f}秒另列，user {audit['CPU_user_seconds']:.6f}/system {audit['CPU_system_seconds']:.6f}秒，peak RSS {audit['peak_RSS_MiB']}MiB。含所有训练/压缩/诊断/评价；不重复加内部时间。累计22轮86次CPU{audit['cumulative_CPU_residence_seconds']:.9f}秒。新增GPU0、历史400036.74851310626不变；client/collector未单测null。",'',
            '没有读取医学images/features/checkpoints/val/test；没有真实预训练encoder或生产checkpoint验证。结果仅为本轮合成非线性适配器证据，不作医学、临床、患者独立或顶级期刊结论。结束后不自动开启下一轮。']
    (root/'ANALYSIS_ZH.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main(Path(sys.argv[1]))
