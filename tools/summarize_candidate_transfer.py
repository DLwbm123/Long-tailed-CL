"""Read completed receipts; validate arithmetic and publish all diagnostic outcomes."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
r=ROOT/'docs/candidate_transfer_design1_20261016'
p=ROOT/'docs/memory_stability_design1_20261016'
read=lambda path:json.loads(path.read_text())
def save(name,value):
    (r/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def near(a,b,tol=1e-10):
    assert abs(a-b)<=tol,(a,b)

raw=read(r/'CPU_RESULTS.json');lock=read(r/'DESIGN_LOCK.json');program=raw['program'];results=raw['results']
assert program['status']=='COMPLETE_CPU_CHECKS' and program['active'] is None
recovery=read(r/'RECOVERY_RESULTS.json');rp=recovery['program'];assert rp['status']=='COMPLETE_CPU_CHECKS' and rp['active'] is None
probes=[read(r/name) for name in ('NUMERICAL_PROBE.json','NUMERICAL_PROBE_MEMORY.json')]
costs=program['costs']+[v['cost'] for v in probes]+rp['costs'];new=sum(c['CPU_process_residence_seconds'] for c in costs)
resource=dict(new_CPU_residence_seconds=new,prior_CPU_residence_seconds=lock['prior_CPU_residence_seconds'],cumulative_CPU_residence_seconds=lock['prior_CPU_residence_seconds']+new,new_CPU_user_seconds=sum(c['CPU_user_seconds'] for c in costs),new_CPU_system_seconds=sum(c['CPU_system_seconds'] for c in costs),peak_RSS_MiB=max([c['peak_RSS_MiB'] for c in costs],default=0),parent_wall_seconds_not_added=[program['supervisor_wall_seconds'],rp['supervisor_wall_seconds']],new_GPU_seconds=0,historical_GPU_seconds=lock['historical_GPU_seconds'],client_preparation_seconds=None,process_costs=costs)
save('RESOURCE_SUMMARY.json',resource)
save('HISTORICAL_COSTS.json',dict(all_already_in_prior=True,reused_four_scene_costs=read(p/'RESOURCE_SUMMARY.json'),prior_round_audit=read(p/'AUDIT.json'),earlier_ledger=read(p/'HISTORICAL_COSTS.json')))
completed=[v for v in results if v.get('status')=='COMPLETED' and 'case' in v]
restored=[v for v in recovery['results'] if v.get('status')=='COMPLETED' and 'case' in v]
assert all(v['adapter_updates']==v['memory_updates']==0 for v in recovery['results'])
for v in restored:
    completed.append(dict(v,adapter_updates=48,recovery_adapter_updates=0,training=dict(updates=48,seconds=None,internal_timing_missing_due_to_original_diagnostic_exception=True),pipeline_seconds=None))
completed.sort(key=lambda v:[c['id'] for c in lock['cases']].index(v['id']))
evaluation=next((v for v in recovery['results'] if v.get('id')=='balanced_evaluation' and v.get('status')=='COMPLETED'),None)
training=[dict(scene=v['id'],training=v['training'],diagnostic_seconds=v['diagnostic_seconds'],pipeline_seconds=v['pipeline_seconds'],adapter_updates=v['adapter_updates'],memory_updates=v['memory_updates']) for v in completed]
gradients=[dict(scene=v['id'],**row) for v in completed for row in v['diagnostics']]
save('TRAINING.json',training);save('GRADIENTS.json',gradients)
if rp['failures'] or len(completed)!=4 or evaluation is None:
    save('REPORT_CHECK.json',dict(status='INCOMPLETE',failures=program['failures'],completed_scenes=len(completed),inference=None))
    (r/'REPORT_ZH.md').write_text('# Candidate机制诊断：未完整完成\n\n存在失败或缺失结果；无重试，不能作完整机制判断。参见CPU_RESULTS、TRAINING、GRADIENTS与费用。\n\n新增CPU驻留'+str(new)+'秒。\n')
    raise SystemExit('Incomplete scientific round; preserve NA and failures')
unit=next(v for v in results if v['id']=='candidate_arithmetic');assert unit['status']=='PASS' and unit['adapter_updates']==unit['memory_updates']==0
assert len(results)==6 and len(costs)==11 and all(not c['timeout'] for c in costs)
assert {(c['id'],c['reason']) for c in program['failures']}=={(sid,'WORKER_FAILED') for sid in ('ISIC_scale_r1','ISIC_scale_r2','balanced_evaluation')}
assert all(c['returncode']==0 for c in rp['costs']) and all(v['cost']['returncode']==0 for v in probes)
assert sum(v['adapter_updates'] for v in completed)==192 and all(v['memory_updates']==v['diagnostic_updates']==0 and v['training']['updates']==48 for v in completed)
assert all(not v['GPU_initialized'] and v['medical_images']==0 and not v['test_accessed'] for v in results+recovery['results'] if 'GPU_initialized' in v)
assert len(gradients)==160
idx={(v['scene'],v['common_state'],v['mode'],v['component']):v for v in gradients};assert len(idx)==160
for row in gradients:
    if row['mode']=='full_support':near(row['absolute_error'],0)
    if row['common_state']=='teacher' and row['component']=='retention':
        assert max(row['norm'],row['reference_norm'],row['absolute_error'])<=1e-12 and row['relative_error'] is None and row['cosine'] is None
    s,t,m=row['scene'],row['common_state'],row['mode'];get=lambda c:idx[s,t,m,c]['value']
    near(get('classification'),get('old_CE')+get('new_CE'));near(get('total'),get('classification')+get('retention'))
classes=evaluation['class_rows'];paired=evaluation['paired'];summaries=evaluation['summaries'];cis=evaluation['intervals'];decisions=evaluation['decisions']
assert (len(classes),len(paired),len(summaries),len(cis),len(decisions))==(42,84,40,24,8)
for row in classes:
    assert row['count']==1000 and all(0<=row[m+'_correct']<=1000 for m in ('moment','selected','candidate'))
for row in paired:
    assert sum(row[k] for k in ('both_correct','rescued','harmed','both_wrong'))==1000
    assert row['baseline_correct']==row['both_correct']+row['harmed']
    assert row['candidate_correct']==row['both_correct']+row['rescued']
    near(row['recall_difference'],(row['rescued']-row['harmed'])/1000)
for row in summaries:
    rr=[v for v in paired if v['scene']==row['scene'] and v['pair']==row['pair'] and (row['group']=='all' or row['group']==v['group'] or row['group']==v['subgroup'])]
    assert len(rr)==row['classes'] and sum(v['count'] for v in rr)==row['count']
    if rr:
        near(row['BA_difference'],sum(v['recall_difference'] for v in rr)/len(rr))
        near(row['candidate_BA'],sum(v['candidate_correct']/1000 for v in rr)/len(rr))
        near(row['baseline_BA'],sum(v['baseline_correct']/1000 for v in rr)/len(rr))
    else:assert row['BA_difference'] is None
for row in cis:
    assert row['bootstrap_draws']==10000 and row['confidence']==.95 and -1<=row['lower']<=row['upper']<=1
    expected=lock['bootstrap_seeds'][row['scene']]+int(row['pair']=='candidate_vs_selected');assert row['bootstrap_seed']==expected
    summary=next(v for v in summaries if (v['scene'],v['pair'],v['group'])==(row['scene'],row['pair'],row['group']));near(row['BA_difference'],summary['BA_difference'])
for row in decisions:
    group={v['group']:v for v in cis if v['scene']==row['scene'] and v['pair']==row['pair']}
    pos=group['all']['BA_difference']>1e-12;point=all(group[k]['BA_difference']>=-.01-1e-12 for k in ('old','new'));cost=all(group[k]['lower']>=-.01-1e-12 for k in ('old','new'));gain=group['all']['lower']>1e-12
    status='SUPPORTED' if pos and point and cost and gain else 'COST_EXCEEDS_LIMIT' if not point else 'NO_POSITIVE_GAIN' if not pos else 'INCONCLUSIVE'
    assert row['status']==status and [row[k] for k in ('positive_overall_BA','point_cost_acceptable','interval_cost_acceptable','interval_gain_positive')]==[pos,point,cost,gain] and not row['confirmatory']
for name,key in [('CLASS_RECALL.json','class_rows'),('PAIRED_PREDICTIONS.json','paired'),('SUMMARY.json','summaries'),('INTERVALS.json','intervals'),('DECISIONS.json','decisions'),('EVALUATION.json','evaluation')]:save(name,evaluation[key])
contrasts=[]
for scene in [v['id'] for v in completed]:
    for state in ('teacher','moment_endpoint'):
        for component in ('old_CE','new_CE','classification','retention','total'):
            base=idx[scene,state,'moment',component]['absolute_error']
            for mode in ('candidate','selected'):
                value=idx[scene,state,mode,component]['absolute_error']
                contrasts.append(dict(scene=scene,common_state=state,component=component,mode=mode,absolute_error=value,moment_absolute_error=base,error_difference=value-base,error_reduction_fraction=(base-value)/base if base>1e-12 else None))
save('GRADIENT_CONTRASTS.json',contrasts)
replication=[]
for scale in ('ISIC_scale','HK_scale'):
    for pair in ('candidate_vs_moment','candidate_vs_selected'):
        rr=[v for v in summaries if v['scene'].startswith(scale) and v['pair']==pair and v['group']=='all'];ds=[v for v in decisions if v['scene'].startswith(scale) and v['pair']==pair]
        vals=[v['BA_difference'] for v in rr]
        replication.append(dict(scale=scale,pair=pair,mean=sum(vals)/2,minimum=min(vals),maximum=max(vals),both_positive=all(v>1e-12 for v in vals),both_supported=all(v['status']=='SUPPORTED' for v in ds),confirmatory=False))
save('REPLICATION.json',replication)
save('REPORT_CHECK.json',dict(status='PASS',class_rows=42,paired_rows=84,summary_rows=40,interval_rows=24,decisions=8,common_state_gradient_rows=160,adapter_updates=192,memory_updates=0,bootstrap_rerun=False,prior_control_counts_reproduced=True,unit=unit))
save('AUDIT.json',dict(status='COMPLETE_MECHANISM_DIAGNOSTIC',source_commit=program['source_commit'],diagnostic_correction_commit=rp['source_commit'],original_failures=program['failures'],recovery_failures=rp['failures'],resources=resource,actual_adapter_updates=192,actual_memory_updates=0,diagnostic_updates=0,independent_confirmation=False,training_retries=0,engineering_diagnostic_continuations=2,numerical_probes=2,stop_after_this_round=True))
lines=['# 未筛选candidate的机制诊断：完整报告','', '本轮仅补四条candidate各48步，累计192次适配器更新、0次记忆优化。原教师、moment、selected和全部记忆复用；初次两个ISIC诊断因精确零断言失败，评估随之未开始。修正舍入检查后仅续算零更新诊断和评估，无训练重试、补种子、调参或科学门槛改变。原失败、两次探针和三进程续算均保留。','', '当前为已观察状态和已观察评估集上的机制诊断；SUPPORTED仅为原规则的算术标签，不能视为新独立确认。差值单位均为pp；条件95%区间只覆盖固定模型的评估采样不确定性。','', '## 分类完整比较','', '|场景|比较|整体差 [95%区间]|旧组差 [95%区间]|新组差 [95%区间]|判定|','|---|---|---|---|---|---|']
for row in decisions:
    cc={v['group']:v for v in cis if v['scene']==row['scene'] and v['pair']==row['pair']}
    fmt=lambda v:f"{100*v['BA_difference']:+.6f} [{100*v['lower']:+.6f}, {100*v['upper']:+.6f}]"
    lines.append('|'+ '|'.join([row['scene'],row['pair'],*[fmt(cc[g]) for g in ('all','old','new')],row['status']])+'|')
lines+=['','## 共同状态总梯度误差','', '参照是原有限训练支持；所有方法在相同参数状态、相同新类支持下查询。保持掩码与目标均由原教师固定。总目标为实际类别均衡CE加保持项。','', '|场景|共同状态|moment|candidate|selected|candidate相对moment误差减少%|','|---|---|---:|---:|---:|---:|']
for scene in [v['id'] for v in completed]:
    for state in ('teacher','moment_endpoint'):
        a=[idx[scene,state,m,'total']['absolute_error'] for m in ('moment','candidate','selected')]
        lines.append(f'|{scene}|{state}|{a[0]:.9f}|{a[1]:.9f}|{a[2]:.9f}|{100*(a[0]-a[1])/a[0]:+.4f}|')
lines+=['','## 完整诊断与核验','', 'CLASS_RECALL包含42条逐类三臂正确数；PAIRED_PREDICTIONS包含84条四格配对、救回/伤害及错向旧/新类；SUMMARY包含40个整体、旧、新、接纳旧、回退旧分组，INTERVALS包含24条区间，DECISIONS包含8条原阈值判定，REPLICATION分规模报告两个固定状态的符号/范围。','', 'GRADIENTS公开160条共同状态×表示×目标分量的目标值、范数、绝对/相对误差及余弦；GRADIENT_CONTRASTS列出全部候选/selected相对moment变化。零范数比值为NA，不填零。TRAINING保存各场景48步计数；HK两场景保留教师覆盖、首末训练诊断和内部计时。ISIC两场景因首次诊断异常，训练内部计时及首末摘要未落盘，明确缺失，不重训补测；封存文件48步元数据和首次进程费用保留。','', '零更新单元验证目标表达式、梯度加性和非零保持项下有限差分；各场景检验共同新类梯度一致、教师处保持项及梯度在1e-12浮点容差内为零，实际残差不归零。旧两臂逐类正确数复现前轮公开记录。报告器重新计算计数、分组与判定，未重做训练、推理或bootstrap。','', '## 费用与边界','', f"新增CPU驻留{new:.12f}秒，历史{lock['prior_CPU_residence_seconds']:.12f}秒，累计{resource['cumulative_CPU_residence_seconds']:.12f}秒。user {resource['new_CPU_user_seconds']:.6f}秒，system {resource['new_CPU_system_seconds']:.6f}秒，峰值RSS {resource['peak_RSS_MiB']:.6f}MiB。父wall {resource['parent_wall_seconds_not_added']}秒另列，不相加。",'',f"新增GPU0，历史GPU {lock['historical_GPU_seconds']}秒。内部训练、诊断、评估耗时已含在进程驻留；ISIC首次训练独立内部计时未落盘为null，但两次初始失败进程驻留全部计入；原教师、moment、权重、旧两臂训练成本引用HISTORICAL_COSTS且已含历史，不重复收费。客户端整理未单测null。",'', '完整代码/匿名合成数值与报告公开GitHub，私有张量及原始日志保留NAS。完成后停止，不自动下一轮。未验证实际医学影像、多阶段、种子总体效应或整个训练轨迹。']
(r/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(status='PASS',resources=resource,decisions=decisions,replication=replication),indent=2))
