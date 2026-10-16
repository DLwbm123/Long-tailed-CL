"""Receipt-only arithmetic, temporal isolation and complete outcome report."""
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];r=ROOT/'docs/query_constraints_design1_20261016';prior=ROOT/'docs/support_reference_design1_20261016'
read=lambda p:json.loads(p.read_text())
def save(n,v):(r/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def near(a,b):assert abs(a-b)<1e-10,(a,b)
raw=read(r/'CPU_RESULTS.json');lock=read(r/'DESIGN_LOCK.json');p=raw['program'];results=raw['results'];costs=p['costs'];assert p['status']=='COMPLETE_CPU_CHECKS' and p['active'] is None
new=sum(c['CPU_process_residence_seconds'] for c in costs)
resources=dict(new_CPU_residence_seconds=new,prior_CPU_residence_seconds=lock['prior_CPU_residence_seconds'],cumulative_CPU_residence_seconds=new+lock['prior_CPU_residence_seconds'],new_CPU_user_seconds=sum(v['CPU_user_seconds'] for v in costs),new_CPU_system_seconds=sum(v['CPU_system_seconds'] for v in costs),peak_RSS_MiB=max([v['peak_RSS_MiB'] for v in costs],default=0),parent_wall_seconds_not_added=p['supervisor_wall_seconds'],new_GPU_seconds=0,historical_GPU_seconds=lock['historical_GPU_seconds'],client_preparation_seconds=None,process_costs=costs)
save('RESOURCE_SUMMARY.json',resources);save('HISTORICAL_COSTS.json',dict(all_already_in_prior=True,prior_round_audit=read(prior/'AUDIT.json'),earlier_ledger=read(prior/'HISTORICAL_COSTS.json')))
training=[v for v in results if v.get('role')=='training' and v['status']=='COMPLETED'];save('TRAINING.json',training)
q=next((v for v in results if v.get('role')=='query' and v['status']=='COMPLETED'),None);e=next((v for v in results if v.get('role')=='evaluation' and v['status']=='COMPLETED'),None)
if p['failures'] or len(training)!=8 or q is None or e is None:
    save('REPORT_CHECK.json',dict(status='INCOMPLETE',failures=p['failures'],training_completed=len(training)))
    (r/'REPORT_ZH.md').write_text('# 约束选择：未完整完成\n\n存在失败或缺失，后续阶段未自动重跑，科学结果为NA。所有已知费用见RESOURCE_SUMMARY，失败见CPU_RESULTS。\n')
    raise SystemExit('Incomplete round')
assert len(costs)==11 and all(v['returncode']==0 and not v['timeout'] for v in costs)
assert sum(v['adapter_updates'] for v in results)==384 and all(v['memory_updates']==0 and v['role_access_guard'] and not v['GPU_initialized'] and v['medical_images']==0 and not v['test_accessed'] for v in results)
assert next(v for v in results if v['role']=='unit')['status']=='PASS'
assert {(v['scene'],v['rho']) for v in training}=={(s['id'],rho) for s in lock['scenes'] for rho in (.5,2.)}
assert all(v['training']['updates']==48 and v['adapter_updates']==48 for v in training)
assert (len(q['class_rows']),len(q['comparisons']),len(q['choices']),len(e['class_rows']),len(e['paired']),len(e['strata']),len(e['summaries']),len(e['intervals']),len(e['decisions']))==(126,12,4,84,42,84,12,12,4)
for v in q['class_rows']+e['class_rows']:
    assert v['count']==1000 and 0<=v['correct']<=1000 and v['CE']>=0
    assert v['errors_to_old']+v['errors_to_new']==1000-v['correct']
for scene in lock['scenes']:
    sid=scene['id'];rr=[v for v in q['comparisons'] if v['scene']==sid];assert {v['rho'] for v in rr}=={.5,1.,2.}
    for v in rr:
        base=next(a for a in rr if a['rho']==1.)
        for g in ('all','old','new'):
            cr=[a for a in q['class_rows'] if a['scene']==sid and a['model']==str(v['rho']) and (g=='all' or a['group']==g)]
            near(v['groups'][g]['BA'],sum(a['correct']/1000 for a in cr)/len(cr))
            near(v['delta_'+g],v['groups'][g]['BA']-base['groups'][g]['BA'])
        assert v['feasible']==all(v['delta_'+g]>=-.01-1e-12 for g in ('old','new'))
    feasible=[v for v in rr if v['feasible']];best=max(v['delta_all'] for v in feasible);expected=1.
    if best>1e-12:expected=min([v for v in feasible if abs(v['delta_all']-best)<=1e-12],key=lambda v:(v['rho']!=1.,abs(math.log(v['rho'])),v['rho']))['rho']
    chosen=next(v for v in q['choices'] if v['scene']==sid);assert chosen['rho']==expected
    assert chosen==next(v for v in e['selection_lock']['choices'] if v['scene']==sid)
for v in e['paired']+e['strata']:
    assert sum(v[k] for k in ('both_correct','rescued','harmed','both_wrong'))==v['count']
    assert v['harmed']==v['correct_to_wrong_old']+v['correct_to_wrong_new'] and v['rescued']==v['wrong_old_to_correct']+v['wrong_new_to_correct']
for v in e['paired']:
    rr=[a for a in e['strata'] if (a['scene'],a['class_index'])==(v['scene'],v['class_index'])];assert sorted(a['count'] for a in rr)==[200,800]
    for k in ('count','both_correct','rescued','harmed','both_wrong','wrong_to_different_wrong'):assert sum(a[k] for a in rr)==v[k]
    for mode,keys in [('baseline',('both_correct','harmed')),('selected',('both_correct','rescued'))]:
        cr=next(a for a in e['class_rows'] if (a['scene'],a['class_index'],a['model'])==(v['scene'],v['class_index'],mode));assert cr['correct']==sum(v[k] for k in keys)
for v in e['summaries']:
    for mode in ('baseline','selected'):
        rr=[a for a in e['class_rows'] if a['scene']==v['scene'] and a['model']==mode and (v['group']=='all' or a['group']==v['group'])]
        near(v[mode]['BA'],sum(a['correct']/1000 for a in rr)/len(rr))
        for key in ('CE','margin'):near(v[mode][key],sum(a[key] for a in rr)/len(rr))
    near(v['BA_difference'],v['selected']['BA']-v['baseline']['BA'])
for v in e['intervals']:
    assert v['bootstrap_draws']==10000 and v['confidence']==.95 and v['bootstrap_seed']==lock['bootstrap_seeds'][v['scene']] and -1<=v['lower']<=v['upper']<=1
    near(v['BA_difference'],next(a['BA_difference'] for a in e['summaries'] if (a['scene'],a['group'])==(v['scene'],v['group'])))
for v in e['decisions']:
    g={a['group']:a for a in e['intervals'] if a['scene']==v['scene']};positive=g['all']['BA_difference']>1e-12;point=all(g[k]['BA_difference']>=-.01-1e-12 for k in ('old','new'));interval=all(g[k]['lower']>=-.01-1e-12 for k in ('old','new'));gain=g['all']['lower']>1e-12
    status='SUPPORTED' if positive and point and interval and gain else 'COST_EXCEEDS_LIMIT' if not point else 'NO_POSITIVE_GAIN' if not positive else 'INCONCLUSIVE'
    assert v['status']==status and [v[k] for k in ('positive_overall_BA','point_cost_acceptable','interval_cost_acceptable','interval_gain_positive')]==[positive,point,interval,gain]
    if v['rho']==1:assert all(a['BA_difference']==a['lower']==a['upper']==0 for a in g.values())
selection=e['selection_lock'];assert selection['source_commit']==p['source_commit'] and selection['locked_at']==q['locked_at']
assert max(v['sealed_at'] for v in training)<=selection['query_started_at']<=min(v['generated_at'] for v in q['receipts'])
assert max(v['generated_at'] for v in q['receipts'])<=selection['locked_at']
assert all(selection['locked_at']<=v['E_seed_created_at']<=v['generated_at'] for v in e['receipts'])
qs={v['seed'] for v in q['receipts']};es={v['seed'] for v in e['receipts']};ss={v['seed'] for v in lock['scenes']}
assert len(qs)==len(es)==4 and not qs&es and not ss&(qs|es) and all(v>=10**9 for v in es)
assert sum(v['samples'] for v in q['receipts'])==sum(v['samples'] for v in e['receipts'])==42000
for name,data in [('Q_CLASS_METRICS.json',q['class_rows']),('Q_COMPARISONS.json',q['comparisons']),('SELECTION.json',selection),('E_CLASS_METRICS.json',e['class_rows']),('E_PAIRED.json',e['paired']),('E_STRATA.json',e['strata']),('E_SUMMARY.json',e['summaries']),('E_INTERVALS.json',e['intervals']),('E_DECISIONS.json',e['decisions']),('DATA_RECEIPTS.json',q['receipts']+e['receipts'])]:save(name,data)
audit=dict(status='COMPLETE_CONSTRAINED_QUERY_ROUND',source_commit=p['source_commit'],failures=[],actual_adapter_updates=384,memory_updates=0,training_retries=0,grid_expansion=False,all_candidates_sealed_before_Q=True,all_choices_locked_before_E_seed_and_generation=True,E_nonselected_evaluated=False,new_Q_samples=42000,new_E_samples=42000,independent_training_seed_confirmation=False,joint_supported=all(v['status']=='SUPPORTED' for v in e['decisions']),resources=resources,stop_after_this_round=True)
save('AUDIT.json',audit);save('REPORT_CHECK.json',dict(status='PASS',query_class_rows=126,evaluation_class_rows=84,paired_rows=42,stratum_rows=84,choices=4,intervals=12,decisions=4,bootstrap_rerun=False,inference_rerun=False,temporal_isolation_verified=True))
lines=['# 更新后约束选择：完整结果','','协议先公开，复用四个既有状态；仅rho=.5和2各48步，共384次更新，rho=1复用moment。八条候选训练、Q选择和E评分全部成功，无重跑、补训练种子或扩网格。','','## Q选择（开发结果，不作独立验证）','','|场景|rho|整体差pp|旧组差pp|新组差pp|可行|','|---|---:|---:|---:|---:|---|']
for v in q['comparisons']:lines.append(f"|{v['scene']}|{v['rho']}|{100*v['delta_all']:+.6f}|{100*v['delta_old']:+.6f}|{100*v['delta_new']:+.6f}|{v['feasible']}|")
lines+=['','## 封存E上的独立评分','','所选模型和基线均在相同新E样本上评分。只评选中候选，未选候选没有E结果。所有差值单位pp，区间为类别×固定混合分层的paired bootstrap条件pointwise95%，不是训练种子总体或同时覆盖。','','|场景|所选rho|整体差 [95%]|旧组差 [95%]|新组差 [95%]|判定|','|---|---:|---|---|---|---|']
for v in e['decisions']:
    g={a['group']:a for a in e['intervals'] if a['scene']==v['scene']};fmt=lambda a:f"{100*a['BA_difference']:+.6f} [{100*a['lower']:+.6f}, {100*a['upper']:+.6f}]"
    lines.append('|'+ '|'.join([v['scene'],str(v['rho']),*[fmt(g[k]) for k in ('all','old','new')],v['status']])+'|')
lines+=['','## 数据与标签隔离','','S只用于训练；Q使用四个新的预定随机流，每已见类1000，共42000样本。所有候选先封存后Q才生成，四个选择全部锁定后supervisor才抽E种子，随后E生成共42000样本。DATA_RECEIPTS和SELECTION保存顺序与种子；角色文件访问审计拒绝Q/E读取历史S状态、拒绝Q读取E、拒绝E读取Q结果和未选模型。模型推理只接收特征，标签只用于当前角色评分；不生成未见类样本。当前是程序级访问边界，不宣称抵御root恶意修改。','','Q是额外开发标签，E是新评估标签；二者各42000，均不是memory免费资源。重复观察的旧bank未读取。本轮只验证合成分布上的有限候选机制；不能作为真实医疗数据、训练种子总体泛化或解决真实尾类独立query不足的证据。回退基线意味着未找到正收益，不算科学成功。','','## 完整记录与费用','','Q_CLASS_METRICS有126条逐类候选记录；E_CLASS_METRICS有84条selected/baseline计数、CE、margin和错向组记录；E_PAIRED有42条配对翻转，E_STRATA有84条混合层配对；E_SUMMARY/E_INTERVALS各12条，E_DECISIONS四条。全部负结果、不可行候选、回退与区间不确定均公开。报告器只核算保存值，不重做推理或bootstrap。','',f"新增CPU驻留{new:.12f}秒，历史{lock['prior_CPU_residence_seconds']:.12f}秒，累计{resources['cumulative_CPU_residence_seconds']:.12f}秒；user {resources['new_CPU_user_seconds']:.6f}秒，system {resources['new_CPU_system_seconds']:.6f}秒，峰值RSS {resources['peak_RSS_MiB']:.6f}MiB。父wall {p['supervisor_wall_seconds']:.12f}秒另列不叠加。",'',f"新增GPU0，历史GPU{lock['historical_GPU_seconds']}秒。训练/Q/E内部耗时已含进程驻留；历史teacher、memory、基线与前轮费用见HISTORICAL_COSTS，已含历史不重复计费。客户端准备未单测null，历史缺失细项不补测。",'', '源码、协议、所有数值、费用与分析推送公开GitHub；模型张量与运行日志保留NAS，不公开运行身份。完成后停止，无自动下一轮。']
(r/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(status='PASS',choices=q['choices'],decisions=e['decisions'],resources=resources),indent=2))
