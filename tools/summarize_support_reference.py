"""Validate saved numeric receipts and expose every frozen comparison without inference reruns."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];r=ROOT/'docs/support_reference_design1_20261016';prior=ROOT/'docs/candidate_transfer_design1_20261016'
read=lambda p:json.loads(p.read_text())
def save(n,v):(r/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def near(a,b):assert abs(a-b)<1e-10,(a,b)
raw=read(r/'CPU_RESULTS.json');lock=read(r/'DESIGN_LOCK.json');p=raw['program'];results=raw['results'];costs=p['costs']
assert p['status']=='COMPLETE_CPU_CHECKS' and p['active'] is None
new=sum(c['CPU_process_residence_seconds'] for c in costs)
resource=dict(new_CPU_residence_seconds=new,prior_CPU_residence_seconds=lock['prior_CPU_residence_seconds'],cumulative_CPU_residence_seconds=new+lock['prior_CPU_residence_seconds'],new_CPU_user_seconds=sum(c['CPU_user_seconds'] for c in costs),new_CPU_system_seconds=sum(c['CPU_system_seconds'] for c in costs),peak_RSS_MiB=max([c['peak_RSS_MiB'] for c in costs],default=0),parent_wall_seconds_not_added=p['supervisor_wall_seconds'],new_GPU_seconds=0,historical_GPU_seconds=lock['historical_GPU_seconds'],client_preparation_seconds=None,process_costs=costs)
save('RESOURCE_SUMMARY.json',resource);save('HISTORICAL_COSTS.json',dict(all_already_in_prior=True,prior_round_audit=read(prior/'AUDIT.json'),earlier_ledger=read(prior/'HISTORICAL_COSTS.json')))
completed=[v for v in results if 'case' in v and v['status']=='COMPLETED'];ev=next((v for v in results if v['id']=='balanced_evaluation' and v['status']=='COMPLETED'),None)
save('TRAINING.json',[dict(scene=v['id'],training=v['training'],old_support_samples=v['old_support_samples'],new_support_samples=v['new_support_samples'],synthetic_counts=v['synthetic_counts'],pipeline_seconds=v['pipeline_seconds'],adapter_updates=v['adapter_updates'],memory_updates=v['memory_updates']) for v in completed])
if p['failures'] or len(completed)!=4 or ev is None:
    save('REPORT_CHECK.json',dict(status='INCOMPLETE',failures=p['failures'],completed_scenes=len(completed)))
    (r/'REPORT_ZH.md').write_text('# 完整支持参照：未完整完成\n\n存在失败或缺失，完整比较为NA。保留原失败和全部已知费用，不科研重跑。\n\n新增CPU驻留'+str(new)+'秒。\n')
    raise SystemExit('Incomplete round')
assert len(costs)==6 and all(c['returncode']==0 and not c['timeout'] for c in costs)
assert sum(v['adapter_updates'] for v in results)==192 and all(v['memory_updates']==0 for v in results)
assert all(v['training']['updates']==48 and v['training']['seconds']>0 and v['sealed'] for v in completed)
assert all(not v['GPU_initialized'] and v['medical_images']==0 and not v['val_accessed'] and not v['test_accessed'] for v in results)
unit=next(v for v in results if v['id']=='support_arithmetic');assert unit['status']=='PASS' and unit['adapter_updates']==0
classes=ev['class_rows'];paired=ev['paired'];summaries=ev['summaries'];groups=ev['groups'];cis=ev['intervals'];decisions=ev['decisions']
assert tuple(map(len,(classes,paired,summaries,groups,cis,decisions)))==(168,126,60,80,36,12)
idx={(v['scene'],v['model'],v['class_index']):v for v in classes};assert len(idx)==168
for v in classes:
    assert v['count']==1000 and 0<=v['correct']<=1000 and v['CE']>=0
    assert v['errors_to_old']+v['errors_to_new']==1000-v['correct']
    assert 0<=v['negative_margin']+v['zero_margin']<=1000 and 0<=v['near_zero_margin']<=1000
    assert v['margin_quantiles']==sorted(v['margin_quantiles']) and len(v['margin_quantiles'])==7
for v in paired:
    assert sum(v[k] for k in ('both_correct','rescued','harmed','both_wrong'))==1000
    assert v['baseline_correct']==v['both_correct']+v['harmed'] and v['full_support_correct']==v['both_correct']+v['rescued']
    for m,prefix in [(v['baseline'],'baseline'),('full_support','full_support')]:
        cr=idx[v['scene'],m,v['class_index']];assert cr['correct']==v[prefix+'_correct']
        assert all(cr['errors_to_'+g]==v[prefix+'_errors_to_'+g] for g in ('old','new'))
    assert v['correct_to_wrong_old']+v['correct_to_wrong_new']==v['harmed']
    assert v['wrong_old_to_correct']+v['wrong_new_to_correct']==v['rescued']
for v in groups:
    rr=[a for a in classes if a['scene']==v['scene'] and a['model']==v['model'] and (v['group']=='all' or v['group']==a['group'] or v['group']==a['subgroup'])]
    assert v['classes']==len(rr) and v['count']==sum(a['count'] for a in rr)
    if rr:
        near(v['BA'],sum(a['correct']/a['count'] for a in rr)/len(rr))
        for key,source in [('macro_CE','CE'),('margin_mean','margin_mean'),('opposite_margin_mean','opposite_margin_mean'),('old_new_gap_mean','old_new_gap_mean')]:near(v[key],sum(a[source] for a in rr)/len(rr))
    else:assert v['BA'] is v['macro_CE'] is None
for v in summaries:
    m=v['pair'].removeprefix('full_support_vs_');a=next(a for a in groups if (a['scene'],a['model'],a['group'])==(v['scene'],m,v['group']));b=next(a for a in groups if (a['scene'],a['model'],a['group'])==(v['scene'],'full_support',v['group']))
    assert v['classes']==a['classes']==b['classes']
    if v['classes']:
        for key,source in [('BA_difference','BA'),('CE_difference','macro_CE'),('margin_difference','margin_mean'),('opposite_margin_difference','opposite_margin_mean')]:near(v[key],b[source]-a[source])
    else:assert v['BA_difference'] is None
for v in cis:
    assert v['bootstrap_draws']==10000 and v['confidence']==.95 and -1<=v['lower']<=v['upper']<=1
    assert v['bootstrap_seed']==lock['bootstrap_seeds'][v['scene']]+lock['comparisons'].index(v['pair'])
    near(v['BA_difference'],next(s['BA_difference'] for s in summaries if (s['scene'],s['pair'],s['group'])==(v['scene'],v['pair'],v['group'])))
for v in decisions:
    g={a['group']:a for a in cis if a['scene']==v['scene'] and a['pair']==v['pair']}
    pos=g['all']['BA_difference']>1e-12;point=all(g[k]['BA_difference']>=-.01-1e-12 for k in ('old','new'));cost=all(g[k]['lower']>=-.01-1e-12 for k in ('old','new'));gain=g['all']['lower']>1e-12
    status='SUPPORTED' if pos and point and cost and gain else 'COST_EXCEEDS_LIMIT' if not point else 'NO_POSITIVE_GAIN' if not pos else 'INCONCLUSIVE'
    assert v['status']==status and not v['confirmatory'] and [v[k] for k in ('positive_overall_BA','point_cost_acceptable','interval_cost_acceptable','interval_gain_positive')]==[pos,point,cost,gain]
for name,key in [('CLASS_METRICS.json','class_rows'),('PAIRED_PREDICTIONS.json','paired'),('SUMMARY.json','summaries'),('MODEL_GROUPS.json','groups'),('INTERVALS.json','intervals'),('DECISIONS.json','decisions'),('EVALUATION.json','evaluation')]:save(name,ev[key])
replication=[]
for scale in ('ISIC_scale','HK_scale'):
    for pair in lock['comparisons']:
        rr=[v for v in summaries if v['scene'].startswith(scale) and v['pair']==pair and v['group']=='all'];ds=[v for v in decisions if v['scene'].startswith(scale) and v['pair']==pair];vals=[v['BA_difference'] for v in rr]
        replication.append(dict(scale=scale,pair=pair,mean=sum(vals)/2,minimum=min(vals),maximum=max(vals),both_positive=all(v>1e-12 for v in vals),both_supported=all(v['status']=='SUPPORTED' for v in ds),confirmatory=False))
save('REPLICATION.json',replication)
save('REPORT_CHECK.json',dict(status='PASS',class_rows=168,paired_rows=126,summary_rows=60,model_group_rows=80,interval_rows=36,decision_rows=12,adapter_updates=192,memory_updates=0,bootstrap_rerun=False,prior_three_counts_reproduced=True,unit=unit))
save('AUDIT.json',dict(status='COMPLETE_FINITE_SUPPORT_REFERENCE',source_commit=p['source_commit'],failures=p['failures'],resources=resource,actual_adapter_updates=192,actual_memory_updates=0,independent_confirmation=False,training_retries=0,stop_after_this_round=True))
lines=['# 完整旧类支持回放：完整报告','','四个既有训练状态各新增48步，共192次适配器更新、0次记忆优化。原teacher、ETF、新类支持、训练函数及目标不变；旧类采用同一有限训练支持全部样本，类内均匀、类间均衡。三个压缩基线直接复用。四个模型及训练费用均先封存，再评估。','','这是已观察合成状态/评估bank上的机制参照，不是性能上界、公平压缩预算比较、真实医疗数据验证或独立确认。SUPPORTED只表示原数值判据；条件区间非训练种子总体区间或联合置信覆盖。','','## BA与原代价判定','','差值单位pp；每类1000个相同评估样本。','','|场景|比较|整体差 [95%区间]|旧组差 [95%区间]|新组差 [95%区间]|判定|','|---|---|---|---|---|---|']
for v in decisions:
    g={a['group']:a for a in cis if a['scene']==v['scene'] and a['pair']==v['pair']};fmt=lambda a:f"{100*a['BA_difference']:+.6f} [{100*a['lower']:+.6f}, {100*a['upper']:+.6f}]"
    lines.append('|'+ '|'.join([v['scene'],v['pair'],*[fmt(g[k]) for k in ('all','old','new')],v['status']])+'|')
lines+=['','## 全部终点的CE与边界','','每类等权，旧新组不各分配一半。CE为seen头logsoftmax；margin为真类分数减最大其他seen类分数。两者是诊断，不替代BA。','','|场景|模型|组|BA%|宏CE|平均margin|平均相反组margin|平均max_old−max_new|','|---|---|---|---:|---:|---:|---:|---:|']
for v in groups:
    if v['group'] in ('all','old','new'):
        lines.append(f"|{v['scene']}|{v['model']}|{v['group']}|{100*v['BA']:.6f}|{v['macro_CE']:.8f}|{v['margin_mean']:.8f}|{v['opposite_margin_mean']:.8f}|{v['old_new_gap_mean']:.8f}|")
lines+=['','## 完整诊断与验证','','CLASS_METRICS包含168条逐类四模型BA原始计数、CE、margin分位数/负值/零值/近零计数、相反组margin和错向旧新类；MODEL_GROUPS包含80条等权组汇总。PAIRED_PREDICTIONS包含126条真实配对四格和错误目的组转移；SUMMARY有60条比较，INTERVALS有36条条件区间，DECISIONS有12条判定，REPLICATION有6条两种子描述性汇总。保留所有正负结果。','','算术单元不执行训练更新，覆盖CE、margin与tie、类权重和配对转移分解。三个基线逐类正确数复现已发布记录；报告器只核算已保存计数、宏平均和判据，不重做推理或bootstrap。全部六个worker成功退出且无超时；TRAINING含每条48步内部时间、teacher覆盖及首末训练前诊断。','','## 费用与交付边界','',f"新增CPU驻留 {new:.12f} 秒；历史 {lock['prior_CPU_residence_seconds']:.12f} 秒；累计 {resource['cumulative_CPU_residence_seconds']:.12f} 秒。user {resource['new_CPU_user_seconds']:.6f} 秒，system {resource['new_CPU_system_seconds']:.6f} 秒，峰值RSS {resource['peak_RSS_MiB']:.6f} MiB。父进程墙钟 {p['supervisor_wall_seconds']:.12f} 秒另列不叠加。",'',f"新增GPU 0，历史GPU {lock['historical_GPU_seconds']} 秒。内部训练及评估耗时已含进程驻留；原teacher、memory、基线训练和历史异常探针费用见HISTORICAL_COSTS，已含历史不重复计费。前轮ISIC训练内部细分仍为NA；客户端准备耗时未独立计量为null。",'', '代码、协议、合成数值、完整费用和分析公开GitHub。私有模型张量和运行日志保留NAS，不发布运行身份/路径；无真实医疗图像、val/test访问。完成后停止，不自动补种子、扫描或下一轮。']
(r/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(status='PASS',resources=resource,decisions=decisions,replication=replication),indent=2))
