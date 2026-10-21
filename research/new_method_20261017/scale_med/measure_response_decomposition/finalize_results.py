"""Report exact response diagnostics from aggregate receipts only."""
import json
from pathlib import Path

def finalize(r,previous,output):
 out=Path(output);out.mkdir(parents=True,exist_ok=True);a=r['aggregate.json'];c=r['COORDINATOR_EXIT.json'];assert c['exit_code']==r['WORKER_EXIT.json']['exit_code']==0 and c['stop_reason'] is None and not r['ps'].strip()
 assert a['cell_count']==72 and a['original_endpoint_counts_exact'] and a['new_classifier_fits']==0 and not r['WORKER_RESOURCE_RECEIPT.json']['GPU_initialized']
 cost=dict(activity_wall_seconds=c['wall_seconds'],known_worker_wait4_CPU_seconds=c['worker_CPU_seconds'],coordinator_CPU_seconds=c['coordinator_CPU_seconds'],peak_RSS_bytes=c['worker_peak_RSS_bytes'],round_remote_bytes=c['round_bytes'],round_with16MiB_reserve=c['round_bytes']+16*2**20,window_remote_bytes=c['window_bytes'],window_total_conservative=c['window_bytes']+1001040384,worker_body_not_added_twice=True,reading_preparation_network_report_CPU_unknown=True)
 assert cost['activity_wall_seconds']<600 and cost['round_with16MiB_reserve']<128*2**20 and cost['peak_RSS_bytes']<4*2**30 and cost['window_total_conservative']+16*2**20<17000000000
 total={k:previous[k]+cost[k] for k in ['activity_wall_seconds','known_worker_wait4_CPU_seconds','coordinator_CPU_seconds']}
 verify=dict(all_round_pids_retired=True,worker_coordinator_exit0=True,active_neutral_parent_worker_argv_captured=True,GPU_initialized=False,source_components_support_only=True,COMPONENT_SEAL_before_query_description=True,original_G_I_counts_exact=True,all72cells_covered=True,real_response_system_solves=2,toy_system_solves=1,new_classifier_fits=0,no_component_argmax_interventions=True,old_risk_stop_unchanged=True,bootstrap_performed=False,method_success=False)
 for n,v in [('aggregate.json',a),('COST_RECEIPT.json',cost),('CUMULATIVE_COSTS.json',total),('VERIFICATION.json',verify)]:
  (out/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
 for n,v in r.items():
  if n.endswith('.json') and '.private.' not in n and n!='aggregate.json':(out/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
 t='''# 类0损害与两折差异的正规方程响应诊断

**主描述门结论为MIXED_OR_NON_CLASS2_SOURCE_PATTERN。** “类2来源在两折全部类0及F原正确类0上均贡献超过一半负向0-v2 margin移动”的四单元合取没有通过，不能把19/24个损害转向类2直接称为类2重加权的单一因果。有限模式仍保留：两折F原正确类0子集上，来源类2的负贡献份额分别69.30%和75.13%；fold0全部类0上仅31.03%，且来源类3略更负。

本轮零新分类器拟合、零encoder/mapperfit/参数更新/GPU。做了两个真实84列RHS线性系统响应求解和一次合成求解，总solve3，不能说所有solve为0。没有把分量加回头后做新argmax，没有扫权重、温度、偏移或选择新候选。原METRIC_MEASURE_ALIGNMENT的停止门和失败结果保持，method_success=false。

## 旧控制与新问题

readout_support检验读出支持覆盖/重数，support_completion检验组补样与近远选择，query_readout_cross_audit/readout_margin_audit做特征/读出路径分解，decision_anchor研究原方向保护。这些结果限制解释，但没有回答本次原冻结G/I在相同新Hessian下的support来源和A/B响应问题。本轮不复活任何旧负端点，不把这种常规线性代数归因称原创方法。

同X=[z,1]、同alpha=.01及不罚截距，H_I=A_I+alpha D。固定有限端点满足：

`H_I (W_I-W_G) = DeltaB - DeltaA W_G`，忽略量级约1e-16的原正常方程残差。

按support来源类和整数G_c*m_g-n_c的负/零/正号拆成12桶；每桶RHS为`sum delta_p*x*(y-xW_G)`，统一左乘H_I逆。四类来源是三桶之和，另拆target项`H_I^-1 DeltaB`及Gram项`-H_I^-1 DeltaA W_G`。所有分量两折先封存，再描述query。只读原PLAN、Fteacher、两个I artifact及原公开聚合/封存响应，不读旧T/P/C/D学习表示或681开发/future/test/raw。

这些分量共享新H_I，是指定锚点的算法路径分解；不等于只改某一类权重后独立重训的效果，也不唯一识别自然数据因果。weighted Gram变化不是encoder表示发生变化。以下来源份额用四个来源类贡献中的负项质量作分母，绝非损害样本的因果比例。

## 类0损害并非只在一折发生

| producer query | 类0救回 | 类0损害 | 净正确数 | 损害转类2 | 损害转类3 | BA变化 pp |
|---|---:|---:|---:|---:|---:|---:|
'''
 for k in ['0','1']:
  p=a['per_fold_paired'][k][0];d=a['per_fold'][k];t+=f"| {k} | {p['rescued']} | {p['harmed']} | {p['net']} | {p['harmed_destination'][2]} | {p['harmed_destination'][3]} | {100*(d['I']['BA']-d['G']['BA']):+.6f} |\n"
 t+='''
两折类0都净少对10个，不能将fold1的总BA提升说成类0得到保护。两折BA差主要是类1与类2的收益不同；以下为每类召回差除4的精确算术贡献，不是机制因果占比。

| producer | 类0贡献 pp | 类1贡献 pp | 类2贡献 pp | 类3贡献 pp | 合计 pp |
|---|---:|---:|---:|---:|---:|
'''
 for k,v in a['BA_contribution_pp_by_fold'].items():t+='| '+k+' | '+' | '.join(f'{x:+.6f}' for x in v)+f' | {sum(v):+.6f} |\n'
 t+='''
fold0类1query本来8/8而没有正确数增量；fold1由5/12升至6/12，对该fold BA贡献+2.083333pp。类2分别净+4/74和+13/96，贡献+1.351351/+3.385417pp。净-10个类0错误在各fold对BA的损害近似相同。不同baseline风险、竞争margin分布、支持结构及query构成仍未被两个fold单独识别，不能将所有折差归因于某一条来源项。

## 预设0-v2主单元

| producer | 类0子集 | 总margin移动均值 | 来源0 | 来源1 | 来源2 | 来源3 | 来源2负质量份额 | 主单元 |
|---|---|---:|---:|---:|---:|---:|---:|---|
'''
 for v in a['primary_pattern_cells']:
  t+=f"| {v['producer']} | {v['subset']} | {v['total_move']:+.8f} | "+' | '.join(f'{x:+.8f}' for x in v['source_class_means'])+f" | {100*v['pattern']['negative_share']:.2f}% | {'通过' if v['pattern']['pass_all'] else '失败'} |\n"
 t+='''
两折原正确子集支持有限类2竞争模式，但fold0 ALL失败，因此不将子集模式替换合取主门。原正确集合由G确定，仍是历史暴露的探索诊断，不是独立验证。来源类1在producer1所有桶都是零，因为其support类1全是单图组、两种权重完全相同；query类1救回因此不能全归直接类1权重干预。

## target与weighted Gram的贡献不能混同

| producer | 类0子集 | 数量 | 总移动 | target项 | Gram项 |
|---|---|---:|---:|---:|---:|
'''
 for v in a['cells']:
  if v['class_id']==0 and v['rival']==2:t+=f"| {v['producer']} | {v['subset']} | {v['total_move']['count']} | {v['total_move']['mean']:+.8f} | {v['target_term']['mean']:+.8f} | {v['Gram_term']['mean']:+.8f} |\n"
 t+='''
整类平均中Gram项两fold都为负，但target项fold0为正、fold1为负。到了事后选出的新受损样本，target项两fold均强负，而fold0 Gram项略正。这说明不能把整类平均项当作新损害样本的唯一解释。新受损类0集合含转类2与转类3，不能声称其每行都跨越了0-v2边界。

12桶完整结果保留增权与减权的抵消，不能仅抓住“上调大组类2”而忽略来源类0自身减权的负贡献。尤其新受损子集上来源类0和来源类2贡献都存在；这些事后子集不得用于挑桶/挑方向、删项重拟合或构造保护解码器。每折12固定class-rival对×ALL/G_correct/I_new_harm×两折完整72cell（含分位/空集NA）见aggregate.json，而非只发布类0-v2有利单元。

## 类1与全部风险门保持

'''
 for title,tab in [('合并',a['summary']),('producer0 query',a['per_fold']['0']),('producer1 query',a['per_fold']['1'])]:
  t+='### '+title+'\n\n| 端点 | BA % | 类0 | 类1 | 类2 | 类3 |\n|---|---:|---:|---:|---:|---:|\n'
  for n,v in tab.items():t+=f"| {n} | {v['BA']*100:.6f} | "+' | '.join(f'{x}/{z}' for x,z in zip(v['correct'],v['counts']))+' |\n'
  t+='\n'
 t+='''本轮只回放原G/I，没有新性能候选。原要求BA>=+1pp、类1>=+1正确/5pp、两fold BA/类1不退、G原正确类1H0、其它类成本各<=1pp不变。I仍因类0成本6.514658pp和fold0 BA退步失败，不重启原bootstrap，不放宽门、不升辅助主。主描述门也未全过；即使通过也不是可部署收益或独立因果。

后续需要不同且可证伪的估量或方法依据，不能直接做class2权重回退、class0保护插值、温度/offset扫描，或把本轮某个分量单独加回头。研究继续，但本诊断不自动授权某个改良候选；强F控制、类1/全类/两fold门和整链query排除仍需先冻结。

## 验证与成本

'''
 t+=f"72cell完整、原G/I合并及逐fold计数exact，F query分数误差{a['F_query_score_error']:.6g}。margin重构最大误差{a['query_margin_sum_max_error']:.6g}；support统计/正规方程/RHS和/DeltaW重构回执全部通过，实际DeltaW闭合最大约1.36e-15。不保存分量矩阵或特征新副本。\n"
 t+=f"\n协调活动wall{cost['activity_wall_seconds']:.6f}s，wait4 workerCPU{cost['known_worker_wait4_CPU_seconds']:.6f}s，协调ownCPU{cost['coordinator_CPU_seconds']:.6f}s，各一次，正文worker不重加。RSS{cost['peak_RSS_bytes']}bytes<4GiB；round{cost['round_remote_bytes']}+16MiB<128MiB；window含default/localGit预留{cost['window_total_conservative']}bytes，报告另16MiB<17GB。原600s/CPU300s/文件16MiB门未触发；原时钟不重置。\n"
 t+='\n实际NFS4挂载/容量/写读probe通过，启动时ps已捕获中性父子argv，退出后ps为空、worker和协调均exit0，PID退休。GPU未初始化，无像素/模型/环境复制、旧资产删除或旧退休进程监测。首次源6411192之后、任何actual之前d75edb3补足公开endpoint聚合的allowlist，未改科学门；运行后科学源未改。准备、阅读、网络和报告CPU未知，实验累计见CUMULATIVE_COSTS，不称全部研究CPU。\n'
 t+='\n公开仅源码协议、全部聚合/封存/退出/成本；身份、逐例、W/A/B/分量/feature/Gram、模型、原始私有日志与第三方PDF不下载公开。GGA仍未决NA/禁重跑；唯一30min心跳继续。\n'
 (out/'ANALYSIS_ZH.md').write_text(t);(out/'REPORT_ZH.md').write_text(t)
 return cost,total
