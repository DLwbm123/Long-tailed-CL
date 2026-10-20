"""Aggregate-only reporting; never loads private rows or feature tensors."""
import json
from pathlib import Path

def finalize(receipts, previous, output):
 out=Path(output);out.mkdir(parents=True,exist_ok=True)
 c=receipts['COORDINATOR_EXIT.json'];w=receipts['WORKER_EXIT.json'];ok=c['exit_code']==w['exit_code']==0 and c['stop_reason'] is None
 a=receipts.get('aggregate.json',dict(decision='ENGINEERING_FACTORIAL_UNRESOLVED',method_success=False,missing_metrics=None))
 assert receipts['ps'].strip()==''
 cost=dict(activity_wall_seconds=c['wall_seconds'],known_worker_wait4_CPU_seconds=c['worker_CPU_seconds'],coordinator_CPU_seconds=c['coordinator_CPU_seconds'],peak_RSS_bytes=c['worker_peak_RSS_bytes'],round_remote_bytes=c['round_bytes'],round_total_with16MiB_reserve=c['round_bytes']+16*2**20,window_remote_bytes=c['window_bytes'],window_total_conservative=c['window_bytes']+1001040384,CPU_gaps_unknown=True,worker_body_not_added_twice=True)
 total={k:previous[k]+cost[k] for k in ['activity_wall_seconds','known_worker_wait4_CPU_seconds','coordinator_CPU_seconds']}
 verify=dict(engineering_complete=ok,all_round_pids_retired=True,original_GGA_remains_unresolved=True,original_GGA_not_retried=True,scientific_method_success=False,budget_pass=c['wall_seconds']<300 and cost['round_total_with16MiB_reserve']<128*2**20 and c['worker_peak_RSS_bytes']<12*2**30 and cost['window_total_conservative']+16*2**20<17000000000)
 assert verify['budget_pass']
 for n,v in [('aggregate.json',a),('VERIFICATION.json',verify),('COST_RECEIPT.json',cost),('CUMULATIVE_COSTS.json',total)]:
  (out/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
 for n in ['COORDINATOR_EXIT.json','WORKER_EXIT.json','WORKER_FAILURE.json','FAILURE.json','WORKER_RESOURCE_RECEIPT.json','SYNTHETIC.json','PRIVATE_PLAN_SEAL.json','FEATURE_SEAL.json','GPU_AVAILABILITY.json']:
  if n in receipts:(out/n).write_text(json.dumps(receipts[n],indent=2,allow_nan=False)+'\n')
 text='''# 四格前向数值研究：batch大小与梯度上下文

独立工程研究，不是GGA梯度诊断的重试。旧GGA仍为UNRESOLVED_SINGLE_IMAGE_FEATURE_PARITY，其缺失误差仍是NA；本研究不认证其旧阈值，不计算组梯度，不启动保护训练或性能实验。method_success=false。

样本为固定seed74320的32个新训练组，每组一张原PLAN首行图，排除了旧GGA全部64组。类计数6/0/2/24，fold计数15/17。因此没有类1，不能推类1机制或总体风险。没有开发/future/test评分。整个研究只做128逻辑图次、66次前向：同一序列在batch32/batch1 × no_grad/enable_grad四格。102912原bias在四格均requires_grad，其他参数冻结，无backward、autograd.grad、CE、teacher、mapperfit或ridge。

四格先全部FEATURE_SEAL，再计算下列差异；所有数值为描述性工程量，没有事后容差选择，也不等于误差的唯一根因。完整指标、interaction、状态exact检查见aggregate.json。

| 比较 | maxabs | RMS |
|---|---:|---:|
'''
 if ok:
  for k,v in a['pairs'].items():text+=f"| {k} | {v['maxabs']:.12g} | {v['RMS']:.12g} |\n"
  text+='\n| 对原cache | maxabs | RMS |\n|---|---:|---:|\n'
  for k,v in a['cache_comparisons'].items():text+=f"| {k} | {v['maxabs']:.12g} | {v['RMS']:.12g} |\n"
  text+='\n差分交互残差：maxabs='+str(a['interaction']['maxabs'])+'，RMS='+str(a['interaction']['RMS'])+'。参数版本、bias及nativeheads均exact不变；四格finite和计数验收通过。\n'
 else:text+='\n运行未完成；缺失数值NA，失败回执保留，不自动重试。\n'
 text+='\n在这32个固定新组和本运行上下文中，梯度记录开关没有可观测特征差异，batch大小变化存在可观测差异；该受控比较未定位内部算子或唯一底层原因，不能概括所有图像、batch布局或FP32实现。batch32与原cache接近不构成所有逐图计算的认证。旧GGA精确失败值仍NA，没有新的类1梯度结果或风险证据。\n'
 text+=f"\n活动wall{cost['activity_wall_seconds']:.6f}s，wait4 workerCPU{cost['known_worker_wait4_CPU_seconds']:.6f}s，协调ownCPU{cost['coordinator_CPU_seconds']:.6f}s，一次并账；worker正文不重复加。RSS{cost['peak_RSS_bytes']}bytes。remote round{cost['round_remote_bytes']}bytes+16MiB<{128*2**20}bytes。window含default/localGit外部预留{cost['window_total_conservative']}bytes，另16MiB报告余量<17GB。所有历史资产保留，无像素/模型/环境复制。准备、阅读、网络与报告CPU缺口未量化。\n"
 text+='\n启动父子完整argv中性已核；活动GPU进程名未捕获，完成后不为补该观察重跑。GPU peak allocated见WORKER_RESOURCE_RECEIPT；原始NFS挂载/容量/写读probe及GPU0 admission已核。\n'
 text+='\n四格特征仅worker RAM，没有特征序列化或下载；私人组/行清单仅NFS，公开聚合、源码、协议和成本/退出回执。父子exit、退役和资源见VERIFICATION。整体研究继续，但本结果不提供新的性能候选，也不支持复活旧CE-bias或GGA。\n'
 (out/'ANALYSIS_ZH.md').write_text(text);(out/'REPORT_ZH.md').write_text(text)
 return a,cost,total
