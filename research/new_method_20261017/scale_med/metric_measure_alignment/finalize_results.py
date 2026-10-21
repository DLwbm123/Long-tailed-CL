"""Public aggregate-only report; never loads private tensors."""
import json
from pathlib import Path

def finalize(receipts,previous,output):
 out=Path(output);out.mkdir(parents=True,exist_ok=True);a=receipts['aggregate.json'];c=receipts['COORDINATOR_EXIT.json'];w=receipts['WORKER_EXIT.json']
 assert w['exit_code']==c['exit_code']==0 and c['stop_reason'] is None and not receipts['ps'].strip()
 assert a['actual_solves']==2 and a['baseline_score_parity']==0 and a['baseline_count_exact'] and a['query_coverage_once'] and not receipts['WORKER_RESOURCE_RECEIPT.json']['GPU_initialized']
 cost=dict(activity_wall_seconds=c['wall_seconds'],known_worker_wait4_CPU_seconds=c['worker_CPU_seconds'],coordinator_CPU_seconds=c['coordinator_CPU_seconds'],peak_RSS_bytes=c['worker_peak_RSS_bytes'],round_remote_bytes=c['round_bytes'],round_with16MiB_reserve=c['round_bytes']+16*2**20,window_remote_bytes=c['window_bytes'],window_total_conservative=c['window_bytes']+1001040384,worker_body_not_added_twice=True,preparation_network_report_CPU_unknown=True)
 total={k:previous[k]+cost[k] for k in ['activity_wall_seconds','known_worker_wait4_CPU_seconds','coordinator_CPU_seconds']}
 verify=dict(all_round_PIDs_retired=True,worker_and_coordinator_exit0=True,resource_guards_triggered=False,synthetic_passed=True,baseline_score_and_counts_exact=True,normal_equation_and_readback_pass=True,all_models_sealed_before_query_risk=True,actual_solves=2,alpha_search=False,mapper_fit=False,encoder_forward=False,query_used_for_selection=False,active_ps_capture_available=False,neutral_argv_defined_by_sealed_launcher=True,GPU_initialized=False,method_success=False)
 assert cost['activity_wall_seconds']<600 and cost['round_with16MiB_reserve']<128*2**20 and cost['peak_RSS_bytes']<4*2**30 and cost['window_total_conservative']+16*2**20<17000000000
 for n,v in [('aggregate.json',a),('COST_RECEIPT.json',cost),('CUMULATIVE_COSTS.json',total),('VERIFICATION.json',verify)]:
  (out/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
 for n,v in receipts.items():
  if n.endswith('.json') and n not in ['LAUNCH.private.json','FIRST_ACTUAL.private.json','aggregate.json']:(out/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
 t='''# 原冻结F的目标测度对齐检验

**结论为STOP_METRIC_MEASURE_ALIGNMENT。** 图像BA从64.713626%升至66.871404%，类1从13/20升至14/20，救回1且损害0；类2从64/170升至81/170。有限正证据保留，但类0下降6.514658pp，且producer fold0 BA下降0.178397pp，未通过完整门。没有bootstrap，区间NA不是0；method_success=false，不能升级为性能成功、完整CL方案或原类1目标已达成。

## 独立假设与旧路线的区别

这次不把“没有新增语义信息”作为自动否决条件。直接检验一个目标测度问题：主指标是类内图像等权BA，旧训练目标却是类内组等权、组内图像等权。原F的四类总权重本已各1/4，不能把再次除原类别频次直接当作BA一致读出。当前ridge分数没有自然频率posterior认证。

强控制G直接复用每fold原冻结F teacher；主臂I只将support权重由1/(4 G_c m_g)改为1/(4 n_c)。原冻结g、support-only mapper、1535维、自由截距、平方损失、总权重、alpha=.01和solver全部相同。每fold仅新增一次solve，共2次，无温度、类offset、alpha/mapper/权重插值或seed搜索。它明确是拟合目标干预，不是新的无拟合解码器、原创方法或旧T适配头救援；没有读取T/P/C/D学习特征。

BA定义参照[官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)；[logit adjustment原文摘要](https://arxiv.org/abs/2007.07314)只用于确认频次后处理是既有路线，不将其全文理论保证搬到本ridge。已有producer/matched-probe/CE/readout-support等协议仍以原类组权为目标，本轮不同于旧T特征上同权重refit或bootstrap覆盖/重数干预。

## 数据隔离与测度差异

只原initial2555和既定seed74290两fold。每producer全部新权重/A/B仅由自身support构造，原mapper已support-only冻结；query退出拟合和选择。两个HEAD_SEAL与总MODEL_SEAL先完成，再读取封存F query响应与评分。无681开发、future/test、rawimages、编码、GPU、反传、encoder更新或新数据。既有query历史暴露与未知患者隔离限制仍在，不称外部独立确认。

以下TV为类内条件权重的总变差，衡量两个经验目标的区别，不是性能收益保证。

| producer | 类 | support图数 | 组数 | 组大小 min/median/max | 条件TV |
|---|---:|---:|---:|---|---:|
'''
 for m in a['measure']:
  for v in m['classes']:t+=f"| {m['producer']} | {v['class_id']} | {v['images']} | {v['groups']} | {' / '.join(map(str,v['group_size_min_median_max']))} | {v['conditional_TV']:.6f} |\n"
 t+='''
producer1支持类1全为单图组，两权在该类完全相同；但其它类权变化仍会改变联合头。因此query类1救回不能全部归因于直接调整类1图权。目标测度不同也不意味着旧组权“实现错误”，它原本是明确的研究选择。

## 全类与两折结果

各类为正确数/分母，BA为百分数。producer编号指该头的支持半组，其query是另一半。

'''
 for title,tab in [('合并',a['summary']),('producer fold0 query',a['per_fold']['0']),('producer fold1 query',a['per_fold']['1'])]:
  t+='### '+title+'''\n\n| 目标 | BA % | 类0 | 类1 | 类2 | 类3 | 组等权BA % |
|---|---:|---:|---:|---:|---:|---:|\n'''
  for n,v in tab.items():t+=f"| {n} | {v['BA']*100:.6f} | "+' | '.join(f'{x}/{z}' for x,z in zip(v['correct'],v['counts']))+f" | {v['group_balanced_BA']*100:.6f} |\n"
  t+='\n'
 t+='| 类 | 救回 | 损害 | 净正确数变化 |\n|---|---:|---:|---:|\n'
 for v in a['paired_vs_G']:t+=f"| {v['class_index']} | {v['rescued']} | {v['harmed']} | {v['net']} |\n"
 t+='''\n完整混淆、各类CE/margin分位、损害去向及逐fold配对计数均在aggregate.json。CE仅是ridge分数上的描述，不称概率校准。

## 冻结门与决定

| 点门 | 结果 |
|---|---|
'''
 for k,v in a['gate'].items():
  if k not in ['pass_all','missing']:t+=f"| {k} | {'通过' if v else '失败'} |\n"
 t+='''
合并BA+2.157779pp、类1+5pp通过幅度门；两fold类1均不降且原正确类1H0通过。但fold0 BA73.741477%→73.563080%未过逐fold不退，类0正确237→217、召回成本6.514658pp超过1pp保护线。类2+10pp或总BA上升不能覆盖这个失败。预设点门未全过，所以两项seed74342/10000draw调整bootstrap未运行，不事后修改家族或计算辅助显著性。

本固定配置停止，不搜索组/图权插值、不只改部分class、不换alpha/mapper/头/seed救门，也不将I直接作为新主候选。此次有效区分了“目标测度干预可改变留组效用”与“能满足全类安全约束的成功方法”；没有建立后者。后续独立路线应解释如何同时满足类1与其它类风险，不能因局部收益自动加保护模块。

## 验证与资源

原G support A/B回放最大误差1.1102230246251565e-16，原正规方程残差最大2.7755575615628914e-17；两个I残差1.1102230246251565e-16/1.3877787807814457e-16，回读score误差0。G query分数误差0、原计数exact，全部query覆盖一次。完整artifact逻辑各9643467bytes<=19058800；allocated放大已由round max(logical,allocated)核算，没有因低于文件逻辑门而隐去物理占用。

'''
 t+=f"协调活动wall{cost['activity_wall_seconds']:.6f}s，wait4 workerCPU{cost['known_worker_wait4_CPU_seconds']:.6f}s，协调ownCPU{cost['coordinator_CPU_seconds']:.6f}s，一次并账；worker正文不重加。RSS{cost['peak_RSS_bytes']}bytes<4GiB。round{cost['round_remote_bytes']}+16MiB<128MiB；window含default/localGit预留{cost['window_total_conservative']}bytes，另16MiB报告余量<17GB。600s原时钟、CPU300s/单文件64MiB/RSS4GiB守卫未触发，无重试。\n"
 t+='\n实际NFS4挂载/容量/小写读probe已核。主子按封存中性入口启动，短任务首次ps检查时已完成，未捕获活动ps，不冒称现场已核完整活动argv；退出后ps为空、worker和协调exit0，全部本轮PID退休。GPU从未初始化。无像素/模型/环境复制或历史删除。阅读、准备、网络、整理CPU缺口未知，见CUMULATIVE_COSTS；本成本不称完整研究CPU。\n'
 t+='\n源码/协议先于actual运行封存。公开内容只有源码、协议、聚合、封存元数据、验证与成本；artifact/A/B/W、身份、逐例预测、features、原私有日志和第三方PDF均留私有NFS。旧GGA仍未决/NA且禁止重跑，旧CE-bias负结论不改。整体研究未暂停，唯一30分钟心跳继续。\n'
 (out/'ANALYSIS_ZH.md').write_text(t);(out/'REPORT_ZH.md').write_text(t)
 return cost,total
