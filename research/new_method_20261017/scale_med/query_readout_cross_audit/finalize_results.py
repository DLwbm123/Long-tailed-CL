"""Report only aggregate diagnostics; no private feature/model loading."""
import json
from pathlib import Path

def finalize(root):
 out=Path(root)/'results'
 def read(n):return json.loads((out/n).read_text())
 def write(n,v):(out/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 a=read('aggregate.json');r=read('COORDINATOR_EXIT.json');wr=read('WORKER_RESOURCE_RECEIPT.json')
 assert r['exit_code']==0 and r['stop_reason'] is None and r['completed_unix']<r['deadline_unix'] and not wr['GPU_initialized']
 assert a['fixed_rival_margin_cells']==24 and a['exact_original_endpoint_count_parity'] and a['common_mapper_exact'] and a['query_coverage_exactly_once']
 assert a['parity_score_max_error']<=1e-8 and a['algebra_max_error']<=1e-8
 assert a['encoder_forwards']==a['new_fits']==a['parameter_updates']==0
 assert not any(a[k] for k in ['method_success','bootstrap_performed','development_used','test_used','future_used'])
 assert r['round_bytes']+16*2**20<128*2**20 and r['worker_peak_RSS_bytes']<4*2**30
 v=dict(query_rows=2555,folds=2,pooled_combinations=4,per_fold_combinations=8,fixed_rival_margin_cells=24,
  original_endpoint_count_parity=True,score_parity_max_error=a['parity_score_max_error'],algebra_max_error=a['algebra_max_error'],
  common_mapper_exact=True,query_coverage_exactly_once=True,all_finite=True,paired_R_H_identity=True,
  both_processes_retired=[1019744,1019745],EXIT0=True,no_failure=True,encoder_forwards=0,new_fits=0,updates=0,GPU_initialized=False,
  method_success=False,not_causal_identification=True,no_new_inference_family=True)
 write('VERIFICATION.json',v)
 c=dict(activity_wall_seconds=r['wall_seconds'],known_worker_wait4_CPU_seconds=r['worker_CPU_seconds'],coordinator_CPU_seconds=r['coordinator_CPU_seconds'],worker_peak_RSS_bytes=r['worker_peak_RSS_bytes'],round_remote_bytes=r['round_bytes'],external_reserve_bytes=16*2**20,round_total_with_reserve=r['round_bytes']+16*2**20,round_limit_bytes=128*2**20,wall_limit_seconds=600,RSS_limit_bytes=4*2**30,window_remote_bytes=r['window_bytes'],window_total_conservative=r['window_bytes']+1001040384,small_report_growth_reserve_bytes=16*2**20,worker_body_not_added_again=True,historical_CPU_gaps_unknown=True,preparation_network_report_CPU_incomplete=True,no_old_assets_deleted=True)
 write('COST_RECEIPT.json',c)
 cum=dict(activity_wall_seconds=31477.319448054623+c['activity_wall_seconds'],known_worker_wait4_CPU_seconds=39930.638934+c['known_worker_wait4_CPU_seconds'],coordinator_CPU_seconds=1196.338634+c['coordinator_CPU_seconds'],this_round_counted_once=True,historical_CPU_gaps_unknown=True,preparation_network_report_CPU_incomplete=True)
 write('CUMULATIVE_COSTS.json',cum)
 tab=a['summary'];rows=[]
 for n in ['F_on_F','F_on_T','T_on_F','T_on_T']:
  t=tab[n];rows.append('|'+n+'|'+f"{100*t['BA']:.6f}"+'|'+'/'.join(map(str,t['correct']))+'|')
 cats={k:sum(a['F_reference_paths'][str(f)][2]['categories'][k] for f in [0,1]) for k in a['F_reference_paths']['0'][2]['categories']}
 write('POOLED_CLASS2_PATH.json',dict(class_index=2,F_original_correct=64,intermediate_F_on_T_final_T_on_T=cats,path_descriptive_not_causal=True))
 text='''# 排组query的2×2交叉读出审计

结论：**固定参考读出未恢复query风险**。REFERENCE_RESTORES_CLASS2/BA及COMPLETE_REFERENCE_RESTORATION_POINT_PATTERN均false，FIXED_REFERENCE_CLASS2_HARM=true。存在部分BA改善，不能因“完全恢复”flag为false将其抹掉；本轮只是描述，任何组合不升主，原方法停门不变。

记法head_on_feature；每行只在未参与该producer训练的query组上评分，两fold合并同2555行。

|组合|query BA %|各类正确数0/1/2/3|
|---|---:|---|
'''+ '\n'.join(rows)+'''

分母307/20/170/2058。两头都用各fold同一个support-only F mapper，经exact核对，未拟合新norm或头。F_on_F/T_on_T回放前轮F_out/T_out的逐类计数，分数误差为0。

## 固定参考也看到的损害

F头固定时，换为T特征：BA64.713626→58.271367，下降6.442258pp；类2 64→48（−9.411765pp）、类1 13→9，类0少37，类3多323。类2救回12、损害28，其中23个损害转向类3。两fold类2分别34→20和30→28，均未保持原参考正确数，不只是一个fold反向。

T头固定时，F特征换为T特征：BA64.342340→56.634201；类2 78→54，类1 12→6。两条读出路径都显示原F特征上的类2风险较低。该陈述只对这两个预定头和这些已排除组成立，不证明任何其他读出无法恢复信息或不可逆信息丢失。

## 更换头仍有作用，但不是单一瓶颈

同T特征换回F头，BA由56.634201升至58.271367（+1.637166pp），但类2由54降至48、类1由6升9，类0少26、类3多73。这是不同类之间的重新分配，不能说换头完全没作用，也不能称尾类恢复。

反向T_on_F类2为78/170，高于F_on_F的64；整体BA仍低.371286pp，类1少1、类0少13。它是事后诊断组合，不能择类2优势升为新主候选或给旧C追认成功。

在F原正确64个类2query中，经F_on_T中间状态再到T_on_T：中间错且最终错24，中间对最终错3，中间错最终对4，二者都对33。原最终27个新增损害中24在固定参考中已错，另3只在换头后错；同时头还救回4个中间损害。此分类依赖中间路径，**不能把24/27或3/27称表示/头的因果比例**；各自救回集合也不能只凭计数当嵌套。

## Margin分解与解释边界

全部2fold×4class×3固定竞争类=24单元都报告F→固定F读T特征→T终点路径，以及先换头再换特征的反向路径；全类、F原正确、最终新增损害三个子集均保留。两路径加和与Δz·ΔW interaction恒等式最大误差'''+f"{a['algebra_max_error']:.3g}"+'''，合成检验亦通过。原始logit尺度不同，代数分解不是干预识别或因果强度排序。未新加bootstrap、显著性或推断家族。

这比旧D在681开发上的交叉读出多了整个producer排组边界：两个参考头/mapper只由各support训练，query没有参与encoder/头适配。本次仍条件于固定两个producer，研究历史使用过这些组，不是外部独立测试。原681开发/future/test/rawimages均未读取或评分。

## 下一步决策

目前没有充分依据把损害缩减成logit校准问题，暂停把“仅改头/加固定F惩罚”作为自动后继。旧P固定头及D方向保护停门保持，不把本轮固定参考损害反过来当加大保护系数的理由。

下一阶段先核无需任务encoder更新的新表示证据：阅读SpatialMoment/中间token相关原文并对照既有G1/层/池化负结果，辨明是否真有未测试的空间信息问题。任何新表示先在训练内已知组排除条件下与冻结F和强简单控制比较，不用训练margin或新logit大尺度替代组外收益。当前只确定此原文/可识别性核查优先级，未授权把已停止h拼接、核、fixed-U、bias旧端点改名重跑；形成不同可证伪问题及独立协议/预算前不启动新训练或编码。

## 完成、成本与隐私

独立协议/源27b3556ec09f45c6daf72edf610ed921e7b7469f在新交叉分数读取前封存。4合并表/8fold表/24margin单元、原两端点分数0误差、mapperexact、query覆盖恰好一次/finite/R-H恒等式全过。0encoder前向/0拟合/0更新，GPU未初始化；父1019744/子1019745退出0退休，无失败/重跑。

活动wall'''+f"{c['activity_wall_seconds']:.6f}"+'''s、wait4CPU'''+f"{c['known_worker_wait4_CPU_seconds']:.6f}"+'''s、协调CPU'''+f"{c['coordinator_CPU_seconds']:.6f}"+'''s一次并账；worker正文不重复加。RSS666968064<4GiB，round283687+16MiB<128MiB，window13619254390+1001040384，报告增长另16MiB<17/20GB；600秒包络未触发。历史CPU及准备/网络/报告整理缺口未知；未删旧资产。

源码协议全聚合分解/成本/中文结果公开；模型、特征、逐例、组身份、原私有日志、论文PDF私有。完成交付以远端SHA/匿名报告与本地NFS双端DELIVERY为准。
'''
 (out/'ANALYSIS_ZH.md').write_text(text);(out/'REPORT_ZH.md').write_text(text)
 return c,cum

if __name__=='__main__':
 c,cum=finalize(Path(__file__).resolve().parent);print(json.dumps(dict(cost=c,cumulative=cum),indent=2))
