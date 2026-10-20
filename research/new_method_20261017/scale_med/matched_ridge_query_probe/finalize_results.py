"""Aggregate-only acceptance/report builder; does not solve or load models."""
import json
from pathlib import Path

def finalize(root,allocation):
 out=Path(root)/'results'
 def read(n):return json.loads((out/n).read_text())
 def write(n,v):(out/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 a=read('aggregate.json');r=read('COORDINATOR_EXIT.json');wr=read('WORKER_RESOURCE_RECEIPT.json')
 assert r['exit_code']==0 and r['stop_reason'] is None and r['completed_unix']<r['deadline_unix'] and not wr['GPU_initialized']
 assert a['actual_ridge_solves']==2 and len(a['fits'])==2 and all(v['alpha']==.01 and not v['mapper_reestimated'] and v['residual']<=1e-7 for v in a['fits'])
 assert a['encoder_forwards']==a['mapper_fits']==a['parameter_updates']==0
 assert a['exact_original_count_parity'] and a['original_score_parity_max_error']<=1e-8 and a['support_stat_max_error']<=1e-10
 assert all(v<=19058800 for v in a['artifact_bytes'].values()) and a['decision']=='STOP_MATCHED_RIDGE_NONCOMPETITIVE'
 assert all(read(k+'.HEAD_SEAL.json')['query_excluded_from_fit'] for k in ['0','1'])
 assert read('MODEL_SEAL.json')['actual_ridge_solves']==2
 assert not any(a[k] for k in ['method_success','bootstrap_performed','development_used','test_used','future_used'])
 assert r['round_bytes']+16*2**20<128*2**20 and r['worker_peak_RSS_bytes']<4*2**30
 assert max(v['logical'] for v in allocation)<16*2**20
 v=dict(actual_ridge_solves=2,alpha=.01,mapper_fits=0,encoder_forwards=0,parameter_updates=0,query_rows=2555,folds=2,
  original_F_CE_count_parity=True,score_parity_max_error=a['original_score_parity_max_error'],support_stat_max_error=a['support_stat_max_error'],
  F_residual_without_resolve=a['F_residual'],probe_residuals=[x['residual'] for x in a['fits']],
  two_head_seals_and_MODEL_SEAL=True,all_finite=True,full_artifact_roundtrip_certified_by_EXIT0=True,
  complete_per_fold_and_all_classes_and_R_H=True,retired_pids=[1022275,1022279],exit_zero=True,no_failure=True,
  no_bootstrap_added=True,method_success=False,decision=a['decision'])
 write('VERIFICATION.json',v);write('ALLOCATION_RECEIPT.json',allocation)
 c=dict(activity_wall_seconds=r['wall_seconds'],known_worker_wait4_CPU_seconds=r['worker_CPU_seconds'],coordinator_CPU_seconds=r['coordinator_CPU_seconds'],worker_peak_RSS_bytes=r['worker_peak_RSS_bytes'],round_remote_bytes=r['round_bytes'],external_reserve_bytes=16*2**20,round_total_with_reserve=r['round_bytes']+16*2**20,round_limit_bytes=128*2**20,prelaunch_estimated_remote_bytes=32*2**20,estimate_exceeded=True,
  largest_file_logical_bytes=max(x['logical'] for x in allocation),RLIMIT_FSIZE_logical_bytes=16*2**20,largest_file_allocated_bytes=max(x['allocated'] for x in allocation),allocated_bytes_charged_in_total=True,
  wall_limit_seconds=600,RSS_limit_bytes=4*2**30,window_remote_bytes=r['window_bytes'],window_total_conservative=r['window_bytes']+1001040384,small_report_growth_reserve_bytes=16*2**20,worker_body_not_added_again=True,historical_CPU_gaps_unknown=True,preparation_network_report_CPU_incomplete=True,no_old_assets_deleted=True)
 write('COST_RECEIPT.json',c)
 cum=dict(activity_wall_seconds=31488.496487201046+c['activity_wall_seconds'],known_worker_wait4_CPU_seconds=39940.236443+c['known_worker_wait4_CPU_seconds'],coordinator_CPU_seconds=1197.574893+c['coordinator_CPU_seconds'],this_round_counted_once=True,historical_CPU_gaps_unknown=True,preparation_network_report_CPU_incomplete=True)
 write('CUMULATIVE_COSTS.json',cum)
 def row(label,z):return '|'+label+'|'+f"{100*z['BA']:.6f}"+'|'+'/'.join(map(str,z['correct']))+'|'+f"{100*z['group_balanced_BA']:.6f}"+'|'
 rows='\n'.join(row(n,a['summary'][n]) for n in ['F','T_CE','T_probe'])
 folds='\n'.join(row('fold'+k+' '+n,v[n]) for k,v in a['per_fold'].items() for n in ['F','T_CE','T_probe'])
 ba=100*(a['summary']['T_probe']['BA']-a['summary']['F']['BA'])
 text='''# 两个匹配ridge探针：完成与下一步

结论：**STOP_MATCHED_RIDGE_NONCOMPETITIVE**。固定原support-only F mapper、alpha=.01和自由截距，两个匹配T探针在合并及两折BA上均低于F，类1未恢复。已满足预设停止分支，停止为这套适配配方的现有终点继续搜索读出。不追加第三个fit、mapper重估、alpha网格或CE步数；不是整个研究暂停。

## 合并结果，类1优先

|端点|图像级BA %|各类正确数0/1/2/3|类组等权BA %（敏感性）|
|---|---:|---:|---:|
'''+rows+'''

合并分母307/20/170/2058。主指标仍图像级BA，不用更好看的加权口径替代；组均衡敏感性也同向变差。

类1：F13/20→probe5/20（召回−40pp），对F救回0、损害8，F原正确仅保留5。原CE头为6/20，探针进一步少1。八个新增损害去向类0为3、类2为1、类3为4，不能泛称都被吸到类3。

类2：F64/170→probe42/170（−12.941176pp），救回12、损害34，净−22。原CE为54/170，probe少12。类0净−11、类3净+236；类3收益不能覆盖类1门失败。合并BA对F'''+f"{ba:.6f}"+'''pp，四项点门均false；这不是模型训练/求解失败，是完整科学负结果。

## 两折完整结果

|端点|图像级BA %|各类正确数0/1/2/3|类组等权BA %|
|---|---:|---:|---:|
'''+folds+'''

fold0 query分母151/8/74/993，fold1为156/12/96/1065。类1分别F8→probe3和F5→probe2，损害5/3且无救回；类2分别34→21和30→21，两折均下降。合并指标从四类计数重算，没有简单平均fold BA。

## 确实匹配了什么

只求解两次。新probe使用原F mapper、同类/组权p=q/4、同六列/自由截距、不罚bias、固定.01；没有再拟合T mapper或扫描alpha。旧T artifact中已存的support A/B未再次除4，先用固定5坐标支持统计核对，误差'''+f"{a['support_stat_max_error']:.3g}"+'''；F旧W只做正规方程残差核对，没有重solve。

两个真实probe残差'''+str([v['residual'] for v in a['fits']])+'''，finite、完整A/B/W/bias回读均通过；两HEAD_SEAL和MODEL_SEAL之后才评分query。原F/CE端点分数回放误差0、计数exact。没有encoder前向、反向更新或mapper拟合，GPU未初始化；synthetic零solve，故总数值solve调用也确实只有2次。

两个完整probe artifact逻辑长度各10076821<=19058800，含继承的absolute bias/mapper/PERM/三角A/B/W和元数据，没有以只存W规避模型容量。

## 如何更新判断

此前固定F头/T头交叉都没有恢复类1；本轮再用预定的匹配ridge规则拟合T特征，仍然更差。现在可以停止这套CE-bias适配配方的读出救援，而非继续把损害简化为头不匹配。它仍不能证明T特征的有用信息不可逆消失，或所有可能读出都无效。

Pro建议的有限诊断顺序已经完成：同producer同mapper交叉读出、两个固定匹配probe。没有把后来提出的4项bootstrap补成旧预设；本轮没有显著性检验，也不宣称总体非劣或独立方法成功。所有query组在研究历史中已暴露，method_success=false。

下一步保留冻结F为强基线，先核真正尚未测试的冻结空间证据问题：对照SpatialMoment/中间token原文及旧G1/层/池化负结果，明确signed ViT能否使用相应统计量。只有独立假设、训练内选择来源、整链排组/强简单控制、类1风险门及硬预算封存后，才启动新编码或训练。不会因本轮失败就直接增大保护项、换旧头或增加epoch/seed。若没有不同且可识别的问题，记录未决并继续原文核查，不拼模块强造原创。

## 资源、异常与交付

CPU父1022275/子1022279均exit0退休，失败回执无，2fit/2artifact/两HEAD_SEAL/MODEL_SEAL、2555query覆盖、全类两fold/配对混淆/点门验收通过。活动wall'''+f"{c['activity_wall_seconds']:.6f}"+'''s、wait4CPU'''+f"{c['known_worker_wait4_CPU_seconds']:.6f}"+'''s、coordCPU'''+f"{c['coordinator_CPU_seconds']:.6f}"+'''s一次并账，正文CPU不重复加；历史及准备/网络/报告CPU缺口未知。

RSS779407360<4GiB、600秒包络未触发。实际远端新增40429187字节，高于启动前<32MiB估算：两个artifact各逻辑10076821，但各allocated20168704，按max(logical,allocated)全计新增，未按逻辑长度少算。source的RLIMIT_FSIZE是单文件逻辑长度16MiB门，两文件逻辑均通过；分配块占用另全计总量，不隐瞒其高于16MiB。总新增40429187+16MiB=57206403<128MiB硬限，窗口13660467488+1001040384，报告增长另16MiB<17/20GB。未改预算、重打包或删除旧资产。ALLOCATION_RECEIPT保留这次低成本疑点核对。

协议/实际源e78b9d371308194ccaa8ae44917b4e3a8a1197df先封存后求解。公开源码/协议/所有聚合（全类/两fold/组权敏感性/混淆/R-H）/成本/失败摘要，模型/feature/组身份/逐例/原私有日志/论文PDF私有。最终以远端SHA/匿名报告及DELIVERY双端回执确认公开交付。
'''
 (out/'ANALYSIS_ZH.md').write_text(text);(out/'REPORT_ZH.md').write_text(text)
 return c,cum

if __name__=='__main__':
 import sys
 c,cum=finalize(Path(__file__).resolve().parent,json.loads(Path(sys.argv[1]).read_text()));print(json.dumps(dict(cost=c,cumulative=cum),indent=2))
