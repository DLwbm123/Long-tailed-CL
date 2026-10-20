"""Create aggregate-only stopped-run report from public resource receipts."""
import json
from pathlib import Path

def finalize(receipts, previous_costs, output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    coord=receipts['COORDINATOR_EXIT.json'];worker=receipts['WORKER_EXIT.json']
    assert worker['exit_code']==coord['exit_code']==1
    assert receipts['WORKER_FAILURE.json']['type']=='AssertionError'
    costs=dict(activity_wall_seconds=coord['wall_seconds'],known_worker_wait4_CPU_seconds=coord['worker_CPU_seconds'],coordinator_CPU_seconds=coord['coordinator_CPU_seconds'],peak_RSS_bytes=coord['worker_peak_RSS_bytes'],peak_GPU_allocated_bytes=receipts['WORKER_RESOURCE_RECEIPT.json']['peak_GPU_allocated_bytes'],round_remote_bytes=coord['round_bytes'],round_total_with16MiB_reserve=coord['round_bytes']+16*2**20,window_remote_bytes=coord['window_bytes'],window_total_conservative=coord['window_bytes']+1001040384,body_worker_resource_not_added_twice=True,historical_CPU_and_preparation_gaps_unknown=True)
    cumulative={key:previous_costs[key]+costs[key] for key in ['activity_wall_seconds','known_worker_wait4_CPU_seconds','coordinator_CPU_seconds']}
    aggregate=dict(decision='UNRESOLVED_SINGLE_IMAGE_FEATURE_PARITY',method_success=False,scientific_gradient_agreement_result_available=False,planned_cells=8,planned_groups=64,selected_training_images=146,completed_cell_coverage=None,actual_gradient_direction_metrics=None,failed_condition='single_image_feature_max_abs_error <= 1e-7',exact_failed_error_value=None,exact_failed_error_value_not_persisted=True,full_batch_replay_guard_passed_by_control_flow=True,exact_full_batch_replay_error_not_persisted=True,synthetic_passed=True,new_ridge_fits=0,optimizer_steps=0,development_used=False,future_used=False,test_used=False,automatic_retry=False)
    verification=dict(worker_and_coordinator_exit1=True,all_round_PIDs_retired=True,GRADIENT_SEAL_present=False,PREFLIGHT_success_receipt_present=False,final_parameter_version_check_reached=False,resource_guards_triggered=False,scientific_acceptance=False,failure_preserved=True)
    for name,value in [('aggregate.json',aggregate),('VERIFICATION.json',verification),('COST_RECEIPT.json',costs),('CUMULATIVE_COSTS.json',cumulative)]:
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    for name in ['COORDINATOR_EXIT.json','WORKER_EXIT.json','WORKER_FAILURE.json','FAILURE.json','WORKER_RESOURCE_RECEIPT.json','SYNTHETIC.json','PRIVATE_PLAN_SEAL.json','GPU_AVAILABILITY.json']:
        (out/name).write_text(json.dumps(receipts[name],indent=2)+'\n')
    text='''# GROUP_GRADIENT_AGREEMENT：逐图特征预检未通过，梯度一致性未决

结论为 **UNRESOLVED_SINGLE_IMAGE_FEATURE_PARITY**，不是类1梯度方向不一致，也不是新的性能负结果。独立协议/科学源ace48f458bc79131fe2f4441b3e5b25acc63413d；既有source e5dcf26之后、任何实际运行之前，已按已知FP32 micro差异区分32图回放与逐图舍入门；实际失败后没有改阈值或重跑。

## 完成与缺失

合成parallel/antiparallel/orthogonal/zero检查通过。固定组清单先封存：两fold每类8组，共64组、146张训练图。没有新拟合、优化器或参数更新，也没有query/开发/future/test评分。

实际32图G回放检查先执行且未抛错；随后首组逐图特征与封存cache差异未满足预定max_abs<=1e-7，AssertionError退出。逐图数值差和此前32图差的具体数值没有在异常前落盘，因此均为NA；不能编造误差大小，也不能把控制流推断写成完整数值回执。PREFLIGHT成功回执和GRADIENT_SEAL均缺失，8cell梯度方向指标/覆盖未认证，最终参数版本检查没有执行到。代码无optimizer.step，不等于已做完退出后exact版本认证。

worker与独立协调均exit1，父子进程已经退休。没有达到时间、存储或RSS门，未触发强制终止；私有traceback保留，公开只发布错误条件与类型。原始1800秒包络没有重置，不追加实际预检或重算。

## 解释和停止

本轮不能回答少数组真实参数梯度是否稳定，更不能从未决结果启动方向保护、SADP模块或旧CE-bias端点。FP32 batch差异是已知可能影响因素，本轮没有定位超过阈值的唯一原因；既不称历史cache损坏，也不称梯度机制失败。未来若研究计算一致性，须有不同且明确的工程问题与独立授权/协议，不以改名绕过本轮无重试门。本轮科学结果仍缺失，method_success=false。

本轮只用初始训练与两个support-only F teacher；固定抽取之外不按梯度或风险挑组。其他类与z-gradient对照尚无完整结果，不挑部分均值作结论。旧CE-bias、读出救援和所有历史停门原样保留。

## 成本与资源

协调活动wall45.904416秒，wait4 worker及后代CPU47.520087秒，协调ownCPU5.231591秒；worker正文wall34.627844/CPU44.788606已在上述口径内，不重复相加。RSS3555631104bytes<12GiB，GPUallocated1018309632bytes；启动GPU0free40445MiB，NFS4实际挂载/容量/写读probe核。启动时完整父子中性argv已现场确认；活动GPU进程名没有捕获，不为了补观察回执重跑。

最终扫描remote round68908bytes，含16MiB外部预留16846124<128MiB；window13660799818+default1040384+localGit1GB=14661840202bytes，报告另16MiB余量仍<17/20GB。无模型/像素/环境复制或历史删除。读取、准备、网络、报告CPU与历史缺口仍未完整计量；累计见CUMULATIVE_COSTS，不能称为总研究CPU。

公开源码/协议/合成与计划封存元数据/失败/资源/成本。64组清单、逐组Gram、图像、身份、特征、模型和原始私有traceback均不下载公开。这里只有计划封存元数据，没有完成的逐组Gram。整体研究未暂停，唯一30min心跳继续；本轮退休且禁止重跑。
'''
    (out/'ANALYSIS_ZH.md').write_text(text);(out/'REPORT_ZH.md').write_text(text)
    return aggregate,costs,cumulative
