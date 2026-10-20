"""Build aggregate-only completion reports; no feature/model loading or fitting."""
import json
from pathlib import Path


def run(root):
    root = Path(root)
    out = root/'results'
    def read(name):
        return json.loads((out/name).read_text())
    def write(name, data):
        (out/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    a = read('aggregate.json')
    d, guard = read('DRIVER_EXIT.json'), read('HARD_DEADLINE_EXIT.json')
    clock, pre = read('ROUND_CLOCK.json'), read('PRECHECK_COST.json')
    enc = read('train_ENCODER_SEAL.json')
    train = read('train_partial.json')['arms']
    assert enc['steps_each'] == 800 and enc['P_head_fixed_exact']
    assert [v['arm'] for v in train] == ['P', 'C']
    assert all(v['steps'] == 800 and v['supervised_slots'] == 25600 for v in train)
    assert all(read(s+'_EXIT.json')['exit_code'] == 0 for s in ['train', 'fit'])
    assert len(d['phases']) == 2 and all(v['worker_exit'] == 0 and v['stop_reason'] is None for v in d['phases'])
    assert not guard['forced'] and not guard['driver_retired_without_final_receipt']
    assert guard['completed_unix'] < clock['deadline_unix']
    assert len(a['fits']) == 6 and len(a['artifact_bytes']) == 4
    assert all(v['status'] == 'CERTIFIED' and v['residual'] <= 1e-7 for v in a['fits'])
    assert all(v <= 19058800 for v in a['artifact_bytes'].values())
    assert a['feature_rows'] == {'P': 3236, 'C': 3236}
    assert read('fit_MODEL_SEAL.json')['total_fits'] == 6
    for arm in ['P', 'C']:
        seal = read('fit_'+arm+'.FEATURE_SEAL.json')
        assert seal['rows'] == 3236 and seal['one_pass']
    assert not a['future_used'] and not a['test_used'] and not a['method_success']
    assert not a['development_gate']['pass_all'] and not a['mechanism_bootstrap_performed']
    assert a['summary']['F']['correct'] == [55,2,22,430]
    assert a['summary']['L']['correct'] == [56,2,21,474]
    # Successful frozen pipeline certifies finite features, exact bias/W/A round trips,
    # six ridge residuals, single-pass seals and initial prediction parity before EXIT0.
    ver = dict(steps={'P':800,'C':800}, supervised_slots=51200,
        actual_ridge_fits=6, actual_artifacts=a['artifact_bytes'], feature_rows=a['feature_rows'],
        all_feature_and_model_seals_present=True, F_L_count_and_prediction_parity=True,
        frozen_pipeline_assertions_reached_EXIT0=True,
        feature_finite_and_W_bias_score_A_roundtrip_certified_by_pipeline=True,
        max_ridge_residual=max(v['residual'] for v in a['fits']),
        all_driver_coordinator_worker_guard_exited=True,
        retired_pids=[999469,999470,999474,999925,1005195,1005196],
        live_ps_empty_and_GPU_no_round_worker=True,
        no_unplanned_failure_receipts=True, planned_injection_failure_retained=True,
        bootstrap_status='NOT_RUN_POINT_GATE_FAILED', CI_status='NA_NOT_ZERO',
        future_used=False,test_used=False,method_success=False,
        development_not_independent=True, decision=a['decision'])
    write('VERIFICATION.json', ver)
    worker = pre['known_worker_CPU']+sum(v['worker_CPU_seconds'] for v in d['phases'])
    coord = pre['coordinator_CPU']+sum(v['coordinator_CPU_seconds'] for v in d['phases'])+d['driver_CPU_seconds']+guard['CPU_seconds']
    wall = pre['precheck_wall']+d['wall_seconds']
    last=d['phases'][-1]
    cost=dict(activity_wall_seconds=wall, known_worker_wait4_CPU_seconds=worker,
        coordinator_driver_guard_CPU_seconds=coord,
        formal_driver_wall_seconds=d['wall_seconds'], precheck_wall_seconds=pre['precheck_wall'],
        hard_guard_CPU_seconds=guard['CPU_seconds'], guard_overlapping_wall_not_added=True,
        worker_body_wall_and_CPU_not_added_again=True,
        original_first_actual_to_guard_retirement_seconds=guard['completed_unix']-clock['first_actual_unix'],
        budget_seconds=10800,worker_peak_RSS_bytes=max(v['worker_peak_RSS_bytes'] for v in d['phases']),
        worker_RSS_limit_bytes=12*2**30,
        peak_GPU_allocated_bytes=max(read(s+'_WORKER_RESOURCE_RECEIPT.json')['peak_GPU_allocated_bytes'] for s in ['train','fit']),
        exit_round_remote_bytes=last['round_bytes'], round_outside_reserve_bytes=16*2**20,
        round_total_with_reserve=last['round_bytes']+16*2**20,round_limit_bytes=2**30,
        window_remote_bytes=last['window_bytes'],default_cache_bytes=1040384,local_Git_reserve_bytes=1000000000,
        window_total_conservative=last['window_bytes']+1001040384,
        subsequent_small_report_growth_reserve_bytes=16*2**20,
        window_soft_stop_bytes=17000000000,window_hard_limit_bytes=20000000000,
        preparation_network_and_report_formatting_CPU_not_fully_measured=True,
        historical_CPU_gaps_unknown=True, no_old_assets_deleted=True)
    assert cost['round_total_with_reserve'] < 2**30
    assert cost['window_total_conservative']+16*2**20 < 17000000000
    assert cost['original_first_actual_to_guard_retirement_seconds'] < 10800
    write('COST_RECEIPT.json',cost)
    cumulative=dict(activity_wall_seconds=23259.597153794806+wall,
        known_worker_wait4_CPU_seconds=31536.761061999998+worker,
        coordinator_driver_guard_CPU_seconds=870.021975+coord,
        this_round_counted_once=True, historical_CPU_gaps_unknown=True,
        network_preparation_report_CPU_incomplete=True)
    write('CUMULATIVE_COSTS.json',cumulative)
    tab=a['summary']
    contrasts={}
    for lhs,rhs in [('P','Q'),('C','R'),('P','C'),('P','F'),('P','L')]:
        contrasts[lhs+'-'+rhs]=dict(BA_pp=100*(tab[lhs]['BA']-tab[rhs]['BA']),
            class2_pp=100*(tab[lhs]['recall'][2]-tab[rhs]['recall'][2]),
            net_correct=[x-y for x,y in zip(tab[lhs]['correct'],tab[rhs]['correct'])],
            descriptive_only=True,posthoc_inference_added=False)
    write('PRO_CONTRASTS.json',contrasts)
    rows='\n'.join('|'+k+'|'+f"{tab[k]['BA']*100:.6f}"+'|'+'/'.join(map(str,tab[k]['correct']))+'|' for k in ['F','L','P','C','Q','R'])
    text=f'''# READOUT_CONSISTENT_CE：结果与机制解释

结论：**STOP_READOUT_CONSISTENT_DEVELOPMENT**。P/C各800步及一次终点编码、6个ridge解和4个完整artifact均完成；原点门失败，不执行预设bootstrap，不追加种子、步数、alpha或参数救援，不择C/Q/R/L升主。method_success=false；仅已暴露初始开发，非独立确认，无future/test/J。

|端点|开发BA %|各类正确数0/1/2/3|
|---|---:|---|
{rows}

开发分母68/3/38/572。Q选alpha=.0001，R选.01；三个预设alpha全部结果在aggregate.json，不择其他网格。

## 与本次Pro分析逐项对应

1. **P−Q：BA −0.077129pp，类2相同15/38，逐例正确/错误集合也相同。** 保留固定训练头并未改善类2，相比事后refit整体近乎持平。P相对Q类0多3例、类3少27例。不能将旧D端点F_on_D的事后优势外推为新P训练收益。
2. **C−R：BA +1.092657pp，类2同为20/38，类0/1相同，类3多25例。** 这是描述性点差；C−R不在预设10项推断家族中，未追加显著性检验。类2计数相同本身不证明逐例相同。
3. **P−C：BA −4.968310pp，类2少5例。** 两者都训练部署一致，本轮固定F读出未优于联合学习头。C比P另多1例类0和30例类3，类1同1/3；不能误写C目标不一致。
4. **P−F/L：BA分别 −10.902397/−12.535227pp。** P对F类2救回1、损害8，净−7；损害4例转类0、4例转类3。类1还损害1例（3例中），类3净增加55例不能掩盖该损失。P唯一主候选不具备竞争力。

因此“训练与部署用同一个头”在此冻结配置下不足以保证开发风险改善。旧D读出审计发现的换头损害仍是真实描述，但不是所有训练失败的共同解释。本轮不证明固定分类器在所有任务无效，也不证伪SADP/DSDR整体。

## 训练信息与解释边界

P采样首/末100批CE均值 .722085/.306059；C为 .521120/.071352。更低采样训练CE并未使任何新端点超过F/L。它们是既定日程中的批次均值，不是全2555训练风险，更不证明具体过拟合机制。类1开发只有3例；反复使用同681开发限制独立科学推断。

固定头仅固定映射参数，不固定encoder输出或预测。未来扩类仍须独立协议；旧列不动也可能被新列抢走。P/C当前A/B不导出其W，这是明确设计，不能当正规方程错误修复；Q/R六解残差均合格，最大{ver['max_ridge_residual']:.3g}。

## 下一步判断

停止把“更换部署读出”作为单一瓶颈继续救援P或旧D。下一独立问题优先是：适配后少数组的训练收益是否能在未参与该适配的组上保持，以及实际类条件梯度支持是否足够稳定。先在已封存产物上设计训练组/开发组的margin与风险审计，并核对已有A2的监督暴露边界；若需训练确认，producer必须真正排除query组，不能把全训练encoder上的小支持ridge当encoder级crossfit。不得用开发受损样本选择保护方向。

这是后继可证伪问题的优先级，不是本轮新增训练或SADP复现。先完成原文实现/可识别性核查与独立协议，明确强简单控制、训练内选择来源、完整家族、停止门和预算再运行。不把共同fixed-U/核/bias旧负端点改名复活，不因本轮失败暂停整个研究路线。

## 验收、预算与交付

2×800、51200监督槽、2×3236行单次FEATURE_SEAL、6解、4完整artifact全部通过；实际容量P/C各10076885、Q/R各10064400字节，均≤19058800。F/L预测parity、finite和完整bias/W/A/score回读由冻结pipeline断言及正常EXIT0认证。全部头先MODEL_SEAL后风险汇总；未盲重编码。driver/TRAIN/FIT退出0、hardguard未强制、六个本轮PID均退休；无新增实际失败，预定注入失败保留。

首预检至guard退休{cost['original_first_actual_to_guard_retirement_seconds']:.3f}s <10800s；本轮含预留{cost['round_total_with_reserve']}bytes <1GiB。窗口保守{cost['window_total_conservative']}bytes，另16MiB小报告余量，低于17/20GB。peakRSS{cost['worker_peak_RSS_bytes']}bytes <12GiB。新增活动墙钟{wall:.6f}s、knownworker/wait4 CPU{worker:.6f}s、协调/driver/guard CPU{coord:.6f}s，一次并账；guard重叠等待与worker正文不重复相加。历史CPU缺口及本机准备/网络/报告整理CPU未完整计量。

科学源493152ef22f13656af8e27fc1b7b5eacf9084e7c；协议f56b173。源码/协议/全部聚合/成本/失败摘要公开；特征、模型、逐例预测、组身份、原私有日志与第三方PDF保留私有。

参考用户本次[Pro分享分析](https://chatgpt.com/share/6ad747bb-b77c-83ea-b079-368b6c3552f5)；上面数值来自本项目冻结回执，不借分享意见代替实验结果。
'''
    (out/'ANALYSIS_ZH.md').write_text(text)
    (out/'REPORT_ZH.md').write_text(text)
    return cost,cumulative

if __name__=='__main__':
    import sys
    cost,cumulative=run(Path(__file__).resolve().parent)
    print(json.dumps(dict(cost=cost,cumulative=cumulative),indent=2))
