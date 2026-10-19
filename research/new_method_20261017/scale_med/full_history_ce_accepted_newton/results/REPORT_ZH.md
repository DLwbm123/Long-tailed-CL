# 完整历史CE接受点验收与Newton-CG修复独立运行结果

同原冻结G/P/C表示、原永久训练与验证划分；CE是特权完整历史参照，不满足退休部署约束，不是BA上界或新方法。原内部测试已研究暴露。

退出回执：worker=1；coordinator={'worker_exit': 1, 'stop_reason': None, 'wall_seconds': 519.1016123481095, 'worker_CPU_seconds': 1780.62505, 'worker_peak_RSS_bytes': 2213859328, 'coordinator_CPU_seconds': 43.483059999999995, 'worker_usage_scope': 'entire_child_including_imports', 'source_commit': '53f01d75bd3c1ccb88392f6e078250f3c7fee755'}。

科学未决。未以缺失结果、未收敛或工程失败充当CE无效证据；不追加预算或重试。

本轮真实数据已完成优化单元数：15。

本轮未完整形成科学比较。实际阶段、已完成单元和失败原因以partial、FAILURE及退出回执为准；不把工程失败当CE无效。

## 失败及成本回执

```json
{
  "failure": {
    "error": "TypeError",
    "phase": "SCIENTIFIC_PIPELINE",
    "decision": "UNRESOLVED_ENGINEERING_NO_EXTENSION",
    "detailed_selfcheck_receipt": true,
    "detailed_fit_receipts": [
      "FIT_C_1_0.0001.json",
      "SELFCHECK_FIT.json",
      "FIT_P_0_0.0001.json",
      "FIT_P_2_0.0001.json",
      "FIT_C_2_0.0001.json",
      "FIT_P_1_0.0001.json",
      "INTERRUPTED_FIT.json",
      "FIT_G_2_0.0001.json"
    ],
    "no_retry": true
  },
  "cost": null
}
```

## 已保存的优化结果

|臂|阶段|lambda|复用|状态|原W梯度max|Newton步|总算子（新增）|
|---|---:|---:|---|---|---:|---:|---:|
|G|0|0.0001|True|CONVERGED|3.46452584e-08|继承|继承|
|G|0|0.01|True|CONVERGED|5.87081094e-08|继承|继承|
|G|0|1.0|True|CONVERGED|2.89747062e-09|继承|继承|
|P|0|0.0001|False|CONVERGED|2.04069643e-08|1|1570|
|P|0|0.01|True|CONVERGED|6.24682684e-08|继承|继承|
|P|0|1.0|True|CONVERGED|2.26212089e-08|继承|继承|
|C|0|0.0001|True|CONVERGED|2.40867312e-08|继承|继承|
|C|0|0.01|True|CONVERGED|2.80673374e-08|继承|继承|
|C|0|1.0|True|CONVERGED|3.68051707e-08|继承|继承|
|G|1|0.0001|True|CONVERGED|8.66040836e-08|继承|继承|
|P|1|0.0001|False|CONVERGED|2.22579576e-08|1|1565|
|C|1|0.0001|False|CONVERGED|2.20594485e-08|1|1537|
|G|2|0.0001|False|CONVERGED|8.15729373e-08|1|1559|
|P|2|0.0001|False|CONVERGED|5.34016214e-08|1|1572|
|C|2|0.0001|False|CONVERGED|7.90789833e-08|1|1547|

15/15优化单元认证并不代表完整科学交付成功：最终aggregate序列化失败；主风险表、区间、门未保存，禁止填为通过或失败。

## 原协议对齐

G/P/C原完整特征维度1535 / 1375+128 / 1318+128；训练前缀2555/4367/4395；初始681验证选择，各臂lambda均0.0001。九项训练统计parity最大2.84217e-14。MODEL_SEAL在测试之前形成。最终测试得分parity数值未保存，不声称有完整逐项回执。

## 完整成本

```json
{
  "total_authorization_wall_cap_seconds": 1800,
  "total_authorization_CPU_cap_seconds": 3600,
  "engineering_wall_seconds": 49.61905137170106,
  "engineering_CPU_seconds": 36.833177,
  "formal_coordinator_wall_seconds": 519.1016123481095,
  "formal_worker_CPU_seconds": 1780.62505,
  "formal_coordinator_CPU_seconds": 43.483059999999995,
  "total_activity_wall_seconds": 568.7206637198105,
  "total_measured_CPU_seconds": 1860.941287,
  "wall_pass": true,
  "CPU_pass": true,
  "RSS_peak_bytes": 2213859328,
  "RSS_cap_bytes": 8589934592,
  "peak_GPU_allocated_bytes": 739425280,
  "GPU_allocated_cap_bytes": 4294967296,
  "GPU_residence_conservative_upper_seconds": 519.1016123481095,
  "GPU_residence_cap_seconds": 1800,
  "GPU_residence_is_billing_measurement": false,
  "measured_directory_bytes_before_reports": 69696266,
  "directory_cap_bytes": 268435456,
  "nested_worker_boot_pipeline_walls_not_added": true,
  "main_COST_RECEIPT_missing": true,
  "cost_source": "wait4 child plus coordinator; boot fallback resources; engineering ledger"
}
```

失败前通用资源回执和父进程wait4保留。正文、boot与协调墙钟是包含关系，不相加。累计成本见CUMULATIVE_COSTS.json；历史失败CPU缺失，累计总CPU仍未知。
