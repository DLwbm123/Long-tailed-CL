# 四条真实零更新 GPU 预检结果

FOUR_PREFLIGHT_PASS_STOPPED_NO_TRAIN

| 项目 | 状态 | GPU驻留秒 | allocated GiB | reserved GiB | 梯度范数 | 最大旧类/约束 |
|---|---|---:|---:|---:|---:|---:|
| pre_ISIC_guard | PASS | 569.213007 | 11.9269199 | 12.6738281 | 0.0166496988 | 6/66038 |
| pre_ISIC_LT_guard | PASS | 560.321662 | 12.1104374 | 12.8925781 | 0.0168296341 | 6/66038 |
| pre_HK_guard | PASS | 561.475058 | 12.2955394 | 17.4570312 | 0.0119551755 | 21/22836 |
| pre_HK_LT_guard | PASS | 576.873325 | 12.4767575 | 17.640625 | 0.0120015321 | 21/22836 |

| 项目 | pool梯度 | few-pool梯度 | 参数/注册buffer数 | operation秒 | guard shadow秒 |
|---|---:|---:|---:|---:|---:|
| pre_ISIC_guard | 0.0123273146 | 0.0111915052 | 839/0 | 558.209391 | 146.654463 |
| pre_ISIC_LT_guard | 0.0124573648 | 0.0113159457 | 839/0 | 548.51899 | 134.837073 |
| pre_HK_guard | 0.00832326082 | 0.00858193166 | 839/0 | 549.073155 | 208.275724 |
| pre_HK_LT_guard | 0.00835813958 | 0.00861268082 | 839/0 | 568.045401 | 232.387585 |

四条全部839个参数tensor精确不变；模型注册buffer数为0，因此buffer集合未变化，不代表非空buffer路径已覆盖。所有首batch为64，双pool梯度有限非零。显存列为PyTorch allocated/reserved实测峰值，不是NVML总物理占用或未来正式训练峰值。逻辑host输入仅成功guard调用的GPU到CPU输入量，不是全部PCIe流量或带宽；fixture负向校验和正向传输/解算耗时均已包含驻留。

{
  "status": "FOUR_PREFLIGHT_PASS_STOPPED_NO_TRAIN",
  "GPU_operation_records": 4,
  "new_GPU_process_residence_seconds": 2267.883051633835,
  "closed_GPU_history_before_seconds": 391490.24710153416,
  "closed_GPU_history_including_this_round_seconds": 393758.130153168,
  "whole_round_wall_seconds": 1146.7084429264069,
  "CPU_harness_residence_seconds": 45.17177395801991,
  "CPU_inside_GPU_residence_not_added": true,
  "formal_training_started": false,
  "formal_updates": 0,
  "val_accessed": false,
  "test_accessed": false,
  "real_first_batch_separate_from_synthetic_highest_bank": true,
  "failures": [],
  "source_commit": "16ad48063895e58fadd93f6eba744d9d82599521",
  "production_math_source": "7a077b69f16d843573b2bc7af5a46d9701c29bb0",
  "GPU_worker_failures": 0,
  "GPU_retries": 0,
  "all_workers_exited_before_collection": true,
  "terminal_collection_count": 1,
  "memory_metric": "PyTorch per-process peak allocated/reserved; not sampled NVML peak or total physical device use",
  "NVML_peak_bytes_unmeasured": null,
  "residence_poll_resolution_seconds": 0.25,
  "CPU_operation_and_guard_shadow_nested_in_GPU_residence": true
}

真实首task第一batch backward与synthetic最大旧bank分别记录；首task没有真实旧bank，不代表真实旧域漂移验证。最大bank shadow中真实模型及梯度仍在GPU驻留；guard沿原CPU full-dual投影，不把传输与CPU等待额外叠加到GPU驻留。全部参数buffer在shadow后再精确比较。四PASS也只停在预检，正式训练0、val/test封存，不构成医学效果或正式全轮预算证明。

源码/参数/门未改；公开范围事故与历史路径元数据残留见ENGINEERING_AUDIT，未强推或改写历史。

完整KKT、真实/合成覆盖、0注册buffer限制、逻辑传输与显存定义及全部费用/工程事件见[ANALYSIS_ZH.md](ANALYSIS_ZH.md)。私有路径配置已从最新版本移除，历史残留如实保留；当前公开结果只有匿名标量，原controller/PREFLIGHT/rawlog保留NAS，不外传。
