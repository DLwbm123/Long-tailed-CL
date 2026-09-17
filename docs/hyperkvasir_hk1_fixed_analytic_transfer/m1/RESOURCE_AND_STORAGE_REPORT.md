# 实际资源与存储

```json
{
  "GPU_process_residence_seconds": 1449.9285484740103,
  "worker_GPU_seconds": 1403.8808133180137,
  "artifact_peak_bytes": 461740575,
  "min_free_bytes": 2821341184,
  "input_staging_bytes": 3146187899,
  "artifact_limit_bytes": 805306368,
  "GPU_limit_seconds": 14400,
  "CPU_analytic_limit_seconds": 7200,
  "peak_GPU_allocated_bytes": 9023454720,
  "peak_RSS_bytes": 3222093824,
  "batch_size": 48,
  "AMP": false,
  "BLAS_threads": 4,
  "torch_threads": 4,
  "loader_workers": 8,
  "mode": "formal",
  "analytic_CPU_seconds": 18.101279427006375,
  "CPU_budget_accounted_seconds": 78.14019344901317,
  "CPU_preflight_unmeasured_reserve_seconds": 60.0
}
```

输入镜像单列；实验产物记账包括源端身份审计及运行代码/清单的保守外部额度。GPU 为单 worker，神经阶段退出后才进入独立 CPU 解析报告进程。删除范围只限本轮已验证恢复的滚动临时文件，见 STORAGE_LEDGER.jsonl；未删除或迁移更多历史资产。
