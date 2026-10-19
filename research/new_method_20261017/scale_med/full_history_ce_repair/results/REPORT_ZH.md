# 完整历史CE独立优化修复结果

同原冻结G/P/C表示、原永久训练与验证划分；CE是特权完整历史参照，不满足退休部署约束，不是BA上界或新方法。原内部测试已研究暴露。

退出回执：worker=1；coordinator={'worker_exit': 1, 'stop_reason': None, 'wall_seconds': 41.116115282289684}。

科学未决。未以缺失结果、未收敛或工程失败充当CE无效证据；不追加预算或重试。

本轮真实数据已完成优化单元数：0。

合成自检未通过认证，原7个已收敛头尚未复核，新增最多8头均未执行。原完整历史CE优化未决结论保留；本轮是此前置工程未决。

## 失败及成本回执

```json
{
  "failure": {
    "status": "STOPPED_ENGINEERING_UNRESOLVED_NO_RETRY",
    "failure_phase": "synthetic_selfcheck_before_real_data",
    "error": "AssertionError",
    "condition": "selfcheck requires fit status CONVERGED",
    "synthetic_fit_status": "UNRESOLVED",
    "detailed_fit_info_persisted": false,
    "exact_internal_stop_cause": "UNKNOWN",
    "source_commit": "f565b7ab2e7f7143a879fde6c418be2c050972d2",
    "parent_pid": 889689,
    "worker_pid": 889693,
    "real_data_fit_attempts": 0,
    "reused_heads_revalidated": 0,
    "planned_reuse_heads": 7,
    "max_planned_new_fits": 8,
    "old_data_features_heads_test_accessed": false,
    "model_seal_created": false,
    "method_success": false,
    "no_retry": true
  },
  "cost": {
    "coordinator_wall_seconds": 41.116115282289684,
    "worker_entry_wall_seconds": 19.86201360821724,
    "worker_CPU_seconds": null,
    "peak_RSS_bytes": null,
    "peak_GPU_allocated_bytes": null,
    "GPU_initialized_by_scientific_run": false,
    "scientific_image_forward_count": 0,
    "directory_bytes_before_reporting": 44295645,
    "worker_CPU_limit_seconds": 3600,
    "cpu_budget_full_receipt_available": false,
    "preflight_synthetic_selfcheck_passed": true,
    "preflight_cost_not_separately_metered": true,
    "preflight_cache_environment_matches_launch": false,
    "retired_pids_verified": true,
    "no_new_real_data_results": true
  }
}
```
