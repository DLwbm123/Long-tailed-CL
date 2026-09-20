最新状态（2026-09-15）：六条轨迹及一次批量测试已完成，结论为 **COMPLETE_MIXED_SIGNAL**。完整报告见 [AUDIT_AND_CONCLUSIONS.md](completed_hb01/AUDIT_AND_CONCLUSIONS.md)。下面的早期预算 BLOCKED 记录保留为历史审计。

# ISIC APART migration V2

**BLOCKED_FORMAL_BUDGET** — data, weights and P1 engineering pass; formal training **0/6**, final session checkpoints **0/18**, model test predictions **0**. This is a budget block, not a negative transfer result.

- [Current status and minimum pending decision](V2_FINAL_RESULTS.md)
- [Cleaning audit](V2_DATA_CLEANING_AUDIT.md) and [aggregate dataset summary](V2_DATASET_SUMMARY.json)
- [P1 evidence and measured budget](P1_TECHNICAL_REPORT.md), [machine-readable P1](P1_TECHNICAL_REPORT.json), [budget calculation](budget_audit.json)
- [Weight lock](WEIGHTS_LOCK.json), [training semantics](TRAINING_SEMANTICS_LOCK.json), [resolved formal configurations — NOT RUN](resolved_formal_configs.json)
- [Class orders](class_orders.json), [planned sessions — NOT RUN](planned_session_matrix_NOT_RUN.csv), [protocol lock status](PROTOCOL_LOCK_V2.json), [resources](resource_usage.json)

The unchanged formal matrix is B/APART and C/APART+ConCM-lite, three paired seeds 1993/1994/1995, 4+2+2 classes, batch 48, ten epochs/session. M0/M1 share B checkpoints without/with head-norm calibration; M2/M3 share C checkpoints without/with calibration. A separate evaluation process is gated behind all 18 locked checkpoints. No Full Dynamic, parameter search or test-based selection is included.

The code reference is `7a089e4b30a3efa6e2943aa4b99c1f5e60652b16`; preserved V1 audit is `45800adb6a14258723d25314d230033491e87411`. The source workspace, original CSV bytes and V1 report were not overwritten. New V2 private manifests, disposition/lesion metadata, images, weights, raw predictions and checkpoints are not published here.

CPU checks run with `tests/test_medical_v2.py` using the recorded Python environment. GPU P1, training and evaluation functions are in `tools/run_medical_v2.py`, `tools/execute_medical_v2.py` and `tools/evaluate_medical_v2.py`; the supervisor starts training and evaluation as separate processes and allows at most one recognized infrastructure recovery during training. Launch through a neutral entry filename and supply the private runtime configuration through the launcher; do not put project/method/user paths in process arguments.

The current protocol file deliberately says `NOT_LOCKED_BLOCKED_BUDGET`. Do not bypass it or reuse P1 state for formal initialization. A budget decision, then the clean runtime commit, final protocol/configuration hash binding and a fresh GPU3 memory check are required before the fixed queue starts.
