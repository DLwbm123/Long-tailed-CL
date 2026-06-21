# APART ConCM-lite Stage 1 Calibration Rescue Results

Date: 2026-06-17

Scope: bounded `max_tasks=2` Stage1-capped rerun with evaluation-time old-class logit scaling. No full 6-task run, no 3-seed run, and no additional Stage1 schedule variants were launched.

## Files Changed

- `third_party/APART/models/apart.py`
  - Added task1-only evaluation-time old-class logit scaling sweep.
  - Saved raw logits, labels, alpha-wise predictions, old class ids, new class ids, and per-alpha metrics.
  - Preserved normal APART inference and training behavior when `eval_old_logit_scale_sweep` is disabled.
- `third_party/APART/trainer.py`
  - Added optional task-end checkpoint saving after `eval_task()`.
- `third_party/APART/exps/apart_cifar_shuffle_concm_stage1_capped_calibration_phase1_gpu0.json`
  - Added the bounded Stage1-capped `max_tasks=2` calibration config.

Local checks:

```bash
python3 -m py_compile third_party/APART/models/apart.py third_party/APART/trainer.py
python3 -m json.tool third_party/APART/exps/apart_cifar_shuffle_concm_stage1_capped_calibration_phase1_gpu0.json
```

Remote checks with `/opt/miniconda3/envs/torchgpu/bin/python` also passed.

## Exact Command

```bash
cd /dev/shm/wangbomin/APART/code
LOG=/dev/shm/wangbomin/APART/logs/apart_concm_stage1_capped_calibration_phase1_20260617-073623.out
nohup timeout 3600s env \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/apart_cifar_shuffle_concm_stage1_capped_calibration_phase1_gpu0.json \
    --text apart_concm_stage1_capped_calibration_phase1 \
  > "$LOG" 2>&1 &
```

PID at launch: `47469` (`timeout`), child Python PID `47471`.

Final process check: no APART/GPA/TaConCM training process remained running, and both GPUs were idle.

## Validity Check

Reference baseline reused from the existing 2-task gate:

| run | curve | Average Accuracy | task1 AccT | old acc | new acc |
|---|---|---:|---:|---:|---:|
| baseline | `[90.42, 88.70]` | 89.560 | 88.70 | 89.40 | 85.20 |

The alpha `1.0` row exactly reproduces the previous Stage1-capped result:

| run | curve | Average Accuracy | task1 AccT | old acc | new acc | new_eval_pred_old_rate | old_eval_pred_new_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| previous Stage1-capped | `[90.42, 88.58]` | 89.500 | 88.58 | 90.86 | 77.20 | 22.20 | 0.76 |
| rerun alpha=1.0 | `[90.42, 88.58]` | 89.500 | 88.58 | 90.86 | 77.20 | 22.20 | 0.76 |

This validates the rerun and confirms the added eval/logit-dump code did not perturb training.

## Alpha Sweep

Old-class logits were multiplied by `alpha`; new-class logits were unchanged.

| alpha | Avg Acc | task1 AccT | old acc | new acc | new_eval_pred_old_rate | old_eval_pred_new_rate |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 89.500 | 88.58 | 90.86 | 77.20 | 22.20 | 0.76 |
| 0.9 | 89.785 | 89.15 | 90.64 | 81.70 | 17.40 | 1.18 |
| 0.8 | 89.935 | 89.45 | 90.30 | 85.20 | 13.70 | 1.82 |
| 0.7 | 89.935 | 89.45 | 89.60 | 88.70 | 9.70 | 2.88 |
| 0.6 | 89.560 | 88.70 | 88.02 | 92.10 | 5.80 | 5.06 |
| 0.5 | 88.370 | 86.32 | 84.64 | 94.70 | 2.90 | 9.58 |
| 0.4 | 86.110 | 81.80 | 78.96 | 96.00 | 1.30 | 16.62 |

Best rescue points:

- `alpha=0.8`: matches baseline new acc (`85.20`) while keeping old acc above baseline (`90.30` vs `89.40`), and improves task1 AccT (`89.45` vs `88.70`).
- `alpha=0.7`: improves new acc above baseline (`88.70` vs `85.20`) while keeping old acc above baseline (`89.60` vs `89.40`), also with task1 AccT `89.45`.

## Loss Diagnostics

Stage1 loss was active and produced nonzero gradients, but its weighted magnitude was tiny relative to CE after the first epoch.

| task1 epoch | raw loss | weighted loss | CE loss | weighted/CE | raw/CE | nonzero grad batches |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.005224 | 0.000261 | 0.774860 | 0.000411 | 0.008218 | 16/16 |
| 2 | 0.000644 | 0.000032 | 0.256744 | 0.000180 | 0.003599 | 16/16 |
| 5 | 0.000007 | 0.000000 | 0.226465 | 0.000003 | 0.000059 | 16/16 |
| 10 | 0.000007 | 0.000000 | 0.212393 | 0.000003 | 0.000060 | 16/16 |

Uncalibrated final head norm diagnostics:

| metric | old | new |
|---|---:|---:|
| `head_weight_norm_mean` | 1.288610 | 0.968578 |
| `head_few_weight_norm_mean` | 1.308362 | 0.996805 |

The uncalibrated model is strongly old-biased on new-class evaluation: `22.20%` of new-class samples are predicted as old classes.

## Saved Artifacts

```text
/dev/shm/wangbomin/APART/artifacts/stage1_capped_calibration/apart_concm_stage1_capped_calibration_phase1_0617-07-36-24-766/task1_old_logit_scale_sweep_metrics.json
/dev/shm/wangbomin/APART/artifacts/stage1_capped_calibration/apart_concm_stage1_capped_calibration_phase1_0617-07-36-24-766/task1_old_logit_scale_sweep_logits.npz
```

NPZ contents:

| key | shape | dtype |
|---|---:|---|
| `alphas` | `(7,)` | `float32` |
| `labels` | `(6000,)` | `int64` |
| `logits` | `(6000, 60)` | `float32` |
| `old_class_ids` | `(50,)` | `int64` |
| `new_class_ids` | `(10,)` | `int64` |
| `predicted_labels` | `(6000,)` | `int64` |
| `predictions_by_alpha` | `(7, 6000)` | `int64` |

Optional task checkpoints were also saved:

```text
/dev/shm/wangbomin/APART/checkpoints/stage1_capped_calibration/apart_concm_stage1_capped_calibration_phase1_0617-07-36-24-766_task0.pt
/dev/shm/wangbomin/APART/checkpoints/stage1_capped_calibration/apart_concm_stage1_capped_calibration_phase1_0617-07-36-24-766_task1.pt
```

## Decision

The calibration rescue diagnostic is positive. Lowering old-class logits at evaluation recovers new-class accuracy to baseline or better while keeping old-class accuracy at or above the baseline old accuracy.

This indicates the Stage1-capped failure is mainly classifier/logit calibration bias, not clear representation damage.

Recommendation:

1. Do not abandon Stage1 prototype augmentation yet.
2. Do not run full 6-task or 3-seed immediately.
3. Next bounded step should be a calibration-aware 2-task gate, using a principled non-oracle calibration mechanism selected without test-label tuning, for example validation-set alpha selection, train-time old/new logit temperature, or norm-balanced head calibration.
