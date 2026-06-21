# APART ConCM-lite Stage 1 Non-Oracle Calibration Gate

Date: 2026-06-17

Scope: evaluation-only non-oracle calibration using the saved Stage1-capped task1 checkpoint and saved task1 logits. No training, full 6-task run, or 3-seed run was launched.

## Inputs

Checkpoint:

```text
/dev/shm/wangbomin/APART/checkpoints/stage1_capped_calibration/apart_concm_stage1_capped_calibration_phase1_0617-07-36-24-766_task1.pt
```

Logits artifact:

```text
/dev/shm/wangbomin/APART/artifacts/stage1_capped_calibration/apart_concm_stage1_capped_calibration_phase1_0617-07-36-24-766/task1_old_logit_scale_sweep_logits.npz
```

Baseline reference:

| run | Avg Acc | task1 AccT | old acc | new acc |
|---|---:|---:|---:|---:|
| APART baseline, max_tasks=2 | 89.560 | 88.70 | 89.40 | 85.20 |

## Files Changed

- `third_party/APART/tools/eval_non_oracle_calibration.py`
  - New evaluation-only script.
  - Reads saved checkpoint and saved logits.
  - Computes head-norm alpha from classifier weights.
  - Evaluates fixed predefined alpha values without using test-set alpha selection.

## Exact Command

```bash
cd /dev/shm/wangbomin/APART/code
/opt/miniconda3/envs/torchgpu/bin/python tools/eval_non_oracle_calibration.py \
  --checkpoint /dev/shm/wangbomin/APART/checkpoints/stage1_capped_calibration/apart_concm_stage1_capped_calibration_phase1_0617-07-36-24-766_task1.pt \
  --logits-npz /dev/shm/wangbomin/APART/artifacts/stage1_capped_calibration/apart_concm_stage1_capped_calibration_phase1_0617-07-36-24-766/task1_old_logit_scale_sweep_logits.npz \
  --metrics-json /dev/shm/wangbomin/APART/artifacts/stage1_capped_calibration/apart_concm_stage1_capped_calibration_phase1_0617-07-36-24-766/task1_old_logit_scale_sweep_metrics.json \
  --output-json /dev/shm/wangbomin/APART/artifacts/stage1_capped_calibration_non_oracle/non_oracle_calibration_results.json \
  --output-md /dev/shm/wangbomin/APART/artifacts/stage1_capped_calibration_non_oracle/non_oracle_calibration_results.md \
  --fixed-alpha 0.8 0.75 0.7
```

## Head-Norm Calibration Rule

APART inference uses `logits = g + g_aux`, so the head-norm rule was applied to the effective classifier weights:

```text
effective_weight = head.weight + head_few.weight
alpha = mean_norm_new / mean_norm_old
```

Computed norms:

| classifier | old mean norm | new mean norm | alpha |
|---|---:|---:|---:|
| `head` | 1.288610 | 0.968579 | 0.751646 |
| `head_few` | 1.308362 | 0.996805 | 0.761873 |
| effective `head + head_few` | 2.360869 | 1.768353 | 0.749026 |

The selected head-norm alpha `0.749026` is inside the oracle-good range `0.7-0.8`.

## Results

| rule | selected alpha | Avg Acc | task1 AccT | old acc | new acc | new_eval_pred_old_rate | old_eval_pred_new_rate | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| head_norm_effective_sum | 0.749026 | 90.020 | 89.62 | 90.14 | 87.00 | 11.60 | 2.14 | promising |
| fixed_0.8 | 0.800000 | 89.935 | 89.45 | 90.30 | 85.20 | 13.70 | 1.82 | promising |
| fixed_0.75 | 0.750000 | 90.010 | 89.60 | 90.14 | 86.90 | 11.70 | 2.14 | promising |
| fixed_0.7 | 0.700000 | 89.935 | 89.45 | 89.60 | 88.70 | 9.70 | 2.88 | promising |

All tested non-oracle rules satisfy the promising criterion:

- task1 AccT > `88.70`
- Avg Acc > `89.560`
- new acc >= `85.20`
- old acc >= `89.40`

The strongest non-oracle rule is head-norm effective-sum calibration:

- Avg Acc: `90.020`, `+0.460` over baseline
- task1 AccT: `89.62`, `+0.92` over baseline
- old acc: `90.14`, `+0.74` over baseline
- new acc: `87.00`, `+1.80` over baseline

## Train/Val Calibration

Status: not run.

Reason: the APART Stage1 run is exemplar-free (`memory_size=0`) and no separate saved validation split exists at task1. Task1 training data contains only new classes under the incremental protocol, so it cannot support old/new balancing. Using old-class train data or test-set sweep results to choose alpha would violate the requested non-oracle gate.

## Saved Outputs

```text
/dev/shm/wangbomin/APART/artifacts/stage1_capped_calibration_non_oracle/non_oracle_calibration_results.json
/dev/shm/wangbomin/APART/artifacts/stage1_capped_calibration_non_oracle/non_oracle_calibration_results.md
```

Final remote status: no APART/GPA/TaConCM training process was running; both GPUs were idle.

## Decision

The non-oracle calibration gate is promising.

Stage1-capped no longer needs test-set alpha selection to beat the 2-task baseline. The head-norm rule independently selects `alpha=0.749026`, which lands in the oracle-good range and improves Avg Acc, task1 AccT, old accuracy, and new accuracy over the baseline reference.

Recommended next step: run one bounded `max_tasks=2` calibrated Stage1 gate where evaluation applies the head-norm effective-sum alpha automatically. Do not run full 6-task or 3-seed until that calibrated evaluation path is integrated and verified end to end.
