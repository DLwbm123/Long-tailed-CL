# APART ConCM-lite Stage 1 Head-Norm Calibrated Gate

Date: 2026-06-17

Scope: fresh from-scratch bounded `max_tasks=2` run with Stage1-capped training and integrated non-oracle head-norm evaluation calibration. No full 6-task run or 3-seed run was launched.

## Files Changed

- `third_party/APART/models/apart.py`
  - Added `concm_stage1_eval_calibration`.
  - Added `calibration_rule=head_norm_effective_sum`.
  - During final evaluation for task1 and later, computes effective classifier weights as `head.weight + head_few.weight`.
  - Computes `alpha = mean_norm_new / mean_norm_old`.
  - Applies alpha only to old-class logits before argmax.
  - Logs calibrated and uncalibrated task metrics.
- `third_party/APART/exps/apart_cifar_shuffle_concm_stage1_capped_headnorm_calibrated_phase1_gpu0.json`
  - Fresh bounded `max_tasks=2` Stage1-capped config with integrated head-norm calibration.

Checks:

```bash
python3 -m py_compile third_party/APART/models/apart.py third_party/APART/tools/eval_non_oracle_calibration.py
python3 -m json.tool third_party/APART/exps/apart_cifar_shuffle_concm_stage1_capped_headnorm_calibrated_phase1_gpu0.json
```

Remote checks with `/opt/miniconda3/envs/torchgpu/bin/python` also passed before launch.

## Exact Command

```bash
cd /dev/shm/wangbomin/APART/code
LOG=/dev/shm/wangbomin/APART/logs/apart_concm_stage1_capped_headnorm_cal_phase1_20260617-085624.out
nohup timeout 3600s env \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/apart_cifar_shuffle_concm_stage1_capped_headnorm_calibrated_phase1_gpu0.json \
    --text apart_concm_stage1_capped_headnorm_cal_phase1 \
  > "$LOG" 2>&1 &
```

PID at launch: `4159` (`timeout`), child Python PID `4160`.

Final remote status: no APART/GPA/TaConCM training process remained running; both GPUs were idle. No `Traceback`, `RuntimeError`, or CUDA OOM appeared in the log.

## Config

Training was unchanged relative to Stage1-capped:

```json
"concm_stage1": true,
"concm_stage1_loss_weight": 0.05,
"concm_stage1_synth_per_class": 4,
"concm_stage1_max_synth_total": 48,
"max_tasks": 2
```

Evaluation-only calibration:

```json
"concm_stage1_eval_calibration": true,
"calibration_rule": "head_norm_effective_sum"
```

## Results

Baseline reference:

| run | Avg Acc | task1 AccT | old acc | new acc | new_eval_pred_old_rate |
|---|---:|---:|---:|---:|---:|
| APART baseline, max_tasks=2 | 89.560 | 88.70 | 89.40 | 85.20 | unavailable |

Fresh integrated Stage1-capped-calibrated run:

| run | curve | Avg Acc | task1 AccT | old acc | new acc | new_eval_pred_old_rate | old_eval_pred_new_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| Stage1-capped, uncalibrated diagnostics | `[90.42, 88.58]` | 89.500 | 88.58 | 90.86 | 77.20 | 22.20 | 0.76 |
| Stage1-capped + head-norm calibration | `[90.42, 89.62]` | 90.020 | 89.62 | 90.14 | 87.00 | 11.60 | 2.14 |

Calibration parameters:

| rule | alpha | old effective norm | new effective norm |
|---|---:|---:|---:|
| `head_norm_effective_sum` | 0.749026 | 2.360869 | 1.768353 |

Comparison to baseline:

| metric | baseline | calibrated Stage1 | delta |
|---|---:|---:|---:|
| Avg Acc | 89.560 | 90.020 | +0.460 |
| task1 AccT | 88.70 | 89.62 | +0.92 |
| old acc | 89.40 | 90.14 | +0.74 |
| new acc | 85.20 | 87.00 | +1.80 |

The prediction-bias diagnostic also improves substantially relative to uncalibrated Stage1-capped:

| metric | uncalibrated Stage1 | calibrated Stage1 |
|---|---:|---:|
| new_eval_pred_old_rate | 22.20 | 11.60 |
| old_eval_pred_new_rate | 0.76 | 2.14 |

## Decision

The fresh integrated run satisfies the promising gate:

- Avg Acc `90.020 > 89.560`
- task1 AccT `89.62 > 88.70`
- old acc `90.14 >= 89.40`
- new acc `87.00 >= 85.20`
- new_eval_pred_old_rate is clearly below uncalibrated Stage1-capped (`11.60 < 22.20`)

This justifies a bounded 3-seed 2-task run for Stage1-capped + head-norm calibration. It does not yet justify a full 6-task or larger multi-seed experiment.
