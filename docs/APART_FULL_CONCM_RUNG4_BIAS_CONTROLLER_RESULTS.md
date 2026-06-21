# APART Full-ConCM Ladder Rung 4: Bias-Adaptive Replay Controller

Date: 2026-06-19

Scope: bounded `max_tasks=2` gate only. No full 6-task run, no 3-seed run, no prototype calibration, no T-DSM, no match loss, no projector, no route-aware loss, and no classifier anchoring were launched or enabled.

Primary reference:

```text
NC-ConCM = APART + Stage1-capped prototype replay + head_norm_effective_sum old-logit calibration
```

Rung 4 candidate:

```text
NC-ConCM + training-time bias-adaptive replay controller
```

The controller estimates current-task old-class prediction bias on current-task training batches, then only reduces Stage1 prototype replay strength:

```text
replay_scale = clip(target_bias / (train_current_pred_old_rate + eps), min_scale, max_scale)
effective_stage1_loss_weight = base_stage1_loss_weight * replay_scale
effective_synth_cap = round(base_synth_cap * replay_scale)
```

First-gate defaults:

```text
target_bias = 0.12
min_scale = 0.5
max_scale = 1.0
base_stage1_loss_weight = 0.05
base_synth_cap = 48
```

## Files Changed

- `third_party/APART/models/apart.py`
  - Added optional `concm_replay_bias_controller`.
  - Added `concm_replay_target_bias`, `concm_replay_min_scale`, `concm_replay_max_scale`, `concm_replay_bias_eps`.
  - Added all-seen-logit current-task old-bias measurement during training.
  - Added adaptive Stage1 loss weight and adaptive synthetic replay cap.
  - Added `ConCMReplay_train_current_pred_old_rate`, `ConCMReplay_scale`, `ConCMReplay_effective_stage1_loss_weight`, and `ConCMReplay_effective_synth_cap` logs.
  - Disabled behavior is identical to NC-ConCM when `concm_replay_bias_controller=false`.
- `third_party/APART/exps/full_concm_ladder/README.md`
- `third_party/APART/exps/full_concm_ladder/rung4_bias_controller_disabled_identity_phase1_seed1993_gpu0.json`
- `third_party/APART/exps/full_concm_ladder/rung4_bias_controller_smoke_seed1993_gpu0.json`
- `third_party/APART/exps/full_concm_ladder/rung4_bias_controller_phase1_seed1993_gpu0.json`
- `docs/APART_FULL_CONCM_RUNG4_BIAS_CONTROLLER_RESULTS.md`

## Static Checks

Local:

```bash
python3 -m py_compile third_party/APART/models/apart.py
for f in third_party/APART/exps/full_concm_ladder/rung4_*.json; do
  python3 -m json.tool "$f" >/dev/null || exit 1
done
```

Remote:

```bash
cd /dev/shm/wangbomin/APART/code
/opt/miniconda3/envs/torchgpu/bin/python -m py_compile models/apart.py
for f in exps/full_concm_ladder/rung4_*.json; do
  /opt/miniconda3/envs/torchgpu/bin/python -m json.tool "$f" >/tmp/$(basename "$f").checked || exit 1
done
```

Result: passed.

## Exact Remote Commands

Disabled identity:

```bash
cd /dev/shm/wangbomin/APART/code
LOG=/dev/shm/wangbomin/APART/logs/rung4_bias_controller_disabled_identity_phase1_seed1993_20260619-024151.out
nohup timeout 3600s env \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/full_concm_ladder/rung4_bias_controller_disabled_identity_phase1_seed1993_gpu0.json \
    --text rung4_bias_controller_disabled_identity_phase1_seed1993 \
  > "$LOG" 2>&1 &
```

Smoke:

```bash
cd /dev/shm/wangbomin/APART/code
LOG=/dev/shm/wangbomin/APART/logs/rung4_bias_controller_smoke_seed1993_20260619-024151.out
nohup timeout 1800s env \
  CUDA_VISIBLE_DEVICES=1 \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/full_concm_ladder/rung4_bias_controller_smoke_seed1993_gpu0.json \
    --text rung4_bias_controller_smoke_seed1993 \
  > "$LOG" 2>&1 &
```

The smoke run was stopped after the requested code-path evidence was logged.

Accuracy gate:

```bash
cd /dev/shm/wangbomin/APART/code
LOG=/dev/shm/wangbomin/APART/logs/rung4_bias_controller_phase1_seed1993_20260619-025641.out
nohup timeout 3600s env \
  CUDA_VISIBLE_DEVICES=1 \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/full_concm_ladder/rung4_bias_controller_phase1_seed1993_gpu0.json \
    --text rung4_bias_controller_phase1_seed1993 \
  > "$LOG" 2>&1 &
```

## Disabled-Mode Identity

`concm_replay_bias_controller=false` reproduced NC-ConCM exactly.

| run | curve | Avg Acc | AccT | old acc | new acc | new_eval_pred_old_rate | old_eval_pred_new_rate | alpha |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NC-ConCM reference | `[90.42, 89.62]` | 90.020 | 89.62 | 90.14 | 87.00 | 11.60 | 2.14 | 0.749026 |
| Rung4 disabled identity | `[90.42, 89.62]` | 90.020 | 89.62 | 90.14 | 87.00 | 11.60 | 2.14 | 0.749026 |

No `ConCMReplay` line appeared in disabled mode.

## Smoke Result

Config: `tuned_epoch=2`, `max_tasks=2`, `concm_replay_bias_controller=true`.

Task1 code-path evidence:

| epoch | train_current_pred_old_rate | replay_scale | effective weight | effective cap | raw Stage1 | weighted Stage1 | CE | Stage1/CE | nonzero grad |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 47.76 | 0.501477 | 0.025074 | 24.06 | 0.011188 | 0.000280 | 0.776026 | 0.000399 | 16/16 |
| 2 | 27.05 | 0.521204 | 0.026060 | 25.06 | 0.001895 | 0.000049 | 0.386959 | 0.000167 | 16/16 |

Smoke passed for code-path validation:

- controller activated only after task0 memory existed;
- `replay_scale` was finite and inside `[0.5, 1.0]`;
- effective cap was finite;
- Stage1 CE received nonzero gradients;
- no `Traceback`, `RuntimeError`, CUDA OOM, or nonzero `nan_or_inf` event.

## Accuracy Gate

Config: normal phase1 budget, `tuned_epoch=10`, `max_tasks=2`, seed `1993`, `concm_replay_target_bias=0.12`, `min_scale=0.5`, `max_scale=1.0`.

Primary comparison is against NC-ConCM.

| run | curve | Avg Acc | AccT | old acc | new acc | new_eval_pred_old_rate | old_eval_pred_new_rate | alpha | old norm | new norm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NC-ConCM reference | `[90.42, 89.62]` | 90.020 | 89.62 | 90.14 | 87.00 | 11.60 | 2.14 | 0.749026 | 2.360869 | 1.768353 |
| Rung4 bias controller | `[90.42, 89.55]` | 89.985 | 89.55 | 90.12 | 86.70 | 11.90 | 2.52 | 0.752717 | 2.349303 | 1.768359 |
| Delta vs NC-ConCM | - | -0.035 | -0.07 | -0.02 | -0.30 | +0.30 | +0.38 | +0.003691 | -0.011566 | +0.000006 |

Uncalibrated diagnostics:

| run | uncalibrated Avg | uncalibrated AccT | uncalibrated old acc | uncalibrated new acc | uncalibrated new_eval_pred_old_rate | uncalibrated old_eval_pred_new_rate |
|---|---:|---:|---:|---:|---:|---:|
| NC-ConCM reference | 89.500 | 88.58 | 90.86 | 77.20 | 22.20 | 0.76 |
| Rung4 bias controller | 89.585 | 88.75 | 90.98 | 77.60 | 21.50 | 0.98 |

## Controller Diagnostics

Task1 controller trajectory:

| epoch | train_current_pred_old_rate | replay_scale | effective weight | effective cap | raw Stage1 | weighted Stage1 | CE | Stage1/CE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 63.43 | 0.500000 | 0.025000 | 24.00 | 0.005177 | 0.000129 | 0.774860 | 0.000236 |
| 2 | 26.70 | 0.549953 | 0.027498 | 26.44 | 0.003379 | 0.000085 | 0.256744 | 0.000923 |
| 3 | 24.23 | 0.568404 | 0.028420 | 27.31 | 0.000138 | 0.000004 | 0.299742 | 0.000015 |
| 4 | 25.15 | 0.561404 | 0.028070 | 27.00 | 0.000118 | 0.000003 | 0.312808 | 0.000017 |
| 5 | 22.27 | 0.593381 | 0.029669 | 28.56 | 0.000242 | 0.000006 | 0.226462 | 0.000029 |
| 6 | 20.43 | 0.665536 | 0.033277 | 31.94 | 0.000019 | 0.000001 | 0.196063 | 0.000003 |
| 7 | 19.31 | 0.640755 | 0.032038 | 30.88 | 0.000006 | 0.000000 | 0.170307 | 0.000001 |
| 8 | 19.88 | 0.647831 | 0.032392 | 31.19 | 0.000015 | 0.000000 | 0.195456 | 0.000006 |
| 9 | 19.82 | 0.686331 | 0.034317 | 32.94 | 0.000018 | 0.000001 | 0.236347 | 0.000002 |
| 10 | 19.52 | 0.655059 | 0.032753 | 31.56 | 0.000004 | 0.000000 | 0.212368 | 0.000002 |

The controller behaved as designed: it reduced replay below NC-ConCM and never amplified beyond the validated base strength. However, the final calibrated metrics did not improve.

## Safety Check

Remote grep counts for disabled identity, smoke, and accuracy logs:

```text
Traceback / RuntimeError / CUDA OOM / nonzero nan_or_inf count = 0
```

Final remote GPU status after runs:

```text
GPU0: 1 MiB, 0%
GPU1: 1 MiB, 0%
```

## Decision

Reject Rung4 as a main method in this form. Keep it as diagnostic-only.

Reason:

- It underperforms NC-ConCM on Avg Acc and AccT.
- It does not reduce calibrated `new_eval_pred_old_rate`; it slightly increases it from `11.60` to `11.90`.
- It slightly reduces new accuracy from `87.00` to `86.70`.
- It only improves uncalibrated diagnostics, but the actual deployed/evaluated NC-ConCM path is calibrated; primary comparison must be against calibrated NC-ConCM.

Do not launch full 6-task or 3-seed validation for Rung4.
