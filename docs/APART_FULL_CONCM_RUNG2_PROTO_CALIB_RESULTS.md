# APART Full-ConCM Ladder Rung 2: Reliability-Aware Prototype Calibration

Date: 2026-06-18

Scope: bounded `max_tasks=2` gate only. No full 6-task run, no 3-seed run, no T-DSM, no match loss, no tail anchors, no projector, no route-aware calibration, and no GPA-style imprinting were launched or implemented.

Reference:

```text
NC-ConCM = APART + Stage1-capped prototype augmentation + head_norm_effective_sum old-logit calibration
```

## Files Changed

- `third_party/APART/models/apart.py`
  - Added optional `concm_proto_calibration`.
  - Added `concm_proto_topk`, `concm_alpha_a`, `concm_alpha_b`, `concm_alpha_min`, `concm_alpha_max`.
  - When enabled, calibrates only Stage1 memory statistics used for synthetic old-class sampling:
    - `mean_main / var_main`
    - `mean_few / var_few`
  - Does not modify APART `head`, `head_few`, routing, classifier logits, or inference code.
  - Keeps disabled mode on the original NC-ConCM memory/log path.
- `third_party/APART/exps/full_concm_ladder/README.md`
- `third_party/APART/exps/full_concm_ladder/rung2_proto_calib_disabled_identity_phase1_seed1993_gpu0.json`
- `third_party/APART/exps/full_concm_ladder/rung2_proto_calib_smoke_seed1993_gpu0.json`
- `third_party/APART/exps/full_concm_ladder/rung2_proto_calib_phase1_seed1993_gpu0.json`
- `docs/APART_FULL_CONCM_RUNG2_PROTO_CALIB_RESULTS.md`

## Static Checks

Local:

```bash
python3 -m py_compile third_party/APART/models/apart.py
for f in third_party/APART/exps/full_concm_ladder/rung2_proto_calib_*.json; do
  python3 -m json.tool "$f" >/dev/null || exit 1
done
```

Remote:

```bash
cd /dev/shm/wangbomin/APART/code
/opt/miniconda3/envs/torchgpu/bin/python -m py_compile models/apart.py
for f in exps/full_concm_ladder/rung2_proto_calib_*.json; do
  /opt/miniconda3/envs/torchgpu/bin/python -m json.tool "$f" >/tmp/$(basename "$f").checked || exit 1
done
```

Result: passed.

## Exact Remote Commands

Disabled identity:

```bash
cd /dev/shm/wangbomin/APART/code
LOG=/dev/shm/wangbomin/APART/logs/rung2_proto_calib_disabled_identity_phase1_seed1993_20260618-113604.out
nohup timeout 3600s env \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/full_concm_ladder/rung2_proto_calib_disabled_identity_phase1_seed1993_gpu0.json \
    --text rung2_proto_calib_disabled_identity_phase1_seed1993 \
  > "$LOG" 2>&1 &
```

Smoke:

```bash
cd /dev/shm/wangbomin/APART/code
LOG=/dev/shm/wangbomin/APART/logs/rung2_proto_calib_smoke_seed1993_20260618-114053.out
nohup timeout 1800s env \
  CUDA_VISIBLE_DEVICES=1 \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/full_concm_ladder/rung2_proto_calib_smoke_seed1993_gpu0.json \
    --text rung2_proto_calib_smoke_seed1993 \
  > "$LOG" 2>&1 &
```

Accuracy gate:

```bash
cd /dev/shm/wangbomin/APART/code
LOG=/dev/shm/wangbomin/APART/logs/rung2_proto_calib_phase1_seed1993_20260618-115152.out
nohup timeout 3600s env \
  CUDA_VISIBLE_DEVICES=1 \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/full_concm_ladder/rung2_proto_calib_phase1_seed1993_gpu0.json \
    --text rung2_proto_calib_phase1_seed1993 \
  > "$LOG" 2>&1 &
```

## Disabled-Mode Identity

`concm_proto_calibration=false` reproduced the NC-ConCM reference exactly.

| run | curve | Avg Acc | AccT | old acc | new acc | new_eval_pred_old_rate | old_eval_pred_new_rate | alpha | old norm | new norm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NC-ConCM reference | `[90.42, 89.62]` | 90.020 | 89.62 | 90.14 | 87.00 | 11.60 | 2.14 | 0.749026 | 2.360869 | 1.768353 |
| disabled identity | `[90.42, 89.62]` | 90.020 | 89.62 | 90.14 | 87.00 | 11.60 | 2.14 | 0.749026 | 2.360869 | 1.768353 |

No `ConCMProtoCalibration` line appeared in disabled mode.

## Smoke Result

Config: `tuned_epoch=2`, `max_tasks=2`, `concm_proto_calibration=true`.

Evidence:

- `ConCMProtoCalibration task=0 active=True ... nan_or_inf=0`
- `ConCMProtoCalibration task=1 active=True ... nan_or_inf=0`
- Task1 synthetic CE had nonzero gradients: `ConCMStage1_nonzero_grad_batches 16/16`.
- Example task1 loss line: raw `1.201444`, weighted `0.060072`, CE `0.774481`, weighted ratio `0.123735`.
- No `Traceback`, `RuntimeError`, CUDA OOM, or nonzero `nan_or_inf` event.

Smoke final calibrated curve was `[83.3, 85.62]`; this was not used for the accuracy decision because it intentionally used only 2 epochs.

## Accuracy Gate

Config: normal APART phase1 budget, `tuned_epoch=10`, `max_tasks=2`, seed `1993`.

Primary comparison is against NC-ConCM, not APART baseline.

| run | curve | Avg Acc | AccT | old acc | new acc | new_eval_pred_old_rate | old_eval_pred_new_rate | alpha | old norm | new norm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NC-ConCM reference | `[90.42, 89.62]` | 90.020 | 89.62 | 90.14 | 87.00 | 11.60 | 2.14 | 0.749026 | 2.360869 | 1.768353 |
| Rung2 proto calibration | `[90.42, 89.42]` | 89.920 | 89.42 | 89.96 | 86.70 | 12.00 | 1.82 | 0.699342 | 2.528086 | 1.767996 |
| Delta vs NC-ConCM | - | -0.100 | -0.20 | -0.18 | -0.30 | +0.40 | -0.32 | -0.049684 | +0.167217 | -0.000357 |

Uncalibrated task1 diagnostics:

| run | uncalibrated Avg | uncalibrated AccT | uncalibrated old acc | uncalibrated new acc | uncalibrated new_eval_pred_old_rate | uncalibrated old_eval_pred_new_rate |
|---|---:|---:|---:|---:|---:|---:|
| NC-ConCM reference | 89.500 | 88.58 | 90.86 | 77.20 | 22.20 | 0.76 |
| Rung2 proto calibration | 88.920 | 87.42 | 90.60 | 71.50 | 27.90 | 0.40 |

## Prototype Calibration Diagnostics

Task0 memory calibration:

| path | alpha_mean | raw_to_calibrated_cos_mean | nan_or_inf |
|---|---:|---:|---:|
| main | 0.499646 | 0.822679 | 0 |
| few | 0.502700 | 0.843456 | 0 |

Task1 memory calibration:

| path | alpha_mean | many alpha | medium alpha | few alpha | raw/calib cos mean | many cos | medium cos | few cos | nan_or_inf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main | 0.499900 | 0.822542 | 0.520345 | 0.150000 | 0.877325 | 0.986759 | 0.908389 | 0.726472 | 0 |
| few | 0.503537 | 0.824786 | 0.527753 | 0.150000 | 0.889331 | 0.987630 | 0.920970 | 0.748845 | 0 |

Class groups in task1: many `3`, medium `4`, few `3`.

## Decision

Reject as a main ladder module for now; keep only as diagnostic code/config.

Reason:

- It underperforms NC-ConCM on Avg Acc and AccT.
- It slightly hurts both old and new accuracy.
- It increases `new_eval_pred_old_rate` from `11.60` to `12.00`.
- It increases the old effective norm and forces headnorm alpha lower (`0.749026 -> 0.699342`) without producing a net accuracy gain.

This does not justify a full 6-task run or 3-seed run. The next useful ladder step should not build on this module as enabled by default. If revisited, it needs a narrower calibration rule that does not increase old-class prediction bias.
