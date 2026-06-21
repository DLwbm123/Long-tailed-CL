# APART ConCM-Lite Stage1 Head-Norm 3-Seed Full Validation

## Scope

- Workline: APART-ConCM only.
- Experiment: paired 3-seed full 6-task CIFAR-100-LT shuffled B50-5 validation.
- Method: ConCM Stage1 capped prototype augmentation plus evaluation-time `head_norm_effective_sum` old-logit calibration.
- No extra variants, alpha sweeps, 2-task gates, or uncalibrated Stage1 full runs were launched.

## Seeds

| seed | baseline | method |
|---:|---|---|
| 1993 | reused existing compatible APART baseline | reused one-seed full gate |
| 1994 | newly run | newly run |
| 1995 | newly run | newly run |

The compatible official APART baseline uses seed `1993`, not numeric seed `0`; seeds `1994` and `1995` extend that seed family.

## Files Changed

- `third_party/APART/models/apart.py`
- `third_party/APART/exps/apart_cifar_shuffle_seed1994_gpu0.json`
- `third_party/APART/exps/apart_cifar_shuffle_seed1995_gpu0.json`
- `third_party/APART/exps/apart_cifar_shuffle_concm_stage1_capped_headnorm_calibrated_full_seed1993_gpu0.json`
- `third_party/APART/exps/apart_cifar_shuffle_concm_stage1_capped_headnorm_calibrated_full_seed1994_gpu1.json`
- `third_party/APART/exps/apart_cifar_shuffle_concm_stage1_capped_headnorm_calibrated_full_seed1995_gpu1.json`
- `third_party/APART/tools/baseline_apart_full_seed1993_reference.json`
- `third_party/APART/tools/summarize_full_headnorm_gate.py`
- `third_party/APART/tools/summarize_full_headnorm_3seed.py`
- `docs/APART_CONCM_STAGE1_HEADNORM_FULL_SEED1993_RESULTS.md`
- `docs/APART_CONCM_STAGE1_HEADNORM_3SEED_FULL_RESULTS.md`

## Exact Commands

Seed1994 paired run:

```bash
cd /dev/shm/wangbomin/APART/code

LOG1=/dev/shm/wangbomin/APART/logs/apart_baseline_full_seed1994_20260618-060705.out
nohup timeout 14400s env \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/apart_cifar_shuffle_seed1994_gpu0.json \
    --text apart_baseline_full_seed1994 \
  > "$LOG1" 2>&1 &

LOG2=/dev/shm/wangbomin/APART/logs/apart_stage1_headnorm_full_seed1994_20260618-060705.out
nohup timeout 14400s env \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/apart_cifar_shuffle_concm_stage1_capped_headnorm_calibrated_full_seed1994_gpu1.json \
    --text apart_stage1_headnorm_full_seed1994 \
  > "$LOG2" 2>&1 &
```

Seed1995 paired run:

```bash
cd /dev/shm/wangbomin/APART/code

LOG1=/dev/shm/wangbomin/APART/logs/apart_baseline_full_seed1995_20260618-073913.out
nohup timeout 14400s env \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/apart_cifar_shuffle_seed1995_gpu0.json \
    --text apart_baseline_full_seed1995 \
  > "$LOG1" 2>&1 &

LOG2=/dev/shm/wangbomin/APART/logs/apart_stage1_headnorm_full_seed1995_20260618-073913.out
nohup timeout 14400s env \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/apart_cifar_shuffle_concm_stage1_capped_headnorm_calibrated_full_seed1995_gpu1.json \
    --text apart_stage1_headnorm_full_seed1995 \
  > "$LOG2" 2>&1 &
```

Final summarizer:

```bash
cd /dev/shm/wangbomin/APART/code

/opt/miniconda3/envs/torchgpu/bin/python tools/summarize_full_headnorm_3seed.py \
  --pair 1993 /dev/shm/wangbomin/APART/logs/apart_cifar_shuffle_20260616-090022.out /dev/shm/wangbomin/APART/logs/apart_stage1_headnorm_full_seed1993_20260618-015443.out \
  --pair 1994 /dev/shm/wangbomin/APART/logs/apart_baseline_full_seed1994_20260618-060705.out /dev/shm/wangbomin/APART/logs/apart_stage1_headnorm_full_seed1994_20260618-060705.out \
  --pair 1995 /dev/shm/wangbomin/APART/logs/apart_baseline_full_seed1995_20260618-073913.out /dev/shm/wangbomin/APART/logs/apart_stage1_headnorm_full_seed1995_20260618-073913.out \
  --output-json /dev/shm/wangbomin/APART/artifacts/stage1_headnorm_full_3seed/final_3seed_results.json \
  --output-md /dev/shm/wangbomin/APART/artifacts/stage1_headnorm_full_3seed/final_3seed_results.md
```

## Per-Seed Summary

| seed | base Avg | method Avg | delta Avg | base AccT | method AccT | delta AccT | base few | method few | delta few | alpha mean | alpha min | alpha max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1993 | 87.157 | 88.067 | +0.910 | 84.90 | 86.22 | +1.32 | 79.93 | 82.63 | +2.70 | 0.760144 | 0.682469 | 0.819368 |
| 1994 | 86.357 | 87.860 | +1.503 | 83.27 | 86.12 | +2.85 | 75.27 | 79.97 | +4.70 | 0.782054 | 0.690441 | 0.962880 |
| 1995 | 86.358 | 86.827 | +0.468 | 83.91 | 84.69 | +0.78 | 72.53 | 73.40 | +0.87 | 0.776202 | 0.732881 | 0.834727 |

## Mean +/- Std

| metric | baseline | method | delta |
|---|---:|---:|---:|
| Avg Acc | 86.624 +/- 0.377 | 87.584 +/- 0.542 | +0.961 +/- 0.424 |
| AccT | 84.03 +/- 0.67 | 85.68 +/- 0.70 | +1.65 +/- 0.88 |
| many | 88.70 +/- 1.00 | 90.18 +/- 0.74 | +1.49 +/- 0.96 |
| medium | 86.31 +/- 1.81 | 87.18 +/- 2.19 | +0.87 +/- 0.52 |
| few | 75.91 +/- 3.06 | 78.67 +/- 3.88 | +2.76 +/- 1.57 |

## Curve Deltas

| phase | seed1993 | seed1994 | seed1995 |
|---:|---|---|---|
| 0 | 90.42 -> 90.42 (+0.00) | 89.88 -> 89.88 (+0.00) | 88.54 -> 88.54 (+0.00) |
| 1 | 88.70 -> 89.62 (+0.92) | 89.55 -> 90.62 (+1.07) | 88.08 -> 88.50 (+0.42) |
| 2 | 88.33 -> 89.61 (+1.28) | 86.34 -> 87.76 (+1.42) | 86.83 -> 87.64 (+0.81) |
| 3 | 85.56 -> 85.92 (+0.36) | 85.46 -> 86.95 (+1.49) | 85.76 -> 86.16 (+0.40) |
| 4 | 85.03 -> 86.61 (+1.58) | 83.64 -> 85.83 (+2.19) | 85.03 -> 85.43 (+0.40) |
| 5 | 84.90 -> 86.22 (+1.32) | 83.27 -> 86.12 (+2.85) | 83.91 -> 84.69 (+0.78) |

## Alpha Stability

| seed | alpha mean | alpha min | alpha max |
|---:|---:|---:|---:|
| 1993 | 0.760144 | 0.682469 | 0.819368 |
| 1994 | 0.782054 | 0.690441 | 0.962880 |
| 1995 | 0.776202 | 0.732881 | 0.834727 |

Alpha remained non-degenerate. Seed1994 task5 had the highest alpha, `0.962880`, but this did not produce a late-task drop: seed1994 final AccT improved by `+2.85`.

## Error Counts

All paired runs completed without Traceback, RuntimeError, or CUDA OOM.

```json
{
  "1993": {"baseline": 0, "method": 0},
  "1994": {"baseline": 0, "method": 0},
  "1995": {"baseline": 0, "method": 0}
}
```

Remote final artifacts:

```text
/dev/shm/wangbomin/APART/artifacts/stage1_headnorm_full_3seed/final_3seed_results.json
/dev/shm/wangbomin/APART/artifacts/stage1_headnorm_full_3seed/final_3seed_results.md
```

## Decision

The 3-seed full validation passes the requested decision rule:

- mean Avg Acc improves by `+0.961`;
- mean AccT improves by `+1.65`;
- all 3 seeds improve Avg Acc and AccT;
- few-class accuracy improves by `+2.76` mean;
- no severe late-task degradation appears in the full curves;
- alpha is stable enough for the current rule and does not require test-set tuning;
- all runs finished without runtime errors.

Recommendation: proceed to paper-level reporting for APART + ConCM-lite Stage1 capped + head-norm effective-sum calibration. Do not add more variants before writing the result section; the next useful validation is packaging the ablation story and, if needed, repeating with a clean run directory for archival reproducibility.
