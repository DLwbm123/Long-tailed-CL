# APART ConCM-lite Stage 1 Head-Norm 3-Seed Phase1 Results

Date: 2026-06-17

Scope: bounded 3-seed, 2-task APART CIFAR shuffle validation. No full 6-task experiment was launched.

## Files Changed

- `third_party/APART/exps/apart_cifar_shuffle_phase1_3seed_gpu0.json`
  - Baseline config, `max_tasks=2`, seeds `[0, 1, 2]`, GPU0.
- `third_party/APART/exps/apart_cifar_shuffle_concm_stage1_capped_headnorm_calibrated_phase1_3seed_gpu1.json`
  - Stage1-capped + `head_norm_effective_sum` calibration config, `max_tasks=2`, seeds `[0, 1, 2]`, GPU1.
- `third_party/APART/tools/summarize_phase1_3seed_gate.py`
  - Log parser and summary generator for per-seed, mean/std, deltas, and error counts.

Pre-run checks:

```bash
python3 -m py_compile third_party/APART/models/apart.py third_party/APART/tools/summarize_phase1_3seed_gate.py
python3 -m json.tool third_party/APART/exps/apart_cifar_shuffle_phase1_3seed_gpu0.json
python3 -m json.tool third_party/APART/exps/apart_cifar_shuffle_concm_stage1_capped_headnorm_calibrated_phase1_3seed_gpu1.json
```

Remote checks with `/opt/miniconda3/envs/torchgpu/bin/python` also passed.

## Exact Commands

Baseline:

```bash
cd /dev/shm/wangbomin/APART/code
nohup timeout 7200s env \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/apart_cifar_shuffle_phase1_3seed_gpu0.json \
    --text apart_phase1_3seed_baseline \
  > /dev/shm/wangbomin/APART/logs/apart_phase1_3seed_baseline_20260617-103104.out 2>&1 &
```

Stage1-capped + head-norm calibration:

```bash
cd /dev/shm/wangbomin/APART/code
nohup timeout 7200s env \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/apart_cifar_shuffle_concm_stage1_capped_headnorm_calibrated_phase1_3seed_gpu1.json \
    --text apart_phase1_3seed_headnorm_cal \
  > /dev/shm/wangbomin/APART/logs/apart_phase1_3seed_headnorm_cal_20260617-103313.out 2>&1 &
```

Summarizer:

```bash
cd /dev/shm/wangbomin/APART/code
/opt/miniconda3/envs/torchgpu/bin/python tools/summarize_phase1_3seed_gate.py \
  --baseline-log /dev/shm/wangbomin/APART/logs/apart_phase1_3seed_baseline_20260617-103104.out \
  --calibrated-log /dev/shm/wangbomin/APART/logs/apart_phase1_3seed_headnorm_cal_20260617-103313.out \
  --output-json /dev/shm/wangbomin/APART/artifacts/stage1_headnorm_3seed/final_results.json \
  --output-md /dev/shm/wangbomin/APART/artifacts/stage1_headnorm_3seed/final_results.md
```

Note: the initial combined launch command only started the baseline correctly. The calibrated run was then launched separately with the command above. No invalid calibrated training process remained.

## Per-Seed Results

| seed | method | curve | Avg Acc | task1 AccT | old acc | new acc | new->old | old->new | alpha | old norm | new norm |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | `[90.7, 85.88]` | 88.290 | 85.88 | 88.72 | 71.70 | 24.10 | 3.30 | n/a | n/a | n/a |
| 0 | stage1_headnorm | `[90.7, 86.07]` | 88.385 | 86.07 | 88.32 | 74.80 | 20.40 | 4.10 | 0.754673 | 2.217427 | 1.673433 |
| 1 | baseline | `[90.82, 89.83]` | 90.325 | 89.83 | 90.10 | 88.50 | 8.30 | 1.46 | n/a | n/a | n/a |
| 1 | stage1_headnorm | `[90.82, 90.65]` | 90.735 | 90.65 | 90.96 | 89.10 | 7.40 | 1.56 | 0.798794 | 2.255254 | 1.801484 |
| 2 | baseline | `[88.74, 87.63]` | 88.185 | 87.63 | 87.28 | 89.40 | 7.50 | 3.10 | n/a | n/a | n/a |
| 2 | stage1_headnorm | `[88.74, 87.93]` | 88.335 | 87.93 | 87.54 | 89.90 | 7.10 | 2.56 | 0.777391 | 2.385568 | 1.854519 |

## Mean +/- Std

| method | Avg Acc | task1 AccT | old acc | new acc | new->old | old->new | alpha |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 88.933 +/- 0.985 | 87.780 +/- 1.616 | 88.700 +/- 1.151 | 83.200 +/- 8.140 | 13.300 +/- 7.644 | 2.620 +/- 0.804 | n/a |
| stage1_headnorm | 89.152 +/- 1.120 | 88.217 +/- 1.881 | 88.940 +/- 1.463 | 84.600 +/- 6.937 | 11.633 +/- 6.200 | 2.740 +/- 1.080 | 0.776953 +/- 0.018015 |

## Seed-Wise Deltas

Delta = stage1_headnorm - baseline.

| seed | Avg Acc | task1 AccT | old acc | new acc | new->old |
|---|---:|---:|---:|---:|---:|
| 0 | +0.095 | +0.19 | -0.40 | +3.10 | -3.70 |
| 1 | +0.410 | +0.82 | +0.86 | +0.60 | -0.90 |
| 2 | +0.150 | +0.30 | +0.26 | +0.50 | -0.40 |

## Uncalibrated vs Calibrated

| seed | uncal AccT | cal AccT | uncal old | cal old | uncal new | cal new | uncal new->old | cal new->old |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 85.13 | 86.07 | 89.82 | 88.32 | 61.70 | 74.80 | 36.50 | 20.40 |
| 1 | 90.30 | 90.65 | 91.48 | 90.96 | 84.40 | 89.10 | 13.40 | 7.40 |
| 2 | 87.68 | 87.93 | 88.70 | 87.54 | 82.60 | 89.90 | 15.10 | 7.10 |

The calibration consistently improves new-class accuracy and reduces new-to-old prediction bias within the Stage1 run, while trading some old accuracy inside the calibrated method.

## Error Counts

| method | Traceback | RuntimeError | CUDA OOM |
|---|---:|---:|---:|
| baseline | 0 | 0 | 0 |
| stage1_headnorm | 0 | 0 | 0 |

Final remote status: no APART/GPA/TaConCM process was running; both GPUs were idle.

Remote artifacts:

```text
/dev/shm/wangbomin/APART/artifacts/stage1_headnorm_3seed/final_results.json
/dev/shm/wangbomin/APART/artifacts/stage1_headnorm_3seed/final_results.md
```

## Decision

The bounded 3-seed 2-task gate is promising:

- Mean Avg Acc improves: `+0.219`.
- Mean task1 AccT improves: `+0.437`.
- Mean new acc improves: `+1.400`.
- Mean old acc improves: `+0.240`.
- Avg Acc, task1 AccT, and new acc improve in all 3 seeds.
- Old acc improves in 2 of 3 seeds and is positive on average.
- New-to-old prediction bias decreases in all 3 seeds.
- Alpha is stable and non-degenerate: mean `0.776953`, std `0.018015`, range `0.754673-0.798794`.

Recommendation: proceed to one seed0 full 6-task run for Stage1-capped + head-norm calibration, paired with the already trusted APART baseline reference or a fresh seed0 full baseline only if strict same-seed comparison is required. Do not launch 3-seed full yet.
