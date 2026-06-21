# APART ConCM-Lite Stage1 Head-Norm Full Gate

## Scope

- Workline: APART-ConCM only.
- Experiment: one full 6-task CIFAR-100-LT shuffled B50-5 gate.
- Method: ConCM Stage1 capped prototype augmentation plus evaluation-time `head_norm_effective_sum` old-logit calibration.
- Baseline: reused existing APART full baseline because the official config, seed, task order, `tuned_epoch`, and evaluation path are compatible.
- Seed note: APART's official baseline config uses seed `1993`; this method run used seed `1993` to match the reused baseline exactly.

## Files Changed

- `third_party/APART/models/apart.py`
- `third_party/APART/exps/apart_cifar_shuffle_concm_stage1_capped_headnorm_calibrated_full_seed1993_gpu0.json`
- `third_party/APART/tools/baseline_apart_full_seed1993_reference.json`
- `third_party/APART/tools/summarize_full_headnorm_gate.py`
- `docs/APART_CONCM_STAGE1_HEADNORM_FULL_SEED1993_RESULTS.md`

## Exact Method Command

Remote working directory:

```bash
cd /dev/shm/wangbomin/APART/code
```

Launched command:

```bash
LOG=/dev/shm/wangbomin/APART/logs/apart_stage1_headnorm_full_seed1993_$(date +%Y%m%d-%H%M%S).out
nohup timeout 14400s env \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/apart_cifar_shuffle_concm_stage1_capped_headnorm_calibrated_full_seed1993_gpu0.json \
    --text apart_stage1_headnorm_full_seed1993 \
  > "$LOG" 2>&1 &
```

Actual log:

```text
/dev/shm/wangbomin/APART/logs/apart_stage1_headnorm_full_seed1993_20260618-015443.out
```

Generated artifacts:

```text
/dev/shm/wangbomin/APART/artifacts/stage1_headnorm_full_seed1993/final_results.json
/dev/shm/wangbomin/APART/artifacts/stage1_headnorm_full_seed1993/final_results.md
/dev/shm/wangbomin/APART/checkpoints/stage1_headnorm_full_seed1993/*_task0.pt ... *_task5.pt
```

## Curve Comparison

| phase | baseline | Stage1 + head-norm | delta |
|---:|---:|---:|---:|
| 0 | 90.42 | 90.42 | 0.00 |
| 1 | 88.70 | 89.62 | +0.92 |
| 2 | 88.33 | 89.61 | +1.28 |
| 3 | 85.56 | 85.92 | +0.36 |
| 4 | 85.03 | 86.61 | +1.58 |
| 5 | 84.90 | 86.22 | +1.32 |

## Final Summary

| metric | baseline | Stage1 + head-norm | delta |
|---|---:|---:|---:|
| Avg Acc | 87.1567 | 88.0667 | +0.9100 |
| AccT | 84.90 | 86.22 | +1.32 |
| many | 89.80 | 91.11 | +1.31 |
| medium | 84.26 | 84.40 | +0.14 |
| few | 79.93 | 82.63 | +2.70 |

## Per-Task Calibration Diagnostics

| phase | AccT | old acc | current acc | new->old | old->new | alpha | old norm | new norm | uncal AccT | uncal new->old |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 90.42 | 0.00 | 90.42 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 1 | 89.62 | 90.14 | 87.00 | 11.60 | 2.14 | 0.749026 | 2.360869 | 1.768353 | 88.58 | 22.20 |
| 2 | 89.61 | 88.90 | 93.90 | 5.40 | 1.97 | 0.789275 | 2.363700 | 1.865610 | 89.06 | 13.20 |
| 3 | 85.92 | 88.43 | 68.40 | 31.20 | 1.77 | 0.760580 | 2.388411 | 1.816576 | 84.26 | 51.90 |
| 4 | 86.61 | 86.76 | 85.40 | 14.20 | 1.88 | 0.682469 | 2.380344 | 1.624511 | 84.17 | 42.70 |
| 5 | 86.22 | 86.91 | 80.00 | 19.00 | 0.87 | 0.819368 | 2.367609 | 1.939942 | 85.33 | 30.70 |

Alpha stats over phases 1-5:

```text
mean=0.760144, std=0.045855, min=0.682469, max=0.819368
```

## Error Counts

```json
{
  "Traceback": 0,
  "RuntimeError": 0,
  "CUDA OOM": 0
}
```

## Decision

This one-seed full gate is promising. It improves Avg Acc and AccT, is not worse at any intermediate task, and the final many/medium/few metrics do not show tail collapse. The calibration alpha remains non-degenerate across tasks.

Recommendation: proceed to a 3-seed full validation before making stronger claims. Do not expand to more variants yet.
