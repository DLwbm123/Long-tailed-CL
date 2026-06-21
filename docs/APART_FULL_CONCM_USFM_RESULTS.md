# APART NC-ConCM + USFM Bounded Gate Results

Date: 2026-06-20

## Scope

This gate tests Uncertainty-Selective Feature Matching (USFM) on top of the validated NC-ConCM reference:

- Stage1-capped prototype replay remains enabled.
- `head_norm_effective_sum` old-logit calibration remains enabled.
- USFM applies only to current-task real samples.
- No classifier-head anchoring or modification is added.
- No full 6-task or 3-seed run was launched.

Primary reference is NC-ConCM seed1993 max_tasks=2:

| method | curve | Avg Acc | AccT | old | new | new->old | old->new | alpha |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NC-ConCM reference | [90.42, 89.62] | 90.020 | 89.62 | 90.14 | 87.00 | 11.60 | 2.14 | 0.749026 |

## Files Changed

- `third_party/APART/models/apart.py`
  - Added optional USFM config switches.
  - Added current-task EMA feature anchors for `pre_logits` and `pre_logits_few`.
  - Added detached Ale/Epi gate and cosine feature-to-anchor loss.
  - Added USFM loss, gate, anchor, uncertainty, and gradient diagnostics.
- `third_party/APART/exps/full_concm_ladder/usfm_disabled_identity_phase1_seed1993_gpu0.json`
- `third_party/APART/exps/full_concm_ladder/usfm_smoke_seed1993_gpu0.json`
- `third_party/APART/exps/full_concm_ladder/usfm_phase1_seed1993_gpu0.json`
- `third_party/APART/exps/full_concm_ladder/README.md`
- `docs/APART_FULL_CONCM_USFM_RESULTS.md`

## Commands

Local static checks:

```bash
python3 -m py_compile third_party/APART/models/apart.py
python3 -m json.tool third_party/APART/exps/full_concm_ladder/usfm_disabled_identity_phase1_seed1993_gpu0.json
python3 -m json.tool third_party/APART/exps/full_concm_ladder/usfm_smoke_seed1993_gpu0.json
python3 -m json.tool third_party/APART/exps/full_concm_ladder/usfm_phase1_seed1993_gpu0.json
```

Remote static checks:

```bash
cd /dev/shm/wangbomin/APART/code
/opt/miniconda3/envs/torchgpu/bin/python -m py_compile models/apart.py
/opt/miniconda3/envs/torchgpu/bin/python -m json.tool exps/full_concm_ladder/usfm_disabled_identity_phase1_seed1993_gpu0.json
/opt/miniconda3/envs/torchgpu/bin/python -m json.tool exps/full_concm_ladder/usfm_smoke_seed1993_gpu0.json
/opt/miniconda3/envs/torchgpu/bin/python -m json.tool exps/full_concm_ladder/usfm_phase1_seed1993_gpu0.json
```

USFM smoke:

```bash
cd /dev/shm/wangbomin/APART/code
nohup timeout 3600s env \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/full_concm_ladder/usfm_smoke_seed1993_gpu0.json \
    --text usfm_smoke_seed1993 \
  > /dev/shm/wangbomin/APART/logs/usfm_smoke_seed1993_20260619-150142.out 2>&1 &
```

Disabled identity and USFM phase1 gate:

```bash
cd /dev/shm/wangbomin/APART/code

nohup timeout 5400s env \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  CUDA_VISIBLE_DEVICES=0 \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/full_concm_ladder/usfm_disabled_identity_phase1_seed1993_gpu0.json \
    --text usfm_disabled_identity_seed1993 \
  > /dev/shm/wangbomin/APART/logs/usfm_disabled_identity_seed1993_20260620-032838.out 2>&1 &

nohup timeout 5400s env \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  CUDA_VISIBLE_DEVICES=1 \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/full_concm_ladder/usfm_phase1_seed1993_gpu0.json \
    --text usfm_phase1_seed1993 \
  > /dev/shm/wangbomin/APART/logs/usfm_phase1_seed1993_20260620-032838.out 2>&1 &
```

## Validation

Static checks passed locally and remotely.

Disabled identity passed. It reproduced the NC-ConCM seed1993 max_tasks=2 reference exactly at logged precision:

| run | curve | Avg Acc | AccT | old | new | new->old | old->new | alpha |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NC-ConCM reference | [90.42, 89.62] | 90.020 | 89.62 | 90.14 | 87.00 | 11.60 | 2.14 | 0.749026 |
| USFM disabled identity | [90.42, 89.62] | 90.020 | 89.62 | 90.14 | 87.00 | 11.60 | 2.14 | 0.749026 |

No `Traceback`, `RuntimeError`, CUDA OOM, or NaN/Inf diagnostics were found in the smoke, disabled identity, or USFM phase1 logs.

## Smoke Diagnostics

Smoke run: `usfm_smoke_seed1993_20260619-150142.out`, `max_tasks=2`, `tuned_epoch=6`.

USFM activated after warmup and produced finite, non-degenerate diagnostics:

| metric | value |
|---|---:|
| `ConCMUSFM_loss_raw` | 0.309603 |
| `ConCMUSFM_loss_weighted` | 0.001548 |
| `ConCMUSFM_to_CE` | 0.028791 |
| `ConCMUSFM_Ale_mean/std` | 0.613119 / 0.702408 |
| `ConCMUSFM_Epi_mean/std` | 0.087037 / 0.124670 |
| `gate_mean/std/min/max` | 0.496009 / 0.155716 / 0.038523 / 0.989967 |
| `gate_correct` | 0.496849 |
| `gate_incorrect` | 0.493063 |
| `gate_new_pred_old` | 0.508811 |
| `anchor_classes` | 10 |
| `anchor_main_norm / anchor_few_norm` | 30.506269 / 31.287113 |
| `anchor_nan_or_inf` | 0 |
| `nonzero_grad_batches` | 16/16 |

## Accuracy Gate

Seed1993, max_tasks=2, normal 10-epoch budget:

| method | curve | Avg Acc | AccT | old | new | new->old | old->new | alpha | old norm | new norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NC-ConCM | [90.42, 89.62] | 90.020 | 89.62 | 90.14 | 87.00 | 11.60 | 2.14 | 0.749026 | 2.360869 | 1.768353 |
| NC-ConCM + USFM | [90.42, 89.60] | 90.010 | 89.60 | 90.12 | 87.00 | 11.60 | 2.16 | 0.748994 | 2.360869 | 1.768278 |
| Delta | [0.00, -0.02] | -0.010 | -0.02 | -0.02 | 0.00 | 0.00 | +0.02 | -0.000032 | 0.000000 | -0.000075 |

Uncalibrated task1 metrics were unchanged at logged precision:

| method | uncal total | uncal old | uncal new | uncal new->old | uncal old->new |
|---|---:|---:|---:|---:|---:|
| NC-ConCM | 88.58 | 90.86 | 77.20 | 22.20 | 0.76 |
| NC-ConCM + USFM | 88.58 | 90.86 | 77.20 | 22.20 | 0.76 |

## USFM Gate Statistics

Final task1 epoch10 USFM training diagnostics:

| metric | value |
|---|---:|
| `ConCMUSFM_loss_raw` | 0.297762 |
| `ConCMUSFM_loss_weighted` | 0.001489 |
| `ConCMUSFM_main_loss` | 0.149796 |
| `ConCMUSFM_few_loss` | 0.147966 |
| `ConCMUSFM_main_cos` | 0.690688 |
| `ConCMUSFM_few_cos` | 0.695989 |
| `ConCMUSFM_to_CE` | 0.018210 |
| `ConCMUSFM_Ale_mean/std` | 0.708234 / 0.752589 |
| `ConCMUSFM_Epi_mean/std` | 0.131492 / 0.158349 |
| `gate_mean/std/min/max` | 0.494483 / 0.158040 / 0.092487 / 0.964737 |
| `gate_correct` | 0.490871 |
| `gate_incorrect` | 0.511518 |
| `gate_new_pred_old` | 0.529820 |
| `gate_correct_n` | 36.94 |
| `gate_incorrect_n` | 10.19 |
| `gate_new_pred_old_n` | 9.19 |
| `selected_n` | 47.12 |
| `anchor_classes` | 10 |
| `anchor_main_norm / anchor_few_norm` | 28.888477 / 28.294580 |
| `anchor_nan_or_inf` | 0 |
| `nonzero_grad_batches` | 16/16 |

Evaluation uncertainty remained mechanism-consistent:

| group | Ale | Epi | n |
|---|---:|---:|---:|
| correct | 0.138939 | 0.042313 | 5376 |
| incorrect | 0.736142 | 0.228649 | 624 |
| new predicted old | 0.743346 | 0.189778 | 116 |
| old predicted new | 1.045676 | 0.299375 | 108 |

Correlation with error remained positive:

| metric | value |
|---|---:|
| `epi_error_corr` | 0.422059 |
| `ale_error_corr` | 0.468078 |
| `epi_uncalibrated_error_corr` | 0.468469 |
| `ale_uncalibrated_error_corr` | 0.497051 |
| `unc_nan_or_inf` | 0 |

## Recommendation

USFM is stable and the diagnostics are useful, but it does not improve over NC-ConCM in the bounded seed1993 max_tasks=2 gate. It slightly lowers Avg Acc, AccT, and old accuracy while preserving new accuracy and new-to-old bias. The gate is non-degenerate and gradients are nonzero, but the selected signal is not strong enough to provide an accuracy or bias benefit.

Recommendation: keep USFM as diagnostic-only for now. Do not launch full 6-task or 3-seed runs for this USFM variant. The current main method should remain NC-ConCM unless a future USFM redesign shows a clear max_tasks=2 gain over NC-ConCM.
