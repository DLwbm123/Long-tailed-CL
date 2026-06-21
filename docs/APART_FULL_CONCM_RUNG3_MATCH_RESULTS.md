# APART Full-ConCM Ladder Rung 3: Dual-Path Feature-Structure Matching

Date: 2026-06-18

Scope: bounded `max_tasks=2` gate only. No full 6-task run, no 3-seed run, no T-DSM margin optimization, no tail weighting, no route-aware module, no projector, and no GPA-style imprinting were launched or implemented.

Primary reference:

```text
NC-ConCM = APART + Stage1-capped prototype replay + head_norm_effective_sum old-logit calibration
```

Rung 3 candidate:

```text
NC-ConCM + dual-path feature-space cosine match loss
```

## Files Changed

- `third_party/APART/models/apart.py`
  - Added optional `concm_use_match_loss`.
  - Added `concm_match_lambda`, `concm_match_warmup_epoch`, `concm_match_paths`, `concm_match_detach_anchors`, `concm_match_current_only`, `concm_match_tail_weight`.
  - Builds detached anchors from Stage1 memory for old classes and phase-start current-task prototypes for current classes.
  - Applies cosine feature-to-anchor matching in `pre_logits` and `pre_logits_few`.
  - Does not anchor or directly optimize `head.weight` or `head_few.weight`.
  - Adds match loss, CE ratio, path-specific loss/cosine, and nonzero-gradient diagnostics.
- `third_party/APART/exps/full_concm_ladder/README.md`
- `third_party/APART/exps/full_concm_ladder/rung3_match_disabled_identity_phase1_seed1993_gpu0.json`
- `third_party/APART/exps/full_concm_ladder/rung3_match_smoke_seed1993_gpu0.json`
- `third_party/APART/exps/full_concm_ladder/rung3_match_phase1_seed1993_gpu0.json`
- `docs/APART_FULL_CONCM_RUNG3_MATCH_RESULTS.md`

## Static Checks

Local:

```bash
python3 -m py_compile third_party/APART/models/apart.py
for f in third_party/APART/exps/full_concm_ladder/rung3_match_*.json; do
  python3 -m json.tool "$f" >/dev/null || exit 1
done
```

Remote:

```bash
cd /dev/shm/wangbomin/APART/code
/opt/miniconda3/envs/torchgpu/bin/python -m py_compile models/apart.py
for f in exps/full_concm_ladder/rung3_match_*.json; do
  /opt/miniconda3/envs/torchgpu/bin/python -m json.tool "$f" >/tmp/$(basename "$f").checked || exit 1
done
```

Result: passed.

## Exact Remote Commands

Disabled identity:

```bash
cd /dev/shm/wangbomin/APART/code
LOG=/dev/shm/wangbomin/APART/logs/rung3_match_disabled_identity_phase1_seed1993_20260618-125919.out
nohup timeout 3600s env \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/full_concm_ladder/rung3_match_disabled_identity_phase1_seed1993_gpu0.json \
    --text rung3_match_disabled_identity_phase1_seed1993 \
  > "$LOG" 2>&1 &
```

Smoke:

```bash
cd /dev/shm/wangbomin/APART/code
LOG=/dev/shm/wangbomin/APART/logs/rung3_match_smoke_seed1993_20260618-125928.out
nohup timeout 1800s env \
  CUDA_VISIBLE_DEVICES=1 \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/full_concm_ladder/rung3_match_smoke_seed1993_gpu0.json \
    --text rung3_match_smoke_seed1993 \
  > "$LOG" 2>&1 &
```

Accuracy gate:

```bash
cd /dev/shm/wangbomin/APART/code
LOG=/dev/shm/wangbomin/APART/logs/rung3_match_phase1_seed1993_20260618-130948.out
nohup timeout 3600s env \
  CUDA_VISIBLE_DEVICES=1 \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/full_concm_ladder/rung3_match_phase1_seed1993_gpu0.json \
    --text rung3_match_phase1_seed1993 \
  > "$LOG" 2>&1 &
```

## Disabled-Mode Identity

`concm_use_match_loss=false` reproduced the NC-ConCM reference exactly.

| run | curve | Avg Acc | AccT | old acc | new acc | new_eval_pred_old_rate | old_eval_pred_new_rate | alpha |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NC-ConCM reference | `[90.42, 89.62]` | 90.020 | 89.62 | 90.14 | 87.00 | 11.60 | 2.14 | 0.749026 |
| rung3 disabled identity | `[90.42, 89.62]` | 90.020 | 89.62 | 90.14 | 87.00 | 11.60 | 2.14 | 0.749026 |

No `ConCMMatch` loss line appeared in disabled mode.

## Smoke Result

Config: `tuned_epoch=2`, `max_tasks=2`, `concm_use_match_loss=true`, `concm_match_warmup_epoch=1`.

Evidence:

- `ConCMMatchAnchors task=1 active=True old_classes=50 current_classes=10 valid_classes=60 ... nan_or_inf=0`.
- Match loss was finite and nonzero:
  - raw `0.381213`
  - weighted `0.003812`
  - main loss `0.381594`
  - few loss `0.380831`
  - match/CE `0.012869`
- Feature-to-anchor cosine was finite:
  - main cosine `0.618406`
  - few cosine `0.619169`
- Match loss had nonzero gradients: `ConCMMatch_nonzero_grad_batches 16/16`.
- No `Traceback`, `RuntimeError`, CUDA OOM, or nonzero `nan_or_inf` event.

Smoke final accuracy was not used for the method decision because it intentionally used only 2 epochs.

## Accuracy Gate

Config: normal APART phase1 budget, `tuned_epoch=10`, `max_tasks=2`, seed `1993`, `concm_match_lambda=0.01`, `concm_match_warmup_epoch=5`, `concm_match_paths=both`.

Primary comparison is against NC-ConCM.

| run | curve | Avg Acc | AccT | old acc | new acc | new_eval_pred_old_rate | old_eval_pred_new_rate | alpha | old norm | new norm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NC-ConCM reference | `[90.42, 89.62]` | 90.020 | 89.62 | 90.14 | 87.00 | 11.60 | 2.14 | 0.749026 | 2.360869 | 1.768353 |
| Rung3 match | `[90.42, 89.58]` | 90.000 | 89.58 | 90.10 | 87.00 | 11.60 | 2.18 | 0.749008 | 2.360887 | 1.768322 |
| Delta vs NC-ConCM | - | -0.020 | -0.04 | -0.04 | +0.00 | +0.00 | +0.04 | -0.000018 | +0.000018 | -0.000031 |

Uncalibrated diagnostics:

| run | uncalibrated Avg | uncalibrated AccT | uncalibrated old acc | uncalibrated new acc | uncalibrated new_eval_pred_old_rate | uncalibrated old_eval_pred_new_rate |
|---|---:|---:|---:|---:|---:|---:|
| NC-ConCM reference | 89.500 | 88.58 | 90.86 | 77.20 | 22.20 | 0.76 |
| Rung3 match | 89.500 | 88.58 | 90.86 | 77.20 | 22.20 | 0.76 |

## Match Diagnostics

Task1 anchor construction:

| old classes | current classes | valid classes | main anchor norm mean | few anchor norm mean | nan_or_inf |
|---:|---:|---:|---:|---:|---:|
| 50 | 10 | 60 | 30.529680 | 28.848194 | 0 |

Final active epoch diagnostics:

| epoch | raw match | weighted match | main loss | few loss | main cosine | few cosine | match/CE | raw match/CE | nonzero grad batches |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.341716 | 0.003417 | 0.343025 | 0.340408 | 0.656975 | 0.659592 | 0.041379 | 4.137905 | 16/16 |

The match path is active and stable, but its effect on final accuracy is negligible and slightly negative relative to NC-ConCM.

## Safety Check

Remote grep counts for all three rung3 logs:

```text
Traceback / RuntimeError / CUDA OOM / nonzero nan_or_inf count = 0
```

Final remote GPU status after runs:

```text
GPU0: 1 MiB, 0%
GPU1: 1 MiB, 0%
```

## Decision

Do not keep rung3 as a main method in its current form, and do not launch full 6-task or 3-seed validation.

Reason:

- It does not beat NC-ConCM on Avg Acc or AccT.
- It preserves new accuracy and does not increase `new_eval_pred_old_rate`, but it slightly reduces old accuracy and AccT.
- The calibrated and uncalibrated metrics are effectively unchanged, so the simple current-only cosine match is not adding a useful mechanism beyond NC-ConCM.

Recommendation: keep the code/config as diagnostic-only. If continuing the full-ConCM ladder, the next useful step should introduce a genuinely structural component with a bounded gate, but not by enabling this rung3 loss as a default module.
