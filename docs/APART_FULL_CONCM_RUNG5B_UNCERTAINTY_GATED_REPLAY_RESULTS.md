# APART NC-ConCM Rung5b Uncertainty-Gated Replay Results

Date: 2026-06-19

## Scope

Rung5b tests a conservative uncertainty-gated replay reduction on top of
NC-ConCM:

```text
NC-ConCM = Stage1-capped prototype replay + head_norm_effective_sum calibration
Rung5b = NC-ConCM + uncertainty-gated Stage1 replay reduction
```

This rung does not implement GUIDE, DERM, `StrategyController`, multi-expert
backbones, validation meta-update, two-timescale optimization, classifier-head
anchoring, feature matching, prototype calibration, or route-aware losses.

## Files Changed

- `third_party/APART/models/apart.py`
- `third_party/APART/exps/full_concm_ladder/rung5b_ugr_disabled_identity_phase1_seed1993_gpu0.json`
- `third_party/APART/exps/full_concm_ladder/rung5b_ugr_smoke_seed1993_gpu0.json`
- `third_party/APART/exps/full_concm_ladder/rung5b_ugr_phase1_seed1993_gpu0.json`
- `third_party/APART/exps/full_concm_ladder/README.md`
- `docs/APART_FULL_CONCM_RUNG5B_UNCERTAINTY_GATED_REPLAY_RESULTS.md`

## Implementation

New config switches:

```json
"concm_uncertainty_gated_replay": true,
"concm_ugr_min_scale": 0.5,
"concm_ugr_max_scale": 1.0,
"concm_ugr_target_ale": 0.6,
"concm_ugr_use_batch_current_only": true
```

During task `t > 0`, the training loop computes APART main/few uncertainty over
the current all-seen class set:

```text
logits_main = head(pre_logits)
logits_few = head_few(pre_logits_few)
Ale/Epi from 0.5 * (softmax(logits_main/T) + softmax(logits_few/T))
```

For current-task real samples:

```text
replay_scale = clip(target_ale / (batch_Ale_current + eps), 0.5, 1.0)
effective_stage1_loss_weight = base_stage1_loss_weight * replay_scale
effective_synth_cap = round(base_synth_cap * replay_scale)
```

Only Stage1 synthetic replay is scaled. CE, APART routing, classifier heads,
prototype memory, and head-norm evaluation calibration are unchanged.

## Commands

Local static checks:

```bash
python3 -m py_compile third_party/APART/models/apart.py
python3 -m json.tool third_party/APART/exps/full_concm_ladder/rung5b_ugr_disabled_identity_phase1_seed1993_gpu0.json
python3 -m json.tool third_party/APART/exps/full_concm_ladder/rung5b_ugr_smoke_seed1993_gpu0.json
python3 -m json.tool third_party/APART/exps/full_concm_ladder/rung5b_ugr_phase1_seed1993_gpu0.json
```

Remote static check:

```bash
cd /dev/shm/wangbomin/APART/code
/opt/miniconda3/envs/torchgpu/bin/python -m py_compile models/apart.py
/opt/miniconda3/envs/torchgpu/bin/python -m json.tool exps/full_concm_ladder/rung5b_ugr_disabled_identity_phase1_seed1993_gpu0.json
/opt/miniconda3/envs/torchgpu/bin/python -m json.tool exps/full_concm_ladder/rung5b_ugr_smoke_seed1993_gpu0.json
/opt/miniconda3/envs/torchgpu/bin/python -m json.tool exps/full_concm_ladder/rung5b_ugr_phase1_seed1993_gpu0.json
```

Smoke:

```bash
cd /dev/shm/wangbomin/APART/code
timeout 2400s env \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/full_concm_ladder/rung5b_ugr_smoke_seed1993_gpu0.json \
    --text rung5b_ugr_smoke_seed1993
```

Disabled identity and accuracy gate:

```bash
cd /dev/shm/wangbomin/APART/code

timeout 5400s env \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/full_concm_ladder/rung5b_ugr_disabled_identity_phase1_seed1993_gpu0.json \
    --text rung5b_ugr_disabled_identity_phase1_seed1993

timeout 5400s env \
  CUDA_VISIBLE_DEVICES=1 \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/full_concm_ladder/rung5b_ugr_phase1_seed1993_gpu0.json \
    --text rung5b_ugr_phase1_seed1993
```

Remote logs:

- `/dev/shm/wangbomin/APART/logs/rung5b_ugr_smoke_seed1993_20260619-135104.out`
- `/dev/shm/wangbomin/APART/logs/rung5b_ugr_disabled_identity_phase1_seed1993_20260619-140400.out`
- `/dev/shm/wangbomin/APART/logs/rung5b_ugr_phase1_seed1993_20260619-140400.out`

## Validation

Static checks passed locally and remotely.

Smoke passed:

- uncertainty values were finite;
- `ConCMUGR_replay_scale` stayed within `[0.5, 1.0]`;
- Stage1 gradients were nonzero: `ConCMStage1_nonzero_grad_batches 16/16`;
- `nan_or_inf=0`;
- no Traceback, RuntimeError, or CUDA OOM.

Smoke evidence:

| epoch | Ale current | Epi current | train current pred old | replay scale | effective weight | effective synth cap | Stage1 grad |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| task1 e1/2 | 1.439290 | 0.112265 | 47.57 | 0.591833 | 0.029592 | 28.38 | 16/16 |
| task1 e2/2 | 0.576082 | 0.056038 | 27.83 | 0.957561 | 0.047878 | 45.94 | 16/16 |

## Disabled Identity

Disabled mode exactly reproduced the NC-ConCM reference.

| run | curve | Avg Acc | AccT | old | new | new->old | old->new | alpha |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NC-ConCM reference | [90.42, 89.62] | 90.020 | 89.62 | 90.14 | 87.00 | 11.60 | 2.14 | 0.749026 |
| Rung5b disabled identity | [90.42, 89.62] | 90.020 | 89.62 | 90.14 | 87.00 | 11.60 | 2.14 | 0.749026 |

## Accuracy Gate

Primary comparison is against NC-ConCM.

| run | curve | Avg Acc | AccT | old | new | new->old | old->new | alpha |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NC-ConCM | [90.42, 89.62] | 90.020 | 89.62 | 90.14 | 87.00 | 11.60 | 2.14 | 0.749026 |
| Rung5b UGR | [90.42, 89.52] | 89.970 | 89.52 | 90.12 | 86.50 | 12.10 | 2.36 | 0.748466 |
| delta | [0.00, -0.10] | -0.050 | -0.10 | -0.02 | -0.50 | +0.50 | +0.22 | -0.000560 |

Uncalibrated task1 comparison:

| run | uncal AccT | uncal old | uncal new | uncal new->old | uncal old->new |
| --- | ---: | ---: | ---: | ---: | ---: |
| NC-ConCM | 88.58 | 90.86 | 77.20 | 22.20 | 0.76 |
| Rung5b UGR | 88.53 | 90.94 | 76.50 | 22.70 | 0.82 |

Rung5b task1 replay-scale summary:

| metric | value |
| --- | ---: |
| mean replay scale | 0.878812 |
| min replay scale | 0.557563 |
| max replay scale | 0.991507 |
| mean effective Stage1 weight | 0.043941 |
| base Stage1 weight | 0.050000 |
| mean effective synth cap | 42.14 |
| base synth cap | 48 |

Final uncertainty diagnostics remained finite:

| group | Ale | Epi |
| --- | ---: | ---: |
| correct | 0.138198 | 0.042833 |
| incorrect | 0.757057 | 0.217888 |
| new predicted old | 0.698142 | 0.173535 |
| old predicted new | 1.100131 | 0.310357 |
| new/current | 0.389875 | 0.103119 |
| old | 0.165715 | 0.052798 |

Remote error grep:

```text
rung5b_ugr_disabled_identity_phase1_seed1993_20260619-140400.out: 0
rung5b_ugr_phase1_seed1993_20260619-140400.out: 0
```

## Decision

Reject Rung5b as a main method.

Reason:

- Avg Acc drops from 90.020 to 89.970.
- AccT drops from 89.62 to 89.52.
- New/current accuracy drops from 87.00 to 86.50.
- New->old bias increases from 11.60 to 12.10.
- Old accuracy is effectively unchanged, so the replay reduction does not buy
  useful retention.

Rung5b should remain diagnostic-only. Do not launch full 6-task or 3-seed runs
for this module.

## Recommendation

Keep NC-ConCM as the current strong method. The uncertainty signal is useful for
analysis, but this conservative replay reduction does not improve the bounded
2-task gate. The next useful step should not be another replay-strength variant
unless there is a more specific mechanism target, such as using uncertainty only
for evaluation diagnostics or post-hoc calibration analysis.
