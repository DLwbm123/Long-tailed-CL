# APART NC-ConCM GUIDE-Style Uncertainty Diagnostics

Date: 2026-06-19

## Scope

This rung inspects GUIDE only for the uncertainty decomposition idea. It does
not port GUIDE's multi-expert DERM architecture into APART.

Explicitly not implemented:

- `AFS_ResNet32Model`
- `DERM_ResNet`
- `StrategyController`
- `gates_per_expert`
- validation-set meta-update
- two-timescale optimization
- competitive expert specialization losses
- new multi-expert backbone
- DERM residual refinement

The implementation uses APART's existing branches as two lightweight views:

- main view: `head(pre_logits)`
- few view: `head_few(pre_logits_few)`

The uncertainty computation is no-grad only and is restricted to the current
all-seen class set.

## Files Changed

- `third_party/APART/models/apart.py`
- `third_party/APART/exps/full_concm_ladder/guide_uncertainty_disabled_identity_phase1_seed1993_gpu0.json`
- `third_party/APART/exps/full_concm_ladder/guide_uncertainty_diag_phase1_seed1993_gpu0.json`
- `third_party/APART/exps/full_concm_ladder/README.md`
- `docs/APART_FULL_CONCM_GUIDE_UNCERTAINTY_DIAGNOSTICS.md`

## GUIDE Code Inspected

- `GUIDE/utils/uncertainty.py`
  - `softmax_with_temperature`
  - `entropy`
  - `get_uncertainty_metrics`
  - `ClasswiseEMA`
- `GUIDE/trainer/trainer.py`
  - confirms GUIDE uses multi-expert logits, controllers, gates, and
    validation-time meta-update. These mechanisms are intentionally not ported.

## Implemented Diagnostic

Config switches:

```json
"concm_uncertainty_diagnostics": true,
"concm_uncertainty_temp": 1.0,
"concm_uncertainty_ema_momentum": 0.9
```

For APART main/few logits over all seen classes:

```text
p_main = softmax(logits_main / T)
p_few = softmax(logits_few / T)
p_bar = 0.5 * (p_main + p_few)
Ale = 0.5 * (entropy(p_main) + entropy(p_few))
Epi = clamp(entropy(p_bar) - Ale, min=0)
```

Logged groups:

- old classes
- current/new classes
- many / medium / few
- correct / incorrect predictions
- new samples predicted as old
- old samples predicted as new
- uncalibrated correct / incorrect
- samples changed by head-norm calibration
- class-wise EMA summaries
- correlation with calibrated and uncalibrated error

## Commands Run

Local static checks:

```bash
python3 -m py_compile third_party/APART/models/apart.py
python3 -m json.tool third_party/APART/exps/full_concm_ladder/guide_uncertainty_disabled_identity_phase1_seed1993_gpu0.json
python3 -m json.tool third_party/APART/exps/full_concm_ladder/guide_uncertainty_diag_phase1_seed1993_gpu0.json
```

Remote sync:

```bash
scp -P 20035 third_party/APART/models/apart.py \
  root@10.12.208.239:/dev/shm/wangbomin/APART/code/models/apart.py

scp -P 20035 \
  third_party/APART/exps/full_concm_ladder/README.md \
  third_party/APART/exps/full_concm_ladder/guide_uncertainty_disabled_identity_phase1_seed1993_gpu0.json \
  third_party/APART/exps/full_concm_ladder/guide_uncertainty_diag_phase1_seed1993_gpu0.json \
  root@10.12.208.239:/dev/shm/wangbomin/APART/code/exps/full_concm_ladder/
```

Remote static checks:

```bash
cd /dev/shm/wangbomin/APART/code
/opt/miniconda3/envs/torchgpu/bin/python -m py_compile models/apart.py
/opt/miniconda3/envs/torchgpu/bin/python -m json.tool \
  exps/full_concm_ladder/guide_uncertainty_disabled_identity_phase1_seed1993_gpu0.json
/opt/miniconda3/envs/torchgpu/bin/python -m json.tool \
  exps/full_concm_ladder/guide_uncertainty_diag_phase1_seed1993_gpu0.json
```

Bounded max_tasks=2 runs:

```bash
cd /dev/shm/wangbomin/APART/code

nohup timeout 5400s env \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/full_concm_ladder/guide_uncertainty_disabled_identity_phase1_seed1993_gpu0.json \
    --text guide_uncertainty_disabled_identity_phase1_seed1993 \
  > /dev/shm/wangbomin/APART/logs/guide_uncertainty_disabled_identity_phase1_seed1993_20260619-125402.out 2>&1 &

nohup timeout 5400s env \
  CUDA_VISIBLE_DEVICES=1 \
  XDG_CACHE_HOME=/dev/shm/wangbomin/APART/cache \
  TORCH_HOME=/dev/shm/wangbomin/APART/cache/torch \
  /opt/miniconda3/envs/torchgpu/bin/python main.py \
    --config ./exps/full_concm_ladder/guide_uncertainty_diag_phase1_seed1993_gpu0.json \
    --text guide_uncertainty_diag_phase1_seed1993 \
  > /dev/shm/wangbomin/APART/logs/guide_uncertainty_diag_phase1_seed1993_20260619-125402.out 2>&1 &
```

## Disabled Identity

The disabled identity run exactly reproduced the NC-ConCM max_tasks=2 reference.

| run | curve | Avg Acc | AccT | old acc | new acc | new->old | old->new | alpha |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NC-ConCM reference | [90.42, 89.62] | 90.020 | 89.62 | 90.14 | 87.00 | 11.60 | 2.14 | 0.749026 |
| uncertainty disabled identity | [90.42, 89.62] | 90.020 | 89.62 | 90.14 | 87.00 | 11.60 | 2.14 | 0.749026 |
| uncertainty diagnostics enabled | [90.42, 89.62] | 90.020 | 89.62 | 90.14 | 87.00 | 11.60 | 2.14 | 0.749026 |

Remote error grep:

```text
Traceback / RuntimeError / CUDA OOM / nan_or_inf count: 0
```

## Task1 Uncertainty Statistics

Final task1 diagnostic run, calibrated predictions:

| group | n | Ale | Epi |
| --- | ---: | ---: | ---: |
| old | 5000 | 0.162544 | 0.052245 |
| new/current | 1000 | 0.392080 | 0.108777 |
| many | 2100 | 0.180606 | 0.059149 |
| medium | 1600 | 0.159986 | 0.061333 |
| few | 2300 | 0.247629 | 0.064197 |
| correct | 5377 | 0.138787 | 0.042276 |
| incorrect | 623 | 0.736021 | 0.229027 |
| new predicted as old | 116 | 0.742515 | 0.189858 |
| old predicted as new | 107 | 1.051433 | 0.302007 |
| uncalibrated correct | 5315 | 0.131365 | 0.038994 |
| uncalibrated incorrect | 685 | 0.739556 | 0.237590 |
| calibration changed | 175 | 0.894399 | 0.297679 |

Correlation diagnostics:

| metric | value |
| --- | ---: |
| Epi vs calibrated error | 0.422717 |
| Ale vs calibrated error | 0.468168 |
| Epi vs uncalibrated error | 0.468640 |
| Ale vs uncalibrated error | 0.497028 |
| nan_or_inf | 0 |

Class-wise EMA at task1:

| group | Ale EMA | Epi EMA |
| --- | ---: | ---: |
| old | 0.307602 | 0.105520 |
| new/current | 0.597152 | 0.139822 |
| many | 0.344018 | 0.112773 |
| medium | 0.364381 | 0.119415 |
| few | 0.360745 | 0.104147 |

## Interpretation

The diagnostic is behavior-preserving: enabled and disabled runs both reproduce
NC-ConCM exactly on the max_tasks=2 gate.

The GUIDE-style uncertainty signal is meaningful in APART:

- Incorrect predictions have much higher uncertainty than correct predictions:
  Epi 0.229027 vs 0.042276, Ale 0.736021 vs 0.138787.
- New samples predicted as old have high uncertainty:
  Epi 0.189858, Ale 0.742515.
- Old samples predicted as new have even higher uncertainty:
  Epi 0.302007, Ale 1.051433.
- Current/new classes have higher uncertainty than old classes:
  Epi 0.108777 vs 0.052245.
- Samples whose predictions are changed by head-norm calibration are high
  uncertainty:
  Epi 0.297679, Ale 0.894399.

This supports using uncertainty as a diagnostic signal for old/new bias and
calibration sensitivity. It does not yet justify changing replay weights or
training dynamics.

## Recommendation

Proceed only to a small, bounded uncertainty-gated replay gate if needed. The
next rung should still compare against NC-ConCM, start with max_tasks=2, and
only reduce replay when current-task old-class attraction is high. Do not
launch full 6-task or 3-seed uncertainty-gated experiments before a max_tasks=2
gate passes.
