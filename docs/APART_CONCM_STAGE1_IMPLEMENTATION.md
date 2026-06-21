# APART-ConCM-lite Stage 1 Implementation

Date: 2026-06-16

Scope: minimal Stage 1 only. This is not full ConCM and does not touch TaConCM or GPA.

## What Changed

Files:

- `third_party/APART/backbone/vision_transformer_adapter_pool_a.py`
- `third_party/APART/models/apart.py`
- `third_party/APART/exps/apart_cifar_shuffle_concm_stage1.json`

Implementation:

- Exposes `pre_logits_few` from the auxiliary adapter path.
- Adds optional feature prototype memory in `models/apart.py`.
- After each phase, stores per-class means and diagonal variances for:
  - main APART feature path, used by `head`;
  - auxiliary APART feature path, used by `head_few`.
- During later phases, samples compact synthetic features from old-class diagonal Gaussian statistics.
- Adds a synthetic feature CE term through `head + head_few`.
- Leaves APART inference unchanged: evaluation still uses raw `logits + logits_few`.

## Config

New config:

```bash
python main.py --config ./exps/apart_cifar_shuffle_concm_stage1.json
```

Stats-only safety config:

```bash
python main.py --config ./exps/apart_cifar_shuffle_concm_stage1_stats.json
```

New options:

| option | default in Stage1 config | meaning |
|---|---:|---|
| `concm_stage1` | `true` | enables prototype memory and synthetic feature CE |
| `concm_stage1_loss_weight` | `0.05` | loss weight for Stage1 CE |
| `concm_stage1_synth_per_class` | `4` | synthetic features sampled per remembered class per batch |
| `concm_stage1_cov_eps` | `1e-6` | diagonal covariance numerical floor |
| `concm_stage1_var_max` | `1.0` | clamps diagonal variance before sampling |

The original official APART config is unchanged and still has Stage1 disabled implicitly.

For stats-only safety validation, `concm_stage1=true` but `concm_stage1_loss_weight=0.0` and `concm_stage1_synth_per_class=0`. This updates prototype statistics after each phase but does not add any prototype augmentation loss.

## Stats-Only Safety Note

The first stats-only attempt was stopped after task 1 started. Task 0 matched the baseline exactly and memory updated correctly, but task 1 was no longer bitwise-aligned with the baseline. The likely cause was phase-end stats extraction consuming RNG through a train-transform DataLoader.

The implementation now captures and restores Python, NumPy, Torch, and CUDA RNG state around `_update_concm_stage1_memory()`, and uses an isolated DataLoader generator for stats extraction. The stats-only validation must be rerun after this fix.

The rerun passed after the RNG fix: stats-only matched the APART baseline exactly with Acc `87.1567`, AccT `84.90`, and curve `[90.42, 88.70, 88.33, 85.56, 85.03, 84.90]`. See `docs/APART_CONCM_STAGE1_STATS_ONLY_RESULTS.md`.

The bounded Stage1 loss smoke also passed: `max_tasks=2`, `tuned_epoch=2`, `concm_stage1_loss_weight=0.05`, and `concm_stage1_synth_per_class=4` produced finite `ConCMStage1_loss` logs in phase 1 with no runtime error or CUDA OOM. See `docs/APART_CONCM_STAGE1_PHASE1_SMOKE_RESULTS.md`.

## Intended Ablation

Run after baseline only:

| method | config | expected comparison |
|---|---|---|
| APART baseline | `apart_cifar_shuffle.json` | reference Acc `87.1567`, AccT `84.90` in this environment |
| APART + Stage1 | `apart_cifar_shuffle_concm_stage1.json` | should improve old/tail retention without changing inference |

Acceptance gate:

- Stage1 should not reduce AccT by more than 0.5 points in the first full run.
- Prefer improvements in final few-shot accuracy and old accuracy.
- If Stage1 hurts Acc/AccT, reduce `concm_stage1_loss_weight` before adding Stage 2.
