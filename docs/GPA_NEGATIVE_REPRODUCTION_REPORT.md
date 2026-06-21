# GPA Negative Reproduction Report

Date: 2026-06-16

This report formally closes the current faithful GPA reproduction route for this workspace. It does not claim a successful GPA reproduction and should not be used to report GPA-paper numbers.

## Process Status

Remote server 35 was checked for long-running GPA/TaConCM jobs matching `finetune_gpa`, `singlehead`, `gpa_phase1`, `GPA_singlehead`, `taconcm`, `TaConCM`, `gpa_on_gvalign`, `run_gpa`, and `cifar100lt_singlehead`.

Result:

```text
No GPA full reproduction process is running.
```

No PID needed to be stopped.

## Final Decision

Stop faithful GPA reproduction in this repository for now.

Reasons:

1. GPA has no available public training implementation. The public repository cannot be used as an official implementation audit source.
2. The paper remains a method specification, not executable source of truth.
3. Multiple independent implementation routes executed the intended GPA paths, but active-anchor GPA did not reproduce the reported CIFAR-100-LT scale.
4. The failure is not explained by dead code: prototype imprinting and old-head freezing/restoration were both verified.
5. Current evidence points to incompatibility among paper bias, dynamic anchor, raw logit calibration, replay/head protocol, and the available training setup.

Do not launch full seed0 GPA reproduction runs from the current implementations.

## Evidence Summary

### Original Long-tailed CL / TaConCM Framework

Run: `/dev/shm/wangbomin/LongTailedCL/runs/taconcm_ablation_20260613-025914_s0`

| method | final_acc | avg_inc_acc | key observation |
|---|---:|---:|---|
| finetune | 4.31 | 11.464 | severe old-task collapse |
| finetune_gpa | 4.87 | 11.698 | GPA does not prevent collapse |
| taconcm_stage4_full | 5.16 | 11.681 | best stage, but only +0.85 over finetune |

The stage4 final accuracy matrix showed that after each new phase all previous tasks dropped to `0.0%`, and the final score was mostly the last task.

Conclusion: this path is dominated by catastrophic forgetting. GPA anchoring and current TaConCM components do not solve all-seen LT-CIL retention in the no-replay setup.

### GVAlign Multi-Head Plugin Route

Protocol: GVAlign / Long-Tailed-CIL, CIFAR-100-LT, fixed shuffled order, `50 + 5x10`, 20 exemplars/class, herding, raw TAg evaluation.

Key phase1 results:

| route | old/base acc | phase1 all-seen TAg | conclusion |
|---|---:|---:|---|
| Finetuning gate | 32.06 | 33.35 | baseline gate |
| default Finetuning+GPA | 29.88 | 31.72 | below gate |
| best active-anchor GPA variant | below gate | below gate | active anchor failed |
| best no-anchor variant | 32.62 | 33.97 | passes gate, but not faithful GPA |
| best no-anchor + cosine eval | 35.08 | 36.17 | calibration helps, but diagnostic only |

Conclusion: GVAlign's per-task multi-head classifier and raw concatenated-head TAg evaluation are a bad match for the paper-style GPA plugin. The only passing variants disabled the core GPA anchor, so they cannot be called faithful GPA.

### Single-Head Scaffold Route

The single-head scaffold used GVAlign data/protocol but replaced the multi-head classifier with one shared classifier over all seen classes.

| method / setting | phase0 acc | phase1 old/base acc | phase1 new acc | phase1 AccT |
|---|---:|---:|---:|---:|
| single-head Finetune | 41.56 | 30.52 | 37.40 | 31.67 |
| single-head GPA, paper bias + mean anchor | 43.32 | 12.92 | 27.50 | 15.35 |
| single-head GPA, paper bias + sum anchor | 43.34 | 13.50 | 12.30 | 13.30 |

Verified diagnostics:

| diagnostic | result |
|---|---:|
| `new_weight_cos_to_frozen_proto_mean` | 1.0 |
| old classifier weight delta after freeze/restore | 0.0 |
| old classifier bias delta after freeze/restore | 0.0 |
| paper-bias + sum-anchor `anchor_to_ce_ratio` | 0.167 |

Eval calibration did not rescue the paper-bias + sum-anchor run:

| eval mode | phase1 AccT |
|---|---:|
| raw | 13.30 |
| no_bias | 13.57 |
| task_bias_center | 13.83 |
| cosine_eval | 14.85 |
| norm_weight_eval | 14.85 |

Conclusion: even when the classifier assumption is made paper-compatible, the current paper-bias + dynamic-anchor implementation performs far below Finetune. This is not a missing-code-path issue.

## Root Cause Assessment

Most likely causes are methodological/protocol mismatches rather than simple implementation omissions:

1. Paper bias is too aggressive under the available CIFAR-100-LT phase/task class counts.
2. Dynamic anchor conflicts with replay CE and current feature learning; stronger `sum` anchor reduced training accuracy and final AccT.
3. Raw logit calibration remains unstable across old and new classes.
4. GPA Table 1 cannot be treated as an immediately reachable baseline without a verified implementation, exact scheduler, exact replay/head semantics, and exact bias/anchor reduction semantics.

## Components To Keep As Non-GPA Ablations

These pieces are useful and should be retained, but not called "faithful GPA":

| component | keep as | reason |
|---|---|---|
| frozen prototype estimation | prototype-imprinting diagnostic | reliably produces class centroids before current-phase training |
| normalized prototype weight init | prototype weight imprinting | verified by cosine-to-prototype = 1.0 |
| zero-bias / old-mean bias init | logit calibration ablation | safer than paper bias in observed phase1 checks |
| old row freeze/restore | classifier-stability control | verified old-row deltas are 0.0 |
| cosine / normalized-weight eval | diagnostic calibration eval | reveals raw-logit calibration mismatch |
| predicted task histograms and old/new margins | collapse diagnostics | quickly identifies latest-task domination |
| GVAlign protocol bridge | data/protocol reference | useful for fixed CIFAR-100-LT order, exemplars, and balanced test evaluation |

Recommended naming for future work:

- `prototype_imprint`
- `zero_bias_imprint`
- `old_mean_bias_imprint`
- `cosine_eval_diagnostic`
- `old_row_freeze_control`

Avoid naming these as GPA unless the full paper behavior is reproduced and validated.

## Back To LT-CIL / TaConCM Work

Recommended next baseline direction:

1. Establish stable LT-CIL baselines under GVAlign-compatible protocol:
   - Finetune with 20 exemplars/class.
   - iCaRL or nearest exemplar-based baseline if available.
   - Single-head Finetune and multi-head GVAlign Finetuning reported separately.
2. Use all-seen top-1 / TAg as the primary metric, not task-masked accuracy.
3. Keep phase1 sanity gates before any full seed0:
   - old/base retention should not drop below Finetune.
   - all-seen phase1 AccT/TAg should not drop below Finetune.
4. Redesign TaConCM as an original LT-CIL method:
   - treat prototype imprinting and calibration as internal ablations;
   - add explicit old-class retention mechanism before full runs;
   - evaluate with and without exemplar replay;
   - report old/new split, many/medium/few, per-task matrix, and predicted-task histograms.

## Do Not Do

- Do not claim current code reproduces GPA Table 1.
- Do not use GPA paper `65.12 / 49.88` as the current baseline target.
- Do not run full GPA seed0 from the current faithful-GPA scaffold.
- Do not label no-anchor variants as GPA.
- Do not mix task-masked, AHM/FA, and LT-CIL Acc/AccT metrics.

## Current Project State

Code retained in this repository:

- `src/gpa_singlehead_repro/`
- `tools/run_cifar100lt_singlehead.py`
- `scripts/run_cifar100lt_singlehead_phase1.sh`

These are retained as diagnostic scaffolds and ablation utilities. They are not evidence of a successful GPA reproduction.
