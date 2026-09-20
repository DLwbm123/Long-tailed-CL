# V7 old-classifier-row retention: complete negative result

Status: COMPLETE; scientific success: **false**. Three I trajectories, 60 new epochs and six incremental session checkpoints are complete. New test predictions: **0**.

I tests one change relative to H: fix the existing main/few classifier rows and biases at each session arrival. Gradients are masked before AdamW; old rows are restored after the step to cancel decay, and old moments are zeroed. The S0 feature extractor, H losses, replay coefficient schedule (2/3), memory, sampling and training lengths are unchanged. I forks from the corresponding original C-S0, not H final weights.

## Same-layout final validation

All scores are percentages, mean ± sample SD across the three fixed seed/order pairs, sorted-ID batch48 seen-class val.

| Variant | BA | Old recall | Current recall | Tail-rank-2 recall |
|---|---:|---:|---:|---:|
| C | 41.848 ± 3.366 | 54.835 ± 3.560 | 2.889 ± 5.003 | 47.489 ± 25.478 |
| E | 25.181 ± 3.084 | 6.309 ± 1.414 | 81.796 ± 10.543 | 28.098 ± 9.379 |
| F | 46.456 ± 5.444 | 37.859 ± 8.140 | 72.246 ± 11.516 | 32.692 ± 13.358 |
| G | 51.017 ± 3.754 | 43.779 ± 7.813 | 72.729 ± 8.447 | 40.224 ± 10.929 |
| H | 52.580 ± 1.031 | 47.526 ± 3.207 | 67.743 ± 6.880 | 45.620 ± 7.008 |
| I | 50.447 ± 4.216 | 44.051 ± 9.068 | 69.636 ± 11.658 | 40.652 ± 1.245 |

| I seed | Final BA | Old recall | Current recall | Tail recall |
|---|---:|---:|---:|---:|
| 1993 | 48.441 | 37.492 | 81.287 | 39.263 |
| 1994 | 47.609 | 40.262 | 69.649 | 41.026 |
| 1995 | 55.292 | 54.400 | 57.971 | 41.667 |

## Decision against the locked gates

I-C final: BA +8.599 pp, current recall +66.747 pp, old recall −10.784 pp. All 3/3 pairs improve BA/current recall and neither incremental stage adds a current zero-recall class. The allowed old-recall loss is at most 5 pp, so the criterion **fails**.

I-H: BA −2.133 pp, old recall −3.474 pp, current recall +1.892 pp, tail recall −4.968 pp. Fixing old rows does not improve mean retention on this protocol. It is not selected as an improvement over H. The result is retained without changing the success definition.

## Stage, head and class detail

Full tables contain 27 session/head rows and 162 per-class rows including inherited S0.

| I stage | Head | BA mean | Old recall mean | Current recall mean |
|---|---|---:|---:|---:|
| 1 | main | 35.298 | 14.730 | 76.435 |
| 1 | few | 38.792 | 53.134 | 10.108 |
| 1 | sum | 56.368 | 48.519 | 72.067 |
| 2 | main | 27.381 | 10.045 | 79.388 |
| 2 | few | 29.207 | 38.792 | 0.450 |
| 2 | sum | 50.447 | 44.051 | 69.636 |

| Original numeric class | I final recall mean | Sample SD |
|---|---:|---:|
| 0 | 63.636 | 19.813 |
| 1 | 60.360 | 22.666 |
| 2 | 77.536 | 4.525 |
| 3 | 44.167 | 2.887 |
| 4 | 33.333 | 17.131 |
| 5 | 43.243 | 15.048 |
| 6 | 57.692 | 7.692 |
| 7 | 23.611 | 8.674 |

Final tail-rank-2 mean recall is 40.652%, 6.838 pp below C. Tail labels are numeric 6/7; do not infer clinical disease names from these identifiers.

## Facts, inference and open questions

Observed: all frozen features remain bitwise equal to C-S0. Every checkpoint and epoch passed the row/moment constraints. S1 old rows exactly match C-S0, and S2 old rows exactly match I-S1 first six rows. Thus failure is not explained by unintended AdamW decay or a broken row freeze. New classes can still win the all-seen competition despite unchanged old scores.

Inference: protecting old rows alone is insufficient here. This weakens the specific hypothesis that old-row movement is the main actionable retention mechanism; it does not prove movement is harmless in other settings. It also does not resolve the mismatch between real loss terms (main, sum, pool-weighted few) and old-synthetic loss (sum only), nor the synthetic/real feature-distribution gap.

The fixed-train probe gradients are raw counterfactual objective gradients. Actual old parameter updates are projected to zero. Do not interpret the nonzero old-row probe norms as evidence that this constraint failed.

A justified next controlled experiment is a separate H-based fork that averages old-synthetic CE for main, few and sum heads, with all H real-loss and sampling rules unchanged. This addresses per-head objective asymmetry; it is not a claim that the hypothesis is correct and is not an increase of replay coefficient. The original ConCM-lite implementation and completed variants remain unchanged.

## Engineering, provenance and resources

Training source `265a6ef93f0fdb6951896486b84b791a3ca255cc`, branch `exp/isic-v7-old-row-retention`. Engineering verified actual updates, zero future CE gradients, old-row retention with AdamW decay/nonzero moments, exact checkpoint/next-step continuation and rejection of wrong seed/parent/anchor. Completion audit checked full matrix coverage, finite records/tensors, parent chain, old Adam moments, exact constrained step counts, paired real streams vs V3 E, class exposures, replay coefficients 2/3 and BA recomputation from per-class recall.

See PROTOCOL_LOCK, per-seed actual_config/FORK_FROM_S0, engineering/ENGINEERING, COMPLETION_STATE_AUDIT and COMPLETION_AUDIT. Wall time: 29.384 minutes; summed GPU-process residence: 0.879 hours (not a utilization integral), maximum two workers. Six compact checkpoint files total 1,851,546 bytes and require preserved C-S0 parent states.

## Limits and release boundary

These are development-validation results after adaptive reuse of 295 val images, not independent test evidence. All new test predictions remain zero. V2 data are unchanged: train18,718/val295/test764, actual training imbalance389.963:1. Clinical class mapping, patient-level isolation and pretraining exposure are unresolved. Three pairs are a small sample; no best epoch, early stop or selective repetition reporting.

Public files contain source, locked configuration, aggregate tables/diagnostics, engineering evidence and resource summaries. Images, row-level manifests/associations, weights, checkpoints, features and per-sample predictions are excluded. All historical artifacts, V1 BLOCKED, V2 COMPLETE_MIXED_SIGNAL and the original completed V3 are preserved.
