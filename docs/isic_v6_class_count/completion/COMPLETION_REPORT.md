# V6 class-count replay completion report

Status: COMPLETE; scientific success: **false**. All three H trajectories, 60 new epochs and six incremental session checkpoints completed. New test predictions: **0**.

H differs from G only in effective replay coefficient: known/current class count, hence 2 at S1 and 3 at S2. The S0 feature extractor remains fixed; both classifiers, all-seen real CE, original replay sampling/loss, optimizer and training lengths are retained. This is a separate post-V3 follow-up, not a revision of the V2/V3 results.

## Final seen-class validation

Sorted sample-ID layout, batch size 48, same V2 validation manifest. Values are percentages; SD is the sample SD over the three prescribed seed/order pairs.

| Variant | BA | Old recall | Current recall | Tail-rank-2 recall |
|---|---:|---:|---:|---:|
| C | 41.848 ± 3.366 | 54.835 ± 3.560 | 2.889 ± 5.003 | 47.489 ± 25.478 |
| E | 25.181 ± 3.084 | 6.309 ± 1.414 | 81.796 ± 10.543 | 28.098 ± 9.379 |
| F | 46.456 ± 5.444 | 37.859 ± 8.140 | 72.246 ± 11.516 | 32.692 ± 13.358 |
| G | 51.017 ± 3.754 | 43.779 ± 7.813 | 72.729 ± 8.447 | 40.224 ± 10.929 |
| H | 52.580 ± 1.031 | 47.526 ± 3.207 | 67.743 ± 6.880 | 45.620 ± 7.008 |

| H seed | Final BA | Old recall | Current recall | Tail recall |
|---|---:|---:|---:|---:|
| 1993 | 53.032 | 46.502 | 72.620 | 42.147 |
| 1994 | 51.401 | 44.956 | 70.736 | 41.026 |
| 1995 | 53.308 | 51.119 | 59.873 | 53.686 |

## Fixed success gates

H − V2-C: BA +10.732 pp; current recall +64.854 pp; old recall −7.309 pp. All 3/3 pairs improved both BA and current recall; there were no additional zero-recall current classes at S1/S2. The old-recall floor is missed by 2.309 pp. **Do not claim success or relax this gate.**

H − G: BA +1.563 pp; old recall +3.747 pp; current recall −4.986 pp; tail recall +5.395 pp. Increased replay pressure partly restores retention, with a new-class tradeoff.

## Per-class and stage evidence

The complete tables contain 27 stage/head metric rows and 162 per-class rows, including inherited S0. Tail classes are original numeric labels 6 and 7; clinical label mapping remains unverified. Final tail mean is 45.620%, still 1.870 pp below C. Different order/seed pairs expose tail classes at different stages; the result does not establish uniform tail protection.

| Original class | H final recall mean | Sample SD |
|---|---:|---:|
| 0 | 53.788 | 20.370 |
| 1 | 56.757 | 16.440 |
| 2 | 73.188 | 3.321 |
| 3 | 51.667 | 10.104 |
| 4 | 47.154 | 14.903 |
| 5 | 46.847 | 14.885 |
| 6 | 55.128 | 14.561 |
| 7 | 36.111 | 16.839 |

## Mechanistic interpretation

Observed: all frozen feature tensors/buffers remain bitwise equal to their C-S0 parents. The final isolated current-train probes show old-row real/replay weighted gradient norms of 9.562/0.047, 5.468/1.250, and 3.573/8.491. These are fixed-batch diagnostic snapshots, not epoch-wide gradient estimates. The main head remains strongly biased toward recent classes, while the few head has very low current recall; sum performance is substantially better than either alone.

Inference: scalar replay pressure helps, but changing existing classifier rows under asymmetric real/synthetic support remains a plausible source of retention loss. Gaussian mismatch and new-head competition are not excluded. A next controlled test should constrain old classifier rows at each session boundary while preserving H losses and coefficient schedule, rather than continue selecting scalar coefficients from validation scores. This hypothesis is not yet verified.

## Engineering, provenance and resources

Training source: `93cdd2bb36527da6bc555f143a60bd1999fcb656`; branch `exp/isic-v6-class-count-replay`. See PROTOCOL_LOCK, actual_config, FORK_FROM_S0, engineering/ENGINEERING, COMPLETION_STATE_AUDIT and COMPLETION_AUDIT for exact provenance and checks. Ordinary parent restore and seed guards remain active. Paired real-image/RNG stream fingerprints match V3 E; full epoch coverage, exposure counts, finite loss/gradients, effective coefficients 2/3 and BA recomputation from class recalls all pass.

Wall time: 29.119 minutes; summed GPU-process residence: 0.866 hours, which is not a GPU-utilization integral. At most two workers. Six compact checkpoint files total 1,658,626 bytes and require their immutable original S0 parents. All parents and historical results are preserved.

## Limits and publication boundary

Repeated use of the same 295-image validation set for adaptive follow-ups introduces selection bias. No held-out confirmation is available; new test predictions remain zero. Data remain the V2 cleaned split: train 18,718, val 295, test 764, actual train imbalance 389.963:1. Patient-level isolation and pretraining exposure are unresolved; clinical class-name mapping is unverified.

This release contains source, configuration, aggregate diagnostics/results, engineering evidence and resource summaries. Images, row-level manifests/associations, weights, checkpoints, feature caches and per-sample predictions are not published. V1 BLOCKED and V2 COMPLETE_MIXED_SIGNAL remain unchanged. All repetitions are reported; no best epoch, early stop or test-based selection.
