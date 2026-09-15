# V8 synthetic per-head objective: completed matrix

Status: COMPLETE; scientific success: **false**. Three J trajectories,60 new epochs and six session checkpoints completed. New test predictions: **0**.

J replaces only H old-synthetic sum CE with the mean of main, few and sum CE over all seen classes. Real losses, fixed S0 features, H replay coefficients2/3, sampler, optimizer and training lengths are unchanged. The I old-row constraint is absent. This separate post-V3 variant does not redefine original ConCM-lite.

## Final validation

Percentages, mean ± sample SD across the three prescribed seed/order pairs. Same seen-class val, sorted-ID batch48.

| Variant | BA | Old recall | Current recall | Tail recall |
|---|---:|---:|---:|---:|
| C | 41.848 ± 3.366 | 54.835 ± 3.560 | 2.889 ± 5.003 | 47.489 ± 25.478 |
| E | 25.181 ± 3.084 | 6.309 ± 1.414 | 81.796 ± 10.543 | 28.098 ± 9.379 |
| F | 46.456 ± 5.444 | 37.859 ± 8.140 | 72.246 ± 11.516 | 32.692 ± 13.358 |
| G | 51.017 ± 3.754 | 43.779 ± 7.813 | 72.729 ± 8.447 | 40.224 ± 10.929 |
| H | 52.580 ± 1.031 | 47.526 ± 3.207 | 67.743 ± 6.880 | 45.620 ± 7.008 |
| I | 50.447 ± 4.216 | 44.051 ± 9.068 | 69.636 ± 11.658 | 40.652 ± 1.245 |
| J | 51.643 ± 4.413 | 45.466 ± 6.316 | 70.172 ± 2.697 | 43.750 ± 15.574 |

| J seed | Final BA | Old recall | Current recall | Tail recall |
|---|---:|---:|---:|---:|
| 1993 | 48.237 | 39.932 | 73.149 | 33.814 |
| 1994 | 50.063 | 44.120 | 67.893 | 35.737 |
| 1995 | 56.628 | 52.346 | 69.475 | 61.699 |

## Locked decision

J-C: BA+9.794pp, current recall+67.284pp, old recall−9.369pp; all3/3 pairs improve BA/current and neither incremental session adds a current zero-recall class. The old-recall loss exceeds the allowed5pp, so the fixed success criterion fails.

J-H: BA−0.937pp, old recall−2.059pp, current recall+2.429pp and tail recall−1.870pp. This particular per-head objective is not supported as an improvement over H; it will not be accumulated into the next candidate. The result does not rule out all forms of objective asymmetry, because this intervention also redistributes the existing replay coefficient among the three CE terms.

## Stage and class results

Complete tables include27 stage/head rows and162 per-class rows, including inherited S0.

| J stage | Head | BA mean | Old recall mean | Current recall mean |
|---|---|---:|---:|---:|
| 1 | main | 56.828 | 50.611 | 69.261 |
| 1 | few | 50.789 | 48.646 | 55.076 |
| 1 | sum | 58.992 | 53.384 | 70.208 |
| 2 | main | 49.000 | 42.148 | 69.557 |
| 2 | few | 45.274 | 43.278 | 51.263 |
| 2 | sum | 51.643 | 45.466 | 70.172 |

| Original numeric class | J final recall mean | Sample SD |
|---|---:|---:|
| 0 | 41.667 | 6.943 |
| 1 | 60.360 | 18.397 |
| 2 | 81.159 | 14.475 |
| 3 | 53.333 | 5.204 |
| 4 | 42.276 | 3.726 |
| 5 | 46.847 | 5.626 |
| 6 | 50.000 | 16.765 |
| 7 | 37.500 | 14.434 |

Tail classes are numeric6/7; final mean43.750%, −3.739pp versus C. Clinical class-name mapping remains unverified.

## Objective and engineering evidence

All60 epoch records pass raw replay = mean(main CE,few CE,sum CE), weighted replay = coefficient2/3 × raw, and per-component diagnostic weight = coefficient/3. The actual shared-current-shift derivative matches (mass_main+mass_few+2mass_sum)/3 when each head receives a unit shift. The legacy old_synthetic shift is explicitly a sum-only reference, not J actual objective. Fixed-batch gradients are snapshots, not complete epoch averages.

Engineering verified exact actual-sampler/head loss and gradient equality to an independent explicit formula, default sum-only regression, future CE gradient exclusion, finite updates, unchanged feature tensors, ordinary parent/seed guards and exact resume/next update. Completion audit verifies every checkpoint, parent/data/code/protocol metadata, full matrix, class exposures, real input stream pairing vs V3 E, coefficients and BA from per-class recall. I row-freeze test fields are not applicable to J.

## Additional training-memory diagnosis

The stored train-only empirical variances are retained before the sampler clips them at1.0. Across48 final class/head states (8classes ×2heads ×3pairs), 99.219–99.870% of dimensions exceed1, and clipping removes an unweighted mean75.600% of summed diagonal variance. Average per-dimension variances range3.163–5.194. This is a direct transformation of training statistics, not a validation-label fit. Final-current class states are included in those48 even though they have not yet been replayed.

For the36 states corresponding to actual S2 old classes, the mean removed-variance fraction is 75.767%, and the capped-dimension fraction ranges 99.479–99.870%.

Example: seed1993 class4 main memory has RMS norm89.937 under the empirical diagonal variance but75.668 after clipping; few RMS changes89.743→72.116. These are analytically implied RMS norms from stored means/variances, not measured validation accuracy. Full aggregate evidence is MEMORY_VARIANCE_DIAGNOSIS.json.

Inference: the inherited cap creates severe synthetic under-dispersion, offering a concrete explanation for very easy replay despite poor old-image retention. This is not causal proof. The next single H-based contrast will remove only the upper variance clip and retain lower clipping/epsilon, Gaussian factorization, sampler RNG, coefficients and H sum-only objective. Correlation loss, non-Gaussian features and train/eval mismatch remain unresolved; no claim that unclipped diagonal Gaussian replay must succeed.

## Provenance and resources

Training source9e9c1b080704969f5850a804aae9f1061f2662a8, branch exp/isic-v8-head-replay-objective. Wall29.370minutes, summed GPU-process residence0.875hours (not utilization integral), at most two workers. Six compact checkpoints total1,659,906bytes and require their preserved immutable C-S0 parent states. See PROTOCOL_LOCK, actual_config, FORK_FROM_S0, ENGINEERING and both completion audits.

## Limits and publication boundary

Development-val has been reused adaptively, so successful future gates would still require independent confirmation. New model test predictions remain0. V2 train18,718/val295/test764, actual train imbalance389.963:1 and all seeds/orders are unchanged. Patient isolation, clinical labels and pretraining exposure remain unresolved. All repetitions and failures are reported without early stop or best epoch.

Public files are source, locked settings, aggregate metrics/diagnostics, engineering and resource evidence. Raw images, row-level associations/manifests, weights, checkpoints, feature caches and per-sample predictions remain private. Preserve V1 BLOCKED, V2 COMPLETE_MIXED_SIGNAL and V3–V7 conclusions; no main merge.
