# V9 final delivery: fixed development-validation gates passed

**Status: COMPLETE — all five agreed development-validation criteria PASS. Stop autonomous experimentation after this release.** This is validation-based success under the agreed rule, not independent test or clinical validation. New model test predictions: **0**.

All3 paired K trajectories completed exactly10epochs atS1 andS2:60new epochs,6final checkpoints. No S0 retraining,early stop,best-epoch selection or selective seed cancellation. The final review independently recomputed the gates from per-class/paired tables and confirmed the stored H training-memory tensors match K bitwise.

## Primary comparison on the same validation layout

Percentages, mean ± sample SD across the three fixed seed/order pairs. V2-C and K share the V2 seen-class validation manifest and sorted-ID batch48 evaluation.

| Metric | V2-C | V9-K | K−C (pp) | Criterion |
|---|---:|---:|---:|---|
| Final BA | 41.848 ± 3.366 | 51.496 ± 4.084 | +9.648 | increase |
| Old recall | 54.835 ± 3.560 | 50.352 ± 5.200 | -4.483 | loss ≤5pp |
| Current recall | 2.889 ± 5.003 | 54.928 ± 1.445 | +52.039 | gain ≥10pp |
| Tail recall | 47.489 ± 25.478 | 42.735 ± 9.206 | -4.754 | reported, not a stopping gate |

The remaining two gates also pass: **3/3** pairs improve both final BA and current recall, and neither S1 norS2 adds a zero-recall current class relative to C. The mean old-recall drop4.483pp passes by only0.517pp. This gate applies to the mean; it does not promise ≤5pp old-recall loss for each seed. Tail recall is still4.754pp below C.

## Complete paired stage results

| Seed | Stage | BA | Old recall | Current recall | Tail recall |
|---|---|---:|---:|---:|---:|
| 1993 | 0 (inherited) | 59.041 | — | 59.041 | 29.167 |
| 1993 | 1 | 64.336 | 58.172 | 76.663 | 62.981 |
| 1993 | 2 | 49.506 | 47.403 | 55.817 | 37.340 |
| 1994 | 0 (inherited) | 50.227 | — | 50.227 | 29.167 |
| 1994 | 1 | 56.323 | 46.217 | 76.536 | 25.000 |
| 1994 | 2 | 48.788 | 47.297 | 53.261 | 37.500 |
| 1995 | 0 (inherited) | 78.114 | — | 78.114 | 69.231 |
| 1995 | 1 | 63.211 | 61.330 | 66.974 | 76.923 |
| 1995 | 2 | 56.194 | 56.357 | 55.707 | 53.365 |

S0 rows are inherited C-S0 evidence, not newly trained epochs. Detailed main/few/sum results cover27 session/head rows and162 per-class rows.

| Seed | Final BA gain vsC | Old-recall change vsC | Current-recall gain vsC |
|---|---:|---:|---:|
| 1993 | +4.997 | -9.054 | +47.150 |
| 1994 | +10.723 | -3.456 | +53.261 |
| 1995 | +13.223 | -0.938 | +55.707 |

## Actual intervention and controls

K differs from H only in synthetic sampling variance: use empirical mean plus independent Gaussian noise with standard deviation sqrt(max(stored variance,0)+1e−6), removing the inherited upper clip1.0. Configuration records synthetic_variance_rule=empirical_unclipped and effective_synthetic_var_max=null. The legacy concm_stage1_var_max1.0 remains recorded but is explicitly overridden only inside the sampler and restored in finally.

Keep H fixed C-S0 feature parameters/buffers; optimize only the existing main/few classifier tensors. Retain all-seen real CE,its original three terms/reductions,/3,pool assignment/few weighting,pull,theta,original optimizer,augmentation and natural current-train sampling. Old replay remains original sum-only CE,4 samples per old class,cap48,with effective coefficient2 atS1 and3 atS2. K does not use I row freezing or J three-head replay. It adds no module,normalization,full covariance,temperature or searched value.

Every K S1/S2 checkpoint has bitwise-identical training memory to the matching H checkpoint, including means,variances,counts and task metadata. Removing only the two explicit variance-rule metadata keys makes K training arguments exactly equal to H. All60 real streams and class exposures match V3E, and the H controls use the same fixed streams.

## What each follow-up established

| Variant | Change | BA | Old recall | Current recall | Tail recall |
|---|---|---:|---:|---:|---:|
| C | V2 reference | 41.848 | 54.835 | 2.889 | 47.489 |
| E | V3 all-seen real CE + original replay | 25.181 | 6.309 | 81.796 | 28.098 |
| F | E with fixed C-S0 features | 46.456 | 37.859 | 72.246 | 32.692 |
| G | F replay coefficient0.05→1 | 51.017 | 43.779 | 72.729 | 40.224 |
| H | G stage replay coefficient2/3 | 52.580 | 47.526 | 67.743 | 45.620 |
| I | H with old classifier rows fixed | 50.447 | 44.051 | 69.636 | 40.652 |
| J | H with three-head synthetic CE mean | 51.643 | 45.466 | 70.172 | 43.750 |
| K | H without upper variance clip | 51.496 | 50.352 | 54.928 | 42.735 |

The final K is not the highest-BA variant: K-H BA−1.084pp and current recall−12.815pp, while old recall+2.827pp. K meets the agreed retention/current-class combination; claiming it dominates H would be wrong. I and J failed to improve the intended balance and were not accumulated into K. F/G/H indicate that fixed features and replay pressure materially affect the tradeoff, but only K among these trained candidates passes every stopping gate.

Original V3P0/P1/P2 and DINOv2 were already delivered: B BA28.861%,D20.064%,E25.181%; frozen AugReg CBRidge55.497%,DINOv2 CBRidge54.193%,NCM43.520%/34.128%. The frozen classifiers are deterministic references, not three independent APART repair runs. Their higher BA is retained as an important comparator, not relabeled as K evidence. See the immutable [V3 completion report](https://github.com/DLwbm123/Long-tailed-CL/blob/cda1b371225de3568f7f07a8fc1658ec5f03fa3d/docs/isic_v3/completion/COMPLETION_REPORT.md).

## Class and tail analysis

| Original numeric class | K final recall mean ±SD | Mean change vsC (pp) |
|---|---:|---:|
| 0 | 58.333 ± 4.731 | -9.091 |
| 1 | 54.054 ± 9.362 | +8.108 |
| 2 | 60.870 ± 11.503 | +58.696 |
| 3 | 55.000 ± 2.500 | +35.833 |
| 4 | 49.593 ± 3.726 | -3.252 |
| 5 | 48.649 ± 4.681 | -3.604 |
| 6 | 57.692 ± 10.176 | -2.564 |
| 7 | 27.778 ± 8.674 | -6.944 |

Tail rank2 consists of original numeric labels6/7. K tail mean42.735% with SD9.206 remains below C47.489%; the stage table shows dependence on when those classes arrive. Clinical disease-name mapping is unverified, so numeric labels are retained. No new-current-zero gate does not imply that every old class is protected or that clinical performance is adequate.

## Mechanism: facts versus inference

Observed: prior stored train statistics lost a mean75.600% of diagonal variance under upper cap1.0, with more than99% of dimensions clipped. K changes only this cap and its old recall rises2.827pp versus H. Mean final-epoch raw replay CE in the three runs is0.175850/0.210731/0.118005, versus H0.054282/0.062949/0.017903, consistent with harder synthetic examples. Norm,margin,head-bias and raw/weighted gradient snapshots are retained in training_components.csv and session diagnostics.

Inference: extreme clipping contributed to the retention/new-class balance in this fixed-feature setup. The controlled comparison supports a role for synthetic dispersion, but the mean BA and tail declines versus H prevent a claim of universal improvement. Diagonal independence,non-Gaussian class distributions,main/few correlation and train/eval differences remain unresolved. The isolated fixed-batch gradient probes are not estimates of whole-epoch gradient balance.

## Engineering and resource audit

PASS: actual empirical-sampling formula and noise/RNG sequence; finite tensors/losses/gradients; no future CE gradients; exact frozen features; ordinary seed/parent/config guards; compact restore and next update bitwise equality; six checkpoint locks; H memory/argument control;60epochs,class exposures,paired real streams,coefficient2/3 and metric recomputation. See COMPLETION_AUDIT.json,COMPLETION_STATE_AUDIT.json,engineering/ENGINEERING.json,PROTOCOL_LOCK.json and per-seed actual_config/FORK_FROM_S0.

Training source:47fe3b6c180bd4e45ce83a779a71f844398cbca7; branch exp/isic-v9-empirical-variance. V9 wall29.370minutes,summed GPU-process residence0.874hours,at most two workers. Six delta checkpoints total1,660,994bytes and require their original C-S0 parents. The15-second sampler recorded117 observations,peak resident3,762MiB,mean sampled GPU utilization84.51%,peak100%. Residence and sampled utilization are different measures; neither is hardware-kernel active time.

| Follow-up | New epochs | Final checkpoints | Wall minutes | GPU-process hours |
|---|---:|---:|---:|---:|
| V4 | 60 | 6 | 29.121 | 0.866 |
| V5 | 60 | 6 | 29.654 | 0.880 |
| V6 | 60 | 6 | 29.119 | 0.866 |
| V7 | 60 | 6 | 29.384 | 0.879 |
| V8 | 60 | 6 | 29.370 | 0.875 |
| V9 | 60 | 6 | 29.370 | 0.874 |

Across V4–V9:360 new epochs,36 final incremental checkpoints,176.016 summed run-wall minutes and 5.239 summed GPU-process hours. This excludes engineering invocations and idle time between hourly reviews. Original V3 including DINO supplement separately reported4.105 GPU-process hours within its8-hour cap. Post-V3 cumulative cap was explicitly removed by the user.

At the final remote check all three trajectories were complete,there were no experiment workers or GPU allocations,all worker exits were0,and the intended data mount had2,368,962,560bytes free. No historical checkpoint,data,weight or unrelated process was removed.

## Data, reproducibility and limitations

V2 locked train18,718/val295/test764; train counts for labels0–7:10529,3263,2835,1306,458,256,44,27; actual imbalance389.963:1. Keep8classes,4+2+2,the three original order/seed pairs and all cleaning decisions. V1 BLOCKED and V2 COMPLETE_MIXED_SIGNAL remain unchanged.

The stopping rule has been applied to a repeatedly reused development validation set,after several adaptive choices. No independent test result has been generated. The old-recall gate passes narrowly and one individual seed can lose more than5pp; the mean-based rule remains unchanged. Small validation size,missing-lesion-ID selection,unverified patient-level isolation,clinical class mapping and pretraining exposure limit inference. This result is not a clinical validation or a generalization guarantee.

Public release contains code,locked configs,engineering evidence,all aggregate stage/class/head tables,paired differences,resource records and negative-result history. Private images,manifests/associations,weights,checkpoints,feature caches and per-sample predictions are excluded. Branches are preserved and main is not merged. No V10,P3,Joint Training,Full Dynamic,extra backbone,hyperparameter search or new test evaluation will be launched.
