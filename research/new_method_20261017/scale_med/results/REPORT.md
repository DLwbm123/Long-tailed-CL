# SCALE-Med v1: NIH author20 development experiment

Frozen ImageNet ViT-B/16; 128 fixed projected coordinates; one Gaussian per class. Patient-equal training weights. Reference covariance is fixed from base classes only. One class order and one seed; no validation parameter search. The author20 single-label development split is not the original multilabel clinical task, an independent test, or evidence of medical safety.

| Arm | Mean stage macro | Final macro | Final micro | Mean forgetting |
|---|---:|---:|---:|---:|
| uniform | 0.2302 | 0.1833 | 0.1833 | 0.0889 |
| entropy | 0.1607 | 0.1200 | 0.1200 | 0.0815 |
| reference | 0.1951 | 0.1633 | 0.1633 | 0.1148 |
| reference_shrink | 0.1920 | 0.1633 | 0.1633 | 0.1111 |
| global | 0.2110 | 0.1600 | 0.1600 | 0.0370 |
| global_local | 0.2190 | 0.1733 | 0.1733 | 0.0444 |
| ridge | 0.3201 | 0.2533 | 0.2533 | 0.1296 |

## Rank diagnostics

Fixed old scores do not prevent new classes from winning. Entropy and ridge can change old scores. The following counts compare predictions restricted to old classes on the same old validation images.

| Arm | Old-winner flips | Harm | Rescue | New-class wins (stage-summed) | Max old-score change |
|---|---:|---:|---:|---:|---:|
| uniform | 0 | 0 | 0 | 154 | 6.66134e-16 |
| entropy | 25 | 6 | 2 | 157 | 5.49649 |
| reference | 0 | 0 | 0 | 216 | 2.77556e-17 |
| reference_shrink | 0 | 0 | 0 | 209 | 2.42861e-17 |
| global | 0 | 0 | 0 | 78 | 8.88178e-16 |
| global_local | 0 | 0 | 0 | 81 | 8.88178e-16 |
| ridge | 127 | 26 | 18 | 116 | 0.204458 |

Stage-summed counts can include the same image at multiple stages. No confidence interval or multi-order robustness claim is supported. Patient identifiers, images, embeddings, raw predictions, and model states are excluded from public delivery.

Training support ranges from 7 to 19524 patients per class.

## Interpretation and execution audit

All 6 stages and all 7 arms have finite metrics in [0,1]. The run completed in 1,614.53 seconds (26.91 minutes), leaving 20 class-statistics checkpoints, 20 validation caches, and 6 stage checkpoints. The recorded main process has exited. No restart, parameter sweep, or additional inference was performed for this analysis.

The fused proposal reaches 17.33% final macro accuracy, 1.00 percentage point below uniform local pooling and 8.00 points below the full-CLS ridge control. Local fusion improves the global Gaussian by 1.33 points but does not beat the simpler controls. Mean stage macro accuracy gives the same ranking of these three principal alternatives: ridge 32.01%, uniform 23.02%, fusion 21.90%. This pilot does not establish a performance advantage for the proposed combination.

The intended stability property holds numerically: fixed arms have zero old-class restricted winner flips, with old-score changes below 9e-16. Entropy pooling produces 25 stage-summed old-winner flips, including 6 harms and 2 rescues. This confirms that the mechanism exists in this development run; it does not establish that it explains most accuracy loss. Uniform pooling already removes this drift and outperforms both reference-based local arms.

Reference and reference+shrinkage both finish at 16.33%, but are not identical: shrinkage changes 9 of 300 final predictions without a net accuracy change. Reference transformation increases stage-summed new-class wins from uniform's 154 to 216; shrinkage reduces that count to 209. These are competition counts, not all harmful events, and repeat images across stages. They show why invariant old scores alone cannot guarantee retention or good classification.

A read-only analysis of saved final scores found no image where every reference class score collapsed to the common mixture floor (within 1e-9), and no image-class aggregate score at that floor. This does not exclude individual-patch saturation. The mean across-class score range is 0.02640 for the local shrinkage arm versus 0.91993 for the global Gaussian. With the frozen 0.5 fusion weight, local influence is relatively small: fusion changes 14 of 300 global predictions. This identifies a score-scale concern for a separately designed future study, not a justification for post-hoc tuning in this run.

The runtime log contains 86 OSError entries; inspected exit traces show multiprocessing temporary-directory cleanup on NFS reporting resource busy. These are an execution caveat: completion is supported by the complete status, saved checkpoints, and finite stage results, not by an error-free log. No rerun or deletion of preserved temporary evidence was done solely to clear these messages.

Next scientific decision: retain uniform pooling and ridge as mandatory controls. Before any further experiment, distinguish representation quality, density calibration, and fusion scaling under a new explicit protocol. No new experiment is launched by this report. One seed/order, 15 validation images per class, and a previously used author20 single-label development split do not support generalization or medical-safety claims.
