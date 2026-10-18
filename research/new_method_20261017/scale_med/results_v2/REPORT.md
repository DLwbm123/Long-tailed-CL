# Fixed-scale ridge plus uniform local evidence: v2

NIH author20 single-label development split; same one seed/order as v1. Reuses v1 frozen features and sufficient statistics; newly extracts only base-training scale-calibration examples. No nonlinear reference mixture. Local covariance shrinkage is unchanged. Scales are frozen base-training class-balanced RMS class-centered scores; lambda=0.5. No validation parameter selection. The global ridge is updated with arriving classes, so only local scores have old-score invariance.

| Arm | Mean stage macro | Final macro | Final micro |
|---|---:|---:|---:|
| ridge | 0.3201 | 0.2533 | 0.2533 |
| uniform | 0.2302 | 0.1833 | 0.1833 |
| local_shrink | 0.2267 | 0.1833 | 0.1833 |
| ridge_local | 0.3428 | 0.2633 | 0.2633 |

| Stage | Fusion changes vs ridge | Harm | Rescue |
|---|---:|---:|---:|
| 0 | 19 | 2 | 6 |
| 1 | 35 | 5 | 8 |
| 2 | 46 | 8 | 14 |
| 3 | 59 | 10 | 16 |
| 4 | 69 | 10 | 18 |
| 5 | 88 | 14 | 17 |

Final fusion minus ridge: +1.00 percentage points. This is an exploratory development-set result, not an independent replication or a medical-safety claim. Control per-class accuracies must match v1 at every stage; the runner checks this before accepting a stage. Original v1 files are read-only. No private data or per-example outputs are included.

## Interpretation and execution audit

All six stages completed in 67.50 seconds. The run reused v1 frozen statistics and validation features; only 295 base-training patient images were newly processed for scale fitting. The log ends with COMPLETE and contains no error output; the original main process has exited. All four arms have finite bounded metrics; ridge and uniform reproduce v1 per-class accuracies at every stage.

Fusion improves macro accuracy at all six stages. Mean stage macro rises from 32.01% to 34.28% (+2.28 percentage points); final macro rises from 25.33% to 26.33% (+1.00 point). The final balanced validation set contains 300 images: fusion gets 79 correct versus ridge's 76. Of 88 changed predictions, 17 rescue ridge errors, 14 spoil correct ridge predictions, and 57 change one wrong answer into another. This is a modest positive result, not strong evidence of a robust performance gain.

Local shrinkage alone reaches the same final 18.33% as unshrunk uniform pooling, and has a slightly lower mean stage macro (22.67% versus 23.02%). Therefore this run supports complementarity of the calibrated local branch with ridge, but does not isolate a benefit from shrinkage. There is no ridge-plus-unshrunk-local arm, so shrinkage cannot be credited for the fusion gain. Relative to v1, global classifier, nonlinear transformation, and fusion scaling all changed; their individual contributions are not isolated by this comparison.

The frozen training scales are 0.1494003430 for ridge and 0.01718945393 for the local branch, with lambda=0.5. No validation selection was used for these values. Uniform and shrunk local scores retain zero measured old-score drift. The full fusion is not invariant: it inherits ridge updates. Across transitions, old-class restricted winner flips total 69 for fusion versus 127 for ridge; counts can repeat images and are not synonymous with harmful forgetting. Absolute ridge/fusion score drift magnitudes are on different scales and should not be compared directly.

This is one class order and seed on an already inspected author20 development split, with 15 validation images per class. Six stages are correlated, not six independent replications. The three-image final net gain does not establish statistical robustness, external generalization, or clinical utility. No additional run, hyperparameter search, or test-set evaluation was performed. The appropriate conclusion is that this minimal fusion is promising enough to retain as a candidate, with benefit still requiring independent confirmation.

Final per-class changes: 7 improved, 7 worsened, 6 unchanged.
