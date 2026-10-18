# V2 ISIC / HyperKvasir matched-query comparison

First transition only: ISIC 4+2 and HyperKvasir 13+2 classes. Exact historical S_old/S_new and Qfit/Qgate rows; no E/test access. Fixed v2 parameters and base-support scale fitting; both models sealed before either query was opened. Four internal arms plus historical frozen A/B; A/B BA and CE must reproduce published results. All observations are reused-development evidence, not independent confirmation.

| Dataset | Query | Arm | All BA % | Old BA % | New BA % | Micro % | Macro CE |
|---|---|---|---:|---:|---:|---:|---:|
| ISIC | Qfit | A | 69.6178 | 63.7805 | 81.2925 | 61.1521 | 0.96014 |
| ISIC | Qfit | B | 68.5860 | 64.9538 | 75.8503 | 59.5273 | 0.81971 |
| ISIC | Qfit | ridge | 55.8698 | 46.2877 | 75.0340 | 67.0606 | 1.46795 |
| ISIC | Qfit | uniform | 51.4586 | 51.4566 | 51.4626 | 62.9247 | 1.71945 |
| ISIC | Qfit | local_shrink | 60.5042 | 50.6202 | 80.2721 | 61.7430 | 1.75138 |
| ISIC | Qfit | ridge_local | 56.7108 | 47.1241 | 75.8844 | 68.3900 | 1.00257 |
| ISIC | Qgate | A | 52.6584 | 38.7438 | 80.4878 | 62.9412 | 1.49689 |
| ISIC | Qgate | B | 54.7189 | 44.4477 | 75.2613 | 63.8235 | 1.09504 |
| ISIC | Qgate | ridge | 58.2171 | 45.6010 | 83.4495 | 71.9118 | 1.49778 |
| ISIC | Qgate | uniform | 46.8284 | 46.5782 | 47.3287 | 67.3529 | 1.74170 |
| ISIC | Qgate | local_shrink | 57.1859 | 45.9705 | 79.6167 | 65.8824 | 1.75909 |
| ISIC | Qgate | ridge_local | 57.4608 | 44.3794 | 83.6237 | 72.5000 | 1.18464 |
| HK | Qfit | A | 59.7233 | 61.2192 | 50.0000 | 82.7068 | 1.41221 |
| HK | Qfit | B | 61.1623 | 62.8795 | 50.0000 | 83.7093 | 1.36757 |
| HK | Qfit | ridge | 62.9522 | 64.9448 | 50.0000 | 88.4712 | 2.22555 |
| HK | Qfit | uniform | 64.7692 | 67.0414 | 50.0000 | 88.9724 | 2.72790 |
| HK | Qfit | local_shrink | 65.0962 | 67.4186 | 50.0000 | 89.2231 | 2.58000 |
| HK | Qfit | ridge_local | 62.7695 | 64.7340 | 50.0000 | 87.9699 | 1.24603 |
| HK | Qgate | A | 69.4050 | 69.8263 | 66.6667 | 84.7118 | 1.12287 |
| HK | Qgate | B | 70.0025 | 70.5157 | 66.6667 | 86.4662 | 1.13843 |
| HK | Qgate | ridge | 61.2926 | 63.0299 | 50.0000 | 87.2180 | 2.22354 |
| HK | Qgate | uniform | 61.6848 | 63.4824 | 50.0000 | 88.2206 | 2.70070 |
| HK | Qgate | local_shrink | 63.1727 | 65.1992 | 50.0000 | 88.7218 | 2.57633 |
| HK | Qgate | ridge_local | 61.7309 | 63.5357 | 50.0000 | 87.4687 | 1.23302 |

## Fusion deltas and paired outcomes

| Dataset | Query | Reference | All delta pp | Old delta pp | New delta pp | Rescue | Harm |
|---|---|---|---:|---:|---:|---:|---:|
| ISIC | Qfit | ridge | +0.8410 | +0.8364 | +0.8503 | 19 | 10 |
| ISIC | Qfit | A | -12.9070 | -16.6564 | -5.4082 | 91 | 42 |
| ISIC | Qfit | B | -11.8751 | -17.8297 | +0.0340 | 93 | 33 |
| ISIC | Qgate | ridge | -0.7563 | -1.2216 | +0.1742 | 16 | 12 |
| ISIC | Qgate | A | +4.8024 | +5.6357 | +3.1359 | 105 | 40 |
| ISIC | Qgate | B | +2.7419 | -0.0683 | +8.3624 | 83 | 24 |
| HK | Qfit | ridge | -0.1826 | -0.2107 | +0.0000 | 1 | 3 |
| HK | Qfit | A | +3.0462 | +3.5148 | +0.0000 | 30 | 9 |
| HK | Qfit | B | +1.6072 | +1.8545 | +0.0000 | 28 | 11 |
| HK | Qgate | ridge | +0.4383 | +0.5058 | +0.0000 | 2 | 1 |
| HK | Qgate | A | -7.6741 | -6.2906 | -16.6667 | 22 | 11 |
| HK | Qgate | B | -8.2716 | -6.9801 | -16.6667 | 12 | 8 |

Descriptive criterion: both queries improve overall BA, with neither old nor new BA losing more than 1 pp. This does not select or deploy a winner.

- ISIC versus ridge: FAIL on the frozen two-query criterion.
- ISIC versus A: FAIL on the frozen two-query criterion.
- ISIC versus B: FAIL on the frozen two-query criterion.
- HK versus ridge: FAIL on the frozen two-query criterion.
- HK versus A: FAIL on the frozen two-query criterion.
- HK versus B: FAIL on the frozen two-query criterion.

## Limits and interpretation

The ridge comparison isolates adding the fixed local term to the current v2 global head. Historical B uses the same support-source rows, but differs in representation, sample weighting, regularization, bias and retained state. A additionally has compressed history. Therefore comparisons against A/B are pipeline comparisons, not equal-memory or single-module ablations. Known identity groups are not verified patients. CE is secondary because score scales differ. Tiny classes can strongly affect BA; full class counts, correct counts and confusion matrices are in results.json, and group rescue/harm counts are in paired.csv. Calibration uses only base support, but uses fitted-training scores and has no held-out calibration guarantee. No claim of old-score invariance for the full ridge fusion. No test-set, multi-seed or clinical generalization claim.

## Outcome analysis

Fusion does not establish a consistent benefit on these matched queries. Versus v2 ridge, overall BA changes are +0.8410/-0.7563 pp on ISIC Qfit/Qgate and -0.1826/+0.4383 pp on HK Qfit/Qgate. Both datasets fail the frozen two-query criterion. On ISIC Qgate, old-group BA falls by 1.2216 pp despite a small new-group increase. The NIH gain therefore did not consistently transfer under the frozen settings.

Comparison with historical B is also mixed: ISIC -11.8751/+2.7419 pp and HK +1.6072/-8.2716 pp. These are whole-pipeline differences; representation, weighting, regularization, bias and memory differ. They cannot be assigned to the added local branch alone. The local-only shrunk arm is stronger than fusion on ISIC Qfit and both HK queries, but this post-hoc observation does not authorize selecting it or tuning fusion weights.

Micro accuracy and BA tell different stories. ISIC Qfit fusion gains 60 net correct images over B but loses 11.8751 pp macro BA; HK Qgate gains 4 net correct images yet loses 8.2716 pp BA. In ISIC Qfit, mapped class1 changes 2/2 to 1/2, contributing -8.3333 pp to six-class BA; class5 changes 5/5 to 4/5 and class2 changes 11/23 to 7/23. In HK Qgate, mapped class12 changes 1/1 to 0/1 (-6.6667 pp of overall BA), and new class13 changes 1/3 to 0/3 (-2.2222 pp). These extreme small-class denominators explain substantial sensitivity; they are retained rather than removed or downweighted after seeing results. Full counts and confusion matrices are available in results.json.

All four conditions completed; metrics are finite, and the stored historical A/B all/old/new BA and CE match their frozen references. No E/test access, parameter search, scientific retry or extra seed was performed. The log contains COMPLETE and no reported error. Both datasets' model seals precede query evaluation, as enforced by the runner.

## Runtime and the idle-GPU observation

Wall time was 993.48 seconds (16 minutes 33 seconds). Cumulative process CPU user/system time was 932.31/27.18 seconds and peak RSS was 1,826,754,560 bytes. This CPU time is not a wall-time bottleneck percentage. The main process had already exited when GPU0 was observed idle. The run did use GPU feature extraction and support-moment accumulation; calibration and query Gaussian scoring use CPU. No per-operator profile was recorded, so the dominant bottleneck and a speedup factor are unmeasured. Future engineering optimization may move Gaussian scoring to GPU after numerical parity checks, but this completed scientific run was not restarted merely to increase GPU utilization. Historical cache-generation costs are excluded from this new run's cost record.

No subsequent model or parameter change is selected by this report. The present conclusion is negative for a consistently beneficial fixed fusion across the two matched development-query datasets.
