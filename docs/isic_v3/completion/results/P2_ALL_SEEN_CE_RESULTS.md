# P2 all-seen real CE results

Actual completed matrix: six S0 forks, twelve final incremental checkpoints, 120 new epochs. No S0 retraining, early stopping, checkpoint selection, head-norm or test inference. All comparisons use the identical sorted validation layout.

| Seed | Branch | Session | BA | Old recall | Current recall | Current→old | Restricted current BA |
|---|---|---|---:|---:|---:|---:|---:|
| 1993 | B | 1 | 45.989 | 50.090 | 37.786 | 60.317 | 94.231 |
| 1993 | B | 2 | 39.920 | 45.374 | 23.561 | 77.108 | 90.247 |
| 1993 | C | 1 | 44.422 | 65.282 | 2.703 | 96.825 | 94.231 |
| 1993 | C | 2 | 44.510 | 56.457 | 8.666 | 91.566 | 92.156 |
| 1993 | D | 1 | 30.769 | 0.000 | 92.308 | 0.000 | 92.308 |
| 1993 | D | 2 | 22.290 | 0.000 | 89.160 | 0.000 | 89.160 |
| 1993 | E | 1 | 32.934 | 5.170 | 88.462 | 0.000 | 88.462 |
| 1993 | E | 2 | 27.293 | 5.769 | 91.863 | 0.000 | 91.863 |
| 1994 | B | 1 | 40.501 | 42.416 | 36.671 | 61.728 | 78.808 |
| 1994 | B | 2 | 20.338 | 8.890 | 54.682 | 26.389 | 80.769 |
| 1994 | C | 1 | 35.746 | 51.593 | 4.054 | 96.296 | 78.808 |
| 1994 | C | 2 | 38.065 | 50.753 | 0.000 | 100.000 | 82.692 |
| 1994 | D | 1 | 27.170 | 0.000 | 81.511 | 0.000 | 81.511 |
| 1994 | D | 2 | 20.192 | 0.000 | 80.769 | 0.000 | 80.769 |
| 1994 | E | 1 | 26.198 | 0.000 | 78.593 | 0.000 | 78.593 |
| 1994 | E | 2 | 26.608 | 7.914 | 82.692 | 0.000 | 82.692 |
| 1995 | B | 1 | 50.833 | 70.498 | 11.503 | 88.462 | 66.051 |
| 1995 | B | 2 | 26.324 | 14.265 | 62.500 | 7.143 | 68.750 |
| 1995 | C | 1 | 51.380 | 77.069 | 0.000 | 100.000 | 66.051 |
| 1995 | C | 2 | 42.971 | 57.295 | 0.000 | 100.000 | 64.583 |
| 1995 | D | 1 | 21.929 | 0.000 | 65.788 | 0.000 | 65.788 |
| 1995 | D | 2 | 17.708 | 0.000 | 70.833 | 0.000 | 70.833 |
| 1995 | E | 1 | 50.030 | 43.370 | 63.349 | 2.564 | 64.568 |
| 1995 | E | 2 | 21.642 | 5.245 | 70.833 | 0.000 | 70.833 |

Paired raw differences and mean ± sample SD (ddof=1) are supplied for all three contrasts, stages and metrics. Per-class and lesion-equal recall and old/current tail coverage are in the aggregate tables. Detailed component losses, fixed-train-probe gradients, margins and exposures are in training_components.csv. Numerical labels are not mapped to unverified disease names.
