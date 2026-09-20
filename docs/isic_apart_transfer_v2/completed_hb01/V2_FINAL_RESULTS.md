# V2 final results

Status: COMPLETE_MIXED_SIGNAL. Six fixed trajectories and 18 final-session checkpoints; one sealed test batch.

| Seed | Variant | Final BA | Average BA | Final tail recall | Lesion-equal BA |
|---|---|---:|---:|---:|---:|
| 1993 | M0 | 35.557 | 44.151 | 22.000 | 36.407 |
| 1993 | M1 | 34.331 | 44.074 | 22.000 | 35.127 |
| 1993 | M2 | 42.839 | 46.404 | 71.000 | 42.677 |
| 1993 | M3 | 42.964 | 48.003 | 71.000 | 42.806 |
| 1994 | M0 | 19.900 | 35.023 | 8.500 | 20.091 |
| 1994 | M1 | 19.839 | 36.160 | 10.000 | 20.129 |
| 1994 | M2 | 34.952 | 39.376 | 27.000 | 35.776 |
| 1994 | M3 | 34.931 | 39.365 | 27.000 | 35.756 |
| 1995 | M0 | 22.076 | 44.355 | 14.000 | 22.973 |
| 1995 | M1 | 22.533 | 44.221 | 13.500 | 23.386 |
| 1995 | M2 | 40.885 | 51.220 | 39.000 | 41.347 |
| 1995 | M3 | 40.885 | 51.164 | 39.000 | 41.347 |

All six paired contrasts and ddof=1 summaries are in paired_results.csv. Per-stage, per-class and eligible-class forgetting tables are supplied separately.

V1 was blocked before training and has no performance results. V2 uses the documented-lesion-disjoint subset, with missing-ID selection bias and UNKNOWN patient isolation/development exposure. Numerical finding IDs have no verified disease-name mapping.

APART (raw-count capacity-adapted), fixed new ImageNet-pretrained initialization, and legacy_effective_v1 optimization. This is not an unchanged APART, historical CIFAR checkpoint, full ConCM, patient-level or external clinical reproduction. Three paired training repeats do not establish statistical significance.

There was no hyperparameter search, checkpoint selection, early stopping for weak results, image replay, or test-driven retraining. No further experiments are authorized by this report.
