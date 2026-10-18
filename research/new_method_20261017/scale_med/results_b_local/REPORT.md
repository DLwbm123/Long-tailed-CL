# Frozen B + local shrink: fixed lambda 0.5

First transition, reused development queries. Frozen B and local_shrink references reproduce exactly. No fitting, image inference, lambda search or E/test access. Both support-derived scales sealed before query scores opened.

| Dataset | Query | B BA % | Local BA % | Fusion BA % | All delta pp | Old delta pp | New delta pp | Rescue | Harm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ISIC | Qfit | 68.5860 | 60.5042 | 70.3186 | +1.7327 | +1.4085 | +2.3810 | 24 | 5 |
| ISIC | Qgate | 54.7189 | 57.1859 | 55.8551 | +1.1362 | +0.9203 | +1.5679 | 21 | 7 |
| HK | Qfit | 61.1623 | 65.0962 | 62.0278 | +0.8655 | +0.9986 | +0.0000 | 5 | 0 |
| HK | Qgate | 70.0025 | 63.1727 | 61.4647 | -8.5378 | -7.2872 | -16.6667 | 2 | 2 |

## Frozen descriptive criterion

- ISIC: PASS (both queries must improve BA; old/new loss cannot exceed 1 pp).
- HK: FAIL (both queries must improve BA; old/new loss cannot exceed 1 pp).

## Disagreement diagnostics

| Dataset | Query | Branch disagreements | B only correct | Local only correct | Both wrong | Fusion changes |
|---|---|---:|---:|---:|---:|---:|
| ISIC | Qfit | 156 | 53 | 68 | 35 | 42 |
| ISIC | Qgate | 182 | 69 | 83 | 30 | 37 |
| HK | Qfit | 40 | 4 | 26 | 10 | 7 |
| HK | Qgate | 24 | 6 | 15 | 3 | 4 |

## Interpretation and limits

Per-class rescue/harm and BA contributions are in classes.csv. results.json includes full confusion matrices and margin quantiles for branch disagreements, fusion changes, rescue and harm. Top2 gaps measure confidence, true margins measure correctness, and B_vs_local margins measure the contest between the two branch winners. All public margin quantiles use normalized branch scores; local margins exclude lambda, fusion margins include it. Per-row margins, labels, predictions and row indices remain private on the approved server. Calibration uses fitted-support scores and all seen model columns; it is not held-out calibration or proof that a fixed scale generalizes. CE depends on score scale. This full-history B diagnostic does not validate compressed-memory deployment. No independent test or multi-seed claim.

## Result analysis

ISIC passes the predefined descriptive criterion: +1.7327/+1.1362 pp overall BA, with positive old and new changes on both queries. Rescue/harm are 24/5 and 21/7. Qfit mapped class4 gains 14 correct images and Qgate class4 gains 9; Qfit class2 also gains one. The tiny class1/class5 counts are preserved relative to B. This supports complementary local information on these reused queries when B is retained. It does not separately identify which representation/readout change caused the previous v2 global gap, and calibration differs from the previous experiment.

HK fails: Qfit gains 0.8655 pp with 5 rescues and zero harms, but Qgate loses 8.5378 pp with 2 rescues and 2 harms. On Qgate class12 loses its only correct image (1/1 to 0/1), contributing -6.6667 pp; class13 falls from 1/3 to 0/3, contributing -2.2222 pp. Class4 gains 1/28 (+0.2381 pp) and class10 gains 1/59 (+0.1130 pp). Their sum is -8.5378 pp even though overall micro accuracy is unchanged. Unlike the earlier raw-CLS comparison, these losses are directly attributable to adding the fixed local term to B, since B itself is unchanged.

The HK Qgate normalized B top2-gap ranges overlap almost completely for rescues [0.07599,0.11043] and harms [0.06929,0.11111]. Therefore simply using low B confidence as a gate is not justified by these observations. For the two harms, local's signed preference for its winner exceeds B's counter-margin after multiplying local by 0.5; the final true-class margins become negative. One harmed prediction selects a third class rather than either branch's winner, illustrating that a full multiclass sum cannot be understood solely as choosing between two top1 labels. These per-row details remain private; aggregate interval statistics are in results.json.

HK also leaves local-only correct disagreements unused: there are 26 on Qfit and 15 on Qgate, while fusion yields only 5 and 2 total rescues. These totals are not a direct recovery fraction: one Qfit rescue occurs where both standalone branches are wrong. Local has useful information, but a global fixed scale/weight neither captures all corrections nor protects rare-class correct decisions. This establishes a fusion failure under the fixed rule, not proof that a different lambda or confidence threshold would solve it. No post-hoc threshold, scan or follow-up experiment was run.

## Validation, provenance and cost

The synthetic check covers signed margins, per-class rescue/harm, disagreement/changed sets and count identities. All frozen B and local_shrink metric dictionaries (including class counts, CE and confusion matrices) reproduce the previous completed transfer run exactly. Positive normalization leaves B argmax unchanged. Input allowlists exclude E/test; both scales were written before any query-score file was opened. Four conditions completed in 4.5243 seconds of runner wall time on CPU using cached inputs, with zero model fits and zero image forwards. Prior feature/statistics generation cost is excluded. Cache reuse explains the short runtime; this is not an end-to-end training speed benchmark.

Sources: previous matched-query delivery db93f25dc8a26e6f57e3b4efc2c17c0a9e176030; historical reference source 74a03f1f7ab61d947c3693e3d4ba6c74ff217cca. New scales: ISIC B=1.7405670166, local=0.0446888900; HK B=1.4104317476, local=0.0898248067. Lambda remains 0.5 in both datasets. Calibration uses only the preselected base support images, with frozen full seen-class columns, and makes no held-out or patient-level generalization claim.
