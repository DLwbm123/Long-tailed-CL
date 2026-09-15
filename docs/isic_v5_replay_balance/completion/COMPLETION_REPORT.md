# V5 unit replay coefficient result

G completed all three paired trajectories, six incremental checkpoints and 60 new epochs. Completion/state audit passed. The agreed scientific success criterion remains **false**. New model test predictions: **0**.

| Method | Final validation BA | Old recall | Current recall | Tail recall |
|---|---:|---:|---:|---:|
| C | 41.848 ± 3.366 | 54.835 ± 3.560 | 2.889 ± 5.003 | 47.489 ± 25.478 |
| E | 25.181 ± 3.084 | 6.309 ± 1.414 | 81.796 ± 10.543 | 28.098 ± 9.379 |
| F | 46.456 ± 5.444 | 37.859 ± 8.140 | 72.246 ± 11.516 | 32.692 ± 13.358 |
| G | 51.017 ± 3.754 | 43.779 ± 7.813 | 72.729 ± 8.447 | 40.224 ± 10.929 |

Mean ± sample SD over the same three seeds/orders, using the same sorted-ID, seen-class validation layout. G final BA by seed 1993/1994/1995 is 47.794%, 50.118%, 55.138%. No validation score is compared against an old test score.

The only intervention versus F was changing the existing old-synthetic CE coefficient from 0.05 to the predeclared 1.0. Features, heads, optimizer, class arrivals, data/augmentation and 10 epochs per session were otherwise identical. No parameter grid or validation-fitted coefficient.

G−F: BA +4.561 pp, old recall +5.920 pp, current recall +0.483 pp, tail recall +7.532 pp. Raising the coefficient improved average retention without an average current-recall penalty in this contrast. This does not establish monotonicity or an optimal weight.

G−V2 C: BA +9.168 pp, current recall +69.840 pp, old recall −11.056 pp. All 3/3 runs improve BA and current recall, and no added current-class zero recall occurs. Old recall still exceeds the allowed 5 pp drop, so the result cannot be declared successful. Tail recall remains below C.

The final raw replay losses fell to 0.0718/0.0704/0.0334. In the final fixed-current-train probes, replay gradients on old/current classifier rows remain smaller than corresponding real-objective gradients. Those local probes do not establish dataset-wide imbalance. Head specialization also persists: main-head old recall is 14.15–27.31%, while few-head current recall is only 0–2.08%; sum-head performance is stronger, but its retained old recall is still insufficient.

A next targeted contrast can test the group-size prior: there are four/six old classes against two current classes. Prescribing old/current class-count coefficients 2 and 3 tests a task-derived weighting rule, rather than selecting another arbitrary fixed scalar from validation. Because the real objective retains natural class imbalance and unequal component reductions, this is not an exactly class-balanced loss. Synthetic approximation, augmentation and main/few objective asymmetry remain possible limits.

All 60 epoch records, six parent-dependent delta checkpoint hashes/states, actual coefficients, frozen-tensor checks, class exposure counts, real-input pairing with E, and BA recomputation from per-class recalls passed. The diagnostic loss/gradient coefficient matches actual training (1.0). Original C S0 parent references and all V2/V3/V4 data/results are retained.

Resources: 29.65 minutes elapsed; 0.880 summed GPU-process residence hours; six session delta checkpoints total 1,657,602 bytes. Overlapping worker time is counted twice, not equated with hardware-kernel active time.

The 295-image validation set is repeatedly used to select follow-up hypotheses; these are adaptive development results, not independent test generalization. Missing-lesion selection, unverified disease-name mapping and pretraining exposure remain limitations. Frozen references are single deterministic fits and are not independent APART repetitions.

Published: source, fixed protocol, actual configs, engineering/state audits, resource records and full aggregate stage/class/component tables. Private: images/manifests, features, checkpoint tensors and per-sample predictions. No earlier result or threshold was overwritten.
