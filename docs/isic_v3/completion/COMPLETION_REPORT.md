# V3 completion audit

V3 P0–P2 and the pinned DINOv2 supplement are complete. The engineering audit passed; the E-versus-C scientific success criterion did not. V2 remains `COMPLETE_MIXED_SIGNAL`. New model test predictions: **0**.

The fixed V2 data contain 18,718 train, 295 validation and 764 sealed test images. Train counts for numerical labels 0–7 are [10529, 3263, 2835, 1306, 458, 256, 44, 27], giving actual imbalance 389.96:1. Three orders, 4+2+2, no split regeneration.

| Method | Final val BA | Old recall | Current recall | Tail recall |
|---|---:|---:|---:|---:|
| B | 28.861 ± 10.035 | 22.843 ± 19.697 | 46.914 ± 20.599 | 22.650 ± 7.464 |
| C | 41.848 ± 3.366 | 54.835 ± 3.560 | 2.889 ± 5.003 | 47.489 ± 25.478 |
| D | 20.064 ± 2.294 | 0.000 ± 0.000 | 80.254 ± 9.174 | 17.201 ± 15.703 |
| E | 25.181 ± 3.084 | 6.309 ± 1.414 | 81.796 ± 10.543 | 28.098 ± 9.379 |

Values above are three trained runs, mean ± sample SD. B/C are the same-layout V2 validation recomputation; D/E are V3. No validation score is subtracted from an old test score.

The all-seen CE intervention recovered current classes but reversed the collapse: D assigns all old validation samples to current classes; E still misassigns 87.44–96.00% of old images to current classes. E−C final deltas are BA −16.667 pp, current recall +78.907 pp and old recall −48.526 pp, with 0/3 paired BA/current improvements. This is a stability–plasticity tradeoff, not success.

Frozen AugReg CBRidge reaches 55.497% BA; DINOv2 CBRidge reaches 54.193%. Both reference classifiers meet the numerical old/current thresholds against the three V2-C layouts. Each is one deterministic final fit, not three independent seed replications, and neither result demonstrates that APART E has been repaired. NCM yields 43.520% and 34.128%, respectively.

P0 native-original features match exactly; zero-adapter final feature relative L2 error is 1.63e−6. The default current-scope loss, gradients and network update match V2 bitwise; all-seen CE exposes old rows while future CE gradients remain zero. The selected batch-peer/order probes showed no logit change; this does not establish universal batch independence. Ridge stream/batch error is below 2.19e−14; class arrival access and final order invariance passed.

Final E training current-class macro recall is 98.51–100%, and synthetic replay loss is below 0.00054. Synthetic old features have negative current-minus-old margins, despite real-old validation collapse. This supports testing feature/statistics mismatch as a next hypothesis; it does not prove its cause or imply that raising replay weight would fix it.

Resources: original pipeline wall time 2.415 h; DINOv2 supplement 43.11 s; summed GPU-process residence including supplement 4.105 h (overlapping workers counted twice), below 8 h. Maximum observed GPU allocation 19286 MiB; at most two experiment processes. These residence measurements are not hardware-kernel active time.

All 18 P0 checkpoints, 6 P2 trajectories, 12 final incremental checkpoints and 120 epochs are accounted for. Fork lineage, same-order paired input streams, actual configs, finite records, per-class BA recomputation, weight/cache locks and train-only ridge fitting passed the completion audit. Historical original and partial reports remain unchanged.

Limitations: only 295 validation images, repeated use of this development set, missing-lesion-ID selection, unverified disease-name mapping and pretraining exposure; no new test inference or external/patient-isolation claim. Frozen encoder differences do not isolate self-supervision. Future adaptive iterations must retain negative results and report validation-selection bias.

The original V3 decision says STOP because that was its locked protocol. The later user authorization permits separate follow-up experiments after this immutable V3 report. No V3 result, threshold or method is relabeled to claim success.

Published: source already on the V3 branch, locked configs, engineering and aggregate diagnostics/results. Private: raw images/manifests, feature caches/associations, checkpoints, and per-sample predictions. See `COMPLETION_AUDIT.json` and the detailed tables.
