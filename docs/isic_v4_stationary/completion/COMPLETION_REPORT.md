# V4 stationary-feature result

Completed three F trajectories, six incremental session checkpoints and 60 new epochs. Engineering and completion audit passed. The scientific success criterion did **not** pass. New test predictions: **0**.

| Method | Final val BA | Old recall | Current recall | Tail recall |
|---|---:|---:|---:|---:|
| C | 41.848 ± 3.366 | 54.835 ± 3.560 | 2.889 ± 5.003 | 47.489 ± 25.478 |
| E | 25.181 ± 3.084 | 6.309 ± 1.414 | 81.796 ± 10.543 | 28.098 ± 9.379 |
| F | 46.456 ± 5.444 | 37.859 ± 8.140 | 72.246 ± 11.516 | 32.692 ± 13.358 |

Mean ± sample SD over the same three paired orders/seeds. C is V2 and E is V3, both recomputed on the same sorted, batch-48 validation layout. The final F values per seed are 43.189%, 43.438%, and 52.740% BA.

F−E: BA +21.275 pp, old recall +31.550 pp, current recall −9.550 pp. Fixing S0 features substantially reduced the old-class collapse. This supports a role for changing features/statistics mismatch in E, while not proving a unique cause: freezing also removes feature adaptation and changes optimization.

F−C: BA +4.607 pp and current recall +69.357 pp, but old recall −16.976 pp exceeds the allowed −5 pp. Two of three paired runs improve BA and current recall, and no new current-class zero recall appears. The remaining old-class loss prevents declaring success. Tail recall remains below C.

The final raw replay losses are 0.5251, 0.5340 and 0.2024, compared with <0.00054 in E. In every final fixed-current-train probe, weighted replay head gradients are smaller than the competing real-image gradients on both old and current rows. These are local gradients, not causal proof or dataset-wide gradient estimates. With the representation held fixed, this motivates one predeclared equal-group replay-weight contrast (1.0 versus 0.05), rather than a hyperparameter grid. The old synthetic distribution could still be inadequate, so increased weight may fail or reduce plasticity.

No train/val split, class order, seed, loss form, augmentation or epoch schedule was changed. The only intervention was freezing all non-head parameters at each matching C S0. Exact non-head tensor equality passed at every completed session. Input-stream fingerprints match the paired V3 E runs. All six compact checkpoint states retain complete optimizer/memory/RNG metadata and reference their immutable C S0 parents; they are not standalone checkpoints.

Resources: 29.12 minutes elapsed for the supervisor; 0.866 summed GPU-process residence hours (overlap counted twice); six session delta files total 1,656,706 bytes. The original parents and all historical results remain retained. Up to two GPU workers ran; no new capacity was purchased.

Limitations remain the small development validation set, adaptive validation selection, synthetic Gaussian approximation and augmentation/context mismatch, unverified disease-name mapping and pretraining exposure. No new model test prediction, independent external cohort or patient-level generalization claim. Full stage/class/component records and failed criteria are included.

Public release contains source, protocol, actual configs, engineering/state audit and aggregate results. Images/manifests, feature caches, checkpoint tensors and per-sample predictions stay private. V2/V3 reports are not overwritten.
