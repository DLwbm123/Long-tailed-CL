# V3 decision

Status: COMPLETE_P0_P1_P2.

## Observed facts

P0 and all six P2 pairs are complete; new model test predictions = 0. V2 remains COMPLETE_MIXED_SIGNAL.

- E-C: final mean BA delta -16.667 pp; current recall +78.907 pp; old recall -48.526 pp; paired BA/current improvements 0/3; predefined resource-review gate False.
- D-B: final mean BA delta -8.797 pp; current recall +33.340 pp; old recall -22.843 pp; paired BA/current improvements 0/3; predefined resource-review gate False.
- E-D: final mean BA delta +5.117 pp; current recall +1.542 pp; old recall +6.309 pp; paired BA/current improvements 2/3; predefined resource-review gate False.
- H1 A-CBRidge vs V2-B: frozen BA 55.497; paired-layout differences [15.577, 35.16, 29.173] pp. This repeats one deterministic frozen result against three trained comparators.
- H1 A-CBRidge vs V2-C: frozen BA 55.497; paired-layout differences [10.988, 17.433, 12.526] pp. This repeats one deterministic frozen result against three trained comparators.
- H1 A-NCM vs V2-B: frozen BA 43.520; paired-layout differences [3.6, 23.182, 17.196] pp. This repeats one deterministic frozen result against three trained comparators.
- H1 A-NCM vs V2-C: frozen BA 43.520; paired-layout differences [-0.99, 5.455, 0.549] pp. This repeats one deterministic frozen result against three trained comparators.
- H1 B-CBRidge vs V2-B: frozen BA 54.193; paired-layout differences [14.273, 33.855, 27.869] pp. This repeats one deterministic frozen result against three trained comparators.
- H1 B-CBRidge vs V2-C: frozen BA 54.193; paired-layout differences [9.683, 16.128, 11.222] pp. This repeats one deterministic frozen result against three trained comparators.
- H1 B-NCM vs V2-B: frozen BA 34.128; paired-layout differences [-5.793, 13.79, 7.804] pp. This repeats one deterministic frozen result against three trained comparators.
- H1 B-NCM vs V2-C: frozen BA 34.128; paired-layout differences [-10.382, -3.937, -8.843] pp. This repeats one deterministic frozen result against three trained comparators.

## Mechanistic interpretation

Mean current recall recovered but old recall fell by more than 5 pp: observed tradeoff, not a resolved collapse.

The shared-logit-shift gradient test establishes a local directional asymmetry of the objectives. The paired validation intervention measures its end-to-end effect. Current recall recovery with old recall loss is a tradeoff, not a resolution of stability/plasticity. A positive overall score alone does not establish success. The three-seed resource thresholds are not statistical significance tests. Frozen-encoder comparisons address separability under the specified simple classifiers, not a universal representation upper bound.

## Unverified questions and limits

No new test inference, external cohort validation or patient-level isolation claim. The small, imbalanced validation set and missing-lesion-ID selection remain limitations. Disease-name mapping and pretraining development exposure remain unverified. Batch-context dependence is recorded without changing routing. The original DINOv2 download block is resolved in this supplement; both approved encoders are available. No claim that encoder differences isolate self-supervision.

Stop here. P3, Joint Training, more encoders, Full Dynamic and hyperparameter searches have not been launched.
