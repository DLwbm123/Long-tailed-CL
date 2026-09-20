# ISIC V3: representation and objective diagnosis

This branch starts from result commit `a18b25a8c582a0ac0f027bb177dcbdb85ddd7596`; the V2 training source remains `6d8bd3e700924c864e229c47efe5f108ce728c78`. V2's `COMPLETE_MIXED_SIGNAL` conclusion and all original artifacts are preserved. No main merge is part of this experiment.

V3 is a development/validation experiment: fixed train 18,718 / val 295 / test 764, eight numerical classes, three locked orders, 4+2+2. Actual train imbalance is 389.96:1. **New model test predictions must remain zero.**

- P0: all 18 V2 checkpoints, deterministic image-ID-sorted train/val batches of 48; train diagnostics are offline forensics. Actual-network component gradients, common-logit shifts, native backbone parity, batch-context routing and feature/head/margin diagnostics.
- P1: original locked AugReg encoder and the specifically authorized DINOv2 revision, each frozen, final norm CLS and sample L2. F-NCM and class-balanced F-CBRidge (lambda 0.001, no bias, float64 sufficient statistics/solve). Extract train/val once per encoder. Missing DINOv2 download does not cancel A or P2.
- P2: D/E x seeds 1993/1994/1995. D forks B S0, E forks C S0 including memory. Ordinary V2 restore validation is retained before an explicit audited transition. Only the three real-image CE candidate slices change to all-seen. Original reductions, /3, pool assignment, pull terms, replay, optimizer and data streams remain unchanged. Twelve incremental checkpoints, 120 new epochs; no S0 retraining.

Implementation reuses `MedicalLearner`, original checkpoint restore, transforms and aggregate metrics. `test_medical_v3.engineering(config)` executes the original frozen training function and the modified default function on disposable copies and checks exact loss/gradient/update equality; it also checks old-row and future-row gradients and inference label independence. `preflight_budget(config)` measures native target-host throughput and combines it with completed V2 timings. The supervisor rechecks the 8 GPU-hour budget after the first complete P2 epoch (retained as a formal epoch), permits at most two workers, generates aggregate reports, then stops.

Entry points take private JSON configuration via neutral wrapper scripts rather than putting project/method names in process arguments. `execute_medical_v3.supervisor(config)` owns the bounded queue. No external monitoring automation, automatic P3, Joint Training, extra backbone, hyperparameter search, head-norm or Full Dynamic is enabled.

At source freeze, engineering default loss, gradient and update regression passed, original-backbone parity was exact, and zero-adapter relative feature error was 1.63e-6. The initial budget projection was 6.52 conservative GPU process-hours including a 50% reserve; the full-epoch gate remains mandatory. DINOv2's fixed Hugging Face URL was inaccessible from both the compute host and local download fallback; this is recorded as `PARTIAL_BACKBONE_COMPARISON` without a substitute.

Actual locks, configs, phase reports and results are generated under the private output root. Public delivery includes code and aggregate evidence only. Images, associations, feature caches, checkpoints, per-sample predictions and stream fingerprints stay private. A source commit or running pipeline is not a completed experiment.
