# Frozen same-split benchmark comparison — 2026-10-18

This protocol replaces the independent-confirmation admission rules for this explicitly authorized FAIR BENCHMARK purpose. It does not relabel previously used data as a new blind test. No external cohort, 50-groups cutoff, power gate or minimum1pp effect is required to execute this comparison. The earlier feasibility audit remains valid as an exposure/precision warning, not a prohibition on same-split benchmarking. The user authorized audit followed by execution without another approval step.

## Scope and exposure

Use existing cleaned ISIC test and HK reserved manifests, restricted to the SAME current first-transition classes: ISIC4+2 and HK13+2. This is not a full8/23-class or full continual-sequence leaderboard, and the cleaned/reserved project protocol is not presented as an untouched universal official benchmark. Include every eligible row: ISIC586 images/500 known groups; HK1271 images/1271 exact-identity groups. No resplitting, sample selection by scores, tail removal or future-class expansion.

Metadata audit must show zero support/train-development overlap in relative paths and known identity components; existing content identifiers are also compared without recomputing hashes. Confirm every test image exists. Patient isolation remains UNKNOWN; HK exact-image identities are not verified patient/exam groups.

ISIC has a previous completed test batch; HK exposure ledger has historical train/test memberships and independent_confirmation=false. Historical membership does not prove every image was actually trained on, but forbids a clean-blind claim. Current B uses only frozen S_old/S_new statistics; A uses compressed H from S_old plus S_new. Their residual adapters are zero. Local models use the same support rows; backbone is fixed pretrained, not fine-tuned on test. Current fusion modifications directly used Qfit/Qgate, not the separate standard test predictions. This conversation inspected historical completion/exposure metadata, not historical test performance. Inherited model/protocol choices may nevertheless reflect prior project exposure; absence of all historical indirect influence cannot be certified. Report this as reused-benchmark evidence.

## Fixed arms and fair interpretation

Primary comparison: frozen B vs consensus implementation61ea835df655feda64f40206f4326e44ce93f21f (shared positive density/distance support, minimum strength, exact zero ties retain B). Freeze all heads, local moments, covariance regularization, projection, original base-support scales, seed74002 and lambda0.5. No training, scale refit, tuning, model selection, entry-average variant or target-specific protection.

Nine prespecified arms:
- B: full-old-statistics1536-D normalized frozen ridge, primary reference.
- consensus: B plus conservative local residual, sole primary candidate.
- B_density and B_distance: simple additive fusion controls on the identical B and local inputs.
- A: compressed-old-memory1536-D ridge, secondary resource-different control.
- ridge: original v2 raw768-D CLS ridge, secondary representation/weighting/head-different control.
- uniform: unshrunk uniform patch Gaussian, local-only ablation.
- local_shrink: shrunk patch Gaussian density, local-only ablation.
- distance_local: same shrunk distribution with volume term removed, local-only ablation.

All arms see identical test rows, labels and preprocessing; paired comparisons use identical sample order. B vs consensus is a controlled frozen-global-head module comparison, not equal compute or equal memory: consensus adds local statistics, projection and an additional feature path. A/ridge/local arms are contextual ablations, not equally resourced competitors. No claim of comparison to newly reproduced external SOTA. Publish logical model-state size excluding encoders and both extraction stage costs; do not invent per-arm latency from aggregate runtime.

## Engineering before scoring

Recreate historical1536-D feature path in its original Python3.10/torch2.1 environment and frozen source. Validate one already-opened Qfit image per class against cached features, max absolute error<1e-6. Recreate local/raw CLS path in the existing Python3.12/torch2.6 environment, validating the first64 Qfit rows per dataset against frozen ridge/uniform/local_shrink scores, max absolute error<1e-5 and identical argmax. These are parity checks, never selection. Use the same fixed checkpoint and preprocessing. Fail before accepting test metrics if parity fails; no tolerance relaxation based on test performance.

Run one frozen evaluation. GPU0 subject to available memory, neutral parent/worker commands, workers0, CPU threads4. Prior-support/cache creation costs excluded but explicitly stated. Expected under30minutes, detached execution preserves results. Partial artifacts and any engineering failure remain recorded; no performance-driven retry.

## Endpoints and uncertainty

Primary: image-level macro recall (BA), consensus minus B, separately ISIC/HK. Report all/old/new BA, micro accuracy, macro CE (descriptive; incompatible logit scales), full class counts/recall/confusion, paired rescue/harm, rescue retention against both additive controls, and the harm-subset invariant. Group-equal BA is a secondary sensitivity metric, not a replacement for the established image-level BA.

Paired class-stratified known-group bootstrap:10000 draws, seed74003, retain every image within a resampled group, same draws for every arm. Report95% percentile intervals for BA differences vs B in all/old/new groups. Extremely small HK class counts can make intervals fragile or degenerate; intervals are conditional on frozen models and observed cohorts and do not adjust away repeated benchmark use. No significance claim from CE changes or unchanged predictions.

All effect estimates and negatives are published. No arbitrary1pp minimum gain required. Descriptive improvement means positive overall delta; old/new losses beyond1pp are separately flagged using the existing tolerance, without suppressing results. A stronger positive-evidence statement requires positive lower CI on both dataset primary contrasts and no material old/new deterioration, and remains a benchmark claim rather than independent generalization. Do not select a new winner if the primary candidate fails.

## Stop and delivery

Stop before accepted scoring on train/test overlap, missing rows/images, mismatched labels/classes, nonfinite scores, parity failure or broken model/scales binding. Check the label-free gate and harm-subset invariant. Preserve failures and costs, fix only engineering defects within the same frozen scientific protocol. Stop after the fixed test pass, whether favorable, negative or inconclusive. No automatic new seeds, threshold scans or formula repair. Publish source, protocol, aggregate results, cost/validation summaries and analysis to the existing GitHub branch via mandatory proxy; keep raw images, identities, per-row predictions/features and model weights private.
