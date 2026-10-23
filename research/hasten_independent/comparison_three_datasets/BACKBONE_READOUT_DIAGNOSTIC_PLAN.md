# Queued diagnostic: backbone/readout separation

Status: PLAN_FROZEN_NOT_STARTED. Execute after the current R14 campaign completes and its frozen gate is evaluated. Do not interrupt or alter R14. Purpose is to decide the next HASTEN mainline change, not optimize historical comparison methods.

Four controls: frozen ImageNet ViT-B/16 global CLS vs frozen BiomedCLIP global image embedding, each with (i) cosine nearest class mean and (ii) one fixed class-balanced ridge readout. Use existing official local checkpoints, model-specific fixed preprocessing, and unit-normalized image features. No old trained APART adapter pools, patch branch, text prompts, task adapter, projection, synthetic replay or fused scores in these controls. Thus this measures encoder-plus-default-preprocessing under matched readouts, not pure pretraining alone.

Use exactly R12 training/development manifest references, class orders and task sizes. No TEST. Record any historical-identity limitation already identified. Reuse cached Biomed TRAIN features; reuse other exact frozen features only after provenance and row mapping match, otherwise extract once. Existing adapted APART features are not frozen ImageNet controls. No validation-based parameter choice, image augmentation search or new samples.

Readouts, fixed before execution:
- Unit input vectors x; NCM weight is the unit-normalized class TRAIN mean, no bias.
- Ridge augments the unit feature with a constant 1. For C arrived classes, per-example weight is 1/(C*n_c), targeting one-hot labels. Let G=X'WX and H=X'WY. Solve (G + 0.01*trace(G)/d * I) B=H, with d including the intercept; all coordinates including intercept are regularized. Use float64, no hyperparameter grid. This exact convention is shared across both feature dimensions; does not guarantee equal effective complexity.
- Accumulate exact sufficient statistics from arrived TRAIN classes only. All readouts for a dataset are sealed before validation. The frozen-feature control has no representation drift; expansion can still change class competition. Report initial, every stage, final, macro tail <250 and class counts; NCM/ridge comparisons are conditional diagnostics, not a ceiling or clinical conclusion.

One deterministic extraction/readout run per backbone, not a fake three-seed replication of deterministic heads. Compare to H0/R14 means with their seed spread shown. No statistical significance claim from this development comparison. Report all four cells, no winner selection by tiny numerical differences.

Execution preparation must first inspect implementation reuse and current resource receipts, then freeze a standalone runtime protocol and controller. Provisional hard maximum: 4h and 8GB newly allocated across both backbones/datasets/outputs including caches, with 256MiB reserve; if forecast cannot fit, reduce to initial-task diagnostic explicitly before any validation or defer, never silently reduce after seeing results. No historical deletion. Use available GPU after R14, neutral process arguments, one existing heartbeat. Small unit checks for sufficient-statistic vs direct ridge parity and cache provenance before dispatch. Publish safe aggregate outcomes, no private images/identities/features/weights.

Decision: consistent encoder ordering across both heads supports prioritizing representation pipeline; adequate Biomed ridge but weak HASTEN prioritizes loss/readout and competition; crossed head ordering implies interaction. Do not infer full learned-HASTEN behavior from frozen controls or claim a numerical causal partition of the old/new gap.
