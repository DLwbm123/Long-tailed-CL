# R14: image-to-text batch multiplicity correction

Purpose: improve the user's HASTEN medical mainline, following D13/R12. NIH tail recognition is initially weak; all three datasets lose recall under expanded class competition. Code-level evidence: the image-to-text CE denominator contains one text entry per real/replay image. Same-class text entries are identical. Their summed denominator mass is multiplied by batch class count m_c. This is a mechanistic hypothesis, not a proven cause of validation errors.

## Single change and controls

For scaled paired logits L_ij, class of column j = y_j, replace only image-to-text paired CE by

`CE(L - log(m[y])[None,:], arange(B)) - mean(log(m[y]))`.

With identical same-class text vectors, this equals CE over the unique classes present in this batch. The subtracted positive-count constant preserves the exact unique-class loss value and does not alter gradients. The text-to-image CE remains unchanged. No all-seen-class negative pool, no sampler/label frequency weighting, no change to fusion or logit scale, no new hyperparameter. This differs from failed r05 balanced sampling and r08 all-seen symmetric class CE. It does not eliminate all long-tail effects or all repeated-pair interference.

H0: reuse sealed r04 NIH H-unit and R12 ISIC/HK H-unit. M: new arm, three seeds 74002/3/4, all stages (18 NIH +9 ISIC+18 HK =45 new). Same exact split, class order, unit input, backbone, prompts/trees, loss coefficients, actual-step projection, natural shuffle, Gaussian replay, 10 epochs, batch64, AdamW.001 and .5 official fusion. The baseline initial-stage checkpoints cannot be reused because the loss changes from the first task. No no-projection or extra tuning arm.

Existing raw NIH caches are normalized read-only; existing normalized ISIC/HK R12 caches are read-only. No backbone/cache copies. Re-encode development images only after all candidate heads for that dataset are sealed, using the same fixed backbone/preprocessing. Both prior and new results remain development evidence; no TEST. No changes based on interim results, all datasets and seeds pre-fixed.

## Fixed exploratory gate

Primary NIH: stage-average BA and final BA each >=H0+1 percentage point; final tail BA >=H0+2.857142857 points (9 net correct prediction instances across seeds); at least 2/3 paired seeds improve in both final overall and final tail BA. Cross-dataset guard: neither ISIC nor HK loses more than 1 point in mean final BA or mean final tail BA. Report every seed/class and branch. These are exploratory resource/promotion gates, not significance or clinical claims. No lowering gates after failure; no dataset-specific variants or correction-strength scan.

## Budget and safety

Independent campaign, 4 hours from first actual execution, 6,000,000,000 bytes hard cap including all new state/logs/outputs, with 256 MiB pre-stop reserve. Expected peak <4GB; existing read-only data/caches excluded but any copies count. All datasets, subprocesses, repairs and retries share the original clock and storage ledger. Controller process group enforces time/storage independently of SSH. GPU0 only after free-memory check; CPU threads4; neutral visible entry and worker paths. Do not delete historical outputs or change original r03–r11/R12 budgets. All existing scientifically negative evidence retained.

## Checks before and during run

Synthetic duplicate-text loss/gradient equivalence to unique-class CE, unique-label no-op, permutation invariance, finite backward. Source replacement must match exactly once. Input cache unit norms, private restore parity, fixed parents/old adapters, full replay, actual-step projection, finite outputs, all heads sealed before validation. Archive loss multiplicity counters. Compare actual per-stage updates to H0. Publish only self-authored source, protocol and safe class aggregates; private checkpoints/scores/identities stay private.
