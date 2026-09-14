# HyperKvasir23 Full Dynamic Branch Closed

## Status

- `FULL_DYNAMIC = CLOSED`
- `RAW_PROTOTYPE_FALLBACK = FAILED`
- `VALIDATION_CHECKPOINT_RESCUE = DIAGNOSTIC_ONLY`
- `NC_SAFE_DYNAMIC_V1 = FAILED`
- `ROOT_CAUSE = OLD_ANCHOR_DRIFT`
- `SECONDARY_FLAG = MIXED_GEOMETRY_AND_TRAJECTORY_FAILURE`
- `PROMOTION = NO_FULL_PROMOTION`

## Why The Branch Is Closed

Full Dynamic produced three collapse sessions in the original 15 incremental-session evaluation. Raw-prototype fallback improved seed1/session5 but did not clear the collapse predicate. Official-fold validation checkpoint selection rescued two targets, but seed1/session5 had no eligible epoch and therefore remained diagnostic-only evidence.

Zero-training forensics showed that seed1/session5 retained 87.5%-97.0% restricted-current accuracy while global current accuracy was suppressed by old-block absorption. The original head was already unit-normalized with shared implicit scale 1 and no bias, temperature, or old/current block scale. Unit-normalization CF1 was ineffective. CF5, which uses fixed NC old anchors with original dynamic current anchors, was the only effective head-only counterfactual.

The bounded `nc_safe_dynamic_v1` microcheck did not validate a repair. Although class9 absorption fell from 83 to 0 and seed1/session5 acquired a legal validation checkpoint, fresh-test current-to-old remained 31.71%, still above the original threshold. The stable control regressed from AccT 50.91 to 39.01 and HM 53.74 to 46.90. The repair branch is therefore closed.

## No More Local Sweeps

No threshold, alpha, or class9 sweep is justified. Class9 was a major absorber but not the sole cause: replacing or eliminating class9 absorption did not restore the global old/current boundary. Searching local thresholds after these results would tune against a failure already observed on test and would violate the frozen development protocol.

## Validation Selector Is Diagnostic Only

The selector uses an official-fold, current-session-only validation split. It can identify trajectory-sensitive epochs, but it has no old-class validation samples and cannot establish old/current global balance. A validation-eligible checkpoint can still fail the original test current-to-old predicate, as seed1/session5 demonstrated. It must not be promoted as a production checkpoint policy.

## Prototype Safety Is Not Sample Safety

The NC-safe gate constrains projected class centers, prototype assignment, and aggregate old-anchor margins. Classification operates on the full feature distribution. Tail samples, covariance spread, and competition from every old row can violate the sample-level boundary even when every prototype-level check passes. Local prototype safety therefore does not imply global sample safety.

## Recommended Mainline

Locked NC-ConCM is the recommended HyperKvasir23 mainline. Its fixed geometry avoids the old-anchor update path that caused Full Dynamic instability. Current test results are closed to further method development and must not be reused for threshold, alpha, class, epoch, or branch selection.

## Equivalence Audits

CF5 is `CF5_NOT_EQUIVALENT` to existing methods:

- Locked NC-ConCM uses fixed NC anchors for both old and current classes.
- Full Dynamic rebuilds both old and current anchors from all seen projected prototypes.
- `nc_anchored` combines normalized NC and dynamic logits for every class; it does not assemble one fixed-old/dynamic-current anchor matrix.
- LTConCM/TDSM regularizes or optimizes retained old anchors but does not use immutable NC old rows with original Full Dynamic current rows.

The existing DSM loss is `DSM_PARTIALLY_OVERLAPPING` with a fixed-old hard-margin objective. Both expose current features to correct-current and old-anchor relations, but DSM uses all-sample/all-anchor softmax contrast, detached dynamic anchors, no explicit hinge margin, and a different reduction. It is not mathematically equivalent to a hardest-fixed-old ReLU margin.

Relevant implementation paths:

- `src/methods/full_concm.py:299-340,343-415`
- `src/methods/concm_dsm.py:18-74,296-403`
- `tools/run_hyperkvasir_full_concm.py:287-295`
- `src/methods/ltconcm.py:227-285`
- `src/methods/gpa.py:243-270`

Because CF5 is not equivalent, a future independent branch may be discussed only as a genuinely new method under a new train/validation protocol. This result does not authorize implementation, test reuse, or further runs in the closed Full Dynamic repair line.
