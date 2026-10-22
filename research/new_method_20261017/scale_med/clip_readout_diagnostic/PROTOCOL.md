# R0: saved-response readout diagnostic (2026-10-22)

Explicit user authorization: diagnose existing ordinary CLIP and BiomedCLIP predictions only. No new training; analysis before any subsequent decision. Project remains paused for training, and no automation is created.

## Inputs and scope

Read only the two existing responses.private.pt tensors, their PLAN_SEAL/MODEL_SEAL/config/aggregate JSON, and the shared S_old/E/Qgate rows metadata. Flat arms seeds74510/74511/74512 only; no selection among true/shuffled/flat or seeds. No images, backbone/adapters, feature cache, downloads, optimizer, image/text forward, ridge fit, or coefficient/prompt/temperature sweep. Execute CPU only with CUDA hidden. Outputs aggregate-only; no row identities or per-example tensors exported.

Compare NCM, recovered pure adapter and original fixed half mixture for E986/Qgate388 separately and pooled1374. Publish all13 classes, denominators, known-group counts, confusion, mean top1 confidence/true-class probability/margin, paired rescues/harms and BA contributions. Count all eight correctness triplets NCM/adapter/mix, including both-branches-wrong but mixture-correct. Cross-backbone comparisons use the exact common row binding and are conditional on the provenance audit. No clinical, causal, statistical-significance or method-success claim. No new tail subset chosen from performance; explicitly discuss tiny evaluation classes1/2/12 and all other classes.

## Provenance and numerical checks

Two configs must bind the same prior_root and class semantics; shared metadata counts/classes/groups/path disjointness checked. Recompute all stored score matrices' aggregate correct/confusion counts against original results, plus exact equality of the common OriginalF scores across the two response files. Responses have no embedded row identifiers: these checks and saved producer code provide conditional row-order provenance, not independent cryptographic proof of every row. No hash/full pixel recheck.

pN=softmax(saved NCM logits) in original CPU float32,4-thread numerical path; saved mix m is float32. Recover a64=2*m.double()-pN.double() without clamp or renormalization. This avoids another float32 subtraction but cannot recover rounding lost in original summation. Also inspect original float32 reconstruction and float64-softmax alternative. Predeclare absolute probability diagnostic envelope1e-6, normalization13e-6; this is a new engineering diagnostic envelope, not a relaxed prior model gate or a rigorous libm error guarantee. Record min/max,row-sum residual, original-mixture reconstruction residual and both path differences. Abort if envelope is violated; preserve observations. Mark adapter rows uncertain if top2 gap<=2e-6 or either sensitivity-path argmax disagrees. Report lower/upper correct-count bounds over classes within2e-6 of max and sensitivity argmax candidates; no silent clipping or dropping uncertain rows. Numerical mixing replay is algebraic consistency, not independent original-adapter parity. Paired/flow tables report how many rows depend on uncertain adapter decisions.

Support metadata audit explicitly checks prior singleton claims against PLAN_SEAL support_group_counts. Known identity_component groups do not establish patient independence; counts of genuinely new/unexposed groups are UNKNOWN, not inferred from disjointness. Historical E/Qgate remain exposed diagnostic cohorts.

## Resources and delivery

Single CPU diagnostic, total600 seconds from launch (TERM590/KILL5), CPU600 seconds,RSS2GiB; new allocation<16MiB including8MiB reserve. No persistent model or image copies. Existing source/response files remain unchanged. No repeated scoring with modified protocol after looking at results. Publish source/protocol before run, then report/aggregate/resource receipts through existing proxied public repository workflow; private rows/tensors remain on NFS. No follow-on training or monitor without new user instruction.
