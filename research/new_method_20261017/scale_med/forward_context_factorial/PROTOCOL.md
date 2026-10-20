# FORWARD_CONTEXT_FACTORIAL — independent engineering protocol

This is a forward-only numerical production study, not a retry of GROUP_GRADIENT_AGREEMENT (GGA). GGA remains retired, scientifically incomplete and its missing error remains NA. No old threshold is changed and no result here certifies or completes GGA. There is no performance candidate or automatic follow-on training.

## Question and estimands

On a new fixed set of 32 training groups, measure the separate observed effects of batch cardinality (32 vs 1) and gradient recording context (no_grad vs enable_grad) on the same original pretrained APART normalized dual-branch G1536 features. Both factors changed together in the old failed GGA comparison. All 102912 original bias parameters require gradients in every cell; all other parameters and native heads remain frozen. There is no backward, autograd.grad, loss, teacher, mapper fit, ridge solve, optimizer, or risk score. Gradient context alone is not a gradient measurement.

## Data and selection

Use only original initial2555 training rows and known group identifiers from the sealed producer PLAN. Exclude every group in GGA's sealed 64-group selection, not merely selected images. Sort the remaining group keys, shuffle once with torch seed74320, take exactly32 distinct groups, and choose each group's first original PLAN row. Save the private plan before GPU setup. No resampling. The old selection exhausts class1 groups, so this design cannot measure class1 or represent class-balanced risk. Report aggregate class/fold counts as selection metadata, not performance. No681 development/future/test/raw images; shared pixel container is filtered immediately by the original training row list. No learned encoder initialization or teacher reading.

## Fixed execution

Original pretrained initialization, eval mode, original pointwise adapter, same FP32 computation and nested float/double normalization as the sealed producer. Exactly four cells, in order B32_NG, B32_GRAD, B1_NG, B1_GRAD, with identical ordered rows:128 logical image forwards and66 calls. Outputs are detached promptly; no graph retained between individual calls. Persist completion receipts for every cell, then a FEATURE_SEAL before any pairwise comparison. Do not serialize/download feature matrices; keep them only in worker RAM.

Report every pair's max absolute difference and RMS and each cell versus cached original G on these new rows. Also report the difference-of-differences interaction residual. No comparison threshold or selection by a favorable comparison; no claim of bitwise reproducibility or unique cause from a nonzero difference. Persist numerical observations before invariant assertions. Parameters' version counters, all selected biases and native heads must remain exact. All four matrices must be finite; exactly128 logical image forwards are required.

## Acceptance and stop

ENGINEERING_FACTORIAL_COMPLETE requires synthetic arithmetic, all32 groups disjoint from old64, four finite sealed cells, all pair summaries, exact unchanged parameter checks, original count coverage, budget and worker/coordinator exit0. Otherwise ENGINEERING_FACTORIAL_UNRESOLVED with missing quantities NA, no automatic retry or tolerance change. method_success=false in all cases; no bootstrap or inference family, no class1 gradient conclusion or risk certification. Cache comparison is descriptive, never substituted for old GGA's missing observation.

## Independent hard budget and costs

One actual run including synthetic checks and setup:300 seconds from coordinator firstactual, TERM at290s, KILL after5s, no reset/retry; worker CPU hard300s. New storage strictly below128MiB including16MiB local/Git/helper reserve; each new file64MiB logical limit, RSS12GiB, old window17GB soft stop/20GB hard bound. Account max(logical, allocated) bytes. Expected remote growth<2MiB. Reuse pixels/weights/environment without copying, new cache/tmp only NFS. Independent coordinator checks time/RSS/storage every second. Admission verifies actual NFS mount/capacity/write-read probe and GPU0 free>=8192MiB. Neutral parent/worker argv and GPU process name verified once. Public source/protocol/aggregate/exit/cost/failure only; all identities and rows remain private NFS. Coordinator wall includes worker and waiting; worker wait4 CPU and coordinator own CPU each counted once. Preparation/network/reporting CPU remains unmeasured.
