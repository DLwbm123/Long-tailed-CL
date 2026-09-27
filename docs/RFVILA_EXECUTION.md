# Execution record

Base: `b01ba12f86706c44dfcdbed81c7305493ce354dc`; branch `exp/nb2-rfvila-12h`.
Run: `NB2_RFVILA_R1_20260926T145517Z`.
Clock starts 2026-09-26 14:55:17 UTC; final deadline 2026-09-27 02:55:17 UTC. Historical ignore-time and multiple-GPU overrides do not apply.
The user explicitly assigned physical GPU 2. One NVIDIA A100-PCIE-40GB only.

The supplied plan is retained in `rfvila_plan/`. Production uses the existing frozen native APART/G/B loaders, independent float64 kernels and the supplied CPU oracle. All 32 oracle tests plus CPU/CUDA production parity passed; the complete 32-method synthetic report checks 1440 stage and 14688 class rows and rejects zero-gain utility. Synthetic results are not empirical evidence.

The original critical inventory was rechecked at both primary and independent NFS destination: 45 files, including six parents, shared AugReg core and G/B assets/locks. A real parent plus bank/W was streamed back from the independent copy and restored without image forward. All six parent loaders then passed. Image archive bytes are outside this claim.

Resource admission uses measured native-image extraction, 4096-dimensional solve/eigendecomposition and an actual 128 MiB independent backup. Methods are chosen solely from resource estimates, with 1.5 safety factor and the supplied phase cutoffs. Float32 encoders with TF32/AMP disabled; float64 R/features/moments/solve/CV/scores. No neural optimization.

Fit and evaluation use separate processes. Python audit hooks deny val scores/manifests, historical run caches and old/future images in fitting; 36 real negative-open tests passed. Current-task train arrays exist only in RAM and are released before the next task. These are application access controls, not an operating-system security boundary against hostile code. Evaluation reuses original APART caches only with verified source, preprocessing, parent, manifest, deterministic sorted row ordering, dtype and original SHA inventory. VLM val features are extracted once per encoder/dataset after the global fit lock.

Each stream stage saves W, metadata and S/Q atomically, with hashes. The local relay streams these directly between the two servers, verifies the written destination stream and returns an acknowledgement before the next task. Only the latest and previous S/Q per stream remain after successor backup; historical runs stay intact. Resume requires the same source/protocol/map hashes and prevents duplicate class append. Interrupted unsealed transactions fail closed and retain evidence for review.

A bounded controller samples heartbeat/resource usage every five minutes, enforces phase deadlines and can terminate only its own child process groups. The local backup relay saves an hourly status summary; both stop with this run. The previous hourly automation remains paused. Source/aggregate reports are committed locally at closeout. The current supplied plan requests commit-only unless explicitly authorized to push this round; no private arrays, identities, model files or PDF enter Git.

## Authorized continuation on 2026-09-27

The user explicitly requested continuation without the time budget. The original clock, source/protocol/map locks, 10 sealed stage locks and failure receipts are retained. An additive `RESUME_AMENDMENT.json` names every accepted legacy stage by hash, and `SOURCE_LOCK_RESUME.json` records the repaired execution code. Future stages reference the amendment and updated source. No model, method, random map, lambda, dataset, order or scoring formula changes.

The original local relay exited on SSH status failure; the fit worker waited for its missing backup acknowledgement until the original deadline. The replacement relay runs on the independent NAS server, reconnects after SSH failure, and can retry a lost acknowledgement without copying the same verified stage again. Its new private SSH key stays on that server in its SSH directory. The public authorization permits only this run's status/archive/ack operations and self-revocation; shell and forwarding are disabled. Successful closeout revokes the key and removes the task key files.

A controlled credential-unavailable fault was injected before resuming any fitting, followed by credential restoration and an actual small backup request. The resume test checks expired-budget rejection before explicit override, retained original time, acceptance of only hash-pinned legacy stages, and rejection of altered locks. The full synthetic report test also checks private traceback redaction. These operational tests do not count as medical results.

Completed ISIC 1993/1994 streams are verified and skipped; ISIC 1995 resumes from Task2's saved sufficient statistics and fixed Task1 lambdas, then proceeds to Task3. Large/private outputs remain in the existing remote-home run and independent NAS backup. Local report collection is independent of the learning/backup path and retries SSH status failures. Publication remains commit-only under this round's supplied prompt.
