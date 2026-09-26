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
