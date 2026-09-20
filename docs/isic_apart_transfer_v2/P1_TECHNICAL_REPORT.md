# P1 technical report

Technical status: PASS. Formal resource status: BLOCKED_FORMAL_BUDGET. No formal training or test inference was launched.

| Branch | Session | Train images / batches | Training seconds | Memory update seconds | Checkpoint seconds | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| B | S0 | 96 / 2 | 11.248 | 0.002 | 7.400 | 8.397 |
| B | S1 | 96 / 2 | 4.975 | 0.002 | 7.533 | 8.418 |
| B | S2 | 96 / 2 | 7.528 | 0.003 | 7.503 | 8.446 |
| C | S0 | 96 / 2 | 4.992 | 5.933 | 7.441 | 8.420 |
| C | S1 | 96 / 2 | 7.121 | 4.180 | 7.877 | 8.419 |
| C | S2 | 96 / 2 | 6.905 | 4.923 | 7.622 | 8.419 |

P1 wall time, including remote serial image decoding: 17.11 minutes, below the 30-minute GPU allowance even as a conservative upper bound.

Validated: all 18,718 remote training images decode; manifest sample/label bijections and actual frequency lists for seeds 1993/1994/1995; deterministic cleaning conflict/unknown-bridge/priority/dedup cases; embedding old-range output equality, out-of-range rejection, new-row and pool gradients; unchanged global RNG; explicit optimizer parameter-update/LR regression; frozen strict loading and exact core-tensor mapping.

The real seed1993 B/C 4→6→8 smoke used at most one epoch and two trajectory batches per session. B/C complete initial and S0 network hashes match. All six real-batch image/label hashes and augmentations match across branches. Labels/count inputs do not change actual main/few evaluation logits. Evaluation is parameter/RNG-pure. Memory uses only the current train dataset and preserves global RNG. Synthetic features use the original sampling distributions on a separate stream.

Checkpoint reload reproduces evaluation logits and the next optimizer update bitwise. The separate discarded resume probes ran twice per session on two train-derived images; they are engineering probes, not additional smoke epochs or formal trajectories. No smoke model, optimizer or statistics may initialize formal runs.

Follow-up hardening explicitly binds checkpoints to manifest/protocol hashes, branch, seed and full resolved arguments. Positive roundtrip, five mismatch-negative cases, and completed-epoch resume with zero extra optimizer steps pass. The original smoke checkpoints and source snapshot remain intact; the follow-up converted a disposable engineering checkpoint in a new directory only. Core losses/updates were not changed by this metadata hardening. Source fingerprints are included in checkpoint_binding_engineering.json.

Synthetic evaluator fixtures validate unequal image/lesion weighting, old/current decomposition, empty groups, 36 session rows, 216 per-class rows and three-repeat paired sample SD (ddof=1). Fixture outputs are not experiment results.

The six trajectories contain 23,480 optimizer steps and 1,123,080 real-image presentations. Worst observed throughput + memory/checkpoint I/O + 25% reserve predicts 47.44 hours. Excluding only the first CUDA-training epoch from steady-state rate and amortizing its excess once per trajectory still predicts 32.33 hours. Both exceed the approved 24 hours.

Only the time budget remains pending. A 48 GPU-hour allowance covers the current conservative estimate; GPU3, seeds, class order, data, batch size, ten epochs/session and methods would remain unchanged. Neither estimate is a guaranteed completion time on a shared GPU.

Including both zero-training-step follow-up checks, the conservative combined P1 wall-time upper bound is 18.35 minutes, still below 30. The final unified CPU suite passes.
