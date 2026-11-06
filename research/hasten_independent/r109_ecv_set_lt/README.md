# R109: ECV-SET-LT engineering implementation

Implements a query-free, class-equivariant finite-memory selector around the existing R107 learner: evidence diagnostics → spatial-risk candidates → paired actual execution → dynamics-assisted verification → near-optimal supervised initialization → optional paired-reference PPO.

**Status: engineering checks completed; method efficacy untested.** See TEST_REPORT.md, DEVIATIONS.md and CONTRACT_LOCK.json. Original R106–R108 outcomes remain closed. Monitor OFF; no P1–P4 experiment has started.

The host is five phases, AdamW, and 256 global/local feature tokens. New candidates preserve host coverage/parent constraints and each anchor's class/global-local quotas. No patient identity, future batch, reward query or query label is a trainable state input. Missing group evidence remains missing. Verification and policy probability are separated from actual reward execution; PG never substitutes predicted reward for executed outcomes.

Scientific development must still establish meaningful decision opportunity and feedback resolution under the R108 findings. The existing engineering scripts do not establish clinical independence, long-horizon benefit on real data, superiority to ZERO/BALANCED, or an RL-specific gain over MORE_SUP.

Source mapping, runnable engineering commands and non-executable core matrix draft are included. Raw images, original identifiers, feature tensors, model/optimizer weights, third-party PDFs and private operational records are excluded from public delivery.

