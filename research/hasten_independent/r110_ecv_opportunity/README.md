<!-- R110_RESULTS_OVERVIEW -->
**P1结果已完成并核验：Ordered未通过、分层order0通过，联合门未通过；未启动P2/RL。**

[中文实验结果总览](RESULTS_OVERVIEW_ZH.md)集中列出实际执行范围、主结果、逐阶段结果、每项门槛和机制对照。正式VAL/TEST尚无新结果。

<!-- /R110_RESULTS_OVERVIEW -->

# R110 ECV opportunity/query pilot

Prospectively frozen P1-only scientific diagnostic. Read PROTOCOL.md and CONFIG.json. The base package is the previously delivered R109 implementation, copied as source only. Original R106–R109 outcomes are unchanged. No method efficacy or RL-specific gain is established by this source release.

Uses two task flows,32 predefined scenario slots,256-token memories, at most16 legal candidates, full100/200/100 shadow updates, and two paired continuation tapes. Larger queries reuse identical branches. Structural reconstructions have exact per-class/global-local quota and edit-magnitude matched random controls. A fresh generic query-disjoint shadow classifier is distinguished from the legal ZERO donor snapshot.

`main.py` is a machine-local neutral launcher reading Q_SOURCE/Q_MODULE/Q_CONFIG from the environment. Source modules are `controller`, `worker`; child arguments contain only a neutral Python/entry path. The native barrier must admit the entire fixed P1 before science. No P2/RL/VAL/TEST or monitoring task is started by this pilot.

Clinical patient independence is unverified. Scenarios and tapes are not independent patients. H=3 is a legal shadow sequence within already arrived TRAIN, not three true unseen future stages. Saved-only verification and influence analysis remain necessary before final scientific delivery.

