# CT3-P workspace and scope

Independent worktree q8m22, branch exp/ct3p-memory-compatibility, based on CT2-D delivery 7e3542d153cc77dcc2bcb4d14b6bc2f8f1b3a7cb. The q8m21 checkout was clean when inspected and was not changed. The established non-git project workspace is preserved. Original CT1 source, checkpoints and results are read-only.

The sole APART modification is an optional post-real-loss hook. Absent the hook, the CT1 loss, CE routing, reductions, /3, optimizer and sampling path are unchanged. The new student subclass adds the fixed pointwise FD term. Teacher state is independent of the synchronizing inference probe. A/T banks do not feed training.

FD_EARLY_GATE and FD_THROUGHPUT_GATE are discarded engineering copies, totaling five optimizer steps. All inputs are current HK/1993 Task2 images. The early paths made 20 wrapper / 60 internal encoder calls and decoded 96 current images; the per-gate inherited counters cover CE calls only, so these supplementary FD/probe counts are carried separately into the main ledger.

No original Run/formal entry is invoked. Parent fork calls the unmodified ordinary CT1 restore validator under its original code/args contract; the child then uses a separate code hash contract and dedicated restore for its own checkpoints. Task2 and Task3 keep the original head capacity and remapped labels. New checkpoints retain nonshared parameters, A/T banks, optimizer/scheduler, RNG and teacher provenance. The rolling epoch resume additionally retains immutable anchors and teacher delta.

New checkpoints are transferred to the existing my-gpu remote-home archive, verified by SHA, readable tensor keys and delta digest. Only verified new local temporary task copies and this run's temporary caches are removed. The archive agent never reads any CT1 checkpoint outside U Task1–3.

Online prediction release precedes a distinct offline process. This process reads only seed1993's arrived six classes for four Q11 fits and cannot write original assets. All scores remain private; only aggregate results are public. No test access, hourly monitor, hyperparameter search or post-prefix training is authorized.

Final I/O admission uses the already-installed zstandard codec at fixed level 3 (four compression threads). The transferred Task1 checkpoint was decoded and matched its original SHA; no tensor conversion occurs. Reuse of the six already-verified Task1 parents plus identity/model/feature-locked baseline validation caches avoids 24 redundant checkpoint fetches. Those validation caches are used only for baseline reproduction until the complete state/W lock; new candidate scores are computed in the subsequent evaluation phase. New parent cache copies are removed after their corresponding fork, and are included in the active/persistent storage guard.

The exact prefix metadata yields 9,452 probe batches. Conservative admission includes 6,620 measured full training steps, prefix probes, scaled compressed checkpoint transfers, all prior measured GPU residence, and 900 seconds miscellaneous reserve: 13,476.272 seconds (<14,400). This is an estimate, not guaranteed completion; the cumulative hard timeout remains enforced. One expired SSH multiplex channel during engineering was replaced with a task-local channel using keepalives; no server/proxy configuration was changed.
