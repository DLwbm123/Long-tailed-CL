# P0 objective diagnosis

18 locked V2 checkpoints evaluated without updates on image-ID-sorted train/val, batch 48. Train access is offline_forensics only; no statistics feed P2.

Gradient probes use actual real-image main/few paths and original synthetic heads. The derivative identities, per-component raw/weighted gradients, totals, row norms and cosines are in gradient_probe.json. These local derivatives do not alone establish the cause of end-to-end collapse.

Batch peer and order dependence is reported in batch_context_probe.json, without changing routing. Native parity uses final normalized CLS and zeroed adapter up projections on a disposable copy.

Memory is extracted after each session using the current training dataset, including its random crop/flip; a separate seed and restored RNG isolate extraction. Old mean/variance entries are retained rather than updated. Main/few Gaussian features are independently sampled. Norm/logit/margin distributions are included in probes and per-checkpoint reports.

New test predictions: 0. V2 COMPLETE_MIXED_SIGNAL remains unchanged.
