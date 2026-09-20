# ISIC APART Transfer V1 — blocked outcome

**BLOCKED_DATA_PROTOCOL + BLOCKED_TRAINING_SEMANTICS**, with pretrained weights
also unlocated in checked paths. See the [executed audit](ISIC_APART_TRANSFER_V1_AUDIT.md).

| phase | result |
|---|---|
| P0 | Audit executed; known lesion leakage and count-embedding incompatibility found |
| P1 | Two audit-tool regression tests and original-component CPU preflight executed; full training acceptance NOT_RUN |
| P2 | 0/6 trajectories; 0/18 session checkpoints |
| P3 | 0/36 test-session metric records; 0 model test predictions |

All r1/r2/r3 B/C training units remain BLOCKED_NOT_STARTED. No zero-valued
performance rows were inserted for missing runs; no failed/low-performing repeat
was removed from an average. No performance average exists.

1. **Were all six trajectories completed?** No; none started because P0 blocked.
2. **M3–M0 Final_BA and tail deltas for r1/r2/r3?** All unavailable.
3. **Training/calibration/interaction contribution?** Not evaluable.
4. **Old-class absorption, current-class confusion, or tails never learned?** Not evaluable without trained checkpoints and the locked test batch.
5. **New zero-recall classes, label dependence, split/weight overlap risks?** Recall and full-model label blindness are untested; 8 identical-content groups and 693 known lesions cross split boundaries; 2 content groups have conflicting labels; exact historical weight provenance/overlap is unknown.
6. **One next action?** Resolve the V1 protocol gate with the user before further implementation or training; do not launch another method or split automatically.

The technical blockers do not constitute evidence for or against medical transfer
performance of APART or ConCM-lite. GPU3 availability cannot override data or
method-definition gates explicitly adopted by the user.
